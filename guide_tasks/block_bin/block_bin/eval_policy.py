"""Shared core of the block_bin policy evaluation: what ``eval_policy_pink`` builds on.

Policy loading, the action-chunk stream (with opt-in real-time chunking), robot and
camera connection, the joint_command publisher, success checks, the stop service and
the episode/zone plan. The joint-space rollout that used to live here, and the MoveIt
Servo sibling, were discontinued; ``eval_policy_pink`` is the evaluation entry point.
"""

import dataclasses
import json
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import draccus
import lerobot.policies  # noqa: F401  -- registers the policy configs (smolvla, ...)
import numpy as np
import torch
from draccus.utils import DecodingError
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

try:  # moved out of lerobot.utils.utils in lerobot 0.6.0
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:  # pragma: no cover -- lerobot <= 0.4.x
    from lerobot.utils.utils import get_safe_torch_device

from block_bin.solve_task import scene_num_zones
from guide_core.types.randomization import zone_plan
from guide_msgs.srv import CheckSuccess

# Dataset image keys (init.yaml `dataset.images`) -> the /cam_* topic each one was
# rendered from. Camera names double as the policy's `observation.images.<name>`.
CAMERAS = {"top": "cam_top", "base": "cam_base", "wrist": "cam_wrist"}

# The gripper is a single dataset dimension (`gripper.pos` == fr3_finger_joint1),
# but the sim's articulation needs both fingers driven, so finger 2 mirrors it.
GRIPPER_MIRROR_JOINT = "fr3_finger_joint2"

# Arm home pose, published as a joint_command at the start of every episode.
# GRIPPER_OPEN matches solve_task's OpenGripper, so a block still held from a failed
# episode is dropped before the scene is re-randomized.
#
# This is the scene's `default_joint_states` (config/init.yaml, and the same vector in
# config/reset.yaml's set_joint), and also the pose
# every demonstration starts and ends at -- solve_task's `rest_pose`, position
# [0, 0, 0.5] with orientation R.from_euler("xyz", [pi, 0, 0]), which its Unclutch
# subtask moves to first and its retreat returns through. Those are the same pose, not
# two: FK of this vector puts fr3_hand_tcp at [0.30689, 0.0, 0.48688] in fr3_link0,
# which is [0.00022, 0.00003, 0.49983] once the measured base offset
# (-0.30667, 0.00003, 1.01295) and the scene's 1 m origin are applied -- 0.17 mm from
# rest_pose, with a rotvec of [-pi, 0, 0] against the dataset's [3.14101, -0.0001,
# -0.001] (same rotation, opposite sign convention). Homing joint-space therefore lands
# exactly where the demonstrations begin.
#
# What that equality depends on is the arm's placement in the scene. Re-mount the robot
# or move Scene_i and the joint vector still reaches the same pose in fr3_link0 while
# rest_pose moves with the scene, and the two silently stop agreeing. Re-measure against
# the dataset's final states (they are all homed) rather than assuming.
HOME_POSITION = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
GRIPPER_OPEN = 0.04

# Commanded gripper position below which the policy is asking for a CLOSE. The
# demonstrations close to 0.01 and open to GRIPPER_OPEN, so halfway between separates
# the two intents without depending on the exact closing value. Used to date the grasp
# attempt: "never asked to close" and "asked at step 40 and caught nothing" are
# different failures and the success rate shows neither.
GRIP_CLOSE_COMMAND = 0.02

# Sim seconds to let the scene settle before the episode's final success check.
SETTLE_SECONDS = 1.0


def state_names(robot) -> list[str]:
    """Joint order behind `observation.state` / `action`.

    The curated training set (``meta/info.json``) names both as
    ``[joint1.pos ... joint7.pos, gripper.pos]``, i.e. the 7 arm joints followed by
    the gripper -- never index a dataset column by position, always by this order.
    """
    return list(robot.config.arm_joint_names) + list(robot.config.gripper_joint_names)


def images_from(observation: dict, mapping: dict | None = None) -> dict:
    """The `observation.images.*` half of a frame -- shared with eval_policy_servo.

    ``mapping`` is {key the policy expects: GUIDE camera}, because a checkpoint trained
    elsewhere names its cameras its own way and may want fewer of them. Defaults to
    GUIDE's own three, keyed by their own names.
    """
    frame = {}
    for key, cam in (mapping or {name: name for name in CAMERAS}).items():
        image = observation.get(cam)
        if image is None:
            # connect_robot proved every camera was delivering before the run started,
            # so this is a stream that STOPPED -- the simulator went down, or its camera
            # graph did. Not the config: pointing at `publish_camera_topics` here sent a
            # previous debugging session to edit the one switch that also slows dataset
            # generation down, and it was never the cause of a mid-run drop-out.
            raise RuntimeError(
                f"Camera '{cam}' ({CAMERAS.get(cam, cam)}) stopped publishing "
                f"mid-episode; it was delivering at start-up. Is the simulator "
                f"still running?"
            )
        frame[f"observation.images.{key}"] = image
    return frame


def connect_robot(robot, timeout: float = 20.0) -> None:
    """Connect, and refuse to start until the first policy frame can actually be built.

    ROS2Camera.connect warns and carries on when a topic stays silent, which is right
    for the recorder -- it captures GUIDE-side through annotators and does not need the
    topics at all. It is wrong here. Every frame handed to the policy REQUIRES all three
    images, so a silent camera is not a degraded evaluation, it is one that cannot take
    a single step.

    Left to `images_from`, that surfaces as a traceback a minute in, after the 450M
    parameters have loaded, the arm has homed and the scene has randomized -- which
    reads like a bug in the evaluation rather than a simulator launched without the
    camera graphs. Under the sweep it is worse: every checkpoint pays the same minute
    to die the same way, and the run ends with an empty report.

    Also replaces a blind `time.sleep(5)`: waiting for the observation the first step
    needs is both the honest check and the exact thing that sleep was guessing at.
    """
    robot.connect()

    joints = state_names(robot)
    deadline = time.perf_counter() + timeout
    while True:
        observation = robot.get_observation()
        blind = [CAMERAS[name] for name in CAMERAS if observation.get(name) is None]
        deaf = [joint for joint in joints if f"{joint}.pos" not in observation]
        if not blind and not deaf:
            break
        if time.perf_counter() >= deadline:
            # Unwind the executor first. Raising through a live ROS2Robot leaves its
            # spin thread scheduling onto an executor the interpreter is tearing down,
            # and the ~70 lines of "cannot schedule new futures after shutdown" that
            # follow push the explanation off the top of the terminal -- which is the
            # entire thing this function exists to prevent.
            robot.disconnect()
            raise SystemExit(complaint(blind, deaf, timeout))
        time.sleep(0.2)

    print("Connected to robot and cameras.")


def complaint(blind: list[str], deaf: list[str], timeout: float) -> str:
    """What is missing from the first observation, and what to do about it."""
    lines = [f"Nothing to evaluate on after {timeout:.0f}s waiting for the first observation."]
    if blind:
        lines += [
            f"  No image on: {', '.join(blind)}",
            "  Isaac publishes the camera topics only when asked. They default to OFF",
            "  because demonstration generation never reads them and they cost a second",
            "  render product per camera plus ~166 MB/s of raw rgb8 over localhost DDS:",
            "      ros2 launch guide_core bringup.launch.py camera_topics:=true",
            "  (or `publish_camera_topics: true` in the task's config/init.yaml, which",
            "  turns them on for dataset generation too -- usually not what you want.)",
        ]
    if deaf:
        lines += [
            f"  No joint_states for: {', '.join(deaf)}",
            "  Check the scene is running and --namespace names it.",
        ]
    return "\n".join(lines)


def command_from_action(action, joints: list[str]) -> tuple[list[str], list[float]]:
    """Policy action tensor -> (joint names, positions) for the joint_command topic."""
    values = [float(v) for v in np.asarray(action).reshape(-1)]
    if len(values) != len(joints):
        raise ValueError(f"Policy returned {len(values)} dims, expected {len(joints)}: {joints}")

    names, positions = list(joints), list(values)
    names.append(GRIPPER_MIRROR_JOINT)
    positions.append(positions[-1])
    return names, positions


def publish_command(robot, names: list[str], positions: list[float]) -> None:
    message = JointState()
    message.header = Header()
    message.header.stamp = robot.node.get_clock().now().to_msg()
    message.header.frame_id = robot.config.frame_id
    message.name = names
    message.position = positions
    robot.joint_command_pub.publish(message)


def add_interpolation_arguments(parser) -> None:
    """Opt-in sub-step interpolation, shared by the eval scripts."""
    parser.add_argument(
        "--interpolate",
        type=int,
        default=2,
        help="Setpoints published per control step. 1 is the old behaviour: one target "
        "per period, which the arm jumps to and then waits out for ~200 ms. Higher "
        "values ramp toward the same target (LERP position, SLERP rotation, IK per "
        "sub-pose). Decision points are unchanged -- only the path between them. "
        "Default 2: measured free, 1 to 12 setpoints spans 31.7-32.2 ms/step with RTF "
        "flat at 0.52, so the only reason not to smooth is if it hurts the policy.",
    )


def is_success(robot, scene_id: int) -> bool:
    """Ask the scene whether its success criterion holds RIGHT NOW.

    The message is logged rather than dropped because the sim reports a criterion
    that came back false and a criterion that RAISED identically, as
    ``success=False`` -- ``guide_ros._is_success_callback`` catches the exception
    and puts the reason in ``message``, nowhere else. Discarded, a broken query (a
    prim renamed, a scene whose randomize never ran, a stage still loading) is
    indistinguishable in the log from a policy that simply missed, and the whole
    run scores 0/N with nothing to explain it.
    """
    response = robot.callService(robot.is_success_client, CheckSuccess.Request(id=scene_id))
    if not response.success and response.message:
        robot.node.get_logger().warn(f"IsSuccess could not be evaluated: {response.message}")
    return bool(response.success)


def final_success(robot, scene_id: int) -> bool:
    """Grade the episode once the scene has stopped moving.

    The criterion is a bounding-box containment test read off the poses PhysX last
    wrote back, and the last action of a successful rollout is the one that RELEASES
    the block -- so at the instant the control loop exits, the block is still falling
    into the bin and is not contained by it yet. The in-loop poll cannot cover this
    either: it only ever runs mid-motion. Without a settle the one check that could
    see a place made at the end of the rollout reads the scene a few milliseconds too
    early, and a completed task scores as a failure.

    ``solve_task`` waits the same second before grading a demonstration, with the
    same comment; this is that wait, on the sim clock the rollout is already paced by.
    """
    sleep_sim(robot, SETTLE_SECONDS)
    return is_success(robot, scene_id)


class RunControl:
    """Operator flags, set from the stop service while a rollout is running."""

    def __init__(self):
        self.abort_episode = threading.Event()
        self.stop_run = threading.Event()


def handle_stop(request, response, control: RunControl, logger):
    """`std_srvs/SetBool`: abort the running episode, `data` decides whether to go on.

    An aborted episode is still scored, so a rollout the operator kills because the
    policy has clearly missed counts as the failure it is, rather than vanishing
    from the denominator.
    """
    control.abort_episode.set()
    if request.data:
        control.stop_run.set()
        response.message = "Aborting this episode and ending the evaluation."
    else:
        response.message = "Aborting this episode; continuing with the next."
    logger.warn(response.message)
    response.success = True
    return response


#: Prefix-attention schedules RTCConfig accepts (lerobot.configs.RTCAttentionSchedule).
RTC_SCHEDULES = ("LINEAR", "EXP", "ONES", "ZEROS")


def add_rtc_arguments(parser) -> None:
    """Opt-in real-time chunking flags, shared by the eval scripts.

    RTC treats each new chunk as an inpainting problem against the previous chunk's
    unexecuted tail, using prefix attention, so the join is continuous instead of a
    jump. That jump is what makes the motion arrive in visible steps at every replan:
    ``ChunkStream._absorb`` currently splices a fresh chunk onto a trajectory it knows
    nothing about.

    It changes only how chunks are stitched, never the action space -- delta positions
    are exactly what they were, which is the point.

    Off by default. Only flow-matching policies carry the hook (smolvla, pi0, pi05,
    pi0_fast, evo1, groot, molmoact2, xvla); anything else falls back to the current
    behaviour with a warning rather than an error.
    """
    parser.add_argument(
        "--rtc",
        action="store_true",
        help="Real-time chunking: inpaint each chunk against the previous one's tail.",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="RTC execution horizon, in actions. Defaults to lerobot's 10.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help="Upper bound on the inpainting guidance weight.",
    )
    parser.add_argument(
        "--rtc-prefix-attention",
        type=str,
        default="LINEAR",
        choices=RTC_SCHEDULES,
        help="How prefix weight decays across the overlap. LINEAR is lerobot's default.",
    )


def rtc_config_from_args(args):
    """An ``RTCConfig`` when --rtc was passed, else None. Imported lazily: the module
    lives under lerobot.policies and pulling it in costs nothing when RTC is off."""
    if not getattr(args, "rtc", False):
        return None

    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    return RTCConfig(
        enabled=True,
        execution_horizon=args.rtc_execution_horizon,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention),
    )


class ChunkStream:
    """Feeds the control loop actions without the arm stalling on every re-plan.

    Predicting a chunk costs 2-3 control periods (measured ~0.5 s against a 0.2 s
    period at 5 Hz, and no knob removes it -- even 2 denoising steps under AMP still
    costs a full period). Predicting only once the queue has run dry therefore
    freezes the arm at every chunk boundary, which is what makes the motion arrive
    in bursts. So the next chunk is predicted on a worker thread as soon as the
    queue is down to `lead` actions, and the actions whose control slots elapsed
    while it computed are dropped -- the arm is never handed a command from the past.

    `lead=0` restores the old blocking behaviour: predict only when nothing is left.
    """

    def __init__(
        self, policy, preprocessor, postprocessor, device, task, robot_type, lead: int, rtc=None
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.task = task
        self.robot_type = robot_type
        self.lead = lead
        self.replans = 0
        # How often RTC actually had a tail to inpaint against, and how long it was.
        # At short horizons the queue can drain during inference, in which case the
        # prediction is plain and "RTC on" did nothing for that chunk -- so a run has
        # to say how much of it was RTC, not just whether the flag was set.
        self.rtc_engaged = 0
        self.rtc_tail_total = 0
        self._queue = deque()
        # The same actions in the policy's own (pre-postprocessor) space. RTC inpaints
        # against the trajectory the model produced, not the unnormalised one the arm
        # is commanded with, so the two have to be kept in lockstep -- extended and
        # popped together, always.
        self._raw_queue = deque()
        self._pending = None  # (future, step whose observation it was launched from)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.rtc = self._attach_rtc(rtc)

    def _attach_rtc(self, rtc):
        """Turn RTC on for this policy, or explain why it stays off."""
        if rtc is None:
            return None
        if not hasattr(self.policy, "init_rtc_processor") or not hasattr(
            self.policy.config, "rtc_config"
        ):
            print(
                f"--rtc ignored: {type(self.policy).__name__} has no real-time chunking "
                f"hook. Chunks will be spliced as before."
            )
            return None
        self.policy.config.rtc_config = rtc
        self.policy.init_rtc_processor()
        print(
            f"RTC on: horizon {rtc.execution_horizon}, "
            f"{rtc.prefix_attention_schedule} prefix attention, "
            f"max guidance {rtc.max_guidance_weight}."
        )
        return rtc

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _rtc_tail(self):
        """Queued actions in the policy's own space, as (batch, time, dim), or None.

        This is what the arm will still be executing while the next chunk is computed,
        and what RTC inpaints the new chunk to continue from.
        """
        if self.rtc is None or not self._raw_queue:
            return None
        return torch.stack(list(self._raw_queue), dim=1)

    def _predict(self, frame: dict, prev_tail=None) -> tuple[list, list]:
        """One action chunk, truncated to the horizon, in both spaces."""
        kwargs = {}
        if self.rtc is not None and prev_tail is not None:
            # inference_delay is how many of those queued slots elapse before the chunk
            # lands. `lead` is that number by construction: the prediction is launched
            # exactly when the queue is down to `lead` actions.
            kwargs = {
                "prev_chunk_left_over": prev_tail,
                "inference_delay": self.lead,
                "execution_horizon": self.rtc.execution_horizon,
            }
        # no_grad rather than inference_mode when RTC is on: its guidance correction
        # is an autograd.grad through the denoiser, wrapped in torch.enable_grad(), and
        # enable_grad cannot lift inference_mode -- tensors created there are barred
        # from autograd for good ("element 0 of tensors does not require grad").
        # inference_mode stays the default: it is the cheaper guard.
        with (
            torch.no_grad() if self.rtc is not None else torch.inference_mode(),
            (
                torch.autocast(device_type=self.device.type)
                if self.device.type == "cuda" and self.policy.config.use_amp
                else nullcontext()
            ),
        ):
            batch = prepare_observation_for_inference(
                frame, self.device, self.task, self.robot_type
            )
            raw = self.policy.predict_action_chunk(self.preprocessor(batch), **kwargs)
            chunk = self.postprocessor(raw)
        horizon = min(self.policy.config.n_action_steps, chunk.shape[1])
        return [chunk[:, i] for i in range(horizon)], [raw[:, i] for i in range(horizon)]

    def _absorb(self, result: tuple, launched_at: int, step: int) -> None:
        """Queue a chunk, skipping the actions whose control slots have already passed."""
        chunk, raw = result
        offset = (step - launched_at) + len(self._queue)
        self._queue.extend(chunk[offset:])
        self._raw_queue.extend(raw[offset:])

    def next_action(self, frame: dict, step: int):
        if self._pending is not None and self._pending[0].done():
            future, launched_at = self._pending
            self._pending = None
            self._absorb(future.result(), launched_at, step)

        if not self._queue and self._pending is not None:
            # Nothing left to execute, so there is no point running ahead: block on
            # the chunk already in flight instead of starting a second one.
            future, launched_at = self._pending
            self._pending = None
            self._absorb(future.result(), launched_at, step)

        if not self._queue:  # first step of the episode, or the chunk arrived fully stale
            self.replans += 1
            self._absorb(self._predict(frame), step, step)
        elif len(self._queue) <= self.lead and self._pending is None:
            self.replans += 1
            tail = self._rtc_tail()
            if tail is not None:
                self.rtc_engaged += 1
                self.rtc_tail_total += int(tail.shape[1])
            self._pending = (self._pool.submit(self._predict, frame, tail), step)

        if self._raw_queue:
            self._raw_queue.popleft()  # kept in lockstep with _queue
        return self._queue.popleft()

    def plan(self) -> list:
        """The actions still queued -- the trajectory ahead of the step just issued.

        Read after ``next_action``, so it excludes the action being executed now.
        This is the policy's INTENT, which is the half a joint-angle log cannot show:
        an arm in the wrong place because the policy aimed there and an arm in the
        wrong place because the IK could not follow look identical in the measurement
        and want opposite fixes.
        """
        return [np.asarray(action).reshape(-1).tolist() for action in self._queue]


def sleep_sim(robot, seconds: float, stall_timeout: float = 30.0) -> None:
    """Block until the ROS clock has advanced ``seconds``.

    The node runs with ``use_sim_time=True``, so this is SIM time. That is the
    baseline that matters: the demonstrations were sampled every 12 world steps at
    ``step_freq: 60``, i.e. 5 Hz of simulated time, and Isaac does not run at
    real time once three cameras are rendering and a VLA is competing for the GPU.
    Pacing on the wall clock would hand the arm a different number of simulated
    seconds per action than it saw in training. The wall-clock guard means a
    stopped or crashed sim raises instead of hanging the run forever.
    """
    clock = robot.node.get_clock()
    target = clock.now() + Duration(seconds=seconds)
    wall_deadline = time.perf_counter() + seconds + stall_timeout
    while clock.now() < target:
        if time.perf_counter() > wall_deadline:
            raise TimeoutError(
                f"ROS clock advanced < {seconds}s in {seconds + stall_timeout}s of wall time; "
                f"is the simulation running (and publishing /clock)?"
            )
        time.sleep(0.002)


def report_rollout(robot, args, policy, step_times, grips, replans) -> None:
    """Loop timing and grasp telemetry for the episode just finished.

    Median rather than any-step is the honest measure: re-planning an action chunk
    costs ~0.5 s while the other steps just pop a queued action for ~10 ms, so a
    per-step warning would fire on every re-plan and mean nothing. The gripper line
    separates the two ways a grasp fails -- never commanding a close (bad
    trajectory) versus commanding it and the fingers not stalling on the block.
    """
    if not step_times:
        return

    median = sorted(step_times)[len(step_times) // 2]
    log = robot.node.get_logger()
    log.info(
        f"Loop: {len(step_times)} steps, median {median * 1000:.0f} ms, "
        f"max {max(step_times) * 1000:.0f} ms, {replans} chunk predictions "
        f"(horizon {policy.config.n_action_steps}, lead {args.lead})"
    )
    # eval_policy_servo derives a scaled period from --time-scale; this script has no
    # such knob and its budget is just the control rate.
    budget = getattr(args, "period", 1.0 / args.fps)
    if median > budget:
        log.warn(
            f"Median step {median * 1000:.0f} ms exceeds the {budget * 1000:.0f} ms "
            f"budget of {1 / budget:g} Hz -- the arm is being driven slower than the "
            f"demonstrations were recorded. Raise --n-action-steps so the policy is "
            f"called less often (each re-plan costs several periods, so a short horizon "
            f"spends most of the rollout predicting), or move inference to a GPU Isaac "
            f"is not using. Do NOT assume cuda:N matches nvidia-smi's numbering: with "
            f"CUDA_DEVICE_ORDER unset the runtime sorts fastest-first, so the indices "
            f"can be reversed. Check torch.cuda.get_device_name(N) first."
        )

    commanded = min(g[0] for g in grips)
    measured = min(g[1] for g in grips)
    log.info(
        f"Gripper: commanded min {commanded:.4f}, measured min {measured:.4f} "
        f"(fingers stalling near the cube half-width means a grasp; closing to ~0 "
        f"means the gripper shut on nothing)"
    )


def episode_plan(zone_spec: str, episodes: int) -> list:
    """One target zone per episode; ``None`` means an unrestricted draw.

    `--zone` speaks the same vocabulary as demonstration generation
    (`solve_task.zoned_request`), so a stratified eval mirrors the stratified
    dataset it is scoring: omitted is a free draw over the whole region, `all` is
    every cell of the scene's grid, `2,16` restricts to those cells, and `2:4,16:10`
    sets per-zone counts. Without explicit counts `--episodes` is PER ZONE, matching
    `all_zones_request(count)` meaning count demos in every zone.
    """
    if not zone_spec:
        return [None] * episodes

    num_zones = scene_num_zones()
    if zone_spec == "all":
        return zone_plan([-1], [episodes], num_zones)

    zones, counts = [], []
    for item in zone_spec.split(","):
        zone, _, count = item.partition(":")
        zones.append(int(zone))
        counts.append(int(count) if count else episodes)

    invalid = [z for z in zones if not 0 <= z < num_zones]
    if invalid:
        raise SystemExit(f"Zones {invalid} are outside this scene's grid (0..{num_zones - 1}).")
    return zone_plan(zones, counts, num_zones)


def load_config(path: str):
    """A checkpoint's policy config, tolerating fields a newer LeRobot wrote.

    `PreTrainedConfig.from_pretrained` decodes strictly, so a checkpoint trained
    against a newer LeRobot dies on fields this one never had (`pretrained_revision`
    at the time of writing). Those extras are hub metadata, not architecture, so drop
    them -- but say which, because a dropped field that DID matter would otherwise
    change behaviour silently.
    """
    try:
        return PreTrainedConfig.from_pretrained(path)
    except DecodingError:
        if not Path(path).is_dir():
            raise

    saved = json.loads((Path(path) / "config.json").read_text())
    config_class = PreTrainedConfig.get_choice_class(saved["type"])
    known = {field.name for field in dataclasses.fields(config_class)}
    ignored = sorted(set(saved) - known - {"type"})
    print(f"Config fields unknown to this LeRobot, ignored: {ignored}")

    with tempfile.NamedTemporaryFile("w+", suffix=".json") as trimmed:
        json.dump({k: v for k, v in saved.items() if k in known}, trimmed)
        trimmed.flush()
        with draccus.config_type("json"):
            return draccus.parse(config_class, trimmed.name, args=[])


def load_policy(path: str, device: str | None):
    """Load a checkpoint plus the pre/post-processing pipelines saved next to it.

    `make_policy` is deliberately not used: it insists on dataset metadata or a
    gym env to derive the features, while a trained checkpoint already carries
    them in its own config.
    """
    config = load_config(path)
    config.pretrained_path = path
    if device:
        config.device = device

    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    # Normalization stats live in the saved preprocessor, so no dataset is needed.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    return policy, preprocessor, postprocessor


# Names of the LIBERO state channels, for naming the one that is out of range.
STATE_CHANNELS = ("x", "y", "z", "wx", "wy", "wz", "gripper.q1", "gripper.q2")


def state_stats(policy_path: str) -> dict | None:
    """The ``observation.state`` normalisation THIS CHECKPOINT was trained with.

    Read out of the checkpoint, never off a dataset on disk. The checkpoint is the only
    record of what the policy actually saw, and it survives the dataset being re-framed
    underneath it -- which is precisely the failure this exists to catch.
    """
    files = sorted(Path(policy_path).glob("*normalizer*.safetensors"))
    if not files:
        return None
    try:
        from safetensors.torch import load_file

        tensors = load_file(str(files[0]))
    except Exception:
        return None
    stats = {}
    for key in ("min", "max", "mean", "std"):
        tensor = tensors.get(f"observation.state.{key}")
        if tensor is not None:
            stats[key] = [float(v) for v in tensor.flatten()]
    return stats or None


def out_of_distribution(state, stats: dict, slack: float = 0.1) -> list:
    """Channels of this state outside the range the checkpoint was trained on.

    Returns (index, value, low, high, sigma) per offending channel. ``slack`` allows a
    margin of the trained range, because the edge of a training set is not a wall.
    """
    low, high = stats.get("min"), stats.get("max")
    if not low or not high:
        return []
    mean, deviation = stats.get("mean"), stats.get("std")
    offending = []
    for index, value in enumerate(np.asarray(state).reshape(-1)[: len(low)]):
        span = high[index] - low[index]
        if low[index] - slack * span <= value <= high[index] + slack * span:
            continue
        sigma = (
            (value - mean[index]) / deviation[index]
            if mean and deviation and deviation[index]
            else float("nan")
        )
        offending.append((index, float(value), low[index], high[index], float(sigma)))
    return offending


def check_state_distribution(state, stats: dict | None, base_offset=None) -> None:
    """Refuse to evaluate on a state the policy has never seen anything like.

    The single most expensive class of bug in this pipeline: a frame or offset that is
    wrong by a constant loads fine, runs fine, drives the arm to plausible places, and
    simply feeds the policy coordinates from a universe it was not trained in. Nothing
    downstream reports it -- not the IK, not the success rate, not the logs. It shows
    up only as a policy that "does not work".

    A real instance: the datasets were re-framed a metre down at source and swept, and
    BASE_OFFSET was not updated, so every evaluation fed z about 7 sigma above anything
    in training. Near the table the policy was told it was at maximum height.
    """
    if not stats:
        return
    offending = out_of_distribution(state, stats)
    if not offending:
        return

    lines = ["The first observation is outside what this checkpoint was trained on:"]
    for index, value, low, high, sigma in offending:
        name = STATE_CHANNELS[index] if index < len(STATE_CHANNELS) else f"dim {index}"
        lines.append(
            f"  {name:<10} = {value:+.4f}   trained range [{low:+.4f}, {high:+.4f}]"
            f"   ({sigma:+.1f} sigma)"
        )
    if base_offset is not None:
        shift = [
            round(mid - value, 5) if index < 3 else 0.0
            for index, value, low, high, _ in offending
            for mid in [(low + high) / 2]
        ]
        lines.append(
            f"  A constant offset would explain this. --base-offset is currently "
            f"{tuple(round(float(v), 5) for v in base_offset)}; only it and "
            f"--state-z-offset change what the policy is shown."
        )
        if any(index < 3 for index, *_ in offending):
            lines.append(f"  Centring the offending position channels needs about {shift}.")
    lines.append(
        "  Fix the frame, or pass --allow-off-distribution if you genuinely mean to "
        "evaluate outside the training range."
    )
    raise SystemExit("\n".join(lines))


def parse_evaluation_args(parser):
    """``parse_known_args``, but a stray ``--flag`` is fatal instead of ignored.

    The tolerance is there for ROS's own arguments -- ``ros2 run`` appends
    ``--ros-args`` and friends that argparse knows nothing about -- and not for typos
    or for flags that belong to a sibling script. Dropped silently, an unrecognised
    flag is indistinguishable from an honoured one: a sweep forwarded
    ``--time-scale 0.25`` to a script with no such flag, ran three hours at full
    speed, and reported the result as though the setting had applied.

    Two seconds of error beats that every time.
    """
    args, unknown = parser.parse_known_args()
    if "--ros-args" in unknown:
        # Everything from here on belongs to rclpy, including bare values.
        unknown = unknown[: unknown.index("--ros-args")]
    stray = [item for item in unknown if item.startswith("-")]
    if stray:
        raise SystemExit(
            f"Unrecognised flag(s): {' '.join(stray)}\n"
            f"Nothing would have applied them, and the run would have looked normal. "
            f"See --help for what this script takes."
        )
    return args
