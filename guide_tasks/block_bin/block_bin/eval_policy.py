"""Roll out a trained LeRobot policy (SmolVLA) on the block_bin scene and score it.

Validation counterpart of ``solve_task.py``: instead of the GUIDE-EX node tree
producing the motion, the fine-tuned policy is queried every control step and its
joint targets are published straight onto the scene's ``joint_command`` topic --
the very topic the dataset's ``action`` was recorded from (see ``config/init.yaml``:
``action: joint_command.fr3_joint*.pos``), so inference replays the training
action space exactly.

Per episode: home the arm -> Randomize (gives the language instruction) ->
closed-loop rollout -> IsSuccess. Prints a success rate at the end.

A rollout that has clearly missed can be cut short without waiting out the timeout:

    ros2 service call <namespace>/stop_episode std_srvs/srv/SetBool "{data: false}"

`data: false` aborts the episode and moves on, `data: true` also ends the run. An
aborted episode is still scored, so it counts as the failure it is.

Before running
--------------
* ``config/init.yaml`` needs ``publish_camera_topics: true`` -- it is off for
  dataset generation, and without it the policy gets no images.
* Nothing else may write ``<ns>/franka/joint_command``. topic_based_ros2_control
  re-publishes its own (now stale) command as soon as the sim drifts away from it,
  which fights the policy, so run the simulator WITHOUT the MoveIt bring-up
  (``bringup.launch.py``) while evaluating.

Example
-------
    ~/ros2_ws/.venv/bin/python -m block_bin.eval_policy \
        --namespace /Sim_0/Scene_0 \
        --policy ~/models/smolvla_fr3/checkpoints/last/pretrained_model \
        --episodes 20
"""

import argparse
import dataclasses
import json
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import draccus
import lerobot.policies  # noqa: F401  -- registers the policy configs (smolvla, ...)
import numpy as np
import torch
from draccus.utils import DecodingError
from irob_lerobot_ros.config import ActionType, FR3RobotConfig, ROS2CameraConfig
from irob_lerobot_ros.ros2robot import ROS2Robot
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_srvs.srv import SetBool

try:  # moved out of lerobot.utils.utils in lerobot 0.6.0
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:  # pragma: no cover -- lerobot <= 0.4.x
    from lerobot.utils.utils import get_safe_torch_device

from block_bin.solve_task import scene_num_zones
from guide_core.types.randomization import zone_plan
from guide_msgs.srv import CheckSuccess, Randomize

# Dataset image keys (init.yaml `dataset.images`) -> the /cam_* topic each one was
# rendered from. Camera names double as the policy's `observation.images.<name>`.
CAMERAS = {"top": "cam_top", "base": "cam_base", "wrist": "cam_wrist"}

# The gripper is a single dataset dimension (`gripper.pos` == fr3_finger_joint1),
# but the sim's articulation needs both fingers driven, so finger 2 mirrors it.
GRIPPER_MIRROR_JOINT = "fr3_finger_joint2"

# Arm home pose from config/reset.yaml, published as a joint_command at the start
# of every episode. GRIPPER_OPEN matches solve_task's OpenGripper, so a block still
# held from a failed episode is dropped before the scene is re-randomized.
HOME_POSITION = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
GRIPPER_OPEN = 0.04


def state_names(robot) -> list[str]:
    """Joint order behind `observation.state` / `action`.

    The curated training set (``meta/info.json``) names both as
    ``[joint1.pos ... joint7.pos, gripper.pos]``, i.e. the 7 arm joints followed by
    the gripper -- never index a dataset column by position, always by this order.
    """
    return list(robot.config.arm_joint_names) + list(robot.config.gripper_joint_names)


def images_from(observation: dict) -> dict:
    """The `observation.images.*` half of a frame -- shared with eval_policy_servo."""
    frame = {}
    for cam in CAMERAS:
        image = observation.get(cam)
        if image is None:
            raise RuntimeError(
                f"No image for camera '{cam}'. Is `publish_camera_topics: true` in init.yaml?"
            )
        frame[f"observation.images.{cam}"] = image
    return frame


def build_frame(observation: dict, joints: list[str]) -> dict:
    """Robot observation -> the dataset-shaped frame `predict_action` expects."""
    state = np.array([observation[f"{j}.pos"] for j in joints], dtype=np.float32)
    return {"observation.state": state, **images_from(observation)}


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


def is_success(robot, scene_id: int) -> bool:
    response = robot.callService(robot.is_success_client, CheckSuccess.Request(id=scene_id))
    return bool(response.success)


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

    def __init__(self, policy, preprocessor, postprocessor, device, task, robot_type, lead: int):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.task = task
        self.robot_type = robot_type
        self.lead = lead
        self.replans = 0
        self._queue = deque()
        self._pending = None  # (future, step whose observation it was launched from)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _predict(self, frame: dict) -> list:
        """One action chunk, truncated to the configured horizon."""
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.policy.config.use_amp
            else nullcontext(),
        ):
            batch = prepare_observation_for_inference(
                frame, self.device, self.task, self.robot_type
            )
            chunk = self.postprocessor(self.policy.predict_action_chunk(self.preprocessor(batch)))
        horizon = min(self.policy.config.n_action_steps, chunk.shape[1])
        return [chunk[:, i] for i in range(horizon)]

    def _absorb(self, chunk: list, launched_at: int, step: int) -> None:
        """Queue a chunk, skipping the actions whose control slots have already passed."""
        offset = (step - launched_at) + len(self._queue)
        self._queue.extend(chunk[offset:])

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
            self._pending = (self._pool.submit(self._predict, frame), step)

        return self._queue.popleft()


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


def run_episode(robot, scene_id, zone, policy, processors, device, control, args) -> bool:
    """Home + randomize the scene, then let the policy drive until it succeeds or times out."""
    joints = state_names(robot)

    # Home through the same joint_command channel the policy uses, NOT the Reset
    # service: Reset's `set_joint` wraps the live scene in a second articulation and
    # initialises it mid-run, which takes Isaac down (nothing else calls Reset --
    # solve_task homes via MoveIt, so the path was never exercised). Randomize
    # re-places every block and both bins regardless, so Reset would only have added
    # the arm pose, and its `set_joint` is an apply_action -- a position target,
    # exactly what publishing here does.
    publish_command(robot, *command_from_action(HOME_POSITION + [GRIPPER_OPEN], joints))

    response = robot.callService(
        robot.randomize,
        Randomize.Request(id=scene_id, use_zone=zone is not None, zone=zone or 0),
    )
    task = args.task or json.loads(response.message)["task"]
    robot.node.get_logger().info(f'Rolling out: "{task}"')

    sleep_sim(robot, 5.0)  # let the randomized scene settle, as solve_task does

    preprocessor, postprocessor = processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    control.abort_episode.clear()
    stream = ChunkStream(
        policy, preprocessor, postprocessor, device, task, robot.name, args.lead
    )

    period = 1.0 / args.fps
    steps = int(args.seconds * args.fps)
    check_interval = max(1, int(args.fps))  # poll the success service ~once a second
    gripper_joint = joints[-1]
    step_times, grips = [], []
    succeeded = False

    clock = robot.node.get_clock()
    next_deadline = clock.now()

    for step in range(steps):
        started = time.perf_counter()

        observation = robot.get_observation()
        action = stream.next_action(build_frame(observation, joints), step)
        names, positions = command_from_action(action, joints)
        publish_command(robot, names, positions)
        grips.append((positions[-1], observation[f"{gripper_joint}.pos"]))

        step_times.append(time.perf_counter() - started)

        if control.abort_episode.is_set():
            robot.node.get_logger().warn(f"Episode aborted after {step + 1} steps.")
            break

        if step % check_interval == check_interval - 1 and is_success(robot, scene_id):
            succeeded = True
            break

        # Hold the cadence in SIM time. Deadlines accumulate so jitter does not drift
        # the trajectory, but an overrun DROPS its slot rather than being made up:
        # catching up would fire several queued setpoints back to back, and a burst
        # like that landing on the approach is how a grasp turns into a swipe.
        next_deadline = next_deadline + Duration(seconds=period)
        if clock.now() > next_deadline:
            next_deadline = clock.now()
        while clock.now() < next_deadline:
            time.sleep(0.002)

    stream.close()
    report_rollout(robot, args, policy, step_times, grips, stream.replans)
    return succeeded or is_success(robot, scene_id)


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
    if median > 1.0 / args.fps:
        log.warn(
            f"Median step {median * 1000:.0f} ms exceeds the {1000 / args.fps:.0f} ms "
            f"budget of {args.fps} Hz -- the arm is being driven slower than the "
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--policy", type=str, required=True, help="Checkpoint dir or HF repo id.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=60.0, help="Rollout timeout per episode.")
    # Demonstrations were captured at step_freq / record_interval = 60 / 12 Hz, so
    # the actions are spaced ~0.2 s apart. The 30 fps in the dataset's info.json is
    # nominal (video metadata), not the rate the arm was actually commanded at --
    # tune this if the policy's motion comes out too fast or too sluggish.
    parser.add_argument("--fps", type=float, default=5.0, help="Control rate, Hz of SIM time.")
    # The checkpoint's chunk_size is 50: one observation would otherwise drive 50
    # blind steps -- 10 s at 5 Hz -- so the grasp happens on a view of the block
    # that is ten seconds stale. Consuming a shorter slice of each chunk makes the
    # policy look again before it closes the gripper. Raise towards chunk_size for
    # smoother but blinder motion, lower for tighter feedback at more re-plans.
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Actions used per policy call (<= the checkpoint's chunk_size).",
    )
    # A chunk takes 2-3 control periods to predict, so the next one has to be started
    # with at least that many actions still queued or the arm stalls waiting for it.
    parser.add_argument(
        "--lead",
        type=int,
        default=3,
        help="Start predicting the next chunk with this many actions left. 0 blocks.",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="",
        help="Cube placement: blank = anywhere, 'all' = every zone, '2,16' = those "
        "zones, '2:4,16:10' = per-zone counts. --episodes is per zone.",
    )
    parser.add_argument("--task", type=str, default="", help="Override the scene's instruction.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu (default: policy's)")
    args = parser.parse_known_args()[0]

    namespace_base = args.namespace or ""
    match = re.search(r"\d+$", namespace_base.split("/")[-1].strip())
    scene_id = int(match.group()) if match else 0
    sim_namespace = "/" + namespace_base.split("/")[1] if namespace_base else ""

    # Built before the policy loads so a bad --zone fails in a second rather than
    # after 450M parameters have been read off disk.
    plan = episode_plan(args.zone, args.episodes)
    print(f"Evaluating {len(plan)} episodes: {args.zone or 'unrestricted placement'}")

    policy, preprocessor, postprocessor = load_policy(args.policy, args.device)
    device = get_safe_torch_device(policy.config.device)

    chunk_size = policy.config.chunk_size
    if not 1 <= args.n_action_steps <= chunk_size:
        raise SystemExit(f"--n-action-steps must be within 1..{chunk_size} (the trained chunk).")
    if args.n_action_steps != policy.config.n_action_steps:
        print(
            f"Action horizon: {args.n_action_steps} of {chunk_size} predicted steps "
            f"({args.n_action_steps / args.fps:.1f}s open-loop, was "
            f"{policy.config.n_action_steps / args.fps:.1f}s)"
        )
        policy.config.n_action_steps = args.n_action_steps

    config = FR3RobotConfig(
        frame_id=namespace_base.split("/")[-1] if namespace_base else "world",
        namespace=f"{namespace_base}/franka",
        # Joint targets go straight onto joint_command: MoveIt planning per control
        # step could never keep up, and the recorded actions are raw joint targets.
        arm_action_type=ActionType.JOINT_POSITION,
        gripper_action_type=ActionType.JOINT_POSITION,
        directly_publish=True,
    )
    config.cameras = {
        name: ROS2CameraConfig(
            namespace=namespace_base, frame_id=name, topic=topic, width=640, height=480
        )
        for name, topic in CAMERAS.items()
    }

    robot = ROS2Robot(config=config)
    robot.connect()
    time.sleep(5)  # wait for connections to establish
    print("Connected to robot and cameras.")

    robot.randomize = robot.node.create_client(
        srv_type=Randomize,
        srv_name=f"{sim_namespace}/Randomize",
        callback_group=robot._reentrant_callback_group,
    )
    robot.is_success_client = robot.node.create_client(
        srv_type=CheckSuccess,
        srv_name=f"{sim_namespace}/IsSuccess",
        callback_group=robot._reentrant_callback_group,
    )

    control = RunControl()
    stop_service_name = f"{namespace_base}/stop_episode"
    robot.stop_service = robot.node.create_service(
        srv_type=SetBool,
        srv_name=stop_service_name,
        callback=lambda request, response: handle_stop(
            request, response, control, robot.node.get_logger()
        ),
        callback_group=robot._reentrant_callback_group,
    )
    print(
        "Abort a rollout that has clearly missed with:\n"
        f"  ros2 service call {stop_service_name} std_srvs/srv/SetBool \"{{data: false}}\"\n"
        "  (data: true also ends the whole evaluation)"
    )

    successes = 0
    attempted = 0
    per_zone = defaultdict(lambda: [0, 0])  # zone -> [successes, attempts]
    try:
        for episode, zone in enumerate(plan):
            label = "anywhere" if zone is None else f"zone {zone}"
            robot.node.get_logger().info(f"--- Episode {episode + 1}/{len(plan)} ({label}) ---")
            success = run_episode(
                robot, scene_id, zone, policy, (preprocessor, postprocessor), device, control, args
            )
            attempted += 1
            successes += bool(success)
            per_zone[zone][0] += bool(success)
            per_zone[zone][1] += 1
            colour, verdict = ("\033[92m", "SUCCESS") if success else ("\033[91m", "FAILURE")
            robot.node.get_logger().info(
                f"{colour}[{verdict}] Episode {episode + 1}: "
                f"{successes}/{attempted} so far\033[0m"
            )
            if control.stop_run.is_set():
                robot.node.get_logger().warn("Evaluation ended early on request.")
                break
    except KeyboardInterrupt:
        robot.node.get_logger().info("Keyboard interrupt received. Exiting...")
    finally:
        print(f"Success rate: {successes}/{attempted}")
        if args.zone:
            for zone in sorted(per_zone, key=lambda z: -1 if z is None else z):
                hits, tries = per_zone[zone]
                print(f"  zone {zone}: {hits}/{tries}")
        robot.disconnect()


if __name__ == "__main__":
    main()
