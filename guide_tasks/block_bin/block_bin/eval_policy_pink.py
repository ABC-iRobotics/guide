"""Roll out a LIBERO-style policy on block_bin through Pink differential IK.

Third sibling of ``eval_policy.py`` and ``eval_policy_servo.py``, and it takes one
half from each. The action space is ``eval_policy_servo``'s -- a dataset built with
``guide_dataset_build --eef-delta-action [--libero-state]``::

    action            = [dx, dy, dz, dwx, dwy, dwz, gripper]   (7)
    observation.state = [x, y, z, wx, wy, wz, g, g]            (8, --libero-state)

but the plumbing is ``eval_policy``'s: joint targets go straight onto
``joint_command``, so no MoveIt, no servo_node and no controller stack are involved.
MoveIt Servo is replaced by Pink, which solves the IK in this process:

    q            <- measured joint_states
    pose         <- FK(q)                       every step, in `fr3_link0`
    target       <- pose (+) delta_t            the policy's action
    q_cmd        <- IK(target, seed=q)          Pink, ~5 QP iterations
    joint_command <- q_cmd + gripper

Doing the IK here rather than in servo_node buys three things that made the servo
path fragile. There is no ``Servo::toPlanningFrame`` conversion to be miscoded
(``eval_policy_servo.twist_frame_for``), no ``incoming_command_timeout`` that has to
be kept equal to the control period or a late step silently travels further than the
policy asked for (``Servo.check_command_lifetime``), and no
topic_based_ros2_control re-publishing a stale ``joint_command`` underneath us.

How an episode starts
---------------------
Home the arm, WAIT for it to actually arrive, then randomize. Publishing a
joint_command returns immediately, so randomizing next scatters the fresh layout off a
still-moving arm; ``--home-settle-seconds`` bounds the wait and it exits as soon as the
arm is there.

When an episode ends
--------------------
Not at the moment the success service says yes. Every episode of the training set was
recorded through the arm's return to home afterwards -- about a sixth of each one is
the homing sweep and another eighth the arm parked there -- so the policy was trained
to fly itself home once the block is placed, and stopping on success alone never
scores a quarter of what it learnt. A rollout therefore runs until the task has
succeeded AND the tool is back within ``--home-return-tolerance`` of home, or until the
timeout, an abort or a held step ends it. Success itself is scored exactly as before:
an episode that places the block and then wanders off is still a success, with a
warning saying it never came home.

Closed loop, deliberately
-------------------------
The target is rebuilt from ``FK(q_measured)`` on every step rather than integrated
from the previous target. That is what the dataset says: ``eef_delta_action``
computes ``delta_t = pose(state_{t+1}) (-) pose(state_t)`` between successive
*measured* poses, so ``FK(q_measured) (+) delta`` is the faithful reconstruction of
what the demonstration did. It also means tracking error can never accumulate. The
cost is the other way round: if the arm fails to reach a target within its period,
that step's motion is dropped rather than carried, so a persistently lagging arm
undershoots. ``--action-scale`` is the knob for that.

Frames
------
All IK is done in ``--base-frame`` (``fr3_link0``). ``--base-offset`` shifts the FK
position into the frame the recorder stored ``observation.state`` in, which is Isaac's
``world``: libero_dataset_250's positions run z 1.04..1.89, a metre up, and every
episode ends homed at [0.00022, 0.00003, 1.49983]. Its default is measured against
that (see ``BASE_OFFSET``) rather than derived from the scene geometry, which is a
centimetre out. Deltas need no conversion of their own -- world, ``Scene_i`` and
``fr3_link0`` differ by pure translations, so a displacement has identical components
in all three.

Do NOT "correct" the state for the checkpoint's normalizer. smolvla_fr3_07_29 was
trained through ``meta/stats.json``, whose ``observation.state`` mean is a metre below
the data it summarises (0.3079 against the frames' own 1.3079 -- the per-episode stats
in ``meta/episodes`` agree with the frames, the aggregate does not). Normalization is
MEAN_STD, so that mean is live: training fed raw world-frame z through it and the
policy learned on the resulting offset. Feeding raw world-frame z here reproduces it
exactly. The same file's stale ``min`` is never read under MEAN_STD.

Running it
----------
1. Isaac with the scene, ``publish_camera_topics: true`` in ``config/init.yaml``.
2. The description publisher -- robot_state_publisher ALONE, no move_group and no
   controllers, because those would fight the direct ``joint_command`` writes::

       ros2 launch block_bin eval_pink.launch.py

   ``--urdf`` skips this and reads a URDF (or xacro) off disk instead.
3. The rollout::

       ~/ros2_ws/.venv/bin/python -m block_bin.eval_policy_pink \
           --namespace /Sim_0/Scene_0 \
           --policy ~/models/smolvla_fr3_libero/checkpoints/last/pretrained_model \
           --episodes 20

Needs ``pin-pink`` and a QP solver: ``uv pip install pin-pink quadprog``. The PyPI
name matters -- plain ``pink`` is an unrelated code formatter.

A rollout that has clearly missed can be cut short:

    ros2 service call <namespace>/stop_episode std_srvs/srv/SetBool "{data: false}"

`data: false` aborts the episode and moves on, `data: true` also ends the run. An
aborted episode is still scored, so it counts as the failure it is.
"""

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pinocchio as pin
import pink
import torch
from irob_lerobot_ros.config import ActionType, FR3RobotConfig, ROS2CameraConfig
from irob_lerobot_ros.ros2robot import ROS2Robot
from pink.tasks import FrameTask, PostureTask
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import SetBool

from block_bin.eval_policy import (
    CAMERAS,
    GRIPPER_OPEN,
    HOME_POSITION,
    ChunkStream,
    RunControl,
    command_from_action,
    episode_plan,
    get_safe_torch_device,
    handle_stop,
    images_from,
    is_success,
    load_policy,
    publish_command,
    report_rollout,
    sleep_sim,
)
from block_bin.eval_policy_servo import DELTA_DIMS, joint_state, libero_state
from guide_msgs.srv import CheckSuccess, Randomize

# world -> fr3_link0, the translation that turns an FK position into the frame the
# recorder stored. MEASURED, not derived: every episode of libero_dataset_250 ends
# homed, and the mean of those 250 final poses is [0.00022, 0.00003, 1.49983] in the
# dataset's frame against an FK of [0.30689, 0, 0.48688] in fr3_link0 (spread 0.5 mm).
# The scene geometry alone -- Scene_0 at -origin = [0, 0, 1] plus the bring-up's
# xyz:="-0.3 0 0" -- predicts [-0.3, 0, 1.0], which is 6.7 mm and 12.9 mm off in x and
# z. Small, and exactly the sort of bias that shifts a grasp; re-measure the same way
# if the scene is re-placed or the arm re-mounted.
BASE_OFFSET = (-0.30667, 0.00003, 1.01295)

# Where the home pose lands in the dataset's frame -- the mean of libero_dataset_250's
# 250 final states, which are all homed, with a spread of 0.5 mm. Every episode here
# starts by commanding HOME_POSITION too, so the first step of a rollout can be checked
# against this directly. It is the one assertion that catches a wrong --base-offset, a
# wrong --base-frame or a re-placed scene, all of which otherwise load fine and simply
# feed the policy positions it never trained on.
HOME_IN_DATASET = (0.00022, 0.00003, 1.499827)
HOME_TOLERANCE = 0.01

# How close the arm has to get back to home before a SUCCEEDED episode is allowed to
# end. Euclidean distance in metres, the same measure as HOME_TOLERANCE, but much
# looser: that one checks a pose the script itself commanded, this one is a policy
# flying itself back, and 0.05 m was never reached in practice -- every successful
# episode ran on to the full timeout instead of ending. 0.30 m is "the arm has clearly
# left the workspace and gone back", which is all this needs to decide.
#
# Every episode of libero_dataset_250 was recorded through that return -- roughly a
# sixth of each one is the homing sweep and another eighth is the arm parked at home --
# so the policy was trained to drive home after placing, and cutting the rollout at the
# moment of success stops scoring a behaviour that makes up a quarter of the training
# data.
HOME_RETURN_TOLERANCE = 0.30

# Metres added to the z the POLICY is shown, and to nothing else. The IK always works
# from the raw FK pose, so this can only change what the policy believes, never where
# the arm is driven.
#
# 0.0 -- MEASURED at the home pose against the recorded data, not argued. Commanding
# HOME_POSITION and reading joint_states back gives an FK of [0.0003, 0.0001, 1.4995]
# once BASE_OFFSET is applied, against the mean final state of libero_dataset_250's 250
# episodes (all homed) of [0.00022, 0.00003, 1.499827], spread 0.0005 m. That is 0.3 mm
# out, well inside the dataset's own [1.4987, 1.5007]. Every other dimension lands
# inside one standard deviation too.
#
# A -1.0 hotfix was tried on the theory that the state should match the frame
# ``meta/stats.json`` is expressed in. It is wrong, and expensively so: it puts z at
# 0.4995 when the dataset's z never goes below 1.0398, i.e. about a thousand standard
# deviations outside anything the policy has seen. The stats file is the thing that is
# inconsistent, not the frames -- the parquet and the per-episode stats in
# ``meta/episodes`` both say world frame, only the AGGREGATE is a metre low, and since
# LeRobot builds the normalizer from that aggregate, training fed raw world z through a
# mean a metre below it and learned on the resulting offset. Feeding raw world z here
# reproduces training exactly. Kept as a flag only so the two can still be A/B'd.
STATE_Z_OFFSET = 1.0

# Ceiling on one step's commanded motion, from libero_dataset_250 -- the set the
# shipped checkpoints were trained on -- which never exceeds 0.1029 m or 0.3302 rad in
# a single step. libero_2000_0_1_2_3_4_6_7_8_12 reaches 0.2552 m and 0.6733 rad, but
# only in frames that are not task motion at all: every one of its 2000 episodes was
# recorded through the arm's return to home, and MoveIt homes through joint space on a
# path whose end-effector traces a wide arc behind the robot (episode 959 sweeps out to
# x = -0.85, half a metre behind the base). A policy trained on that reproduces those
# sweeps, and here that would ask the IK to cross a quarter of a metre in one control
# step. Scaling the step back keeps its direction and its translation/rotation ratio.
MAX_STEP_LINEAR = 0.10
MAX_STEP_ANGULAR = 0.33

# Last-resort guard on the IK's own output, in joint space. A fully-clamped step is
# worth up to ~1.05 rad on a single joint near a singularity, so this is NOT a
# smoothness limit -- it only catches a solution that has nothing to do with where the
# arm is: a stale or empty joint_states read (which would seed the IK at zeros and
# command the arm to fold straight up), or the QP settling into a different IK branch.
# Publishing either as a position target is what makes an arm launch.
MAX_JOINT_STEP = 1.2

# Pink integrates a QP velocity solution towards the target. These are the inner
# solver's own steps, not control steps -- nothing is published until it converges.
# 0.01 s x 40 leaves plenty of headroom over the ~5 iterations a 0.2 s control step's
# delta actually needs, so the cap only bites on a target the arm cannot reach.
IK_DT = 0.01
IK_MAX_ITERATIONS = 40

# quadprog is the dense active-set solver from the install line above. Pink would
# otherwise pick whatever qpsolvers found first, which is not reproducible across
# machines -- and for a 7-DOF arm the choice shows up as different redundancy
# resolution, i.e. a visibly different elbow.
QP_SOLVER = "quadprog"


def wait_for_robot_description(node, timeout: float = 30.0) -> str:
    """The URDF ``robot_state_publisher`` latched on ``<namespace>/robot_description``.

    Published TRANSIENT_LOCAL and never again, so a late subscriber still gets it --
    but only if it asks with matching durability, which is why the QoS is spelled out
    rather than left at the default VOLATILE (that combination silently receives
    nothing, forever).
    """
    received = []
    node.create_subscription(
        String,
        "robot_description",
        lambda message: received.append(message.data),
        QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        ),
    )
    deadline = time.perf_counter() + timeout
    while not received and time.perf_counter() < deadline:
        time.sleep(0.05)
    if not received:
        raise SystemExit(
            f"Nothing published '{node.get_namespace().rstrip('/')}/robot_description' "
            f"within {timeout:g}s. Start the description publisher first:\n"
            f"  ros2 launch block_bin eval_pink.launch.py\n"
            f"or pass --urdf <file.urdf|file.urdf.xacro> to skip ROS entirely."
        )
    return received[0]


def urdf_from_file(path: str) -> str:
    """A URDF off disk, running xacro first if it needs it."""
    if not Path(path).is_file():
        raise SystemExit(f"--urdf {path}: no such file.")
    if not path.endswith(".xacro"):
        return Path(path).read_text()
    finished = subprocess.run(["xacro", path], capture_output=True, text=True)
    if finished.returncode != 0:
        raise SystemExit(f"xacro failed on {path}:\n{finished.stderr}")
    return finished.stdout


class ArmIK:
    """Forward and inverse kinematics for the arm alone, via Pinocchio and Pink.

    The fingers are locked out of the model (``buildReducedModel``) rather than left
    in and ignored: they are prismatic joints on the same kinematic tree, so an IK
    that can move them will happily "reach" a target by opening the gripper. What is
    left is exactly the 7 revolute joints the policy's delta has to be resolved into.

    Joint order is resolved BY NAME against ``arm_joints``. Pinocchio orders its
    configuration vector by the URDF's tree, which happens to match the dataset's
    ``fr3_joint1..7`` today -- and would keep matching silently right up until a
    description reorders them, at which point every command would be a permutation of
    the right one.

    ``PostureTask`` is what resolves the redundancy of a 7-DOF arm against a 6-DOF
    target. It is the counterpart of the centering the servo path got from pick_ik
    (``config/kinematics.yaml``): weak enough not to fight the pose, strong enough
    that the null space drifts back towards the home posture instead of wandering
    into a joint limit over a 60 s rollout.
    """

    def __init__(
        self,
        urdf_xml: str,
        arm_joints: list[str],
        ee_frame: str,
        base_frame: str,
        posture_cost: float,
        posture: np.ndarray,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
    ):
        full = pin.buildModelFromXML(urdf_xml)
        missing = [j for j in arm_joints if not full.existJointName(j)]
        if missing:
            raise SystemExit(f"The robot description has no joints {missing}.")
        locked = [
            full.getJointId(name)
            for name in full.names
            if name != "universe" and name not in arm_joints
        ]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.data = self.model.createData()

        for frame in (ee_frame, base_frame):
            if not self.model.existFrame(frame):
                raise SystemExit(f"The robot description has no frame '{frame}'.")
        self.ee_frame = ee_frame
        self.base_frame = base_frame

        model_order = [self.model.names[i + 1] for i in range(self.model.nq)]
        self._to_model = [arm_joints.index(name) for name in model_order]
        self._to_ros = [model_order.index(name) for name in arm_joints]

        self.frame_task = FrameTask(
            ee_frame, position_cost=position_cost, orientation_cost=orientation_cost
        )
        self.posture_task = PostureTask(cost=posture_cost)
        self.posture_task.set_target(np.asarray(posture, dtype=np.float64)[self._to_model])

    def _configuration(self, q_ros) -> pink.Configuration:
        q = np.asarray(q_ros, dtype=np.float64)[self._to_model]
        return pink.Configuration(self.model, self.data, q)

    def _pose(self, configuration) -> tuple[np.ndarray, np.ndarray]:
        transform = configuration.get_transform(self.ee_frame, self.base_frame)
        return transform.translation.copy(), Rotation.from_matrix(transform.rotation).as_rotvec()

    def _to_root(self, configuration, target: pin.SE3) -> pin.SE3:
        """A target given in ``base_frame`` re-expressed in the model's ROOT frame.

        Not optional, and the reason is easy to miss: ``fk`` reports the tool relative
        to ``base_frame`` (``fr3_link0``), but Pink's ``FrameTask`` always measures its
        error against whatever the URDF is rooted at. Those are the same frame only by
        accident. A description built without ``ros2_control`` roots at ``base``, which
        sits on ``fr3_link0``, and the two agree; the description
        ``eval_pink.launch.py`` actually publishes is built the way the MoveIt bring-up
        builds it and roots at ``Scene_0``, with the arm mounted at ``xyz="-0.3 0 0"``.
        Feeding a base-frame target straight to the task then asks for a pose 0.3 m
        forward of the one the policy requested -- on the first action of the episode,
        from a standing start, which looks exactly like the arm launching. The solver
        does not complain: it converges, on the wrong target.
        """
        return configuration.get_transform_frame_to_world(self.base_frame) * target

    def fk(self, q_ros) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotvec) of the end effector in ``base_frame``, from joint angles."""
        return self._pose(self._configuration(q_ros))

    def solve(self, q_seed, position, rotvec, tolerance: float) -> tuple[np.ndarray, float]:
        """Joint angles reaching ``(position, rotvec)``, and the residual pose error.

        Seeded from the measured configuration, so the solution is the one nearest to
        where the arm already is -- a 7-DOF arm has a continuum of them, and picking a
        far one would be a joint-space jump executed as a single position command.

        ``safety_break=False``: Pink otherwise raises as soon as the seed is outside
        the URDF's joint limits, and the seed is a *measurement*, so a joint resting a
        hair past its limit would end the rollout with an exception instead of a
        command. The limits still constrain the QP, so the solution respects them
        either way.
        """
        target = pin.SE3(Rotation.from_rotvec(rotvec).as_matrix(), np.asarray(position, float))
        configuration = self._configuration(q_seed)
        self.frame_task.set_target(self._to_root(configuration, target))
        residual = float("inf")
        for _ in range(IK_MAX_ITERATIONS):
            velocity = pink.solve_ik(
                configuration,
                [self.frame_task, self.posture_task],
                IK_DT,
                solver=QP_SOLVER,
                safety_break=False,
            )
            configuration.integrate_inplace(velocity, IK_DT)
            residual = float(np.linalg.norm(self.frame_task.compute_error(configuration)))
            if residual < tolerance:
                break
        return configuration.q[self._to_ros], residual


def delta_from_action(action, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """7-dim LIBERO action -> (position delta, rotation delta as rotvec, gripper).

    The gripper dimension stays an ABSOLUTE finger position in metres (GUIDE's
    convention, ~0..0.04), not LIBERO's +-1, and ``scale`` deliberately does not touch
    it -- stretching the motion must not stretch how far the fingers close.
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != DELTA_DIMS:
        raise ValueError(
            f"Policy returned {values.size} dims, expected {DELTA_DIMS} "
            f"[dx dy dz dwx dwy dwz gripper]. Was this checkpoint trained on a dataset "
            f"built with --eef-delta-action? A joint-space policy belongs in eval_policy.py."
        )
    return values[0:3] * scale, values[3:6] * scale, float(values[6])


def clamp_delta(dposition, drotvec, max_linear: float, max_angular: float):
    """Scale one step's motion back under the ceilings, keeping its direction.

    Both halves are scaled by the SAME factor, so a clamped step is a shortened
    version of the motion the policy asked for rather than a different one: scaling
    the translation alone would leave the arm rotating at full rate while barely
    travelling, which is not a pose the policy ever predicted.

    Returns the scaled delta and whether it was clamped. Either ceiling at 0 disables
    that half.
    """
    dposition = np.asarray(dposition, dtype=np.float64)
    drotvec = np.asarray(drotvec, dtype=np.float64)
    linear = float(np.linalg.norm(dposition))
    angular = float(np.linalg.norm(drotvec))

    factor = 1.0
    if max_linear > 0 and linear > max_linear:
        factor = min(factor, max_linear / linear)
    if max_angular > 0 and angular > max_angular:
        factor = min(factor, max_angular / angular)
    return dposition * factor, drotvec * factor, factor < 1.0


def joint_step(q_command, q_measured) -> float:
    """The largest single-joint move a command asks for, in radians."""
    return float(np.abs(np.asarray(q_command) - np.asarray(q_measured)).max())


def apply_delta(position, rotvec, dposition, drotvec) -> tuple[np.ndarray, np.ndarray]:
    """Compose one action onto a pose, the way ``eef_delta_action`` decomposed it.

    Position adds. Orientation LEFT-composes -- ``R_target = dR * R`` -- because the
    delta was built as ``(R_{t+1} * R_t^T).as_rotvec()``, a rotation about the base
    frame's axes. Adding rotation vectors componentwise instead would be wrong twice
    over: rotations do not commute, and the rotation vector wraps at +-pi.
    """
    target = Rotation.from_rotvec(drotvec) * Rotation.from_rotvec(rotvec)
    return np.asarray(position, float) + np.asarray(dposition, float), target.as_rotvec()


def measured_joints(observation: dict, arm_joints: list[str]) -> np.ndarray:
    return np.array([observation[f"{joint}.pos"] for joint in arm_joints], dtype=np.float64)


def wait_until_home(
    robot, ik, arm_joints, offset, timeout: float, tolerance: float, poll: float = 0.25
) -> float:
    """Block until the tool is actually back at home. Returns the error it got to.

    Homing is a published position target, not a service call, so it returns
    instantly while the arm is still crossing the workspace. Randomizing during that
    flight is what drops fresh blocks and bins into a moving arm: the collision
    scatters them, and the episode starts from a scene the randomizer never intended.
    Waiting also gives a block still held from a failed episode time to fall clear,
    since the same command opens the fingers.

    Timed in SIM time via ``sleep_sim``, which brings its own wall-clock stall guard --
    Isaac does not run at real time with three cameras rendering and a VLA on the GPU.
    """
    error = float("inf")
    for _ in range(max(1, int(timeout / poll))):
        position, _ = ik.fk(measured_joints(robot.get_observation(), arm_joints))
        error = home_pose_error(position + offset)
        if error < tolerance:
            return error
        sleep_sim(robot, poll)
    return error


def append_record(path: str, record: dict) -> None:
    """One episode, one JSON line, flushed. Appended so a sweep can resume onto it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def episode_is_finished(succeeded: bool, home_error: float, tolerance: float) -> bool:
    """Whether a rollout may stop: the task is done AND the arm is back home.

    Success alone is not enough. ``tolerance <= 0`` restores the old behaviour of
    stopping the moment the success service says yes.
    """
    if not succeeded:
        return False
    return tolerance <= 0 or home_error < tolerance


def home_pose_error(position) -> float:
    """How far the first state of an episode is from where the dataset put home.

    The rollout commands ``HOME_POSITION`` and waits for the scene to settle before
    step 0, so this compares like with like: a frame or offset that disagrees with the
    training data shows up here as a hard number instead of as a rollout that merely
    behaves badly.
    """
    return float(np.linalg.norm(np.asarray(position) - np.asarray(HOME_IN_DATASET)))


def run_episode(
    robot, scene_id, zone, policy, processors, device, control, ik, args, seed=None
) -> dict:
    """Home + randomize the scene, then let the policy drive the IK until success.

    Returns the episode's record -- ``success`` plus everything the run printed as it
    went, so a sweep over checkpoints has numbers to aggregate instead of a log to
    scrape. See ``--results``.
    """
    arm_joints = list(robot.config.arm_joint_names)
    gripper_joint = robot.config.gripper_joint_names[0]
    joints = arm_joints + [gripper_joint]
    offset = np.asarray(args.base_offset, dtype=np.float64)

    # Homed through joint_command, like eval_policy.py: with no MoveIt running there is
    # no planning pipeline to home through, and opening the fingers first drops a block
    # still held from a failed episode before the scene is re-randomized.
    publish_command(robot, *command_from_action(HOME_POSITION + [GRIPPER_OPEN], joints))

    # Home FIRST and wait for the arm to get there, THEN randomize. The order matters:
    # publishing a joint_command returns immediately, so randomizing next places the
    # blocks and bins around an arm that is still flying home, and whatever it clips on
    # the way gets knocked out of the layout the randomizer just chose.
    settled = wait_until_home(
        robot, ik, arm_joints, offset, args.home_settle_seconds, HOME_TOLERANCE
    )
    if settled > HOME_TOLERANCE:
        robot.node.get_logger().warn(
            f"Arm still {settled * 100:.1f} cm from home after "
            f"{args.home_settle_seconds:g}s; randomizing anyway. Raise "
            f"--home-settle-seconds if the scene comes out disturbed."
        )

    # A seed makes THIS episode's layout repeatable, so the same episode index of two
    # different checkpoints is the same scene and the two can be compared as paired
    # samples instead of as two independent draws. Reproducible within one simulator
    # session -- the seed is combined with the scene's master seed, which is drawn from
    # OS entropy at registration unless it was injected there.
    response = robot.callService(
        robot.randomize,
        Randomize.Request(
            id=scene_id,
            use_zone=zone is not None,
            zone=zone or 0,
            use_seed=seed is not None,
            seed=int(seed) if seed is not None else 0,
        ),
    )
    task = args.task or json.loads(response.message)["task"]
    robot.node.get_logger().info(f'Rolling out: "{task}"')

    sleep_sim(robot, 5.0)  # let the randomized scene settle, as solve_task does

    preprocessor, postprocessor = processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    control.abort_episode.clear()
    stream = ChunkStream(policy, preprocessor, postprocessor, device, task, robot.name, args.lead)

    period = 1.0 / args.fps
    steps = int(args.seconds * args.fps)
    check_interval = max(1, int(args.fps))  # poll the success service ~once a second
    step_times, grips, home_errors = [], [], []
    # Where the tool went and what the fingers did, kept so a FAILED episode can still
    # say which way it failed. An arm that never left home, one that crossed the table
    # but never closed, and one that closed on empty air are three different problems
    # and only the first is cheap to see in the log.
    travelled, previous_position = 0.0, None
    unreachable = clamped = refused = 0
    succeeded = returned_home = False
    success_step = 0
    home_error = float("inf")

    clock = robot.node.get_clock()
    next_deadline = clock.now()
    started_wall, started_sim = time.perf_counter(), clock.now()

    for step in range(steps):
        started = time.perf_counter()

        observation = robot.get_observation()
        q = measured_joints(observation, arm_joints)
        position, rotvec = ik.fk(q)

        dataset_position = position + offset
        home_error = home_pose_error(dataset_position)
        home_errors.append(home_error)
        if previous_position is not None:
            travelled += float(np.linalg.norm(position - previous_position))
        previous_position = position
        if args.state == "libero":
            # --state-z-offset shifts ONLY what the policy is shown. The IK below keeps
            # working from `position`, so a wrong guess here cannot move the arm to the
            # wrong place -- it can only feed the policy the wrong picture of where it is.
            state = libero_state(
                observation,
                dataset_position + np.array([0.0, 0.0, args.state_z_offset]),
                rotvec,
                gripper_joint,
            )
        else:
            state = joint_state(observation, joints)
        if step == 0:
            robot.node.get_logger().info(
                f"First {args.state} state: [{', '.join(f'{v:.4f}' for v in state)}] "
                f"({len(state)} dims, EEF in '{args.base_frame}' + {tuple(offset)}"
                + (f", state z {args.state_z_offset:+g} m" if args.state_z_offset else "")
                + ")"
            )
            # The arm was homed and the scene has settled, so this pose is directly
            # comparable with the dataset's own home. Warned, not fatal: a block still
            # wedged in the fingers can hold the arm short of home without the frames
            # being wrong at all.
            if home_error > HOME_TOLERANCE:
                robot.node.get_logger().warn(
                    f"\033[91mHome is {home_error * 100:.1f} cm from where the training data "
                    f"has it ({np.round(position + offset, 4).tolist()} against "
                    f"{list(HOME_IN_DATASET)}). Every state fed to the policy is off by "
                    f"that much, which is off-distribution and will not be visible in "
                    f"anything except the success rate. Check --base-offset and "
                    f"--base-frame.\033[0m"
                )

        action = stream.next_action({"observation.state": state, **images_from(observation)}, step)
        dposition, drotvec, grip = delta_from_action(action, args.action_scale)
        # Clamped AFTER --action-scale: the scale is a calibration knob the operator
        # sets, the ceiling is what the training data actually contains.
        dposition, drotvec, was_clamped = clamp_delta(
            dposition, drotvec, args.max_step, args.max_rotation_step
        )
        clamped += was_clamped
        target_position, target_rotvec = apply_delta(position, rotvec, dposition, drotvec)
        q_command, residual = ik.solve(q, target_position, target_rotvec, args.ik_tolerance)

        if residual > args.ik_tolerance:
            # Not fatal and not silent: the QP always returns its best effort, so an
            # unreachable target (a singularity, a joint against its stop) would
            # otherwise look exactly like a policy that stopped asking to move.
            unreachable += 1

        jump = joint_step(q_command, q)
        if args.max_joint_step > 0 and jump > args.max_joint_step:
            # Hold instead of publishing. Nothing else re-publishes joint_command here,
            # so the arm simply stays where it is for this slot -- a dropped step costs
            # one period of motion, where a bad position target costs the episode and
            # slams the arm across the workspace.
            refused += 1
            robot.node.get_logger().warn(
                f"Step {step}: IK asked joint {int(np.argmax(np.abs(q_command - q)))} to "
                f"move {jump:.2f} rad in one period (limit {args.max_joint_step:g}). "
                f"Holding. Measured q={np.round(q, 3).tolist()}"
            )
        else:
            publish_command(robot, *command_from_action(list(q_command) + [grip], joints))

        grips.append((grip, observation[f"{gripper_joint}.pos"]))
        step_times.append(time.perf_counter() - started)

        if control.abort_episode.is_set():
            robot.node.get_logger().warn(f"Episode aborted after {step + 1} steps.")
            break

        # Polled until it first says yes, then latched: the rollout carries on so the
        # arm can fly itself home, and re-asking a service that has already answered
        # costs a round trip per second for nothing.
        if not succeeded and step % check_interval == check_interval - 1:
            if is_success(robot, scene_id):
                succeeded = True
                success_step = step
                robot.node.get_logger().info(
                    f"Task succeeded at step {step + 1}. Continuing until the arm is "
                    f"back within {args.home_return_tolerance:g} m of home."
                )

        if episode_is_finished(succeeded, home_error, args.home_return_tolerance):
            returned_home = True
            robot.node.get_logger().info(
                f"Arm back home at step {step + 1}, "
                f"{step - success_step} steps after succeeding."
            )
            break

        # Hold the cadence in SIM time, as eval_policy.py does: deadlines accumulate so
        # jitter does not drift the trajectory, but an overrun DROPS its slot rather
        # than being made up -- catching up would fire several setpoints back to back,
        # and a burst like that landing on the approach turns a grasp into a swipe.
        next_deadline = next_deadline + Duration(seconds=period)
        if clock.now() > next_deadline:
            next_deadline = clock.now()
        while clock.now() < next_deadline:
            time.sleep(0.002)

    stream.close()
    wall_seconds = time.perf_counter() - started_wall
    sim_seconds = (clock.now() - started_sim).nanoseconds / 1e9
    report_rollout(robot, args, policy, step_times, grips, stream.replans)
    if unreachable:
        robot.node.get_logger().warn(
            f"IK missed its {args.ik_tolerance:g} tolerance on {unreachable}/{len(step_times)} "
            f"steps -- the policy asked for poses this arm cannot reach from where it was "
            f"(singularity, joint limit, or a --action-scale that overshoots)."
        )
    if clamped:
        robot.node.get_logger().info(
            f"Clamped {clamped}/{len(step_times)} steps to {args.max_step:g} m / "
            f"{args.max_rotation_step:g} rad. A handful is the policy reproducing the "
            f"homing sweeps in its training set; most of the episode means the motion is "
            f"scaled wrong -- check --fps against the rate the demonstrations were "
            f"recorded at before reaching for --action-scale."
        )
    if refused:
        robot.node.get_logger().warn(
            f"\033[91mHeld {refused}/{len(step_times)} steps: the IK solution was more "
            f"than {args.max_joint_step:g} rad from the measured configuration. That is "
            f"not a policy problem -- check that joint_states is live and that "
            f"--base-frame matches the description.\033[0m"
        )
    if succeeded and not returned_home and args.home_return_tolerance > 0:
        # Scored a success either way -- the task is the pick and place, not the tidy-up
        # afterwards -- but worth seeing, because returning home is a quarter of what the
        # policy was trained on and failing to do it says the rollout drifted after the
        # place rather than finishing the demonstrated behaviour.
        robot.node.get_logger().warn(
            f"Task succeeded but the arm never got back within "
            f"{args.home_return_tolerance:g} m of home "
            f"({home_error:.3f} m at the last step)."
        )

    # Asked once more if the loop never got to poll (a rollout shorter than
    # check_interval), which is also what the old bool return did.
    success = bool(succeeded or is_success(robot, scene_id))
    return {
        "zone": zone,
        "seed": seed,
        "task": task,
        "success": success,
        "succeeded_in_loop": succeeded,
        "returned_home": returned_home,
        "aborted": control.abort_episode.is_set(),
        "steps": len(step_times),
        "success_step": success_step + 1 if succeeded else None,
        "wall_seconds": round(wall_seconds, 3),
        "sim_seconds": round(sim_seconds, 3),
        # Sim seconds to the moment of success, which is what "how fast is this
        # checkpoint" means -- wall time is whatever the GPU and the renderer were
        # doing, and total sim time now includes the homing return.
        "success_seconds": round(success_step / args.fps, 3) if succeeded else None,
        "median_step_ms": round(1000 * sorted(step_times)[len(step_times) // 2], 1)
        if step_times
        else None,
        "replans": stream.replans,
        "unreachable": unreachable,
        "clamped": clamped,
        "held": refused,
        "home_error": round(home_error, 4),
        "home_settle_error": round(settled, 4),
        # How far the tool got from home at its furthest, and the total path it walked.
        # A rollout that never leaves home has not failed the task so much as declined
        # to attempt it, which is a different thing to fix.
        "max_home_error": round(max(home_errors), 4) if home_errors else 0.0,
        "travelled": round(travelled, 3),
        # The two ways a grasp fails, as report_rollout already prints them: the policy
        # never commanding a close at all, versus commanding one and the fingers not
        # stalling on anything (they shut to ~0 on empty air, and to about the block's
        # 0.025 m half-width when there is a block between them).
        "grip_commanded_min": round(min(g[0] for g in grips), 4) if grips else None,
        "grip_measured_min": round(min(g[1] for g in grips), 4) if grips else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--policy", type=str, required=True, help="Checkpoint dir or HF repo id.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=60.0, help="Rollout timeout per episode.")
    # Demonstrations were captured at step_freq / record_interval = 60 / 12 Hz, so the
    # deltas span ~0.2 s each. Unlike the servo path nothing here divides by the period
    # -- a delta is applied as a displacement, not re-integrated as a velocity -- so
    # this rate only sets how often the policy looks.
    parser.add_argument("--fps", type=float, default=5.0, help="Control rate, Hz of SIM time.")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Actions used per policy call (<= the checkpoint's chunk_size).",
    )
    parser.add_argument(
        "--lead",
        type=int,
        default=3,
        help="Start predicting the next chunk with this many actions left. 0 blocks.",
    )
    parser.add_argument(
        "--state",
        choices=("libero", "joint"),
        default="libero",
        help="observation.state layout: 'libero' = the 8-dim [pos, axisangle, gripper x2] "
        "written by --libero-state, 'joint' = the 7 arm joints plus the gripper. Both are "
        "8 dims, so a wrong choice loads fine and simply drives badly -- check the first "
        "state line printed each episode against the dataset.",
    )
    # The one physical calibration knob: deltas are metric and each is executed as a
    # single position command, so a policy that consistently undershoots the block can
    # be nudged without retraining. Leave at 1.0 unless the motion is visibly short.
    parser.add_argument(
        "--action-scale", type=float, default=1.0, help="Multiplier on the commanded delta."
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=MAX_STEP_LINEAR,
        help=f"Metres the end effector may be asked to travel in one control step "
        f"(default {MAX_STEP_LINEAR}, the ceiling in libero_dataset_250). 0 disables.",
    )
    parser.add_argument(
        "--max-rotation-step",
        type=float,
        default=MAX_STEP_ANGULAR,
        help=f"Radians of end-effector rotation per control step (default "
        f"{MAX_STEP_ANGULAR}). Scaled with --max-step so the motion keeps its shape.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=MAX_JOINT_STEP,
        help=f"Radians a single joint may be commanded from its measured position "
        f"(default {MAX_JOINT_STEP}). Exceeding it holds the arm rather than "
        f"publishing. 0 disables the guard.",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="",
        help="URDF or xacro to load instead of subscribing to <ns>/robot_description.",
    )
    parser.add_argument(
        "--base-frame",
        type=str,
        default="",
        help="Frame the IK and FK are expressed in (default: the config's fr3_link0).",
    )
    parser.add_argument(
        "--base-offset",
        type=float,
        nargs=3,
        default=list(BASE_OFFSET),
        metavar=("X", "Y", "Z"),
        help=f"Added to the FK position to reach the dataset's 'world' frame "
        f"(default: {BASE_OFFSET}, derived from the scene origin and the arm mounting).",
    )
    parser.add_argument(
        "--home-settle-seconds",
        type=float,
        default=5.0,
        help="Seconds of SIM time to wait for the arm to reach home BEFORE the scene is "
        "randomized, so it cannot collide with the blocks and bins being placed. Polling "
        "stops as soon as it arrives, so a short trip costs nothing.",
    )
    parser.add_argument(
        "--home-return-tolerance",
        type=float,
        default=HOME_RETURN_TOLERANCE,
        help=f"Metres from home the arm must get back to before a SUCCEEDED episode "
        f"stops (default {HOME_RETURN_TOLERANCE:g}). The demonstrations all end homed, "
        f"so this scores the return the policy was trained to make. 0 stops at success, "
        f"as before. Scoring is unchanged either way -- a success that never comes home "
        f"is still a success, just a logged one.",
    )
    parser.add_argument(
        "--state-z-offset",
        type=float,
        default=STATE_Z_OFFSET,
        help=f"Metres added to the z the POLICY is shown (default {STATE_Z_OFFSET:g} = "
        f"the raw world-frame z the dataset stores, measured correct at the home pose; "
        f"the IK is never affected). -1 is the stats.json frame, which is off by ~1000 "
        f"standard deviations -- see STATE_Z_OFFSET.",
    )
    parser.add_argument(
        "--ik-tolerance",
        type=float,
        default=1e-4,
        help="Pose error Pink must reach before the solution is published. Steps that "
        "miss it are still commanded, and counted in the per-episode report.",
    )
    parser.add_argument(
        "--posture-cost",
        type=float,
        default=1e-3,
        help="Weight pulling the null space back towards the home posture. Raise if the "
        "elbow wanders into a joint limit over a rollout, lower if it fights the pose.",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="",
        help="Cube placement: blank = anywhere, 'all' = every zone, '2,16' = those "
        "zones, '2:4,16:10' = per-zone counts. --episodes is per zone.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="",
        help="Append one JSON object per episode to this file (JSON Lines). Written as "
        "each episode ends, so a run that is aborted or crashes still leaves everything "
        "it got through. This is what sweep_checkpoints reads.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=-1,
        help="Draw episode N's layout from seed <base>+N instead of the scene's own "
        "episode counter, so two runs with the same base see IDENTICAL scenes. That makes "
        "two policies comparable as paired samples, which is worth far more than the same "
        "number of independent episodes. -1 (default) leaves every episode a fresh draw. "
        "Only reproducible within one simulator session unless the scene's master seed was "
        "pinned at registration.",
    )
    parser.add_argument("--task", type=str, default="", help="Override the scene's instruction.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu (default: policy's)")
    args = parser.parse_known_args()[0]

    namespace_base = args.namespace or ""
    match = re.search(r"\d+$", namespace_base.split("/")[-1].strip())
    scene_id = int(match.group()) if match else 0
    sim_namespace = "/" + namespace_base.split("/")[1] if namespace_base else ""

    # Built before the policy loads so a bad --zone fails in a second rather than after
    # 450M parameters have been read off disk.
    plan = episode_plan(args.zone, args.episodes)
    print(f"Evaluating {len(plan)} episodes: {args.zone or 'unrestricted placement'}")

    policy, preprocessor, postprocessor = load_policy(args.policy, args.device)
    device = get_safe_torch_device(policy.config.device)
    if device.type == "cuda":
        # Named, not numbered: with CUDA_DEVICE_ORDER unset the runtime sorts
        # fastest-first, so torch's cuda:0 is not necessarily nvidia-smi's GPU 0.
        print(f"Inference on {device} = {torch.cuda.get_device_name(device)}")

    chunk_size = policy.config.chunk_size
    if not 1 <= args.n_action_steps <= chunk_size:
        raise SystemExit(f"--n-action-steps must be within 1..{chunk_size} (the trained chunk).")
    if args.n_action_steps != policy.config.n_action_steps:
        print(
            f"Action horizon: {args.n_action_steps} of {chunk_size} predicted steps "
            f"({args.n_action_steps / args.fps:.1f}s open-loop, was "
            f"{policy.config.n_action_steps / args.fps:.1f}s)"
        )
        # ONLY n_action_steps -- chunk_size is a trained quantity that sizes the action
        # token sequence, and _predict truncates to the horizon anyway. See the same
        # comment in eval_policy_servo.py.
        policy.config.n_action_steps = args.n_action_steps

    config = FR3RobotConfig(
        frame_id=namespace_base.split("/")[-1] if namespace_base else "world",
        namespace=f"{namespace_base}/franka",
        # Same as eval_policy.py: the IK's output is a joint position target, and it
        # goes onto joint_command directly. No MoveIt is running to plan through.
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

    args.base_frame = args.base_frame or config.base_link_name
    urdf = urdf_from_file(args.urdf) if args.urdf else wait_for_robot_description(robot.node)
    ik = ArmIK(
        urdf,
        list(config.arm_joint_names),
        config.end_effector_name,
        args.base_frame,
        args.posture_cost,
        np.array(HOME_POSITION, dtype=np.float64),
    )
    home_position, _ = ik.fk(np.array(HOME_POSITION))
    print(
        f"Pink IK on {ik.model.nq} joints, '{args.base_frame}' -> "
        f"'{config.end_effector_name}' ({'--urdf' if args.urdf else 'robot_description'}). "
        f"Home EEF at {np.round(home_position, 4).tolist()} in '{args.base_frame}', "
        f"{np.round(home_position + np.asarray(args.base_offset), 4).tolist()} in 'world'."
    )

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
            record = run_episode(
                robot,
                scene_id,
                zone,
                policy,
                (preprocessor, postprocessor),
                device,
                control,
                ik,
                args,
                seed=None if args.seed_base < 0 else args.seed_base + episode,
            )
            success = record["success"]
            if args.results:
                append_record(
                    args.results,
                    {"policy": args.policy, "episode": episode, "time": time.time(), **record},
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
