"""Roll out a LIBERO-style policy on block_bin through MoveIt Servo.

Sibling of ``eval_policy.py``. That script replays the recorded joint-space action
space -- it publishes joint targets straight onto ``joint_command`` and no MoveIt is
involved. This one is for a policy trained the LIBERO way, i.e. on a dataset built
with ``guide_dataset_build --eef-delta-action [--libero-state]``::

    action            = [dx, dy, dz, dwx, dwy, dwz, gripper]   (7)
    observation.state = [x, y, z, wx, wy, wz, g, g]            (8, --libero-state)

so the policy emits a *forward end-effector pose delta* per control step instead of
joint targets. A displacement over a control period is a velocity, which is exactly
what MoveIt Servo eats: every step publishes ``delta / period`` as a ``TwistStamped``
and servo owns the IK, joint limits, singularity handling and collision checking,
streaming a trajectory into ``fr3_arm_controller``. The arm is homed between episodes
through the MoveIt planning pipeline (``move_to_configuration``) and the gripper goes
through the ``franka_gripper`` GripperCommand action -- both the same paths
``solve_task.py`` uses to record demonstrations.

Rotation deltas are ``(R_{t+1} R_t^T).as_rotvec()`` (see
``guide_dataset_tools.build.eef_delta_action``), a LEFT-composed rotation about the
base-frame axes applied at the tool point -- which is precisely what servo's
``poseFromCartesianDelta`` does, so the deltas go across unmodified. The gripper
dimension stays an ABSOLUTE finger position in metres (GUIDE's convention, ~0..0.04),
not LIBERO's +-1.

Two frames matter and neither is negotiable. The state is read in ``world``, the frame
the recorder stored (via ``Scene_i`` it is a metre low, straight out of the training
distribution). The twist is published in ``fr3_link0``, servo's planning frame, because
any other frame sends it through a frame conversion that is miscoded in moveit_servo
2.12.4 -- see ``twist_frame_for``.

Running it
----------
1. Isaac with the scene, ``publish_camera_topics: true`` in ``config/init.yaml``.
2. MoveIt + servo, without the demonstration solver::

       ros2 launch block_bin eval_servo.launch.py

3. The rollout::

       ~/ros2_ws/.venv/bin/python -m block_bin.eval_policy_servo \
           --namespace /Sim_0/Scene_0 \
           --policy ~/models/smolvla_fr3_libero/checkpoints/last/pretrained_model \
           --episodes 20

Unlike ``eval_policy.py`` this one *needs* the MoveIt bring-up: servo is what turns
the deltas into joint commands. Nothing else may drive the arm at the same time, so
do not run ``solve_task`` alongside it.

A rollout that has clearly missed can be cut short:

    ros2 service call <namespace>/stop_episode std_srvs/srv/SetBool "{data: false}"

`data: false` aborts the episode and moves on, `data: true` also ends the run. An
aborted episode is still scored, so it counts as the failure it is.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from queue import Empty, Full, Queue
from threading import Thread, Event

import numpy as np
import torch
from geometry_msgs.msg import TwistStamped
from irob_lerobot_ros.config import ActionType, FR3RobotConfig, ROS2CameraConfig
from irob_lerobot_ros.ros2robot import ROS2Robot
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from block_bin.eval_policy import (
    CAMERAS,
    GRIPPER_OPEN,
    HOME_POSITION,
    ChunkStream,
    RunControl,
    episode_plan,
    get_safe_torch_device,
    handle_stop,
    images_from,
    is_success,
    load_policy,
    report_rollout,
    sleep_sim,
)
from guide_msgs.srv import CheckSuccess, Randomize

# [dx, dy, dz, dwx, dwy, dwz, gripper] -- guide_dataset_tools.build.DELTA_NAMES.
DELTA_DIMS = 7

# Servo's own topics, relative to the robot namespace the node is launched into.
SERVO_TWIST_TOPIC = "servo_node/delta_twist_cmds"
SERVO_SWITCH_SERVICE = "servo_node/switch_command_type"
SERVO_PAUSE_SERVICE = "servo_node/pause_servo"
SERVO_STATUS_TOPIC = "servo_node/status"
SERVO_PARAMETERS_SERVICE = "servo_node/get_parameters"

# The ServoStatus codes that mean the arm is not executing what it was told. Servo keeps
# publishing trajectories through all of them, so nothing downstream notices; the rollout
# just stops moving. Everything else (NO_WARNING, the DECELERATE_* warnings) still makes
# progress and is left alone -- decelerating near a singularity is servo working, not
# failing.
HALTING_STATUS = {
    ServoStatus.JOINT_BOUND: "a joint hit its limit and servo is halting",
    ServoStatus.HALT_FOR_SINGULARITY: "the arm ran into a singularity",
    ServoStatus.HALT_FOR_COLLISION: "servo halted for a collision",
    ServoStatus.INVALID: "servo rejected the command (no IK solution, or a bad twist frame)",
}
# ServoNode builds its ParamListener with the "moveit_servo" prefix, so that is the
# name the parameter is declared under -- asking for the bare name returns no values.
SERVO_TIMEOUT_PARAMETER = "moveit_servo.incoming_command_timeout"


def twist_frame_for(config, requested: str = "") -> str:
    """The frame a twist MUST be published in: servo's planning frame.

    This is not cosmetic. ``Servo::toPlanningFrame`` converts any twist whose frame
    differs from the planning frame, and in moveit_servo 2.12.4 that conversion is
    broken: for a ``[linear; angular]`` twist the adjoint blocks should be
    ``R, skew(t)R, 0, R`` but are coded as ``skew(t)R, R, R, 0``, which SWAPS the
    linear and angular halves (``omega_out = R*v_in``). Sent in the scene frame, 0.3 m
    from the arm base, the policy's wrist rotation ``dwz`` came out as vertical
    velocity and the arm climbed into a singularity while the policy was asking it to
    descend.

    Servo's planning frame is the IK solver's base frame -- ``fr3_arm``'s SRDF chain
    starts at ``fr3_link0`` -- so publishing there makes ``toPlanningFrame`` a no-op.
    The deltas need no conversion of their own: world, ``Scene_i`` and ``fr3_link0``
    differ by pure translations, so a delta has identical components in all three.
    """
    return requested or config.base_link_name


def libero_state(observation: dict, position, rotvec, gripper_joint: str) -> np.ndarray:
    """`[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]` -- the `--libero-state` layout.

    Mirrors ``guide_dataset_tools.build.libero_state``: GUIDE logs one finger and the
    Franka Hand's two mirror, so the gripper position is repeated rather than measured
    twice.
    """
    grip = float(observation[f"{gripper_joint}.pos"])
    return np.array([*position, *rotvec, grip, grip], dtype=np.float32)


def joint_state(observation: dict, joints: list[str]) -> np.ndarray:
    """The joint-space state layout, for a policy trained with delta actions only."""
    return np.array([observation[f"{j}.pos"] for j in joints], dtype=np.float32)


def twist_from_delta(action, period: float, scale: float = 1.0):
    """7-dim LIBERO action -> (linear m/s, angular rad/s, absolute gripper position).

    The action is a displacement over one control period, so dividing by the period is
    what turns it into the velocity servo wants -- and why `--fps` has to match the rate
    the demonstrations were sampled at, or every delta is executed at the wrong speed.
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != DELTA_DIMS:
        raise ValueError(
            f"Policy returned {values.size} dims, expected {DELTA_DIMS} "
            f"[dx dy dz dwx dwy dwz gripper]. Was this checkpoint trained on a dataset "
            f"built with --eef-delta-action? A joint-space policy belongs in eval_policy.py."
        )
    return values[0:3] * scale / period, values[3:6] * scale / period, float(values[6])


class EndEffector:
    """Absolute end-effector pose from TF, in the frame the dataset's poses used.

    The recorder stored the EEF prim's Isaac WORLD pose (``scene_orchestrator``), and
    Isaac places ``Scene_i`` at a pure translation of world, so the scene frame carries
    the same orientation and the same position up to that offset -- zero for Scene_0.
    Point ``--state-frame`` elsewhere if a scene sits at an offset and the absolute
    position matters to the policy.

    ``TransformListener`` is not used: it hard-codes the absolute ``/tf``, while every
    publisher here (Isaac's TF graphs, MoveIt's robot_state_publisher) is remapped into
    the scene's namespace, so it would listen to an empty topic forever. Subscribing
    relatively puts the same messages into the same Buffer.
    """

    def __init__(self, node, base_frame: str, ee_frame: str, offset=None):
        self.node = node
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
        self.buffer = Buffer(node=node)
        group = ReentrantCallbackGroup()
        node.create_subscription(
            TFMessage,
            "tf",
            lambda message: self._absorb(message, static=False),
            QoSProfile(depth=100, durability=DurabilityPolicy.VOLATILE),
            callback_group=group,
        )
        node.create_subscription(
            TFMessage,
            "tf_static",
            lambda message: self._absorb(message, static=True),
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=group,
        )

    def _absorb(self, message, static: bool) -> None:
        for transform in message.transforms:
            if static:
                self.buffer.set_transform_static(transform, "eval_policy_servo")
            else:
                self.buffer.set_transform(transform, "eval_policy_servo")

    def pose(self, timeout: float = 2.0):
        """(position xyz, orientation rotvec) of the end effector, latest available.

        The offset only shifts position -- the scene prim is placed by a pure
        translation, so TF and the dataset share an orientation.
        """
        transform = self.buffer.lookup_transform(
            self.base_frame, self.ee_frame, Time(), Duration(seconds=timeout)
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rotvec = Rotation.from_quat(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        ).as_rotvec()
        position = np.array([translation.x, translation.y, translation.z]) + self.offset
        return position, rotvec


class Servo:
    """Client for the Jazzy ``moveit_servo`` node.

    pymoveit2's ``MoveIt2Servo`` is deliberately not used: it drives the pre-Jazzy API
    (a ``start_servo`` Trigger that ``ServoNode`` no longer advertises, so ``enable()``
    just warns and drops every command) and pins the twist frame at construction.
    Jazzy instead wants a one-off ``switch_command_type`` to TWIST, and takes SetBool
    on ``pause_servo``.
    """

    def __init__(self, robot, frame_id: str, stall_grace: float = 2.0):
        self.robot = robot
        self.frame_id = frame_id
        self.stall_grace = stall_grace
        self._halted_since = None
        self._halt_reason = ""
        node = robot.node
        group = robot._reentrant_callback_group
        self.publisher = node.create_publisher(TwistStamped, SERVO_TWIST_TOPIC, 10)
        # SystemDefaultsQoS, matching ServoNode's status publisher (servo_node.cpp:127).
        node.create_subscription(
            ServoStatus, SERVO_STATUS_TOPIC, self._absorb_status, 10, callback_group=group
        )
        self.switch = node.create_client(
            ServoCommandType, SERVO_SWITCH_SERVICE, callback_group=group
        )
        self.pause_client = node.create_client(
            SetBool, SERVO_PAUSE_SERVICE, callback_group=group
        )
        self.parameters = node.create_client(
            GetParameters, SERVO_PARAMETERS_SERVICE, callback_group=group
        )

    def start(self, timeout: float = 30.0) -> None:
        """Wait for servo and put it in twist mode (it boots expecting joint jogs)."""
        for client in (self.switch, self.pause_client):
            if not client.wait_for_service(timeout_sec=timeout):
                raise SystemExit(
                    f"MoveIt Servo service '{client.srv_name}' never appeared. Start the "
                    f"bring-up first: ros2 launch block_bin eval_servo.launch.py"
                )
        request = ServoCommandType.Request(command_type=ServoCommandType.Request.TWIST)
        if not self.robot.callService(self.switch, request).success:
            raise SystemExit("MoveIt Servo refused to switch to TWIST commands.")
        self.pause(False)

    def _absorb_status(self, message) -> None:
        """Track how long servo has been continuously unable to execute."""
        reason = HALTING_STATUS.get(message.code)
        if reason is None:
            self._halted_since = None  # any healthy cycle clears it
            return
        if self._halted_since is None:
            self._halted_since = self.robot.node.get_clock().now()
            self._halt_reason = reason

    def clear_status(self) -> None:
        """Forget a halt from the previous episode, before a new one is scored on it."""
        self._halted_since = None

    def stalled(self) -> str:
        """Why servo has been stuck past the grace period, or '' while it is executing.

        Debounced rather than tripped on the first message, because these codes fire
        for a single cycle in normal operation -- a joint touching its margin on one
        tick, a twist arriving mid-transition -- and servo recovers on the next command.
        What is worth ending an episode over is the state persisting, which is what a
        saturated joint or a misconfigured IK solver actually looks like: every cycle,
        for the rest of the rollout.
        """
        if self.stall_grace <= 0 or self._halted_since is None:
            return ""
        stuck_for = self.robot.node.get_clock().now() - self._halted_since
        if stuck_for < Duration(seconds=self.stall_grace):
            return ""
        return self._halt_reason

    def check_command_lifetime(self, period: float) -> None:
        """Fail unless servo drops a twist exactly when its control period is up.

        This is what actually bounds how far one action travels, and it is not this
        script. A twist is a displacement over one period published as ``delta /
        period``, but ``ServoNode::processTwistCommand`` re-integrates the LAST twist
        it holds every ``publish_period`` (50 Hz here) until either a newer one
        arrives or the held one's header stamp ages past ``incoming_command_timeout``.
        So that timeout is the deadline the arm actually obeys whenever a publish is
        late -- a chunk re-plan ``ChunkStream`` could not hide, or a GIL stall while
        the policy runs inference in this same process. Set longer than the period,
        as the config shipped, and that step silently runs on at full commanded
        velocity for the difference: 0.3 s against a 0.2 s slot is half again the
        distance the policy asked for, on exactly the steps that were already late.
        Shorter, and the arm halts inside every slot instead.

        Checked rather than set, because the parameter cannot be changed once servo
        is up: ``ServoNode`` copies the whole parameter struct by value at
        construction (``servo_node.cpp:98``) and only ``Servo`` ever refreshes its own
        copy (``servo.cpp:476``), so the staleness test keeps reading the startup
        value no matter what is written to the parameter afterwards. It has to come
        from ``config/servo.yaml``, which means that file and ``--fps`` have to agree
        -- and nothing about the motion reveals the mismatch short of measuring the
        overshoot.
        """
        if not self.parameters.wait_for_service(timeout_sec=10.0):
            raise SystemExit(f"MoveIt Servo service '{SERVO_PARAMETERS_SERVICE}' never appeared.")
        response = self.robot.callService(
            self.parameters, GetParameters.Request(names=[SERVO_TIMEOUT_PARAMETER])
        )
        if not response.values:
            raise SystemExit(
                f"servo_node does not declare '{SERVO_TIMEOUT_PARAMETER}'. An unknown "
                f"name comes back as an empty list rather than an error, so this is "
                f"most likely the parameter prefix having moved in another moveit_servo "
                f"release: check `ros2 param list <ns>/servo_node`."
            )
        timeout = response.values[0].double_value
        if abs(timeout - period) > 1e-6:
            raise SystemExit(
                f"servo_node holds a twist for incoming_command_timeout={timeout}s but "
                f"the control period is {period}s (--fps {1 / period:g}), so a late step "
                f"would keep executing for {timeout / period:.2f}x its slot and travel "
                f"that much further than the policy asked for.\n"
                f"Fix one of the two:\n"
                f"  incoming_command_timeout: {period}   in block_bin/config/servo.yaml, "
                f"then relaunch eval_servo.launch.py (the value is latched when "
                f"servo_node starts)\n"
                f"  --fps {1 / timeout:g}   to match the servo already running"
            )

    def pause(self, paused: bool) -> None:
        """Hand the arm controller back to move_group (True) or to servo (False).

        Both write to ``fr3_arm_controller``: servo streams onto its ``joint_trajectory``
        topic while move_group executes through its action. Homing while servo is live
        means two writers fighting over the same joints, so the episode boundary pauses.
        """
        self.robot.callService(self.pause_client, SetBool.Request(data=paused))

    def send(self, linear, angular) -> None:
        message = TwistStamped()
        message.header.stamp = self.robot.node.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = (
            float(v) for v in linear
        )
        message.twist.angular.x, message.twist.angular.y, message.twist.angular.z = (
            float(v) for v in angular
        )
        self.publisher.publish(message)

    def stop(self) -> None:
        self.send(np.zeros(3), np.zeros(3))


class Gripper:
    """Forwards the action's gripper dimension, but only when it actually moves.

    The delta action carries an absolute finger position every single step. Sending all
    of them would fire a GripperCommand goal five times a second, each preempting the
    last mid-close, so the fingers never get to stall on the block and nothing is ever
    gripped. A deadband collapses that stream to the open/close transitions the policy
    means, which is all the binary LIBERO gripper ever encodes.
    """

    def __init__(self, robot, joint: str, deadband: float):
        self.robot = robot
        self.joint = joint
        self.deadband = deadband
        self.last = None

    def set(self, position: float) -> bool:
        if self.last is not None and abs(position - self.last) < self.deadband:
            return False
        self.last = position
        self.robot.send_action({f"{self.joint}.pos": float(position)})
        return True


def home(robot, servo: Servo, gripper: Gripper) -> None:
    """Plan back to the start pose with MoveIt, fingers open.

    Uses the planning pipeline rather than a joint_command like ``eval_policy.py``:
    with the MoveIt bring-up running, topic_based_ros2_control owns ``joint_command``
    and would fight a direct write. Opening first drops a block still held from a
    failed episode before the scene is re-randomized.
    """
    servo.stop()
    servo.pause(True)
    gripper.last = None  # a new episode re-commands the gripper even if unchanged
    gripper.set(GRIPPER_OPEN)
    joints = robot.config.arm_joint_names
    robot.send_action(
        {f"{joint}.pos": float(value) for joint, value in zip(joints, HOME_POSITION)},
        wait_for_execution=True,
    )
    servo.pause(False)


def run_servo(queue: Queue, stop_event: Event, arm, robot, args) -> None:
    """Publish one twist per control period, in SIM time, until stopped.

    The cadence lives here and inference lives on the control loop, with the queue the
    only thing between them: predicting a chunk costs ~0.5 s against a 0.2 s period
    (see ``ChunkStream``), so a twist published inline arrives late and the arm moves
    in bursts. Servo keeps executing whatever it was last handed, so a twist and its
    period are one unit -- publishing anything else before that period is up cancels
    the motion rather than adding to it.
    """
    servo, _, gripper = arm
    period = 1.0 / args.fps
    clock = robot.node.get_clock()
    next_deadline = clock.now()

    while not stop_event.is_set():
        try:
            linear, angular, grip = queue.get_nowait()
        except Empty:
            # The control loop missed its slot (a chunk re-plan it could not hide).
            # Hold still rather than coast for another period on a velocity the
            # policy has not re-confirmed.
            servo.stop()
        else:
            servo.send(linear, angular)
            gripper.set(grip)

        # Same cadence rule as eval_policy's loop, and BOTH halves of it matter.
        # Deadlines accumulate so jitter does not drift the trajectory, but an
        # overrun DROPS its slot instead of banking the debt: without the reset the
        # deadline stays in the past for the rest of the episode, and this loop then
        # publishes on every single pass -- a flood of zero twists landing
        # microseconds after each real one, which is exactly the arm not moving at
        # all. The sleep is what makes a pass cost a period instead of nothing.
        next_deadline = next_deadline + Duration(seconds=period)
        if clock.now() > next_deadline:
            next_deadline = clock.now()
        while clock.now() < next_deadline and not stop_event.is_set():
            time.sleep(0.002)


def run_episode(robot, scene_id, zone, policy, processors, device, control, arm, args) -> bool:
    """Home + randomize, then let the policy servo the arm until success or timeout."""
    servo, end_effector, gripper = arm
    gripper_joint = gripper.joint
    joints = list(robot.config.arm_joint_names) + [gripper_joint]

    home(robot, servo, gripper)

    response = robot.callService(
        robot.randomize,
        Randomize.Request(id=scene_id, use_zone=zone is not None, zone=zone or 0),
    )
    task = args.task or json.loads(response.message)["task"]
    robot.node.get_logger().info(f'Rolling out: "{task}"')

    sleep_sim(robot, 0.5)  # let the randomized scene settle, as solve_task does

    # maxsize=1: one action waiting behind the one executing, so inference hides
    # behind execution without the loop running so far ahead that it commands the arm
    # from an observation several periods old. A blocking put() IS the pacing -- it
    # returns when the servo thread takes the previous action, i.e. once per period.
    action_queue = Queue(maxsize=1)
    action_stop_event = Event()
    action_thread = Thread(
        target=run_servo,
        args=(action_queue, action_stop_event, arm, robot, args),
        daemon=True,
    )
    action_thread.start()

    preprocessor, postprocessor = processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    control.abort_episode.clear()
    servo.clear_status()  # a halt from the last episode must not end this one
    stream = ChunkStream(policy, preprocessor, postprocessor, device, task, robot.name, args.lead)

    period = 1.0 / args.fps
    steps = int(args.seconds * args.fps)
    check_interval = max(1, int(args.fps))  # poll the success service ~once a second
    step_times, grips = [], []
    succeeded = False

    try:
        for step in range(steps):
            started = time.perf_counter()

            observation = robot.get_observation()
            if args.state == "libero":
                # Only this layout needs the pose, so --state joint does not need TF.
                state = libero_state(observation, *end_effector.pose(), gripper_joint)
            else:
                state = joint_state(observation, joints)
            if step == 0:
                robot.node.get_logger().info(
                    f"First {args.state} state: [{', '.join(f'{v:.4f}' for v in state)}] "
                    f"({len(state)} dims, EEF in '{args.state_frame}')"
                )

            action = stream.next_action(
                {"observation.state": state, **images_from(observation)}, step
            )
            linear, angular, grip = twist_from_delta(action, period, args.action_scale)
            grips.append((grip, observation[f"{gripper_joint}.pos"]))
            # Timed before the handover, so this stays a measure of inference alone
            # and report_rollout's budget warning still means what it says.
            step_times.append(time.perf_counter() - started)

            try:
                action_queue.put((linear, angular, grip), timeout=10.0)
            except Full:
                raise RuntimeError(
                    "The servo thread stopped consuming actions -- nothing is driving "
                    "the arm. Check for an exception logged above."
                ) from None

            if control.abort_episode.is_set():
                robot.node.get_logger().warn(f"Episode aborted after {step + 1} steps.")
                break

            stalled = servo.stalled()
            if stalled:
                # Scored as the failure it is, but distinguishable in the log from a
                # policy that simply missed: without this the arm sits frozen for the
                # rest of the timeout and both look identical in the success rate.
                robot.node.get_logger().warn(
                    f"Episode ended after {step + 1} steps: {stalled}, and it has not "
                    f"recovered in {args.servo_stall}s. The remaining "
                    f"{steps - step - 1} steps would not have moved the arm."
                )
                break

            if step % check_interval == check_interval - 1 and is_success(robot, scene_id):
                succeeded = True
                break
    finally:
        # Every way out -- success, timeout, abort, exception -- has to stop the
        # thread, and stop it BEFORE the final zero twist or it publishes over that
        # zero and the arm keeps moving. Setting the event only on the abort path
        # left join() waiting on a loop that was never asked to end.
        action_stop_event.set()
        action_thread.join(timeout=5.0)
        servo.stop()

    stream.close()
    report_rollout(robot, args, policy, step_times, grips, stream.replans)
    return succeeded or is_success(robot, scene_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--policy", type=str, required=True, help="Checkpoint dir or HF repo id.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=60.0, help="Rollout timeout per episode.")
    # Demonstrations were captured at step_freq / record_interval = 60 / 12 Hz, so the
    # deltas span ~0.2 s each. This rate is what converts them back into a velocity --
    # get it wrong and every motion is executed proportionally too fast or too slow.
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
    # The one physical calibration knob: the deltas are metric and executed open-loop
    # over a period, so a policy that consistently undershoots the block can be nudged
    # without retraining. Leave at 1.0 unless the motion is visibly short or long.
    parser.add_argument(
        "--action-scale", type=float, default=1.0, help="Multiplier on the commanded delta."
    )
    parser.add_argument(
        "--state-frame",
        type=str,
        default="",
        help="TF frame for the absolute EEF pose (default: 'world', the frame the "
        "recorder stored). Via 'Scene_0' instead, add --state-offset 0 0 1.",
    )
    parser.add_argument(
        "--state-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Added to the looked-up EEF position (default: none, 'world' is already "
        "the dataset's frame).",
    )
    parser.add_argument(
        "--twist-frame",
        type=str,
        default="",
        help="TF frame the twist is published in (default: fr3_link0, servo's planning "
        "frame). Anything else triggers a broken frame conversion in servo 2.12.4.",
    )
    parser.add_argument(
        "--servo-stall",
        type=float,
        default=2.0,
        help="Seconds servo may keep reporting a halting status (joint limit, "
        "singularity, collision, or a rejected command) before the episode is "
        "abandoned and scored a failure. 0 runs the timeout out instead.",
    )
    parser.add_argument(
        "--gripper-deadband",
        type=float,
        default=0.005,
        help="Metres of commanded finger travel before a new gripper goal is sent.",
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
    # 'Scene_0' -- Isaac's scene prim, and the root link of the MoveIt model (the
    # bring-up mounts the arm with connected_to/base_frame:=Scene_i), so it is both the
    # planning frame and the frame the recorded poses are expressed in.
    # The recorder stored the EEF's Isaac WORLD pose, and Isaac publishes that frame
    # itself (its TF graph roots the scene prim at `world`), so read the state straight
    # out of it. Going through the scene frame instead costs a metre: SceneManager puts
    # `Scene_i` at -origin, which is [0, 0, 1.0] for block_bin.
    args.state_frame = args.state_frame or "world"
    offset = np.zeros(3) if args.state_offset is None else np.asarray(args.state_offset)

    # Built before the policy loads so a bad --zone fails in a second rather than after
    # 450M parameters have been read off disk.
    plan = episode_plan(args.zone, args.episodes)
    print(f"Evaluating {len(plan)} episodes: {args.zone or 'unrestricted placement'}")

    policy, preprocessor, postprocessor = load_policy(args.policy, args.device)
    device = get_safe_torch_device(policy.config.device)
    if device.type == "cuda":
        # Named, not numbered: with CUDA_DEVICE_ORDER unset the runtime sorts
        # fastest-first, so torch's cuda:0 is not necessarily nvidia-smi's GPU 0. Sharing
        # a GPU with Isaac's rendering is the difference between a 0.5 s chunk and a
        # multi-second one, and the index alone will not tell you whether you are.
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
        # ONLY n_action_steps. chunk_size is a trained quantity, not a preference: it
        # sizes the action-token sequence the expert denoises
        # (`actions_shape = (bsize, chunk_size, max_action_dim)`, modeling_smolvla.py:827,
        # and the matching `att_masks += [1] * chunk_size`). Overwriting it made a
        # checkpoint trained on 50 action tokens run on 10 -- off-distribution in exactly
        # the dimension the policy predicts. It also bought nothing: _predict already
        # truncates with `horizon = min(n_action_steps, chunk.shape[1])`, so the extra
        # actions are dropped either way. eval_policy.py never touched it.
        policy.config.n_action_steps = args.n_action_steps

    config = FR3RobotConfig(
        frame_id='Scene_0',
        namespace=f"{namespace_base}/franka",
        # MoveIt owns the arm here: JOINT_POSITION without directly_publish gives
        # move_to_configuration for homing, and the gripper action for the fingers.
        # The rollout itself bypasses both and goes to servo.
        arm_action_type=ActionType.JOINT_POSITION,
        gripper_action_type=ActionType.JOINT_POSITION,
        directly_publish=False,
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

    args.twist_frame = twist_frame_for(config, args.twist_frame)
    servo = Servo(robot, args.twist_frame, args.servo_stall)
    servo.start()
    servo.check_command_lifetime(1.0 / args.fps)
    print(
        f"MoveIt Servo in TWIST mode, commanding in frame '{args.twist_frame}', "
        f"dropping each twist after its {1 / args.fps:.3f}s slot."
    )
    if args.twist_frame != config.base_link_name:
        print(
            f"\033[91mWARNING: twists are not in servo's planning frame "
            f"('{config.base_link_name}'), so Servo::toPlanningFrame will convert them. "
            f"In moveit_servo 2.12.4 that conversion swaps the linear and angular halves "
            f"of the twist -- the arm will climb and spin instead of following the "
            f"policy. See twist_frame_for().\033[0m"
        )

    # Built once: each one adds subscriptions/clients to the robot's node, and rebuilding
    # them per episode would stack a fresh TF listener on top of the last.
    arm = (
        servo,
        EndEffector(robot.node, args.state_frame, robot.config.end_effector_name, offset),
        Gripper(robot, robot.config.gripper_joint_names[0], args.gripper_deadband),
    )
    if args.state == "libero":
        print(f"EEF pose: TF '{args.state_frame}' -> '{robot.config.end_effector_name}' + {offset}")

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
                robot,
                scene_id,
                zone,
                policy,
                (preprocessor, postprocessor),
                device,
                control,
                arm,
                args,
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
        servo.stop()
        servo.pause(True)
        print(f"Success rate: {successes}/{attempted}")
        if args.zone:
            for zone in sorted(per_zone, key=lambda z: -1 if z is None else z):
                hits, tries = per_zone[zone]
                print(f"  zone {zone}: {hits}/{tries}")
        robot.disconnect()


if __name__ == "__main__":
    main()
