import time
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

# Same as test_eval_policy: this only imports under the Isaac venv with the ROS
# workspace sourced (lerobot, moveit_msgs, tf2_ros). Skip elsewhere.
es = pytest.importorskip("block_bin.eval_policy_servo")

JOINTS = [f"j{i}" for i in range(1, 8)] + ["fr3_finger_joint1"]


def test_libero_state_repeats_the_gripper():
    position = np.array([0.4, -0.1, 0.25])
    rotvec = np.array([3.1, 0.02, -0.05])

    state = es.libero_state({"fr3_finger_joint1.pos": 0.031}, position, rotvec, "fr3_finger_joint1")

    assert state.shape == (8,)
    assert state.dtype == np.float32
    # [eef_pos(3), eef_axisangle(3), gripper_qpos(2)] -- the two fingers mirror.
    assert state == pytest.approx([0.4, -0.1, 0.25, 3.1, 0.02, -0.05, 0.031, 0.031], abs=1e-6)


class FakeConfig:
    base_link_name = "fr3_link0"


def test_twist_frame_defaults_to_servos_planning_frame():
    # Not cosmetic: any other frame goes through Servo::toPlanningFrame, whose adjoint
    # is miscoded in moveit_servo 2.12.4 and swaps the twist's linear and angular
    # halves -- the arm climbs on the policy's wrist rotation instead of descending.
    assert es.twist_frame_for(FakeConfig()) == "fr3_link0"


def test_twist_frame_can_still_be_overridden():
    assert es.twist_frame_for(FakeConfig(), "world") == "world"


def test_servo_config_takes_the_rotation_only_conversion_path():
    # If a twist ever does arrive in a foreign frame, this keeps it off the broken
    # adjoint branch.
    import yaml
    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("block_bin")
    config = yaml.safe_load(open(f"{share}/config/servo.yaml"))
    assert config["apply_twist_commands_about_ee_frame"] is True


def test_joint_state_follows_the_dataset_order():
    observation = {**{f"j{i}.pos": float(i) for i in range(1, 8)}, "fr3_finger_joint1.pos": 0.03}

    state = es.joint_state(observation, JOINTS)

    assert state == pytest.approx([1, 2, 3, 4, 5, 6, 7, 0.03])


def test_delta_becomes_a_velocity_over_the_control_period():
    action = np.array([0.02, -0.01, 0.005, 0.1, 0.0, -0.2, 0.04], dtype=np.float32)

    linear, angular, grip = es.twist_from_delta(action, period=0.2)

    # A 2 cm step per 0.2 s slot is 0.1 m/s; the gripper stays an absolute position.
    assert linear == pytest.approx([0.1, -0.05, 0.025], abs=1e-6)
    assert angular == pytest.approx([0.5, 0.0, -1.0], abs=1e-6)
    assert grip == pytest.approx(0.04)


def test_action_scale_stretches_only_the_motion():
    action = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04])

    linear, _, grip = es.twist_from_delta(action, period=0.2, scale=1.5)

    assert linear == pytest.approx([0.15, 0.0, 0.0], abs=1e-6)
    assert grip == pytest.approx(0.04)  # not a distance, so never scaled


def test_batched_policy_action_is_accepted():
    action = np.array([[0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    linear, _, _ = es.twist_from_delta(action, period=0.2)

    assert linear == pytest.approx([0.1, 0.0, 0.0], abs=1e-6)


def test_joint_space_action_is_rejected_with_a_pointer_to_the_other_script():
    # 8 dims = the joint-space action space; eval_policy.py is the script for that.
    with pytest.raises(ValueError, match="eval_policy.py"):
        es.twist_from_delta(np.zeros(8), period=0.2)


class FakeRobot:
    def __init__(self, node=None):
        self.node = node
        self.sent = []

    def send_action(self, action):
        self.sent.append(action)


def test_gripper_sends_the_first_command_then_only_real_moves():
    robot = FakeRobot()
    gripper = es.Gripper(robot, "fr3_finger_joint1", deadband=0.005)

    assert gripper.set(0.04) is True  # nothing commanded yet, so this is a move
    assert gripper.set(0.039) is False  # 1 mm of policy noise, not an intent to close
    assert gripper.set(0.01) is True  # an actual close

    assert robot.sent == [{"fr3_finger_joint1.pos": 0.04}, {"fr3_finger_joint1.pos": 0.01}]


def test_gripper_deadband_is_measured_from_the_last_command_not_the_last_step():
    robot = FakeRobot()
    gripper = es.Gripper(robot, "fr3_finger_joint1", deadband=0.005)
    gripper.set(0.04)

    # Four 2 mm steps: each is inside the deadband on its own, but they add up past it.
    # Comparing against the last COMMANDED position lets the run through at 0.034;
    # comparing against the previous step would swallow the close entirely.
    for target in (0.038, 0.036, 0.034, 0.032):
        gripper.set(target)

    assert robot.sent == [{"fr3_finger_joint1.pos": 0.04}, {"fr3_finger_joint1.pos": 0.034}]


class FakeClock:
    """Steps itself forward on every read, so the cadence is checked without waiting."""

    def __init__(self, tick: float = 0.01):
        self.seconds = 0.0
        self.tick = tick

    def now(self):
        self.seconds += self.tick
        return es.Time(seconds=self.seconds)


class FakeServo:
    def __init__(self, clock):
        self.clock = clock
        self.sent = []  # (sim time of the publish, commanded vz)

    def send(self, linear, angular):
        self.sent.append((self.clock.seconds, float(np.asarray(linear)[2])))

    def stop(self):
        self.send(np.zeros(3), np.zeros(3))


def test_servo_thread_never_publishes_twice_inside_one_control_period():
    """The regression that made servo look like it never started.

    A twist stays in effect until the next one replaces it, so a second message
    inside the same period does not add to the motion, it cancels it. The deadline
    used to advance one period per action CONSUMED, so as soon as the policy fell
    behind its 0.2 s budget it stayed in the past for the rest of the episode and
    the loop published on every pass: each real twist was overwritten by a flood of
    zeros microseconds later and the arm never moved.
    """
    clock = FakeClock()
    servo = FakeServo(clock)
    robot = FakeRobot(node=SimpleNamespace(get_clock=lambda: clock))
    gripper = es.Gripper(robot, "fr3_finger_joint1", deadband=0.005)
    args = SimpleNamespace(fps=5.0)  # 0.2 s period

    queue = Queue()
    for vz in (-0.01, -0.02):
        queue.put((np.array([0.0, 0.0, vz]), np.zeros(3), 0.04))

    stop = Event()
    thread = Thread(
        target=es.run_servo,
        args=(queue, stop, (servo, None, gripper), robot, args),
        daemon=True,
    )
    thread.start()
    wall_deadline = time.monotonic() + 10.0
    while len(servo.sent) < 5 and time.monotonic() < wall_deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "run_servo ignored its stop event"
    assert len(servo.sent) >= 5, f"the thread stopped publishing after {len(servo.sent)}"
    # Both queued actions went out, in order, once each -- the rest are the starved
    # slots, which hold still rather than coasting on a stale velocity.
    assert [vz for _, vz in servo.sent if vz != 0.0] == [-0.01, -0.02]
    gaps = [later - earlier for (earlier, _), (later, _) in zip(servo.sent, servo.sent[1:])]
    assert min(gaps) >= 0.2 - 3 * clock.tick, (
        f"published {min(gaps) * 1000:.0f} ms apart, the control period is 200 ms"
    )


class FakeParamRobot:
    """Answers GetParameters with one double, the way servo_node's node does."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.node = None

    def callService(self, client, request):
        assert list(request.names) == [es.SERVO_TIMEOUT_PARAMETER]
        # An unknown name comes back as an empty list, not an error.
        values = [] if self.timeout is None else [SimpleNamespace(double_value=self.timeout)]
        return SimpleNamespace(values=values)


def servo_with_reported_timeout(timeout):
    servo = es.Servo.__new__(es.Servo)  # the real __init__ needs a live ROS node
    servo.robot = FakeParamRobot(timeout)
    servo.parameters = SimpleNamespace(wait_for_service=lambda timeout_sec: True)
    return servo


def test_a_twist_may_not_outlive_the_period_that_scaled_it():
    """0.3 s against a 0.2 s slot is 1.5x the distance the policy asked for.

    servo re-integrates the last twist until this timeout expires, so it -- not the
    control loop -- is what stops the arm whenever the next publish is late. The
    parameter is latched at servo_node startup, so a mismatch with --fps cannot be
    corrected at runtime and has to fail loudly instead.
    """
    with pytest.raises(SystemExit, match="1.50x its slot"):
        servo_with_reported_timeout(0.3).check_command_lifetime(0.2)


def test_a_timeout_shorter_than_the_period_is_rejected_too():
    # Halts the arm inside every slot rather than overshooting past it.
    with pytest.raises(SystemExit, match="incoming_command_timeout"):
        servo_with_reported_timeout(0.1).check_command_lifetime(0.2)


def test_matching_timeout_and_period_is_accepted():
    assert servo_with_reported_timeout(0.2).check_command_lifetime(0.2) is None


def test_the_shipped_config_matches_the_default_fps():
    """--fps default is 5.0, so servo.yaml has to hold a twist for exactly 0.2 s."""
    import yaml
    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("block_bin")
    config = yaml.safe_load(open(f"{share}/config/servo.yaml"))
    assert config["incoming_command_timeout"] == pytest.approx(0.2)


def test_an_unknown_parameter_name_is_not_mistaken_for_a_match():
    # GetParameters answers an unknown name with an empty list rather than an error,
    # so the prefix moving in a future moveit_servo would otherwise pass silently.
    with pytest.raises(SystemExit, match="does not declare"):
        servo_with_reported_timeout(None).check_command_lifetime(0.2)


def _config(name):
    import yaml
    from ament_index_python.packages import get_package_share_directory

    return yaml.safe_load(open(f"{get_package_share_directory('block_bin')}/config/{name}"))


def test_servo_resolves_redundancy_with_a_centering_solver():
    """fr3_arm is 7-DOF on a 6-DOF task; KDL leaves the spare DOF to random-walk.

    All three of these weights default to 0.0 in pick_ik, which reproduces exactly the
    KDL behaviour this file exists to replace -- so the plugin being selected is not on
    its own enough, and a zeroed weight would look like a working config.
    """
    arm = _config("kinematics.yaml")["fr3_arm"]

    assert arm["kinematics_solver"] == "pick_ik/PickIkPlugin"
    # `global` runs a population search that can return a different IK branch between
    # consecutive 20 ms cycles -- a joint jump straight into the arm controller.
    assert arm["mode"] == "local"


def test_the_ik_plugin_name_matches_what_pluginlib_exports():
    # A typo here only surfaces as a servo_node that fails to come up at launch.
    from ament_index_python.packages import get_package_share_directory

    description = open(
        f"{get_package_share_directory('pick_ik')}/pick_ik_kinematics_description.xml"
    ).read()
    assert f'name="{_config("kinematics.yaml")["fr3_arm"]["kinematics_solver"]}"' in description


def test_one_saturated_joint_does_not_freeze_the_whole_arm():
    # Servo::haltJoints replaces the target with the current state when this is true, so
    # a wrist the policy keeps driving outward stops all seven joints for the rest of the
    # episode and the rollout scores as a failure that never ran.
    assert _config("servo.yaml")["halt_all_joints_in_cartesian_mode"] is False


def test_servo_is_wired_to_the_projects_kinematics_not_frankas():
    """The pick_ik config only reaches servo_node through describe_robot().

    move_group is brought up separately by guide_moveit.launch.py and keeps stock KDL,
    which is deliberate -- planning and demonstration recording must not change.
    """
    from ament_index_python.packages import get_package_share_directory

    launch = open(
        f"{get_package_share_directory('block_bin')}/launch/eval_servo.launch.py"
    ).read()
    assert "load_yaml('block_bin', 'config/kinematics.yaml')" in launch


def servo_watching_status(grace, clock):
    """A Servo with only the status-watching half wired up."""
    servo = es.Servo.__new__(es.Servo)
    servo.robot = SimpleNamespace(node=SimpleNamespace(get_clock=lambda: clock))
    servo.stall_grace = grace
    servo._halted_since = None
    servo._halt_reason = ""
    return servo


class StepClock:
    """Sim time the test advances by hand."""

    def __init__(self):
        self.seconds = 0.0

    def now(self):
        return es.Time(seconds=self.seconds)


def test_a_one_cycle_halt_does_not_end_the_episode():
    # These codes fire for a single cycle in normal operation -- a joint touching its
    # margin, a twist arriving mid-transition -- and servo recovers on the next command.
    clock = StepClock()
    servo = servo_watching_status(2.0, clock)

    servo._absorb_status(SimpleNamespace(code=es.ServoStatus.JOINT_BOUND))
    clock.seconds = 0.5
    assert servo.stalled() == ""

    servo._absorb_status(SimpleNamespace(code=es.ServoStatus.NO_WARNING))
    clock.seconds = 10.0
    assert servo.stalled() == "", "a healthy cycle must clear the halt"


def test_a_sustained_joint_limit_ends_the_episode():
    clock = StepClock()
    servo = servo_watching_status(2.0, clock)

    servo._absorb_status(SimpleNamespace(code=es.ServoStatus.JOINT_BOUND))
    clock.seconds = 1.9
    assert servo.stalled() == ""
    clock.seconds = 2.1
    assert "joint hit its limit" in servo.stalled()


def test_the_ik_failure_that_error_31_reports_is_caught_too():
    # servo sets StatusCode::INVALID when searchPositionIK fails, so a misconfigured
    # solver ends the episode instead of silently running out the clock.
    clock = StepClock()
    servo = servo_watching_status(1.0, clock)

    servo._absorb_status(SimpleNamespace(code=es.ServoStatus.INVALID))
    clock.seconds = 1.5
    assert "no IK solution" in servo.stalled()


def test_decelerating_near_a_singularity_is_not_a_stall():
    # Servo is still making progress here; only the HALT_* codes mean it is not.
    clock = StepClock()
    servo = servo_watching_status(1.0, clock)

    servo._absorb_status(
        SimpleNamespace(code=es.ServoStatus.DECELERATE_FOR_APPROACHING_SINGULARITY)
    )
    clock.seconds = 100.0
    assert servo.stalled() == ""


def test_the_stall_check_can_be_turned_off():
    clock = StepClock()
    servo = servo_watching_status(0.0, clock)

    servo._absorb_status(SimpleNamespace(code=es.ServoStatus.JOINT_BOUND))
    clock.seconds = 1000.0
    assert servo.stalled() == ""


def test_secondary_ik_weights_cannot_outrank_the_pose_they_are_gated_on():
    """The -31 regression: weight^2 * goal_eval must stay under the pose cost.

    pick_ik minimises pose_cost + sum(goal_eval * weight^2) and then accepts only if the
    pose is still within position_threshold. At the acceptance boundary the pose term is
    position_threshold^2, so a weight big enough to make the goal term exceed that buys
    posture at the cost of the pose and every solve is rejected.
    """
    arm = _config("kinematics.yaml")["fr3_arm"]
    pose_cost = (arm["position_threshold"] * arm["position_scale"]) ** 2
    # Measured on this arm: 0.054 centred at home, 0.126 in the drifted posture.
    worst_goal_eval = 0.13

    # A weight may legitimately be absent or 0.0 -- that just turns its goal off, which
    # is pick_ik behaving like KDL for redundancy. What must never happen is a weight
    # large enough to outrank the pose, because then NOTHING solves.
    for name in ("center_joints_weight", "avoid_joint_limits_weight",
                 "minimal_displacement_weight"):
        if not arm.get(name):
            continue
        goal_cost = worst_goal_eval * arm[name] ** 2
        assert goal_cost < pose_cost, (
            f"{name}={arm[name]} gives a goal cost of {goal_cost:.2e}, "
            f"{goal_cost / pose_cost:.0f}x the {pose_cost:.0e} pose cost -> NO_IK_SOLUTION"
        )
        # And it must still clear pick_ik's own per-goal veto, which squares the threshold.
        assert goal_cost < arm["cost_threshold"] ** 2
