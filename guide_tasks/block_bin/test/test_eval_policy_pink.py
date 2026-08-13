import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# Same as the other eval tests: this only imports under the Isaac venv with the ROS
# workspace sourced (lerobot, pinocchio, pink, guide_msgs). Skip elsewhere.
ep = pytest.importorskip("block_bin.eval_policy_pink")

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
TOLERANCE = 1e-4


def _xacro(*arguments: str) -> str:
    import os
    import subprocess

    from ament_index_python.packages import get_package_share_directory

    path = os.path.join(
        get_package_share_directory("franka_description"), "robots", "fr3", "fr3.urdf.xacro"
    )
    finished = subprocess.run(
        ["xacro", path, "hand:=true", "ee_id:=franka_hand", *arguments],
        capture_output=True,
        text=True,
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout


@pytest.fixture(scope="module")
def urdf() -> str:
    """The bare FR3 description -- rooted at ``base``, which sits on ``fr3_link0``."""
    return _xacro()


@pytest.fixture(scope="module")
def scene_urdf() -> str:
    """What ``eval_pink.launch.py`` actually publishes: rooted at ``Scene_0``.

    The arm hangs off the scene at ``xyz="-0.3 0 0"``, so this description's root is
    NOT ``fr3_link0``. Every IK test that uses the bare URDF above is blind to any
    frame confusion between the two, because there they coincide.
    """
    return _xacro(
        "ros2_control:=true",
        "use_topic_based:=true",
        "connected_to:=Scene_0",
        "base_frame:=Scene_0",
        'xyz:=-0.3 0 0',
        'rpy:=0 0 0',
        "joint_states_topic:=/js",
        "joint_commands_topic:=/jc",
    )


@pytest.fixture(scope="module")
def scene_ik(scene_urdf):
    return ep.ArmIK(scene_urdf, ARM_JOINTS, "fr3_hand_tcp", "fr3_link0", 1e-3, HOME)


@pytest.fixture(scope="module")
def ik(urdf):
    return ep.ArmIK(urdf, ARM_JOINTS, "fr3_hand_tcp", "fr3_link0", 1e-3, HOME)


def test_the_fingers_are_locked_out_of_the_ik(ik):
    # Prismatic finger joints on the same tree would let the IK "reach" a target by
    # opening the gripper instead of moving the arm.
    assert ik.model.nq == 7
    assert [ik.model.names[i + 1] for i in range(ik.model.nq)] == ARM_JOINTS


def test_a_zero_delta_solves_back_to_where_the_arm_already_is(ik):
    position, rotvec = ik.fk(HOME)

    solution, residual = ik.solve(HOME, position, rotvec, TOLERANCE)

    assert residual < TOLERANCE
    assert solution == pytest.approx(HOME, abs=1e-3)


def test_a_known_delta_lands_on_the_pose_it_asked_for(ik):
    position, rotvec = ik.fk(HOME)
    target_position, target_rotvec = ep.apply_delta(
        position, rotvec, [0.0, 0.02, -0.05], [0.0, 0.0, 0.1]
    )

    solution, residual = ik.solve(HOME, target_position, target_rotvec, TOLERANCE)

    assert residual < TOLERANCE
    reached_position, reached_rotvec = ik.fk(solution)
    assert reached_position == pytest.approx(target_position, abs=1e-3)
    relative = Rotation.from_rotvec(reached_rotvec) * Rotation.from_rotvec(target_rotvec).inv()
    assert relative.magnitude() < 1e-2


def test_the_scene_rooted_description_is_the_one_the_launch_file_publishes(scene_urdf, scene_ik):
    # Guards the fixture itself: if this ever stopped being rooted at Scene_0 the
    # regression tests below would silently go back to testing nothing.
    assert 'link name="Scene_0"' in scene_urdf
    assert scene_ik.model.existFrame("Scene_0")
    assert not scene_ik.model.existFrame("base")


def test_a_scene_rooted_description_solves_the_pose_it_was_actually_given(scene_ik):
    # THE regression. fk() reports relative to fr3_link0, but Pink's FrameTask measures
    # against the model ROOT -- Scene_0 here, 0.3 m behind fr3_link0. Without the
    # conversion the solver chases a target 0.3 m forward and converges on it, which is
    # the arm launching on the first action of an episode.
    position, rotvec = scene_ik.fk(HOME)
    target_position, target_rotvec = ep.apply_delta(
        position, rotvec, [0.0, 0.0, -0.0226], np.zeros(3)
    )

    solution, residual = scene_ik.solve(HOME, target_position, target_rotvec, TOLERANCE)

    assert residual < TOLERANCE
    reached, _ = scene_ik.fk(solution)
    assert reached == pytest.approx(target_position, abs=1e-3)


def test_a_small_move_stays_a_small_joint_move_on_the_scene_rooted_model(scene_ik):
    # The observable symptom: 2.3 cm down became a 0.96 rad joint jump, and
    # MAX_JOINT_STEP was set loose enough (1.2) that the guard never fired.
    position, rotvec = scene_ik.fk(HOME)
    target_position, target_rotvec = ep.apply_delta(
        position, rotvec, [0.0, 0.0, -0.0226], np.zeros(3)
    )

    solution, _ = scene_ik.solve(HOME, target_position, target_rotvec, TOLERANCE)

    assert ep.joint_step(solution, HOME) < 0.15


def test_both_descriptions_agree_on_the_solution(ik, scene_ik):
    # Where the arm is mounted in the scene must not change the joint angles that
    # reach a pose expressed relative to fr3_link0.
    position, rotvec = ik.fk(HOME)
    scene_position, scene_rotvec = scene_ik.fk(HOME)
    assert scene_position == pytest.approx(position, abs=1e-9)

    target_position, target_rotvec = ep.apply_delta(
        position, rotvec, [0.01, -0.02, -0.03], [0.0, 0.05, 0.0]
    )
    plain, _ = ik.solve(HOME, target_position, target_rotvec, TOLERANCE)
    scene, _ = scene_ik.solve(HOME, target_position, target_rotvec, TOLERANCE)

    assert scene == pytest.approx(plain, abs=1e-3)


def test_joint_order_is_resolved_by_name_not_position(urdf):
    # A description that reorders its joints must not silently permute every command.
    shuffled = list(reversed(ARM_JOINTS))
    straight = ep.ArmIK(urdf, ARM_JOINTS, "fr3_hand_tcp", "fr3_link0", 1e-3, HOME)
    reversed_ik = ep.ArmIK(urdf, shuffled, "fr3_hand_tcp", "fr3_link0", 1e-3, HOME[::-1])

    position, rotvec = straight.fk(HOME)
    reversed_position, reversed_rotvec = reversed_ik.fk(HOME[::-1])

    assert reversed_position == pytest.approx(position, abs=1e-9)
    assert reversed_rotvec == pytest.approx(rotvec, abs=1e-9)


def test_solutions_come_back_in_the_callers_joint_order(urdf):
    reversed_ik = ep.ArmIK(urdf, list(reversed(ARM_JOINTS)), "fr3_hand_tcp", "fr3_link0", 1e-3,
                           HOME[::-1])
    position, rotvec = reversed_ik.fk(HOME[::-1])

    solution, residual = reversed_ik.solve(HOME[::-1], position, rotvec, TOLERANCE)

    assert residual < TOLERANCE
    assert solution == pytest.approx(HOME[::-1], abs=1e-3)


def test_a_missing_joint_is_rejected_rather_than_silently_dropped(urdf):
    with pytest.raises(SystemExit, match="panda_joint1"):
        ep.ArmIK(urdf, ["panda_joint1"], "fr3_hand_tcp", "fr3_link0", 1e-3, HOME[:1])


def test_a_missing_frame_is_rejected(urdf):
    with pytest.raises(SystemExit, match="tool0"):
        ep.ArmIK(urdf, ARM_JOINTS, "tool0", "fr3_link0", 1e-3, HOME)


def test_the_delta_is_composed_the_way_the_dataset_decomposed_it():
    # eef_delta_action stores (R_{t+1} * R_t^T).as_rotvec(), so the inverse is a LEFT
    # composition. Adding rotation vectors componentwise passes for small deltas and
    # then wraps at +-pi; this delta is large enough to tell the two apart.
    first = Rotation.from_rotvec([3.1, 0.02, -0.05])
    second = Rotation.from_rotvec([2.6, 0.4, 0.3])
    delta = (second * first.inv()).as_rotvec()

    _, rebuilt = ep.apply_delta([0.0, 0.0, 0.0], first.as_rotvec(), [0.0, 0.0, 0.0], delta)

    assert (Rotation.from_rotvec(rebuilt) * second.inv()).magnitude() < 1e-9


def test_position_delta_adds():
    position, _ = ep.apply_delta(
        [0.4, -0.1, 0.25], [0.0, 0.0, 0.0], [0.01, 0.02, -0.03], np.zeros(3)
    )

    assert position == pytest.approx([0.41, -0.08, 0.22])


def test_action_scale_stretches_the_motion_but_not_the_gripper():
    action = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.035]

    dposition, drotvec, grip = ep.delta_from_action(action, scale=2.0)

    assert dposition == pytest.approx([0.02, 0.04, 0.06])
    assert drotvec == pytest.approx([0.2, 0.4, 0.6])
    # Stretching how far the arm travels must not stretch how far the fingers close.
    assert grip == pytest.approx(0.035)


def test_batched_policy_action_is_accepted():
    dposition, _, grip = ep.delta_from_action(np.zeros((1, 7)) + 0.1)

    assert dposition == pytest.approx([0.1, 0.1, 0.1])
    assert grip == pytest.approx(0.1)


def test_joint_space_action_is_rejected_with_a_pointer_to_the_other_script():
    with pytest.raises(ValueError, match="eval_policy.py"):
        ep.delta_from_action(np.zeros(8))


def test_a_delta_inside_the_ceiling_is_left_alone():
    dposition, drotvec, clamped = ep.clamp_delta([0.0, 0.0, 0.02], [0.0, 0.1, 0.0], 0.10, 0.33)

    assert not clamped
    assert dposition == pytest.approx([0.0, 0.0, 0.02])
    assert drotvec == pytest.approx([0.0, 0.1, 0.0])


def test_the_homing_sweep_is_scaled_back_to_the_ceiling():
    # 0.2552 m is the largest single step in libero_2000_0_1_2_3_4_6_7_8_12, and it is
    # part of a return-to-home arc, not task motion.
    dposition, _, clamped = ep.clamp_delta([0.2552, 0.0, 0.0], np.zeros(3), 0.10, 0.33)

    assert clamped
    assert np.linalg.norm(dposition) == pytest.approx(0.10)


def test_clamping_keeps_the_direction_and_the_translation_to_rotation_ratio():
    dposition = np.array([0.2, -0.1, 0.05])
    drotvec = np.array([0.4, 0.0, -0.2])

    scaled_position, scaled_rotvec, clamped = ep.clamp_delta(dposition, drotvec, 0.10, 0.33)

    assert clamped
    factor = np.linalg.norm(scaled_position) / np.linalg.norm(dposition)
    assert scaled_position == pytest.approx(dposition * factor)
    # The same factor on both halves, or the arm rotates at full rate while barely moving.
    assert scaled_rotvec == pytest.approx(drotvec * factor)


def test_the_rotation_ceiling_can_bind_on_its_own():
    _, drotvec, clamped = ep.clamp_delta([0.001, 0.0, 0.0], [0.6733, 0.0, 0.0], 0.10, 0.33)

    assert clamped
    assert np.linalg.norm(drotvec) == pytest.approx(0.33)


def test_a_zero_ceiling_disables_that_half():
    dposition, drotvec, clamped = ep.clamp_delta([5.0, 0.0, 0.0], [5.0, 0.0, 0.0], 0.0, 0.0)

    assert not clamped
    assert dposition == pytest.approx([5.0, 0.0, 0.0])
    assert drotvec == pytest.approx([5.0, 0.0, 0.0])


def test_the_state_z_offset_does_not_reach_the_ik(ik):
    # The whole point of the hotfix knob: it may only change what the policy is shown.
    # If it ever leaked into the IK target the arm would be driven a metre off.
    position, rotvec = ik.fk(HOME)
    shown = position + np.asarray(ep.BASE_OFFSET) + np.array([0.0, 0.0, ep.STATE_Z_OFFSET])

    target_position, _ = ep.apply_delta(position, rotvec, np.zeros(3), np.zeros(3))

    assert target_position == pytest.approx(position)          # IK works from raw FK
    assert shown[2] == pytest.approx(position[2] + ep.BASE_OFFSET[2] + ep.STATE_Z_OFFSET)


def test_the_default_shows_the_policy_the_frame_the_dataset_stores(ik):
    # Measured at the home pose: commanding HOME_POSITION and reading joint_states back
    # puts the EEF within 0.3 mm of where all 250 episodes end. Anything but 0 here
    # moves the policy's z off that.
    position, _ = ik.fk(HOME)
    shown = position + np.asarray(ep.BASE_OFFSET) + np.array([0.0, 0.0, ep.STATE_Z_OFFSET])

    assert ep.STATE_Z_OFFSET == pytest.approx(0.0)
    assert ep.home_pose_error(shown) < 1e-3


def test_the_minus_one_hotfix_lands_outside_the_datasets_entire_z_range(ik):
    # Why the -1 hotfix is not merely suboptimal: libero_dataset_250's z never goes
    # below 1.0398, and shifting by a metre puts the home pose at ~0.4995.
    position, _ = ik.fk(HOME)
    shifted = (position + np.asarray(ep.BASE_OFFSET) + np.array([0.0, 0.0, -1.0]))[2]

    assert shifted < 1.0398


class FakeRobot:
    """Reports a tool that walks in to home over a fixed number of polls."""

    def __init__(self, arrivals):
        self.arrivals = list(arrivals)
        self.reads = 0

    def get_observation(self):
        self.reads += 1
        return {}


class FakeIK:
    def __init__(self, robot, offsets):
        self.robot = robot
        self.offsets = offsets

    def fk(self, q):
        i = min(self.robot.reads - 1, len(self.offsets) - 1)
        return np.array([0.0, 0.0, self.offsets[i]]), np.zeros(3)


def _wait(monkeypatch, offsets, timeout=5.0):
    """wait_until_home against a scripted arm, with sim sleeps stubbed out."""
    monkeypatch.setattr(ep, "sleep_sim", lambda robot, seconds: None)
    monkeypatch.setattr(ep, "measured_joints", lambda observation, joints: np.zeros(7))
    robot = FakeRobot(offsets)
    ik = FakeIK(robot, offsets)
    base = np.asarray(ep.HOME_IN_DATASET)
    error = ep.wait_until_home(robot, ik, ARM_JOINTS, base, timeout, ep.HOME_TOLERANCE)
    return error, robot.reads


def test_the_wait_returns_as_soon_as_the_arm_arrives(monkeypatch):
    # Third poll is within tolerance, so it must not burn the remaining budget.
    error, reads = _wait(monkeypatch, [0.4, 0.1, 0.002, 0.002, 0.002])

    assert error < ep.HOME_TOLERANCE
    assert reads == 3


def test_an_arm_that_never_arrives_gives_up_and_reports_the_error(monkeypatch):
    error, reads = _wait(monkeypatch, [0.3] * 40, timeout=1.0)

    assert error == pytest.approx(0.3, abs=1e-6)
    assert reads == int(1.0 / 0.25)          # bounded by --home-settle-seconds


def test_an_arm_already_home_costs_a_single_read(monkeypatch):
    error, reads = _wait(monkeypatch, [0.0])

    assert error < ep.HOME_TOLERANCE
    assert reads == 1


def test_success_alone_does_not_end_the_episode():
    # The change: the policy was trained to fly home after placing, so the rollout has
    # to keep going long enough to score that.
    assert not ep.episode_is_finished(True, home_error=0.4, tolerance=0.05)


def test_an_episode_ends_once_it_has_succeeded_and_come_home():
    assert ep.episode_is_finished(True, home_error=0.02, tolerance=0.05)


def test_coming_home_without_succeeding_does_not_end_the_episode():
    # The arm starts AT home, so this would otherwise end every episode on step 0.
    assert not ep.episode_is_finished(False, home_error=0.0, tolerance=0.05)


def test_a_zero_tolerance_restores_stopping_at_success():
    assert ep.episode_is_finished(True, home_error=99.0, tolerance=0.0)
    assert not ep.episode_is_finished(False, home_error=0.0, tolerance=0.0)


def test_the_return_tolerance_is_looser_than_the_start_of_episode_check():
    # HOME_TOLERANCE checks a commanded pose against the recorded one; this is a policy
    # flying itself back and will not land within the demonstrations' 0.5 mm.
    assert ep.HOME_RETURN_TOLERANCE > ep.HOME_TOLERANCE


def test_joint_step_reports_the_worst_single_joint():
    assert ep.joint_step(HOME + np.array([0, 0, 0, 0.4, 0, -0.9, 0]), HOME) == pytest.approx(0.9)


def test_a_stale_joint_read_would_trip_the_joint_guard():
    # The failure the guard exists for: joint_states not yet flowing seeds the IK at
    # zeros, and publishing that folds the arm straight up.
    assert ep.joint_step(np.zeros(7), HOME) > ep.MAX_JOINT_STEP


def test_a_fully_clamped_step_stays_under_the_joint_guard(ik):
    # Measured worst case for a 0.10 m + 0.33 rad step is ~1.05 rad near a singularity,
    # so the guard must not fire on motion the clamp already allows.
    position, rotvec = ik.fk(HOME)
    target_position, target_rotvec = ep.apply_delta(
        position, rotvec, [0.0, 0.0, -0.10], [0.0, 0.0, 0.33]
    )

    solution, _ = ik.solve(HOME, target_position, target_rotvec, TOLERANCE)

    assert ep.joint_step(solution, HOME) < ep.MAX_JOINT_STEP


def test_measured_joints_follow_the_configs_order():
    observation = {f"fr3_joint{i}.pos": float(i) for i in range(1, 8)}

    assert ep.measured_joints(observation, ARM_JOINTS) == pytest.approx([1, 2, 3, 4, 5, 6, 7])


def test_the_default_base_offset_puts_the_home_pose_where_the_dataset_has_it(ik):
    # HOME_IN_DATASET is the mean final pose of libero_dataset_250's 250 episodes,
    # which all end homed (spread 0.5 mm). A wrong offset loads fine and simply drives
    # off-distribution, so this is the only thing that pins it.
    position, _ = ik.fk(HOME)

    assert ep.home_pose_error(position + np.asarray(ep.BASE_OFFSET)) < 1e-4


def test_a_metre_of_frame_error_is_caught(ik):
    # Reading the state in Scene_0 instead of world -- the single most likely mistake,
    # and worth exactly one metre of z.
    position, _ = ik.fk(HOME)
    scene_frame = position + np.asarray(ep.BASE_OFFSET) - np.array([0.0, 0.0, 1.0])

    assert ep.home_pose_error(scene_frame) > ep.HOME_TOLERANCE


def test_the_scene_geometry_alone_would_have_been_a_centimetre_out():
    # Kept as the reason BASE_OFFSET is measured rather than computed: Scene_0 at
    # -origin plus the bring-up's xyz:="-0.3 0 0" predicts [-0.3, 0, 1.0], and the arm
    # is not mounted exactly where the launch file says.
    import yaml
    from ament_index_python.packages import get_package_share_directory

    share = get_package_share_directory("block_bin")
    origin = yaml.safe_load(open(f"{share}/config/init.yaml"))["origin"]
    derived = -np.asarray(origin, dtype=float) + np.array([-0.3, 0.0, 0.0])

    error = np.asarray(ep.BASE_OFFSET) - derived

    assert np.abs(error).max() > 5e-3
    assert np.abs(error).max() < 5e-2
