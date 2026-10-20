"""GUIDE-EX step nodes against a fake ROS2Robot: each test pins one fixed defect."""

import logging
from types import SimpleNamespace

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot

from guide_ex.core.states import DemoStatus
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveWithCartesianVelocity
from guide_ex.steps.simulation.isaac.prim import GetPrimPose
from guide_msgs.srv import Pose as PoseSrv


class FakeRobot(ROS2Robot):
    def __del__(self):  # never connected, nothing to disconnect
        pass


def fake_robot(**attrs):
    robot = FakeRobot.__new__(FakeRobot)
    robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("test"))
    for name, value in attrs.items():
        setattr(robot, name, value)
    return robot


def test_get_prim_pose_fails_when_the_simulator_has_no_such_prim():
    no_prim = PoseSrv.Response(success=False, message="no prim at /Scene_0/blocks/pink_block")
    robot = fake_robot(
        pose=object(),
        config=SimpleNamespace(frame_id="fr3_link0"),
        callService=lambda client, request, message: no_prim,
    )

    result = GetPrimPose().run(robot, "/Sim_0", "/Scene_0", "/blocks/pink_block")

    assert result.status == DemoStatus.FAILURE


def test_a_cartesian_velocity_command_that_went_through_is_sent_once():
    sent = []
    robot = fake_robot(
        config=SimpleNamespace(arm_action_type=ActionType.CARTESIAN_VELOCITY),
        send_action=lambda action, **kwargs: sent.append(action) or True,
    )

    result = MoveWithCartesianVelocity().run(robot, {"vx": 0.1})

    assert result.status == DemoStatus.PERFECT and len(sent) == 1


def gripper_on_sim_time(finger, sent):
    """A robot on sim time: only sleep advances its clock, and its joint_state follows; finger(t) is
    fr3_finger_joint1's position at t s."""
    clock = SimpleNamespace(ns=0)
    clock.now = lambda: SimpleNamespace(nanoseconds=clock.ns)

    def sleep(seconds):
        clock.ns += round(seconds * 1e9)
        robot.joint_state = SimpleNamespace(name=["fr3_joint1", "fr3_finger_joint1"], position=[0.0, finger(clock.ns / 1e9)])

    def get_observation():  # also reads every camera: 5 s of wall time each while no image topic is up
        raise AssertionError("the settle read the cameras")

    robot = fake_robot(
        config=SimpleNamespace(gripper_action_type=ActionType.JOINT_POSITION, gripper_joint_names=["fr3_finger_joint1"]),
        send_action=lambda action, **kwargs: sent.append(action) or True,
        sleep=sleep,
        get_observation=get_observation,
    )
    robot.node.get_clock = lambda: clock
    sleep(0)
    return robot, lambda: clock.ns / 1e9


def test_set_gripper_state_returns_once_the_fingers_stop_and_the_residual_is_over(caplog):
    sent = []
    robot, now = gripper_on_sim_time(lambda t: 0.04 * min(t / 0.6, 1.0), sent)  # opens, still from 0.6 s

    result = SetGripperState().run(robot, {"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04})

    assert result.status == DemoStatus.PERFECT
    assert sent == [{"fr3_finger_joint1.pos": 0.04, "fr3_finger_joint2.pos": 0.04}]
    assert 0.6 + 0.15 + 0.2 <= now() < 0.6 + 0.15 + 0.2 + 0.05  # still at 0.6, 3 still polls, the residual
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_set_gripper_state_counts_a_stall_on_the_cube_as_settled(caplog):
    # commanded 0.01; the cube stops the fingers at 0.0205, where contact jitters them by 0.1 mm
    def finger(t):
        return max(0.04 - 0.05 * t, 0.0205 + 1e-4 * (round(t / 0.05) % 2))

    robot, now = gripper_on_sim_time(finger, [])

    result = SetGripperState().run(robot, {"fr3_finger_joint1": 0.01})

    assert result.status == DemoStatus.PERFECT
    assert 0.39 + 0.15 + 0.2 <= now() < 0.39 + 0.15 + 0.2 + 0.05  # stalled at 0.39 s
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_set_gripper_state_gives_up_on_fingers_that_never_stop_with_one_warning(caplog):
    robot, now = gripper_on_sim_time(lambda t: 0.04 - 0.01 * t, [])  # creeping 0.5 mm a poll, forever

    result = SetGripperState().run(robot, {"fr3_finger_joint1": 0.01})

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert result.status == DemoStatus.PERFECT
    assert 2.0 <= now() <= 2.0 + 0.2  # the timeout, plus the residual at most
    assert len(warnings) == 1 and "fr3_finger_joint1" in warnings[0].getMessage()


def test_set_gripper_state_on_another_robot_only_sends_the_goal():
    sent = []
    robot = SimpleNamespace(send_action=sent.append)  # no sleep, no observation: nothing else may be called

    result = SetGripperState().run(robot, {"gripper": 0.04})

    assert result.status == DemoStatus.PERFECT and sent == [{"gripper": 0.04}]


def test_get_prim_pose_asks_for_the_robots_own_frame():
    asked = []
    robot = fake_robot(
        pose=object(),
        config=SimpleNamespace(frame_id="fr3_link0"),
        callService=lambda client, request, message: asked.append(request) or PoseSrv.Response(success=True),
    )

    result = GetPrimPose().run(robot, "/Sim_0", "/Scene_0", "/blocks/red_block")

    assert result.status == DemoStatus.PERFECT
    assert (asked[0].path, asked[0].frame_id) == ("/Scene_0/blocks/red_block", "fr3_link0")
