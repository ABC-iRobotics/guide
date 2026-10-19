"""MoveToCartesianPose / MoveToJointConfiguration only talk to the robot through send_action."""

import logging
from types import SimpleNamespace

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration

TARGET = Pose(position=Point([0.2, 0.0, 0.3]))


class FakeRobot(ROS2Robot):
    def __del__(self):  # never connected, nothing to disconnect
        pass


def robot_answering(result):
    """A ROS2Robot with no MoveIt at all, whose send_action answers `result` and records its kwargs."""
    robot = FakeRobot.__new__(FakeRobot)
    robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("test"))
    robot.config = SimpleNamespace(arm_action_type=ActionType.CARTESIAN_POSE, frame_id="fr3_link0")
    robot._moveit2 = None  # the seam: a step that reaches into MoveIt fails here
    robot.sent = []
    robot.send_action = lambda action, **kwargs: robot.sent.append(kwargs) or result
    return robot


def test_speed_and_straightness_go_to_the_robot():
    robot = robot_answering(True)

    result = MoveToCartesianPose().run(robot, TARGET, speed=0.2, cartesian=True, min_fraction=1.0)

    assert result.status == DemoStatus.PERFECT
    assert robot.sent == [{"cartesian": True, "wait_for_execution": True, "speed": 0.2, "min_fraction": 1.0}]


def test_a_refused_move_fails_after_three_tries():
    robot = robot_answering(False)

    assert MoveToCartesianPose().run(robot, TARGET, cartesian=True).status == DemoStatus.FAILURE
    assert len(robot.sent) == 3


def test_a_joint_move_passes_its_speed():
    robot = robot_answering(True)
    robot.config.arm_action_type = ActionType.JOINT_POSITION

    result = MoveToJointConfiguration().run(robot, {"fr3_joint1.pos": 0.0}, speed=0.3)

    assert result.status == DemoStatus.PERFECT
    assert robot.sent == [{"wait_for_execution": True, "speed": 0.3}]
