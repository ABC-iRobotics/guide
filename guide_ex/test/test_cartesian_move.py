"""MoveToCartesianPose refuses a straight-line path MoveIt cannot finish, before moving."""

import logging
from types import SimpleNamespace

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose

TARGET = Pose(position=Point([0.2, 0.0, 0.3]))


def robot_planning(path):
    """A ROS2Robot whose MoveIt answers `path` (None: under the threshold) and records sends."""
    robot = ROS2Robot.__new__(ROS2Robot)
    robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("test"))
    robot.config = SimpleNamespace(arm_action_type=ActionType.CARTESIAN_POSE, frame_id="Scene_0")
    robot.joint_state = None
    robot.plans, robot.sent = [], []
    robot._moveit2 = SimpleNamespace(
        max_velocity=1.0, plan=lambda **kwargs: robot.plans.append(kwargs) or path
    )
    robot.send_action = lambda action, **kwargs: robot.sent.append(action) or True
    return robot


def test_a_path_moveit_cannot_finish_is_not_executed():
    robot = robot_planning(path=None)

    result = MoveToCartesianPose().run(robot, TARGET, cartesian=True)

    assert result.status == DemoStatus.FAILURE
    assert robot.sent == []
    assert robot.plans[0]["cartesian"] and robot.plans[0]["cartesian_fraction_threshold"] == 0.95
    assert robot.plans[0]["frame_id"] == "Scene_0"


def test_a_complete_path_is_executed():
    robot = robot_planning(path=object())

    result = MoveToCartesianPose().run(robot, TARGET, cartesian=True, min_fraction=1.0)

    assert result.status == DemoStatus.PERFECT
    assert len(robot.sent) == 1 and robot.plans[0]["cartesian_fraction_threshold"] == 1.0


def test_a_planned_move_is_not_checked():
    robot = robot_planning(path=None)

    assert MoveToCartesianPose().run(robot, TARGET).status == DemoStatus.PERFECT
    assert robot.plans == [] and len(robot.sent) == 1
