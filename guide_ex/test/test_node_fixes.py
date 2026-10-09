"""GUIDE-EX step nodes against a fake ROS2Robot: each test pins one fixed defect."""

import logging
from types import SimpleNamespace

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot

from guide_ex.core.states import DemoStatus
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
    robot = fake_robot(pose=object(), callService=lambda client, request, message: no_prim)

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
