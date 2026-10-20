"""block_bin's tree on a fake robot (C3b review M5): what GUIDE is asked, every move, every wait."""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import block_bin.solve_task as st  # noqa: E402
from guide_core.types.geometry import Point, Pose  # noqa: E402
from guide_ex.steps.end_effector.gripper_control import SetGripperState  # noqa: E402
from guide_msgs.srv import Pose as PoseSrv  # noqa: E402
from irob_lerobot_ros.config import ActionType  # noqa: E402
from irob_lerobot_ros.ros2robot import ROS2Robot  # noqa: E402

CUBE, BIN = [0.5, 0.1, 0.012], [0.45, -0.25, -0.013]  # in fr3_link0, where GUIDE answers


class FakeRobot(ROS2Robot):
    def __del__(self):  # never connected
        pass


@pytest.mark.parametrize("startup", [0.0, 2.0])  # dataset.camera_transition off, and on for 2 s
def test_block_bin_moves_in_the_base_frame_and_waits_on_sim_time(monkeypatch, startup):
    robot = FakeRobot.__new__(FakeRobot)
    robot.node = SimpleNamespace(get_logger=lambda: logging.getLogger("dry"),
                                 get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0)))
    robot.config = SimpleNamespace(frame_id="fr3_link0", arm_joint_names=[f"fr3_joint{i}" for i in range(1, 8)], gripper_joint_names=["fr3_finger_joint1"],
                                   arm_action_type=ActionType.CARTESIAN_POSE, gripper_action_type=ActionType.JOINT_POSITION)
    robot._reentrant_callback_group = None
    moves, asked, slept = [], [], []
    robot.sleep = lambda seconds: slept.append((seconds, len(moves)))
    robot.joint_state = SimpleNamespace(name=["fr3_finger_joint1"], position=[0.04])

    def send_action(action, **kwargs):
        moves.append(action)
        if isinstance(action, dict):  # the fingers jump to the goal: each gripper step settles at once
            robot.joint_state = SimpleNamespace(name=["fr3_finger_joint1"], position=[action["fr3_finger_joint1.pos"]])
        return True

    robot.send_action = send_action
    for name in ("randomize", "pose", "collision", "is_success", "start_recording", "stop_recording", "set_prompt"):
        setattr(robot, name, SimpleNamespace(srv_name=name))

    def call(client, request=None, message=None, **kwargs):
        if client.srv_name == "randomize":
            return SimpleNamespace(success=True, message=json.dumps({"target": "/blocks/red_block", "goal": "/bin_0"}))
        if client.srv_name == "pose":
            asked.append((request.path, request.frame_id))
            return PoseSrv.Response(pose=Pose(Point(CUBE if "block" in request.path else BIN)).to_ros(), success=True)
        return SimpleNamespace(success=True, message="", collision=True)

    robot.callService = call
    monkeypatch.setattr(st.time, "sleep", lambda seconds: None)  # a wall-clock wait would show by its absence in `slept`
    monkeypatch.setattr(st, "namespace_base", "/Sim_0/Scene_0", raising=False)
    monkeypatch.setattr(st, "startup_padding", startup)

    assert st.solveTask(0, robot)

    assert asked == [("/Scene_0/blocks/red_block", "fr3_link0"), ("/Scene_0/bin_0", "fr3_link0")]
    xyz = [tuple(round(v, 3) for v in (m.position.x, m.position.y, m.position.z)) for m in moves if hasattr(m, "position")]
    # rest, unclutch, cube approach / grasp / retreat, bin approach, rest: all in fr3_link0
    assert xyz == [(0.3, 0.0, 0.5), (0.3, 0.0, 0.4), (0.5, 0.1, 0.112), (0.5, 0.1, 0.022), (0.5, 0.1, 0.292),
                   (0.45, -0.25, 0.137), (0.3, 0.0, 0.5)]
    settle = [SetGripperState.POLL] * SetGripperState.STILL_SAMPLES + [SetGripperState.RESIDUAL]
    # the post-Randomize pause, settles and WaitForSeconds; with the camera transition, its wait right
    # after StartRecording, before any move but Unclutch's (rest, unclutch, open the hand): the
    # duration plus one 10 fps frame, as t = 0 is the first frame, up to 1/fps after the reply
    padding = [startup + 1 / 10] if startup else []
    assert [s for s, _ in slept] == [5, *settle, 0.5, *padding, *settle, 0.5, *settle, 0.5, 1.0]
    assert [m for s, m in slept if s in padding] == ([3] if startup else [])
