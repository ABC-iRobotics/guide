"""The library's PutOn and PickAndPlace run against a fake robot: the motion rules hold in the order the moves run."""
import numpy as np
import pytest

from guide_core.types.geometry import Point, Pose
from guide_ex.core.states import DemoStatus, ExecutionResult
from guide_ex.recipe import load
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility import recording
from guide_ex.utility.wait import WaitForSeconds

VALUES = {"robot": {"gripper_joint": "fr3_finger_joint1", "home": {f"fr3_joint{i}.pos": 0.0 for i in range(1, 8)}},
          "episode": {"camera_hold": False, "hold_seconds": 0.0}}
PERFECT = ExecutionResult(DemoStatus.PERFECT)
POSE = Pose(position=Point(np.array([0.4, 0.0, 0.1])))
CONTEXT = {"robot": object(), "sim_namespace": "sim", "scene_namespace": "scene", "scene_id": "s", "robot_prim": "/r",
           "target": "/t", "top": "/t", "support": "/s", "goal": "/s", "pick": "pick", "place": "place",
           "rest_pose": Pose(position=Point(np.array([0.3, 0.0, 0.5]))), "scene_pose_inv": POSE, "dataset_path": "d"}


@pytest.fixture
def moves(monkeypatch):
    log, failing = [], []

    def move(self, robot, target_pose, speed=1.0, cartesian=False, perturb=False, precise_end=True, jitter=None, aim=False):
        log.append((self.name, perturb, precise_end, aim))
        if self.name in failing:
            failing.remove(self.name)
            return ExecutionResult(DemoStatus.FAILURE, error_message="no path")
        return PERFECT

    patches = {
        MoveToCartesianPose: move,
        MoveToJointConfiguration: lambda self, *a, **k: PERFECT,
        GetPrimPose: lambda self, *a, **k: ExecutionResult(DemoStatus.PERFECT, outputs={"pose": POSE}),
        IsPrimClashing: lambda self, *a, **k: ExecutionResult(DemoStatus.PERFECT, outputs={"has_collided": True}),
        IsTaskSuccessful: lambda self, *a, **k: ExecutionResult(DemoStatus.PERFECT, outputs={"success": True, "reason": ""}),
        SetGripperState: lambda self, *a, **k: PERFECT,
        WaitForSeconds: lambda self, *a, **k: PERFECT,
        recording.SetPrompt: lambda self, *a, **k: PERFECT,
        recording.StartRecording: lambda self, *a, **k: PERFECT,
        recording.StopRecording: lambda self, *a, **k: PERFECT,
    }
    for cls, run in patches.items():
        monkeypatch.setattr(cls, "run", run)
    return log, failing


def check(log):
    assert any(not precise for _, _, precise, _ in log), "no free end at all: the blend rule is untested"
    ends = [precise for _, _, precise, _ in log]
    assert not any(not a and not b for a, b in zip(ends, ends[1:])), log  # a robot cannot queue two free tails
    assert all(not precise for _, _, precise, aim in log if aim)  # aim only rides a free end


@pytest.mark.parametrize("recipe", ["PutOn", "PickAndPlace"])
def test_no_two_free_ends_in_a_row_and_aim_only_on_free_ends(moves, recipe):
    log, _ = moves
    assert load({"ref": recipe}, values=VALUES).execute(dict(CONTEXT)).status == DemoStatus.PERFECT
    check(log)
    assert {n for n, *_, aim in log if aim} <= {"MoveToPrePlace"}


@pytest.mark.parametrize("failing", ["MoveToCubeRetreat", "MoveOverSupport", "MoveToPrePlace"])
def test_the_recovery_routes_keep_the_rules_too(moves, failing):
    log, to_fail = moves
    to_fail.append(failing)
    assert load({"ref": "PutOn"}, values=VALUES).execute(dict(CONTEXT)).status == DemoStatus.PERFECT
    check(log)
