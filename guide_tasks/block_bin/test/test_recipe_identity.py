"""The recipe tree is today's Python tree, node by node (kept until the Isaac regression passes)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import block_bin.solve_task as st  # noqa: E402
from legacy_tree import describe, legacy_tree  # noqa: E402

ROBOT = SimpleNamespace(config=SimpleNamespace(gripper_joint_names=["fr3_finger_joint1"]))
CONTEXT_KEYS = ["sim_namespace", "scene_namespace", "scene_id", "robot", "robot_prim", "target", "goal",
                "rest_pose", "dataset_path"]


@pytest.mark.parametrize("startup", [0.0, 2.0])  # dataset.camera_transition off, and on for 2 s
def test_the_recipes_build_todays_tree(monkeypatch, startup):
    monkeypatch.setattr(st, "startup_padding", startup)
    old = legacy_tree(ROBOT, dict.fromkeys(CONTEXT_KEYS), startup, st.record_fps)
    new = st.episode_tree(ROBOT)

    assert describe(new) == describe(old)
