"""guide_ex's recipe library: the paper's Fig. 3 loads verbatim and every recipe loads where it lives."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from guide_ex.core.composite_node import RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.recipe import RecipeError, load
from guide_ex.recipe.registry import LIBRARY

FIG3 = Path(__file__).parent / "data" / "fig3_pick_and_place.yaml"
VALUES = {  # what the library interpolates (see CATALOGUE.md's inputs)
    "robot": {"gripper_joint": "fr3_finger_joint1",
              "home": {f"fr3_joint{i}.pos": 0.0 for i in range(1, 8)}},
    "episode": {"camera_hold": True, "hold_seconds": 2.1},
}


def test_the_papers_fig3_loads_verbatim():
    tree = load(FIG3, values=VALUES)

    assert (tree.name, tree.level) == ("PickAndPlace", Layer.TASK)
    assert [c.name for c in tree.children] == ["Unclutch", "GetTargetPose", "Pick", "Place", "Return"]
    recovery = tree.fallbacks["Pick"]
    assert isinstance(recovery, RecoveryNode)
    assert (recovery.name, recovery.level, recovery.resume_target) == ("PickRecovery", Layer.SUBTASK, "Pick")
    assert [c.name for c in recovery.children] == ["GetTargetPose", "OpenGripperRecovery", "RotateCubePose"]
    turn = recovery.children[2]
    assert (turn.dynamic_map, turn.output_map) == ({"l_pose": "target_pose"}, {"pose": "target_pose"})
    assert np.allclose(turn.static_args["r_pose"].orientation.to_scipy().as_matrix(),
                       R.from_euler("z", 1.57).as_matrix())


@pytest.mark.parametrize("path", sorted(p for folder in LIBRARY for p in folder.rglob("*.yaml")), ids=lambda p: p.stem)
def test_every_library_recipe_loads_at_the_level_of_its_folder(path):
    node = load(path, values=VALUES)

    assert node.level == Layer[path.parent.name.removesuffix("s").upper()]  # steps/ -> STEP, utility/ -> UTILITY


def test_a_robot_profile_feeds_the_library():
    """Plan B's robot_profile.resolve() entry (here a Panda's) gives the robot values (Contract addition 7)."""
    entry = {"joint_names": [f"panda_joint{i}" for i in range(1, 8)], "gripper_joint": "panda_finger_joint1",
             "default_joint_states": [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398, 0.0, 0.0]}
    robot = {"gripper_joint": entry["gripper_joint"],
             "home": {f"{j}.pos": q for j, q in zip(entry["joint_names"], entry["default_joint_states"])}}

    home = load({"ref": "MoveHome"}, values={"robot": robot})
    grip = load({"ref": "OpenGripper"}, values={"robot": robot})

    assert list(home.static_args["target_configuration"]) == [f"panda_joint{i}.pos" for i in range(1, 8)]
    assert grip.static_args["gripper_goal_pos"] == {"panda_finger_joint1": 0.04}


def test_a_value_missing_deep_in_the_library_names_its_file():
    """A task that forgets the episode values hears which template wanted them."""
    with pytest.raises(RecipeError, match=r"CameraHold\.yaml: .*episode\.camera_hold"):
        load({"ref": "PickAndPlace"}, values={"robot": VALUES["robot"]})


def test_an_override_through_a_fallback_key_is_refused():
    """Override paths walk node names: PickRecovery's nodes are under PickRecovery, not under the
    fallback's key, Pick (which names the child Pick)."""
    turn = {"ref": "TransformPose", "alias": "RotateCubePose", "dynamic_map": {"l_pose": "cube_pose"},
            "static_args": {"r_pose": {"orientation": {"euler": ["z", 0.0]}}}, "output_map": {"pose": "cube_pose"}}

    with pytest.raises(RecipeError, match="override 'Pick.RotateCubePose' names no node below PickAndPlace"):
        load({"ref": "PickAndPlace", "overrides": {"Pick.RotateCubePose": turn}}, values=VALUES)
    assert load({"ref": "PickAndPlace", "overrides": {"PickRecovery.RotateCubePose": turn}}, values=VALUES)
