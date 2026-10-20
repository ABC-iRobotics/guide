import json
from pathlib import Path

import numpy as np

from guide_ex.recipe.solve import context, dataset_path, episodes_saved, recipe_values

REPLY = json.dumps({"task": "put the bowl on the plate", "target": "/objects/b", "goal": [{"on": ["/objects/b", "/objects/p"]}],
                    "subtasks": ["Pick up the bowl.", "Place it on the plate."]})
ENTRY = {"path": "/robot", "profile": "fr3"}


def test_context_feeds_the_library_from_a_recipe_scene_reply():
    ctx = context("/Sim_0/Scene_2", "R", ENTRY, "/home/user/dataset/t", REPLY)
    assert (ctx["sim_namespace"], ctx["scene_namespace"], ctx["scene_id"]) == ("/Sim_0", "/Scene_2", 2)
    assert (ctx["target"], ctx["top"], ctx["support"]) == ("/objects/b", "/objects/b", "/objects/p")
    assert (ctx["pick"], ctx["place"]) == ("Pick up the bowl.", "Place it on the plate.")
    assert (ctx["robot"], ctx["robot_prim"], ctx["dataset_path"]) == ("R", "/robot/fr3_rightfinger", "/home/user/dataset/t")
    np.testing.assert_allclose(ctx["rest_pose"].position.to_numpy(), [0.3, 0.0, 0.5])
    np.testing.assert_allclose(ctx["scene_pose_inv"].to_numpy(), np.eye(4))


def test_a_reply_that_is_not_json_leaves_the_goal_keys_empty():
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", "Randomized.")
    assert (ctx["scene_id"], ctx["target"], ctx["support"], ctx["pick"], ctx["place"]) == (0, None, None, "", "")


def test_dataset_defaults_to_home_dataset_task_dir_name():
    home = Path("~").expanduser()
    assert dataset_path(Path("/x/tasks/put_bowl"), "") == str(home / "dataset/put_bowl")
    assert dataset_path(Path("/x/tasks/put_bowl"), "~/d") == str(home / "d")


def test_recipe_values_from_a_profile_entry():
    from guide_core.scene.robot_profile import resolve

    entry = resolve({"profile": "fr3", "path": "/robot", "default_joint_states": [0.1] * 7 + [0.04, 0.04]})
    assert recipe_values(entry, {"dataset": {"fps": 20}}) == {
        "robot": {"gripper_joint": "fr3_finger_joint1", "home": {f"fr3_joint{i}.pos": 0.1 for i in range(1, 8)}},
        "episode": {"camera_hold": False, "hold_seconds": 0.0}}
    held = recipe_values(entry, {"dataset": {"fps": 20, "camera_transition": {"enabled": True, "duration": 2.0}}})
    assert held["episode"] == {"camera_hold": True, "hold_seconds": 2.05}


def test_episodes_saved_counts_what_the_recorder_wrote(tmp_path):
    assert episodes_saved(str(tmp_path / "none")) == 0  # no dataset dir: nothing written
    for name, n in (("dataset_0_0_a", 2), ("dataset_0_0_b", 1)):
        (tmp_path / name / "meta").mkdir(parents=True)
        (tmp_path / name / "meta/info.json").write_text(json.dumps({"total_episodes": n}))
    (tmp_path / "dataset_0_0_c/meta").mkdir(parents=True)
    (tmp_path / "dataset_0_0_c/meta/info.json").write_text("{")  # unreadable: skipped
    assert episodes_saved(str(tmp_path)) == 3
