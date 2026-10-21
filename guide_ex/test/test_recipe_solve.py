import json
from pathlib import Path

import numpy as np

from guide_ex.recipe.solve import context, dataset_path, draw, episodes_saved, recipe_values, run_episodes

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


class Reply:
    def __init__(self, success, message):
        self.success, self.message = success, message


class DrawRobot:
    randomize = "client"

    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def callService(self, client, request, timeout_sec):
        if self.error:
            raise self.error
        return self.reply


def test_draw_passes_a_successful_reply_through():
    assert draw(DrawRobot(Reply(True, REPLY)), None) == (REPLY, "")


def test_a_failed_draw_is_a_failed_attempt_with_its_message():
    message, why = draw(DrawRobot(Reply(False, "no free spot")), None)
    assert "no free spot" in why


def test_a_randomize_timeout_is_a_failed_attempt_not_a_crash():
    _, why = draw(DrawRobot(error=TimeoutError("timed out after 600s")), None)
    assert "timed out" in why


def test_run_episodes_falls_back_to_a_free_draw_after_the_arrangement_fails():
    calls, logs = [], []

    def try_once(start):  # demo_1 never settles; the free draw works on its 2nd try
        calls.append(start)
        return start == "demo_0" or (start == "" and calls.count("") == 2)

    assert run_episodes(["demo_0", "demo_1"], 2, try_once, logs.append) == (2, 5, 1)
    assert calls == ["demo_0", "demo_1", "demo_1", "", ""]
    assert logs == ["[solve] fallback demo_1 -> free draw"]


def test_run_episodes_counts_an_episode_lost_when_the_free_draw_fails_too_and_plain_free_has_no_fallback():
    assert run_episodes(["demo_0"], 2, lambda s: False, lambda m: None) == (0, 4, 1)
    assert run_episodes(["", ""], 2, lambda s: False, lambda m: None) == (0, 4, 0)


def _reply(goal, subtasks, target="/objects/b"):
    return json.dumps({"task": "t", "target": target, "goal": goal, "subtasks": subtasks})


JOINT = {"joint": "/objects/c/base/cabinet_top/top_level", "body": "/objects/c/base/cabinet_top", "value": -0.14, "direction": -1}
ZONE = {"obj": "/objects/p", "region": "", "center": [-0.05, 0.21, -0.005], "half": [0.04, 0.04, 0.007]}


def test_context_for_pull_turn_push_put_in_and_put_on_a_region():
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"open": JOINT}], ["Open the top drawer."], "/objects/c/base/cabinet_top"))
    assert (ctx["joint"], ctx["joint_body"], ctx["joint_value"], ctx["direction"]) == (JOINT["joint"], JOINT["body"], -0.14, -1)
    assert (ctx["subtask"], ctx["pick"], ctx["place"], ctx["top"], ctx["zone"]) == ("Open the top drawer.", "", "", None, None)
    # Pull then PutIn: the drawer's open goal rides along the success's `in`; the first label is Pull's
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"in": ["/objects/b", "/objects/c/r"]}, {"open": JOINT}],
                                                            ["Open the top drawer.", "Pick up the bowl.", "Put it in the drawer."]))
    assert (ctx["subtask"], ctx["pick"], ctx["place"]) == ("Open the top drawer.", "Pick up the bowl.", "Put it in the drawer.")
    assert (ctx["top"], ctx["support"], ctx["goal"], ctx["direction"]) == ("/objects/b", "/objects/c/r", "/objects/c/r", -1)
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"turn_on": {**JOINT, "direction": 1, "value": 0.5}}], ["Turn on the stove."]))
    assert (ctx["direction"], ctx["joint_value"], ctx["subtask"]) == (1, 0.5, "Turn on the stove.")
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"on_region": ZONE}], ["Push the plate to the front of the stove."]))
    assert (ctx["top"], ctx["support"], ctx["zone"], ctx["subtask"]) == ("/objects/p", None, ZONE, "Push the plate to the front of the stove.")
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"on_region": {"obj": "/objects/b", "region": "/objects/c/top_side"}}],
                                                            ["Pick up the bowl.", "Place it on the wooden cabinet."]))
    assert (ctx["top"], ctx["support"], ctx["zone"], ctx["subtask"], ctx["pick"]) == ("/objects/b", "/objects/c/top_side", None, "", "Pick up the bowl.")


def test_a_plain_on_task_keeps_its_keys_and_has_no_joint_or_zone():
    ctx = context("/Sim_0/Scene_2", "R", ENTRY, "", REPLY)
    assert (ctx["pick"], ctx["place"], ctx["subtask"], ctx["joint"], ctx["zone"]) == ("Pick up the bowl.", "Place it on the plate.", "", None, None)
