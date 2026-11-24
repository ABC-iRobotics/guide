import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from guide_ex.recipe import solve
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
ZONE = {"obj": "/objects/p", "region": "/stove_front_region"}


def test_context_for_pull_turn_push_put_in_and_put_on_a_region():
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"open": JOINT}], ["Open the top drawer."], "/objects/c/base/cabinet_top"))
    assert ctx["joint_body"] == JOINT["body"]
    assert (ctx["subtask"], ctx["pick"], ctx["place"], ctx["top"]) == ("Open the top drawer.", "", "", None)
    # Pull then PutIn: the drawer's open goal rides along the success's `in`; the first label is Pull's
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"in": ["/objects/b", "/objects/c/r"]}, {"open": JOINT}],
                                                            ["Open the top drawer.", "Pick up the bowl.", "Put it in the drawer."]))
    assert (ctx["subtask"], ctx["pick"], ctx["place"]) == ("Open the top drawer.", "Pick up the bowl.", "Put it in the drawer.")
    assert (ctx["top"], ctx["support"], ctx["goal"]) == ("/objects/b", "/objects/c/r", "/objects/c/r")
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"turn_on": {**JOINT, "direction": 1, "value": 0.5}}], ["Turn on the stove."]))
    assert (ctx["joint_body"], ctx["subtask"], ctx["pick"]) == (JOINT["body"], "Turn on the stove.", "")
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"on_zone": ZONE}], ["Push the plate to the front of the stove."]))
    assert (ctx["top"], ctx["support"], ctx["subtask"], ctx["pick"]) == ("/objects/p", "/stove_front_region", "Push the plate to the front of the stove.", "")
    ctx = context("/Sim_0/Scene_0", "R", ENTRY, "", _reply([{"on_region": {"obj": "/objects/b", "region": "/objects/c/top_side"}}],
                                                            ["Pick up the bowl.", "Place it on the wooden cabinet."]))
    assert (ctx["top"], ctx["support"], ctx["subtask"], ctx["pick"]) == ("/objects/b", "/objects/c/top_side", "", "Pick up the bowl.")


def test_a_plain_on_task_keeps_its_keys_and_has_no_joint_or_zone():
    ctx = context("/Sim_0/Scene_2", "R", ENTRY, "", REPLY)
    assert (ctx["pick"], ctx["place"], ctx["subtask"], ctx["joint_body"]) == ("Pick up the bowl.", "Place it on the plate.", "", None)


def test_make_robots_builds_every_init_robot_and_still_one_fr3():
    import pytest

    pytest.importorskip("irob_lerobot_ros.config")
    from irob_lerobot_ros.config import ActionType

    from guide_ex.recipe.solve import make_robots

    three = {"robots": {"psm1": {"profile": "psm", "in_scene": True, "path": "/PSM1/psm"},
                        "psm2": {"profile": "psm", "in_scene": True, "path": "/PSM2/psm"},
                        "ecm": {"profile": "ecm", "in_scene": True, "path": "/ECM/ecm"}}}
    robots = make_robots(three, "/Sim_0/Scene_1")
    assert {k: (r.config.id, r.config.namespace, r.config.frame_id) for k, r in robots.items()} == {
        "psm1": ("psm1", "/Sim_0/Scene_1/psm1", "psm1/psm_base_link"),
        "psm2": ("psm2", "/Sim_0/Scene_1/psm2", "psm2/psm_base_link"),
        "ecm": ("ecm", "/Sim_0/Scene_1/ecm", "ecm/ecm_base_link")}
    assert robots["psm1"].config.arm_action_type == ActionType.CARTESIAN_POSE
    assert robots["ecm"].config.arm_action_type == ActionType.JOINT_POSITION  # joints only
    (fr3,) = make_robots({"robots": {"franka": {"profile": "fr3", "path": "/robot"}}}, "/Sim_0/Scene_0").values()
    assert (type(fr3.config).__name__, fr3.config.namespace, fr3.config.frame_id) == (
        "FR3RobotConfig", "/Sim_0/Scene_0/franka", "fr3_link0")
    assert fr3.config.arm_action_type == ActionType.CARTESIAN_POSE and fr3.config.controller == "oscbf"


ONE = {"failures": {"counts": [1, 2], "weights": [0.5, 0.5]},
       "stages": {"grasp": {"grasp": {"axes": "y", "r": [0.01, 0.02]}}, "handover": {"receive": {"axes": "y", "r": [0.01, 0.02]}},
                  "place": {"place": {"axes": "xy", "r": [0.004, 0.005]}}}}


def test_planned_misses_one_slot_draws_distinct_stages_keyed_on_its_default():
    assert solve.planned_misses({"miss": {"grasp": {"p": 1.0}}}, [{"*": "red"}], None) == ({"grasp": {"p": 1.0}}, [])
    counts = set()
    for seed in range(40):
        miss, stages = solve.planned_misses(ONE, [{"*": "red"}], np.random.default_rng(seed))
        assert len(stages) == len(set(stages)) == len(miss) and set(stages) <= set(ONE["stages"])
        assert miss == {t: {"at": {"red": s}} for st in stages for t, s in ONE["stages"][st].items()}
        counts.add(len(stages))
    assert counts == {1, 2}


def test_planned_misses_many_slots_draw_distinct_cells_keyed_per_target():
    order = ["a", "b", "c", "d"]
    slots = [{"pre_grasp": order[i], "grasp": order[i], "*": order[i - 1]} for i in range(1, len(order))]
    v = {"failures": {"counts": [3], "weights": [1.0]},
         "stages": {"air_grasp": {"grasp": {"axes": "x", "r": [0.045, 0.06]}}, "askew": {"place": {"axes": "xy", "r": [0.01, 0.02]}},
                    "fall": {"place": {"axes": "xy", "r": [0.03, 0.04]}}}}
    for seed in range(50):
        miss, cells = solve.planned_misses(v, slots, np.random.default_rng(seed))
        assert len(cells) == len(set(cells)) == 3
        for cell in cells:  # placement i puts order[i] on order[i - 1]
            stage, i = cell.split("@")
            for target, spec in v["stages"][stage].items():
                assert miss[target]["at"][order[int(i)] if target == "grasp" else order[int(i) - 1]] == spec
        assert sum(len(t["at"]) for t in miss.values()) == 3  # askew and fall never at one placement


def test_variant_reads_the_task_config(tmp_path):
    (tmp_path / "pkg/config/recipes").mkdir(parents=True)
    (tmp_path / "pkg/config/variants").mkdir()
    (tmp_path / "pkg/config/variants/flawless.yaml").write_text("miss: {}\n")
    task = SimpleNamespace(__package__="pkg", __file__=str(tmp_path / "pkg/pkg/solve_task.py"))
    assert solve.config_dir(task) == tmp_path / "pkg/config" and solve.variant(task, "flawless") == {"miss": {}}


def test_log_moves_records_pose_and_joint_goals_with_flags_not_jaws():
    sent = []
    robot = SimpleNamespace(config=SimpleNamespace(arm_joint_names=["j1", "j2"]), send_action=lambda a, *x, **k: sent.append(a) or True)
    solve.log_moves("psm1", robot)
    solve.MOVES.clear()
    assert robot.send_action(SimpleNamespace(position=SimpleNamespace(x=0.1, y=0.2, z=0.3)), speed=0.1, perturb=True,
                             precise_end=False, aim=True)
    robot.send_action({"j1.pos": 0.5, "j2.pos": -0.5}, speed=0.2)
    robot.send_action({"jaw.pos": 0.1})  # the jaw alone: not an arm goal
    assert len(sent) == 3 and solve.MOVES == [
        {"robot": "psm1", "position": [0.1, 0.2, 0.3], "speed": 0.1, "perturb": True, "precise_end": False, "aim": True},
        {"robot": "psm1", "joints": {"j1.pos": 0.5, "j2.pos": -0.5}, "speed": 0.2}]


def _offline(monkeypatch, outcomes, runs):
    """solve_episode, discard and finalize faked; the robots one with a silent log."""
    monkeypatch.setattr(solve, "solve_episode", lambda task, *a, **k: runs.append((a, k)) or (
        next(outcomes), {"recipe": k["recipe"], "misses": [], "master": 1, "failed": []}))
    monkeypatch.setattr(solve, "discard", lambda *a: None)
    monkeypatch.setattr(solve, "finalize", lambda *a: None)
    log = SimpleNamespace(info=lambda m: None, error=lambda m: None)
    return {"r": SimpleNamespace(node=SimpleNamespace(get_logger=lambda: log))}


def test_generate_redraws_a_failed_attempt_and_writes_the_provenance_check_smoke_reads(monkeypatch, tmp_path):
    runs = []
    robots = _offline(monkeypatch, iter([False, True, True]), runs)  # episode 0 fails once

    solve.generate("task", [None, 4], 0, robots, "/Sim_0", {}, str(tmp_path / "set"), ["flawless", "light_fail"], seed_base=7)

    assert [(a[5:], k["clean"], k["recipe"], k["zone"]) for a, k in runs] == [
        ((1, 7), True, "flawless", None), ((2, 8), True, "flawless", None), ((3, 1007), False, "light_fail", 4)]
    lines = [json.loads(s) for s in (tmp_path / "set.provenance.jsonl").read_text().splitlines()]
    # check_smoke: seed = seed_base + episode * SEED_BLOCK + attempt, recipe and misses on every saved line
    assert [(p["episode"], p["attempt"], p["seed"], p["saved"]) for p in lines] == [(0, 0, 7, False), (0, 1, 8, True), (1, 0, 1007, True)]
    assert all({"recipe", "misses", "master", "failed", "draw", "scene_id", "clean", "moves"} <= set(p) for p in lines)
    assert "zone" not in lines[0] and lines[2]["zone"] == 4


def test_scenarios_replace_a_failing_layout_and_redraw_a_failing_trajectory(monkeypatch, tmp_path):
    runs = []
    robots = _offline(monkeypatch, iter([False, True, True, False, True]), runs)  # seed 0 fails clean; trajectory 2 once

    solve.generate_scenarios("task", [{"red_block": 3}], 1, 3, 0, robots, "/Sim_0", {}, str(tmp_path / "set"), ["flawless"])

    assert [(a[6], k["clean"], k["no_recovery"]) for a, k in runs] == [
        (0, True, True), (1, True, True), (1, False, True), (1, False, True), (1, False, True)]
    assert all(k["zones"] == {"red_block": 3} for _, k in runs) and [a[5] for a, _ in runs] == [1, 2, 3, 4, 5]
    lines = [json.loads(s) for s in (tmp_path / "set.provenance.jsonl").read_text().splitlines()]
    assert [p["saved"] for p in lines] == [False, True, True, False, True] and lines[0]["zones"] == {"red_block": 3}
