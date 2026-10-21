"""RecipeScene (S3 v2 contract B -> C): a Scene that is all data -- task.yaml and success predicates."""
import json
import logging
from types import SimpleNamespace

import pytest
import yaml

from guide_core.scene.recipe_scene import RecipeScene, load_task, separated_by_radii
from guide_core.scene.scene_manager import SceneManager
from guide_core.scene.scene_orchestrator import SceneOrchestrator

TASK = {"language": {"task": "put the plate on its mark", "subtasks": ["pick up the plate"]},
        "target": "/objects/plate", "goal": [{"on": ["/objects/plate", "/marks/home"]}]}
SUCCESS = [{"cmd": "on", "kwargs": {"a": "/objects/plate", "b": "/marks/home"}},
           {"cmd": "joint_below", "kwargs": {"joint": "/objects/cab/top_level", "value": -0.1}}]


def recipe_task(tmp_path, task=None, success=None):
    root = tmp_path / "demo"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "scene.py").write_text("from guide_core.scene.recipe_scene import RecipeScene as Scene\n")
    config = root / "config"
    config.mkdir()
    (config / "init.yaml").write_text(yaml.safe_dump(
        {"usd_path": "/stage.usda", "limits": [[-1, 1], [-1, 1], [-1, 1]], "origin": [0, 0, 0]}))
    (config / "randomize.yaml").write_text("instructions: []\n")
    (config / "reset.yaml").write_text("instructions: []\n")
    (config / "success.yaml").write_text(yaml.safe_dump({"instructions": SUCCESS if success is None else success}))
    (config / "task.yaml").write_text(yaml.safe_dump(TASK if task is None else task))
    return root


def register(root):
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(root))
    return manager, sid, cfg


def test_a_recipe_scene_is_all_data(tmp_path):
    manager, sid, cfg = register(recipe_task(tmp_path))
    assert sid == 0, cfg
    scene = manager._scenes[0]
    assert isinstance(scene, RecipeScene)
    assert scene.zone_target() == "/Scene_0/objects/plate"
    assert scene.success_instructions[0]["kwargs"]["a"] == "/Scene_0/objects/plate"
    assert json.loads(scene.randomize_postprocess([])) == {
        "task": "put the plate on its mark", "target": "/objects/plate",
        "goal": [{"on": ["/objects/plate", "/marks/home"]}], "subtasks": ["pick up the plate"]}
    assert scene.task == "put the plate on its mark"


def test_success_is_every_predicate(tmp_path, caplog):
    manager, sid, cfg = register(recipe_task(tmp_path))
    scene = manager._scenes[0]
    caplog.set_level(logging.DEBUG)
    assert scene.is_success_postprocess([{"success": True, "reason": "a"}, {"success": True, "reason": "b"}])
    assert not scene.is_success_postprocess([{"success": True, "reason": "a"}, {"success": False, "reason": "xy 0.05 m"}])
    assert "xy 0.05 m" in caplog.text


@pytest.mark.parametrize("task, success, message", [
    ({"language": {"subtasks": []}}, None, "language.task"),
    ({**TASK, "colour": "red"}, None, "colour"),
    (None, [], "success predicates"),
    (None, [{"cmd": "is_prim_contained", "kwargs": {"prim_path": "/a"}}], "is_prim_contained"),
])
def test_a_recipe_scene_refuses_what_it_cannot_run(tmp_path, task, success, message):
    _, sid, cfg = register(recipe_task(tmp_path, task, success))
    assert sid == -1 and message in cfg["error"]


def test_separation_is_libero_s_horizontal_test():
    assert separated_by_radii([[0, 0, 0], [0.2, 0, 5]], [0.06, 0.06])
    assert not separated_by_radii([[0, 0, 0], [0.1, 0, 0]], [0.06, 0.06])
    assert not separated_by_radii([[0, 0, 0], [0.12, 0, 0]], [0.06, 0.06])  # touching counts as overlap


def test_a_free_draw_is_redrawn_until_the_separation_holds(tmp_path, monkeypatch):
    root = recipe_task(tmp_path, {**TASK, "separation": {"/objects/a": 0.06, "/objects/b": 0.06}})
    (root / "config" / "randomize.yaml").write_text("events: {}\n")  # the Replicator dialect
    manager, sid, cfg = register(root)
    assert sid == 0, cfg
    scene = manager._scenes[0]
    draws = []
    monkeypatch.setattr(SceneOrchestrator, "randomize", lambda self, **kw: draws.append(kw) or SimpleNamespace())
    layouts = iter([[[0, 0, 0], [0.1, 0, 0]], [[0, 0, 0], [0.2, 0, 0]]])
    monkeypatch.setattr(RecipeScene, "_positions", lambda self, prims: next(layouts))
    scene.randomize()
    assert len(draws) == 2
    draws.clear()
    scene.randomize(arrangement="a")  # a fixed start is taken as it is
    assert len(draws) == 1 and draws[0]["arrangement"] == "a"


def test_separation_needs_a_replicator_randomize_file(tmp_path):
    _, sid, cfg = register(recipe_task(tmp_path, {**TASK, "separation": {"/objects/a": 0.06}}))
    assert sid == -1 and "separation needs a Replicator randomize.yaml" in cfg["error"]


def test_an_unquoted_on_goal_key_is_read_as_on(tmp_path):
    root = recipe_task(tmp_path)
    (root / "config" / "task.yaml").write_text(
        "language: {task: put it}\ngoal:\n  - on: [/objects/plate, /marks/home]\n")
    assert load_task(root / "config" / "task.yaml")["goal"] == [{"on": ["/objects/plate", "/marks/home"]}]


JOINT_GOAL = {"joint": "/objects/cab/top_level", "body": "/objects/cab/base/cabinet_top", "value": -0.14, "direction": -1}


def test_goal_items_name_the_skills_inputs(tmp_path):
    root = tmp_path / "t.yaml"
    goal = [{"on": ["/a", "/b"]}, {"in": ["/a", "/region"]}, {"on_region": {"obj": "/a", "region": "/zone"}}, {"on_zone": {"obj": "/a", "region": "/zone"}},
            {"open": JOINT_GOAL}, {"turn_on": {**JOINT_GOAL, "inclusive": True}}]
    root.write_text(yaml.safe_dump({"language": {"task": "x"}, "goal": goal}))
    assert load_task(root)["goal"] == goal


@pytest.mark.parametrize("item", [{"open": ["/a"]}, {"open": {"joint": "/j"}}, {"on": ["/a"]}, {"on_region": {"obj": "/a"}},
                                  {"spin": ["/a", "/b"]}, {"in": {"a": 1}}])
def test_a_malformed_goal_item_is_refused(tmp_path, item):
    root = tmp_path / "t.yaml"
    root.write_text(yaml.safe_dump({"language": {"task": "x"}, "goal": [item]}))
    with pytest.raises(ValueError, match="goal item"):
        load_task(root)
