"""Scene recipes (S3 v2 contract B -> C): config/scene.yaml, create.* only, run once at registration."""
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from guide_core.scene import robot_profile
from guide_core.scene.scene_manager import SceneManager
from guide_core.types.randomization import replicator_guide as rg

DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"
RECIPE = "cab:\n  create.from_usd: {usd: a.usda, name: cab, parent: /objects}\n"


def test_parents_are_scene_prefixed_like_path_patterns():
    doc = {"cab": {"create.from_usd": {"usd": "x.usda", "name": "cab", "parent": "/objects"}},
           "rel": {"create.from_usd": {"usd": "y.usda", "parent": "objects"}}}
    out = rg.prefixed(doc, "/Scene_2")
    assert out["cab"]["create.from_usd"]["parent"] == "/Scene_2/objects"
    assert out["rel"]["create.from_usd"]["parent"] == "objects"


def test_a_scene_recipe_takes_create_calls_only():
    rg.check_scene_recipe({"cab": {"create.from_usd": {"usd": {"property": "file_path", "value": "../a.usda"},
                                                       "parent": "/objects"}}}, "scene.yaml")
    for bad in ({"e": {"trigger.on_custom_event": {"event_name": "x"}}},
                {"w": {"with.e": {"modify.pose": {"position": [0, 0, 0]}}}},
                {"c": {"create.from_usd": {"usd": "a.usda",
                                           "position": {"distribution.uniform": {"lower": [0], "upper": [1]}}}}},
                {"z": {"guide.zone": {}}}):
        with pytest.raises(ValueError, match="create"):
            rg.check_scene_recipe(bad, "scene.yaml")
    with pytest.raises(ValueError, match="Replicator"):
        rg.check_scene_recipe({"instructions": []}, "scene.yaml")


def task(root, recipe=None, robots=None):
    shutil.copytree(DUMMY, root)
    if recipe is not None:
        (root / "config" / "scene.yaml").write_text(recipe)
    if robots is not None:
        config = yaml.safe_load((root / "config" / "init.yaml").read_text())
        config["robots"] = robots
        (root / "config" / "init.yaml").write_text(yaml.safe_dump(config))
    return root


def test_registration_finds_and_checks_the_scene_recipe(tmp_path):
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task(tmp_path / "a", RECIPE)))
    assert sid == 0, cfg
    assert manager._scenes[0].scene_recipe.name == "scene.yaml"
    sid, _, cfg = manager.add_scene(str(task(tmp_path / "b")))
    assert sid == 1 and manager._scenes[1].scene_recipe is None
    sid, _, cfg = manager.add_scene(str(task(tmp_path / "c", "e:\n  trigger.on_custom_event: {event_name: x}\n")))
    assert sid == -1 and "trigger.on_custom_event" in cfg["error"]


def test_build_stage_runs_the_scene_recipe_after_the_robots(tmp_path, monkeypatch):
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task(tmp_path / "t", RECIPE, {"franka": {"profile": "fr3"}})))
    assert sid == 0, cfg
    omni = SimpleNamespace(usd=SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: None)),
                           kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda: SimpleNamespace(update=lambda: None))))
    for name, module in {"omni": omni, "omni.usd": omni.usd, "omni.kit": omni.kit, "omni.kit.app": omni.kit.app,
                         "isaacsim": SimpleNamespace(), "isaacsim.core": SimpleNamespace(),
                         "isaacsim.core.utils": SimpleNamespace(),
                         "isaacsim.core.utils.stage": SimpleNamespace(is_stage_loading=lambda: False)}.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    monkeypatch.setattr(robot_profile, "mount", lambda stage, prefix, entry: calls.append(("mount", prefix)))
    monkeypatch.setattr(rg, "run_scene_recipe", lambda path, prefix: calls.append(("recipe", path.name, prefix)))
    manager._scenes[0].build_stage()
    assert calls == [("mount", "/Scene_0"), ("recipe", "scene.yaml", "/Scene_0")]
