"""Arrangements (S3 v2 contract B -> C): Randomize {arrangement} loads config/arrangements/<id>.yaml
and pins it into the Replicator fire."""
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from guide_core.scene.scene_manager import SceneManager
from guide_core.types.randomization import replicator_guide as rg

DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"


@pytest.fixture
def registered(tmp_path):
    """The dummy task registered (no Isaac, no recorder) with a Replicator randomize state stubbed in."""
    task = tmp_path / "task"
    shutil.copytree(DUMMY, task)
    (task / "config" / "arrangements").mkdir()
    (task / "config" / "arrangements" / "demo_7.yaml").write_text(
        "plate_position: [0.1, 0.2, 0.0]\ncabinet_top_state: -0.12\n")
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task))
    assert sid == 0, cfg
    scene = manager._scenes[0]
    scene.replicator["randomize"] = {"named": {}}
    return manager, scene


def test_an_arrangement_is_read_and_pinned_into_the_fire(registered, monkeypatch):
    _, scene = registered
    seen = {}

    def fire(state, seed, zone, zone_target, values=None):
        seen.update(values=values, zone=zone)
        return {"plate_position": [[0.1, 0.2, 0.0]]}

    monkeypatch.setattr(rg, "fire", fire)
    context = scene.randomize(arrangement="demo_7")
    assert seen == {"values": {"plate_position": [0.1, 0.2, 0.0], "cabinet_top_state": -0.12}, "zone": None}
    assert context.arrangement == "demo_7"
    assert context.record.values["replicator/plate_position"] == [[0.1, 0.2, 0.0]]


def test_a_free_draw_pins_nothing(registered, monkeypatch):
    _, scene = registered
    seen = {}
    monkeypatch.setattr(rg, "fire", lambda state, seed, zone, zone_target, values=None: seen.update(values=values) or {})
    assert scene.randomize().arrangement is None and seen["values"] is None


@pytest.mark.parametrize("arrangement, error", [("missing", FileNotFoundError), ("../demo_7", ValueError), ("empty", ValueError)])
def test_an_arrangement_that_is_not_a_file_of_values_fails(registered, monkeypatch, arrangement, error):
    """Review focus 2: no file, a path for an id, an empty file -- nothing fires."""
    _, scene = registered
    (scene._path / "config" / "arrangements" / "empty.yaml").write_text("")
    monkeypatch.setattr(rg, "fire", lambda *a, **k: pytest.fail("nothing may fire"))
    with pytest.raises(error):
        scene.randomize(arrangement=arrangement)


def test_an_arrangement_needs_replicator_and_no_zone(registered, monkeypatch):
    _, scene = registered
    monkeypatch.setattr(rg, "fire", lambda *a, **k: pytest.fail("nothing may fire"))
    with pytest.raises(ValueError, match="zone"):
        scene.randomize(arrangement="demo_7", zone=3)
    del scene.replicator["randomize"]
    with pytest.raises(ValueError, match="Replicator"):
        scene.randomize(arrangement="demo_7")


def test_the_sidecar_names_the_arrangement(registered, monkeypatch):
    manager, scene = registered
    monkeypatch.setattr(rg, "fire", lambda *a, **k: {})
    scene.randomize(arrangement="demo_7")
    meta = {}
    scene.recorder = SimpleNamespace(set_pending_episode_meta=meta.update)
    manager._capture_episode_meta(0, "{}")
    assert meta["arrangement"] == "demo_7"


def test_a_scene_whose_randomize_predates_arrangements_is_called_as_before():
    """Review focus 5: cube_stack's randomize(*, seed, inject, zone) override takes no arrangement."""
    class Old:
        randomize_instructions = []

        def randomize(self, *, seed=None, inject=None, zone=None):
            self.got = (seed, inject, zone)

    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    manager._scenes = [Old()]
    manager.randomize_preprocess(0, seed=5, zone=2)
    assert manager._scenes[0].got == (5, None, 2)
