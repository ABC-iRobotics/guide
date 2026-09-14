"""SceneManager.add_scene resolves a task given as a filesystem path in both layouts."""
import logging
import shutil
from pathlib import Path

import pytest

from guide_core.scene.scene_manager import SceneManager

DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"


@pytest.fixture
def manager(monkeypatch):
    # add_scene only needs the orchestrator's file parsing here; keep Isaac and the recorder out.
    import guide_core.scene.scene_orchestrator as so

    monkeypatch.setattr(so.SceneOrchestrator, "setup_recorder", lambda self, *a, **k: None, raising=False)
    return SceneManager(sim_id=0, logger=logging.getLogger("test"))


def test_flat_layout(manager, tmp_path):
    flat = tmp_path / "flat"
    shutil.copytree(DUMMY, flat)
    sid, _offset, cfg = manager.add_scene(str(flat))
    assert sid == 0
    assert cfg.get("error") is None, cfg


def test_source_package_layout(manager, tmp_path):
    pkg = tmp_path / "dummy_scene"
    (pkg / "dummy_scene").mkdir(parents=True)
    shutil.copy(DUMMY / "scene.py", pkg / "dummy_scene" / "scene.py")
    shutil.copytree(DUMMY / "config", pkg / "config")
    sid, _offset, cfg = manager.add_scene(str(pkg))
    assert sid == 0
    assert cfg.get("error") is None, cfg
