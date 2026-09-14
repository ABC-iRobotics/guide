"""Pure parts of the Replicator-YAML spike: dialect detection, prefixing, axis mapping."""
import pytest

from guide_core.types.randomization import replicator_guide as rg


def test_dialect_detection(tmp_path):
    (tmp_path / "old.yaml").write_text("instructions: []\n")
    (tmp_path / "new.yaml").write_text("blocks:\n  get.prims: {path_pattern: '/blocks/.*'}\n")
    (tmp_path / "empty.yaml").write_text("")
    assert not rg.is_replicator_yaml(tmp_path / "old.yaml")
    assert rg.is_replicator_yaml(tmp_path / "new.yaml")
    assert not rg.is_replicator_yaml(tmp_path / "empty.yaml")


def test_prefixing_touches_only_path_patterns():
    doc = {
        "g": {"randomizer.register": {"g": {"blocks": {"get.prims": {"path_pattern": "/blocks/.*"}},
                                            "with.blocks": {"modify.pose": {"position": [0, 0, 0]}}}}},
        "z": {"guide.grid": {"distribution": "blocks_position", "resolution": 0.1}},
    }
    out = rg.prefixed(doc, "/Scene_3")
    assert out["g"]["randomizer.register"]["g"]["blocks"]["get.prims"]["path_pattern"] == "/Scene_3/blocks/.*"
    assert out["z"]["guide.grid"]["distribution"] == "blocks_position"
    assert doc["g"]["randomizer.register"]["g"]["blocks"]["get.prims"]["path_pattern"] == "/blocks/.*"  # untouched


def test_principal_axis():
    assert rg.principal_axis([0, 0, 2]) == 2
    assert rg.principal_axis([-1, 0, 0]) == 0
    with pytest.raises(ValueError):
        rg.principal_axis([1, 1, 0])


def test_event_name_and_scene_suffix():
    doc = {"a": {"trigger.on_custom_event": {"event_name": "guide_reset"}},
           "b": {"trigger.on_custom_event": {"event_name": "guide_reset"}}}
    out = rg.prefixed(doc, "/Scene_2")
    assert out["a"]["trigger.on_custom_event"]["event_name"] == "guide_reset_Scene_2"
    assert rg._event_name(out) == "guide_reset_Scene_2"
    with pytest.raises(ValueError):
        rg._event_name({"x": {"get.prims": {}}})
