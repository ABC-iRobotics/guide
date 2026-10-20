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


class FakeRng:
    """omni.replicator.core rng.set_global_seed: an unchanged value is a no-op, so no sampler restarts."""

    def __init__(self, current):
        self.current, self.restarts = current, []

    def set_global_seed(self, seed):
        if seed == self.current:
            return
        self.current = seed
        self.restarts.append(seed)


@pytest.mark.parametrize("current, seed", [(0, 0), (5, 5), (3, 5)])
def test_reseed_restarts_the_samplers_even_for_an_unchanged_seed(current, seed):
    rng = FakeRng(current)
    rg._reseed(rng, seed, current)
    assert rng.current == seed and rng.restarts[-1] == seed


def test_event_name_and_scene_suffix():
    doc = {"a": {"trigger.on_custom_event": {"event_name": "guide_reset"}},
           "b": {"trigger.on_custom_event": {"event_name": "guide_reset"}}}
    out = rg.prefixed(doc, "/Scene_2")
    assert out["a"]["trigger.on_custom_event"]["event_name"] == "guide_reset_Scene_2"
    assert rg.events(out) == ("guide_reset_Scene_2", None)
    with pytest.raises(ValueError):
        rg.events({"x": {"get.prims": {}}})


def test_zone_grid_reads_either_dialect(tmp_path):
    (tmp_path / "rep.yaml").write_text(
        "g:\n"
        "  randomizer.register:\n"
        "    g:\n"
        "      with.x:\n"
        "        modify.pose:\n"
        "          position:\n"
        "            distribution.uniform: {lower: [-0.25, 0.0, 0.025], upper: [0.25, 0.4, 0.025], name: p}\n"
        "w:\n"
        "  with.e:\n"
        "    guide.zone: {distribution: p, path_pattern: '/x', resolution: 0.1}\n"
    )
    (tmp_path / "old.yaml").write_text(
        "instructions:\n"
        "  - cmd: set_local_poses\n"
        "    kwargs:\n"
        "      pose:\n"
        "        position: {value: [0, 0, 0.025], random: [[-0.25, 0.25], [0, 0.4], [0, 0]],"
        " grid: {enabled: true, resolution: 0.1}}\n"
    )
    (tmp_path / "free.yaml").write_text("instructions: []\n")

    assert rg.zone_grid(tmp_path / "rep.yaml").num_zones == 20
    assert rg.zone_grid(tmp_path / "old.yaml").num_zones == 20
    assert rg.zone_grid(tmp_path / "free.yaml") is None


from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402


class Node:
    """An OmniGraph node: its attribute values and its type name."""

    def __init__(self, attrs, kind=rg.UNIFORM):
        self.attrs, self.kind = dict(attrs), kind

    def get_type_name(self):
        return self.kind


@pytest.fixture
def og(monkeypatch):
    monkeypatch.setattr(rg, "_get", lambda node, attr: node.attrs[attr])


def test_events_split_the_pose_event_from_the_joint_event():
    doc = rg.prefixed({"e": {"trigger.on_custom_event": {"event_name": "guide_randomize"}},
                       "j": {"trigger.on_custom_event": {"event_name": "guide_randomize_joints"}}}, "/Scene_1")
    assert rg.events(doc) == ("guide_randomize_Scene_1", "guide_randomize_joints_Scene_1")
    with pytest.raises(ValueError):  # a joint event alone has no pose event to follow
        rg.events({"j": {"trigger.on_custom_event": {"event_name": "guide_randomize_joints"}}})


@pytest.mark.parametrize("physics_hz, render_hz, updates", [(120, 60, 2), (60, 60, 3), (240, 60, 1)])
def test_the_joint_event_waits_three_physics_frames(physics_hz, render_hz, updates):
    assert rg.joint_delay_updates(1 / physics_hz, 1 / render_hz) == updates


def test_a_guide_argument_is_its_distributions_sample_or_the_constant(og):
    vector = SimpleNamespace(node=Node({"inputs:lower": [0.0, -0.3, 0.005], "inputs:upper": [0.04, -0.26, 0.005],
                                        "outputs:samples": [[0.01, -0.29, 0.005]]}))
    scalar = SimpleNamespace(node=Node({"inputs:lower": [-0.16], "inputs:upper": [0.0], "outputs:samples": [-0.12]}))
    assert rg.drawn(vector).tolist() == [0.01, -0.29, 0.005]
    assert rg.midpoint(vector).tolist() == pytest.approx([0.02, -0.28, 0.005])
    assert float(np.ravel(rg.drawn(scalar))[0]) == -0.12
    assert rg.drawn([0, 0, 154]).tolist() == [0, 0, 154] and float(rg.midpoint(-0.12)) == -0.12
    assert rg.attribute_type(scalar.node) == "float" and rg.attribute_type(vector.node) == "float3"


def test_guide_nodes_take_uniform_distributions_only(og):
    normal = SimpleNamespace(node=Node({"inputs:mean": [0.0]}, kind="omni.replicator.core.OgnSampleNormal"))
    with pytest.raises(ValueError, match="distribution.uniform"):
        rg.midpoint(normal)
    two = Node({"inputs:lower": [0.0, 0.0], "inputs:upper": [1.0, 1.0]})
    with pytest.raises(ValueError, match="one or three"):
        rg.attribute_type(two)
