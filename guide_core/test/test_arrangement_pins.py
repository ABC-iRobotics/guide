"""fire(values=...): an arrangement pins named uniform distributions for one fire (contract B -> C)."""
import pytest

from guide_core.types.randomization import replicator_guide as rg
from guide_core.types.randomization.grid import Grid


class Node:
    """An OmniGraph node: its type name and its attribute values."""

    def __init__(self, attrs, kind=rg.UNIFORM):
        self.attrs, self.kind = dict(attrs), kind

    def get_type_name(self):
        return self.kind


@pytest.fixture
def og(monkeypatch):
    monkeypatch.setattr(rg, "_get", lambda node, attr: node.attrs[attr])
    monkeypatch.setattr(rg, "_set", lambda node, attr, value: node.attrs.__setitem__(attr, value))


def named():
    return {"plate_position": Node({"inputs:lower": [-0.1, 0.1, 0.0033], "inputs:upper": [0.1, 0.25, 0.0033]}),
            "cabinet_top_state": Node({"inputs:lower": [-0.16], "inputs:upper": [0.0]})}


def test_pinning_sets_lower_and_upper_and_unpinning_restores_them(og):
    nodes = named()
    pins = rg._pin(nodes, {"plate_position": [0.05, 0.2, 0.0033], "cabinet_top_state": -0.12})
    plate = nodes["plate_position"].attrs
    assert plate["inputs:lower"] == plate["inputs:upper"] == [0.05, 0.2, 0.0033]
    assert nodes["cabinet_top_state"].attrs["inputs:upper"] == [-0.12]
    rg._unpin(pins)
    assert plate == {"inputs:lower": [-0.1, 0.1, 0.0033], "inputs:upper": [0.1, 0.25, 0.0033]}
    assert nodes["cabinet_top_state"].attrs == {"inputs:lower": [-0.16], "inputs:upper": [0.0]}


@pytest.mark.parametrize("values, message", [
    ({"plate_position": [0.0, 0.0, 0.0], "no_such": 1.0}, "no_such"),
    ({"plate_position": [0.05, 0.2]}, "plate_position takes 3"),
    ({"cabinet_top_state": [0.0, 1.0]}, "cabinet_top_state takes 1"),
])
def test_a_bad_arrangement_fails_before_any_node_moves(og, values, message):
    """Review focus 2: the culprit named, nothing pinned."""
    nodes = named()
    before = {name: dict(node.attrs) for name, node in nodes.items()}
    with pytest.raises(ValueError, match=message):
        rg._pin(nodes, values)
    assert {name: node.attrs for name, node in nodes.items()} == before


def test_only_uniform_distributions_can_be_pinned(og):
    nodes = {"spread": Node({"inputs:mean": [0.0], "inputs:std": [1.0]}, kind="omni.replicator.core.OgnSampleNormal")}
    with pytest.raises(ValueError, match="spread.*distribution.uniform"):
        rg._pin(nodes, {"spread": 0.0})


def test_a_zoned_scene_takes_an_arrangement_only_with_its_zone_distribution():
    """Ruling R8: the zone redraw would move the target off its arranged pose."""
    g = Grid([-0.15, 0.0, 0.042], [0.15, 0.4, 0.042], 0.1)
    state = {"grid": g, "zone_distribution": "bowl_position"}
    assert rg.zone_pin(state, {"bowl_position": [0.05, 0.2, 0.042]}, "/Scene_0/bowl") == [0.05, 0.2, 0.042]
    assert rg.zone_pin(state, {}, "/Scene_0/bowl") is None  # a free draw
    assert rg.zone_pin({"grid": None}, {"x": 1.0}, "/Scene_0/bowl") is None  # no zone
    with pytest.raises(ValueError, match="bowl_position"):
        rg.zone_pin(state, {"plate_position": [0, 0, 0]}, "/Scene_0/bowl")


def test_the_zone_redraw_bounds():
    g = Grid([-0.15, 0.0, 0.042], [0.15, 0.4, 0.042], 0.1)
    low, high = rg.zone_bounds(g, None, [0.05, 0.2, 0.042])
    assert low.tolist() == high.tolist() == [0.05, 0.2, 0.042]
    assert [b.tolist() for b in rg.zone_bounds(g, 4, None)] == [b.tolist() for b in g.cell_bounds(4)]
    assert [b.tolist() for b in rg.zone_bounds(g, None, None)] == [g.low.tolist(), g.high.tolist()]



def test_a_failure_on_the_second_node_restores_both(monkeypatch):
    nodes = named()
    before = {name: dict(node.attrs) for name, node in nodes.items()}
    monkeypatch.setattr(rg, "_get", lambda node, attr: node.attrs[attr])

    def write(node, attr, value):
        node.attrs[attr] = value
        if node is nodes["cabinet_top_state"] and attr == "inputs:upper":
            raise RuntimeError("write failed")  # after inputs:lower was written: it must come back too

    monkeypatch.setattr(rg, "_set", write)
    with pytest.raises(RuntimeError, match="write failed"):
        with rg.pinning(nodes, {"plate_position": [0.05, 0.2, 0.0033], "cabinet_top_state": -0.12}):
            pytest.fail("the block must not run")
    assert {name: node.attrs for name, node in nodes.items()} == before


def test_pinning_restores_when_the_block_raises(og):
    nodes = named()
    before = {name: dict(node.attrs) for name, node in nodes.items()}
    with pytest.raises(KeyError):
        with rg.pinning(nodes, {"plate_position": [0.0, 0.2, 0.0033]}):
            assert nodes["plate_position"].attrs["inputs:lower"] == [0.0, 0.2, 0.0033]
            raise KeyError("the fire failed")
    assert {name: node.attrs for name, node in nodes.items()} == before


def test_unpin_goes_on_past_a_failing_node_and_raises_the_first_error(monkeypatch):
    nodes = named()
    pins = [(nodes["plate_position"], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]), (nodes["cabinet_top_state"], [-1.0], [1.0])]

    def write(node, attr, value):
        if node is nodes["plate_position"]:
            raise RuntimeError("first")
        node.attrs[attr] = value

    monkeypatch.setattr(rg, "_set", write)
    with pytest.raises(RuntimeError, match="first"):
        rg._unpin(pins)
    assert nodes["cabinet_top_state"].attrs == {"inputs:lower": [-1.0], "inputs:upper": [1.0]}
