"""Grid/zone model + zoned draw pipeline. Pure Python (NumPy/SciPy) — no Isaac/ROS."""

from __future__ import annotations

import numpy as np
import pytest

from guide_core.types.randomization import (
    Grid,
    PoseDist,
    RandomizationRecord,
    Randomizer,
    UniformVec,
    grid_from_yaml,
    single_grid,
    zone_plan,
)
from guide_core.types.randomization.apply import draw_instructions
from guide_core.types.randomization.distributions import AxisAngle


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #
def test_grid_shape_and_num_zones():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    assert g.ncols == 3 and g.nrows == 2
    assert g.num_zones == 6


def test_grid_partial_edge_cells_use_ceil_and_clamp():
    g = Grid([0.0, 0.0, 0.0], [0.25, 0.15, 0.0], 0.1)  # not a multiple of 0.1
    assert g.ncols == 3 and g.nrows == 2  # ceil
    lo, hi = g.cell_bounds(2)  # last column, row 0
    assert np.isclose(lo[0], 0.2) and np.isclose(hi[0], 0.25)  # clamped to high


def test_zone_numbering_row_major_from_min_corner():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    lo0, _ = g.cell_bounds(0)
    assert np.allclose(lo0[:2], [0.0, 0.0])  # zone 0 = (min-x, min-y)
    lo1, _ = g.cell_bounds(1)
    assert np.allclose(lo1[:2], [0.1, 0.0])  # +1 advances x (column)
    lo3, _ = g.cell_bounds(3)
    assert np.allclose(lo3[:2], [0.0, 0.1])  # +ncols advances y (row)


def test_cell_bounds_tile_without_gaps_or_overlap():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    seen = set()
    for z in range(g.num_zones):
        lo, hi = g.cell_bounds(z)
        assert hi[0] > lo[0] and hi[1] > lo[1]
        seen.add((round(float(lo[0]), 6), round(float(lo[1]), 6)))
    assert len(seen) == g.num_zones  # every cell distinct


def test_zone_of_round_trips_cell_centers():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    for z in range(g.num_zones):
        lo, hi = g.cell_bounds(z)
        center = (lo + hi) / 2.0
        assert g.zone_of(center) == z


def test_out_of_range_zone_raises():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    for bad in (-1, 6, 99):
        with pytest.raises(ValueError):
            g.cell_bounds(bad)


def test_restrict_samples_inside_the_cell():
    g = Grid([0.0, 0.0, 0.0], [0.3, 0.2, 0.0], 0.1)
    base = PoseDist(UniformVec([0, 0, 0.025], [0.3, 0.2, 0.025]), AxisAngle([0, 0, 1], 0.0))
    rng = np.random.default_rng(0)
    zone = 4  # col 1, row 1 -> x∈[0.1,0.2], y∈[0.1,0.2]
    for _ in range(200):
        v = g.restrict(base, zone).sample(rng)
        assert 0.1 <= v[0] <= 0.2 and 0.1 <= v[1] <= 0.2


# --------------------------------------------------------------------------- #
# grid_from_yaml
# --------------------------------------------------------------------------- #
def test_grid_from_yaml_builds_region_like_pose_from_yaml():
    spec = {
        "value": [0.45, 0.0, 0.025],
        "random": [[-0.15, 0.15], [-0.2, 0.2], [0.0, 0.0]],
        "grid": {"enabled": True, "resolution": 0.1},
    }
    g = grid_from_yaml(spec)
    assert np.allclose(g.low[:2], [0.30, -0.20]) and np.allclose(g.high[:2], [0.60, 0.20])
    assert g.ncols == 3 and g.nrows == 4 and g.num_zones == 12


def test_grid_from_yaml_absent_or_disabled_returns_none():
    assert grid_from_yaml(None) is None
    assert grid_from_yaml({"random": [[0, 1], [0, 1], [0, 0]]}) is None
    assert grid_from_yaml({"grid": {"enabled": False}, "random": [[0, 1], [0, 1], [0, 0]]}) is None


# --------------------------------------------------------------------------- #
# Zoned draw through draw_instructions (dynamic target)
# --------------------------------------------------------------------------- #
def _builder(v):
    return list(np.asarray(v, dtype=float))


def _rzr(seed=0):
    return Randomizer(np.random.default_rng(seed), RandomizationRecord(seed=seed))


def _blocks_instruction():
    g = Grid([0.30, -0.20, 0.025], [0.60, 0.20, 0.025], 0.1)  # 3x4 = 12 zones
    return [
        {
            "kwargs": {"prim_path": "/Scene_0/blocks/*"},
            "pose_dist": PoseDist(
                UniformVec([0.30, -0.20, 0.025], [0.60, 0.20, 0.025]),
                AxisAngle([0, 0, 1], np.deg2rad(180)),
            ),
            "grid": g,
        }
    ]


def test_only_zone_target_prim_lands_in_the_cell():
    prims = ["/Scene_0/blocks/blue_block", "/Scene_0/blocks/red_block"]
    instr = _blocks_instruction()
    g = instr[0]["grid"]
    zone = 5
    draw_instructions(
        instr, _rzr(0), _builder, prim_resolver=lambda _: prims,
        zone=zone, zone_target="/Scene_0/blocks/red_block",
    )
    poses = {p: pose for p, pose in zip(instr[0]["kwargs"]["prim_path"], instr[0]["kwargs"]["pose"])}
    lo, hi = g.cell_bounds(zone)

    def _in_cell(p):
        return lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]

    assert _in_cell(poses["/Scene_0/blocks/red_block"])  # target in cell

    # The non-target prim must be genuinely free, not merely "sometimes elsewhere":
    # across many seeds the target stays in the cell every time while blue escapes it
    # at least once. A single-seed check would pass even if blue were also zoned.
    blue_escaped = False
    for s in range(30):
        i = _blocks_instruction()
        draw_instructions(
            i, _rzr(s), _builder, prim_resolver=lambda _: prims,
            zone=zone, zone_target="/Scene_0/blocks/red_block",
        )
        p = dict(zip(i[0]["kwargs"]["prim_path"], i[0]["kwargs"]["pose"]))
        assert _in_cell(p["/Scene_0/blocks/red_block"])
        blue_escaped |= not _in_cell(p["/Scene_0/blocks/blue_block"])
    assert blue_escaped, "non-target prim never left the zone cell — it is being zoned too"


def test_zone_none_is_free_for_all():
    prims = ["/Scene_0/blocks/red_block"]
    instr = _blocks_instruction()
    draw_instructions(instr, _rzr(0), _builder, prim_resolver=lambda _: prims,
                      zone=None, zone_target="/Scene_0/blocks/red_block")
    # a single draw over the full range — just assert it produced a per-prim list
    assert np.asarray(instr[0]["kwargs"]["pose"]).shape == (1, 7)


def test_non_grid_instruction_ignores_zone():
    instr = [{
        "kwargs": {"prim_path": "/Scene_0/bin_0"},
        "pose_dist": PoseDist(UniformVec([0, 0, 0], [1, 1, 0]), AxisAngle([0, 0, 1], 0.0)),
    }]  # no "grid"
    draw_instructions(instr, _rzr(0), _builder, prim_resolver=lambda _: [], zone=3, zone_target=None)
    assert np.asarray(instr[0]["kwargs"]["pose"]).shape == (7,)


def test_negative_zone_is_free():
    """`zones: [-1]` is the documented escape hatch for free demos on a gridded scene.

    `_dist_for` gates on `zone >= 0`, so a negative zone must fall through to the full
    range rather than restricting to a cell or raising. The README documents this, so
    it needs pinning: `Grid.cell_bounds(-1)` raises, meaning a regression here would
    surface as a hard failure mid-generation.
    """
    prims = ["/Scene_0/blocks/red_block"]
    g = _blocks_instruction()[0]["grid"]
    lo, hi = g.cell_bounds(0)
    escaped = False
    for s in range(30):
        instr = _blocks_instruction()
        draw_instructions(
            instr, _rzr(s), _builder, prim_resolver=lambda _: prims,
            zone=-1, zone_target="/Scene_0/blocks/red_block",
        )
        p = instr[0]["kwargs"]["pose"][0]
        escaped |= not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1])
    assert escaped, "zone=-1 confined the draw to a cell; it must be a free draw"


def test_zoned_draw_reproduces_under_injection():
    prims = ["/Scene_0/blocks/red_block", "/Scene_0/blocks/blue_block"]
    i1 = _blocks_instruction()
    r1 = _rzr(0)
    draw_instructions(i1, r1, _builder, prim_resolver=lambda _: prims, zone=7,
                      zone_target="/Scene_0/blocks/red_block")
    inject = RandomizationRecord.from_json(r1.record.to_json())
    i2 = _blocks_instruction()
    r2 = Randomizer(np.random.default_rng(999), RandomizationRecord(seed=999), inject=inject)
    draw_instructions(i2, r2, _builder, prim_resolver=lambda _: prims, zone=7,
                      zone_target="/Scene_0/blocks/red_block")
    assert r1.record.values == r2.record.values


# --------------------------------------------------------------------------- #
# Scene validation: at most one grid
# --------------------------------------------------------------------------- #
def test_single_grid_returns_the_only_grid():
    g = Grid([0, 0, 0], [0.2, 0.2, 0], 0.1)
    instrs = [{"grid": g}, {"kwargs": {}}, {"grid": None}]
    assert single_grid(instrs) is g


def test_single_grid_returns_none_when_ungridded():
    assert single_grid([{"kwargs": {}}, {"grid": None}]) is None
    assert single_grid([]) is None


def test_single_grid_rejects_a_second_grid():
    g = Grid([0, 0, 0], [0.2, 0.2, 0], 0.1)
    with pytest.raises(ValueError, match="At most one grid"):
        single_grid([{"grid": g}, {"grid": g}])


# --------------------------------------------------------------------------- #
# Demonstration request -> per-episode zone plan
# --------------------------------------------------------------------------- #
def test_zone_plan_free_demos_on_ungridded_scene():
    # No grid (num_zones == 1): an empty `zones` means N free episodes, not N zoned ones.
    assert zone_plan([], [3], num_zones=1) == [None, None, None]


def test_zone_plan_empty_zones_sweeps_every_zone():
    # The multiplying case: 2 counts over 4 zones is 8 episodes, ascending by zone.
    assert zone_plan([], [2], num_zones=4) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_zone_plan_explicit_zones_and_counts():
    assert zone_plan([2, 16], [4, 1], num_zones=20) == [2, 2, 2, 2, 16]


def test_zone_plan_short_counts_falls_back_to_first():
    # Fewer counts than zones: the remaining zones reuse counts[0].
    assert zone_plan([1, 2, 3], [2], num_zones=20) == [1, 1, 2, 2, 3, 3]


def test_zone_plan_negative_zone_passes_through():
    # The free-demo escape hatch survives planning untouched (see test_negative_zone_is_free).
    assert zone_plan([-1], [3], num_zones=20) == [-1, -1, -1]


def test_zone_plan_empty_counts_is_empty():
    assert zone_plan([], [], num_zones=20) == []
    assert zone_plan([5], [], num_zones=20) == []
