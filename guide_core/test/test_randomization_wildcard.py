"""Per-prim wildcard randomization (draw_instructions pattern expansion).

Pure Python (NumPy/SciPy) — no Isaac/ROS, since the prim resolver is injected.

Regression target: a joker/pattern prim_path (e.g. ``/Scene_0/blocks/*``) used to
draw ONE pose and apply it to every matched prim, stacking all blocks at the same
point (which the physics solver then shoves apart into a line). The fix expands the
pattern and draws an independent pose per prim.
"""

from __future__ import annotations

import numpy as np
import pytest

from guide_core.types.randomization import RandomizationRecord, Randomizer
from guide_core.types.randomization.apply import draw_instructions, is_prim_pattern
from guide_core.types.randomization.distributions import AxisAngle, PoseDist, UniformVec


def _pose_dist() -> PoseDist:
    # Mirrors block_bin randomize.yaml: x,y uniform (z fixed) + yaw about Z.
    return PoseDist(
        UniformVec([-0.25, 0.0, 0.025], [0.25, 0.25, 0.025]),
        AxisAngle([0.0, 0.0, 1.0], np.deg2rad([-180.0, 180.0])),
    )


def _builder(v):
    # Keep the raw realized 7-vector so tests can inspect positions.
    return list(np.asarray(v, dtype=float))


def _rzr(seed=0, inject=None):
    return Randomizer(np.random.default_rng(seed), RandomizationRecord(seed=seed), inject=inject)


def _instr():
    return [{"kwargs": {"prim_path": "/Scene_0/blocks/*"}, "pose_dist": _pose_dist()}]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/Scene_0/blocks/red_block", False),
        ("/bin_0", False),
        ("/Scene_0/blocks/*", True),
        ("/Scene_0/block_[0-9]", True),
        (["/a", "/b"], False),
        (None, False),
    ],
)
def test_is_prim_pattern(path, expected):
    assert is_prim_pattern(path) is expected


def test_pattern_draws_distinct_pose_per_prim():
    prims = [
        "/Scene_0/blocks/blue_block",
        "/Scene_0/blocks/green_block",
        "/Scene_0/blocks/red_block",
    ]
    instr = _instr()
    rzr = _rzr(0)

    draw_instructions(instr, rzr, _builder, prim_resolver=lambda _: prims)

    kw = instr[0]["kwargs"]
    # prim_path expanded to the concrete list; one pose per prim, same order
    assert kw["prim_path"] == prims
    assert np.asarray(kw["pose"]).shape == (len(prims), 7)
    # record keyed by each concrete prim, not the wildcard
    assert set(rzr.record.values) == set(prims)
    assert "/Scene_0/blocks/*" not in rzr.record.values
    # THE FIX: xy positions are all distinct -> blocks scatter, don't stack
    xy = [tuple(np.round(p[:2], 6)) for p in kw["pose"]]
    assert len(set(xy)) == len(prims), f"positions must all differ, got {xy}"


def test_single_literal_path_unchanged():
    instr = [{"kwargs": {"prim_path": "/Scene_0/bin_0"}, "pose_dist": _pose_dist()}]
    rzr = _rzr(0)
    # resolver present but must be ignored for a literal path
    draw_instructions(instr, rzr, _builder, prim_resolver=lambda _: ["should", "not", "use"])
    kw = instr[0]["kwargs"]
    assert kw["prim_path"] == "/Scene_0/bin_0"
    assert np.asarray(kw["pose"]).shape == (7,)  # a single pose, not a list of poses
    assert list(rzr.record.values) == ["/Scene_0/bin_0"]


def test_empty_match_falls_back_to_single_draw():
    instr = _instr()
    rzr = _rzr(0)
    draw_instructions(instr, rzr, _builder, prim_resolver=lambda _: [])
    kw = instr[0]["kwargs"]
    assert np.asarray(kw["pose"]).shape == (7,)
    assert list(rzr.record.values) == ["/Scene_0/blocks/*"]  # keyed by the pattern (fallback)


def test_no_resolver_is_backward_compatible():
    instr = _instr()
    rzr = _rzr(0)
    draw_instructions(instr, rzr, _builder)  # 3-arg call, as before
    assert np.asarray(instr[0]["kwargs"]["pose"]).shape == (7,)
    assert list(rzr.record.values) == ["/Scene_0/blocks/*"]


def test_repeated_randomization_keeps_expanding():
    """Regression: the 2nd/3rd randomize must still scatter, not collapse to a line.

    The first expansion overwrites prim_path with the concrete list; the original
    pattern must be remembered so later passes re-expand instead of falling through
    to the single-draw path (which would stack every prim at one point).
    """
    prims = [
        "/Scene_0/blocks/blue_block",
        "/Scene_0/blocks/green_block",
        "/Scene_0/blocks/red_block",
    ]
    calls = []

    def resolver(expr):
        calls.append(expr)
        return prims

    instr = _instr()  # a single, reused instruction (as in a live scene)
    for i in range(3):
        rzr = _rzr(seed=i)  # fresh record each episode
        draw_instructions(instr, rzr, _builder, prim_resolver=resolver)
        kw = instr[0]["kwargs"]
        assert np.asarray(kw["pose"]).shape == (len(prims), 7), f"pass {i}: not per-prim"
        xy = [tuple(np.round(p[:2], 6)) for p in kw["pose"]]
        assert len(set(xy)) == len(prims), f"pass {i}: poses must differ, got {xy}"
        assert set(rzr.record.values) == set(prims), f"pass {i}: per-prim keys expected"

    # The resolver is always given the ORIGINAL pattern, never the expanded list.
    assert calls == ["/Scene_0/blocks/*"] * 3, calls


def test_per_prim_reproduces_under_injection():
    prims = ["/Scene_0/blocks/blue_block", "/Scene_0/blocks/red_block"]
    dist = _pose_dist()

    instr1 = [{"kwargs": {"prim_path": "/Scene_0/blocks/*"}, "pose_dist": dist}]
    r1 = _rzr(0)
    draw_instructions(instr1, r1, _builder, prim_resolver=lambda _: prims)

    # Replay from the recorded values via injection, with a *different* base seed
    # (injection must win, per-prim).
    inject = RandomizationRecord.from_json(r1.record.to_json())
    instr2 = [{"kwargs": {"prim_path": "/Scene_0/blocks/*"}, "pose_dist": dist}]
    r2 = _rzr(999, inject=inject)
    draw_instructions(instr2, r2, _builder, prim_resolver=lambda _: prims)

    # The drawn per-prim values reproduce exactly (the seed field differs by design).
    assert r1.record.values == r2.record.values
    assert instr1[0]["kwargs"]["pose"] == instr2[0]["kwargs"]["pose"]
