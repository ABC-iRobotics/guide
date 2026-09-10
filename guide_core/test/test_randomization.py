"""Unit tests for guide_core.types.randomization.

Pure-Python (NumPy + SciPy) — runs without Isaac Sim or ROS. Mirrors the test
plan in SCENE_REPRODUCE_PLAN.md section 7.
"""

from __future__ import annotations

import numpy as np
import pytest

from guide_core.types.randomization import (
    AxisAngle,
    Categorical,
    Constant,
    PoseDist,
    RandomizationRecord,
    Randomizer,
    SeedTree,
    UniformVec,
    from_spec,
    pose_from_yaml,
)
from guide_core.types.randomization import _quat


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def test_seedtree_determinism():
    tree = SeedTree(master=20260629)
    g1, s1 = tree.generator(0, 7)
    g2, s2 = tree.generator(0, 7)
    assert s1 == s2
    assert g1.uniform(size=5).tolist() == g2.uniform(size=5).tolist()
    assert tree.generator(1, 7)[1] != s1
    assert tree.generator(0, 8)[1] != s1


def test_master_seed_resolution():
    assert SeedTree.create(123).master == 123
    a = SeedTree.create()
    b = SeedTree.create()
    assert isinstance(a.master, int)
    assert a.master != b.master
    reborn = SeedTree.create(a.master)
    assert reborn.generator(2, 3)[1] == a.generator(2, 3)[1]


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
def test_uniform_vec_bounds_and_determinism():
    dist = UniformVec([-1, -2, 0], [1, 2, 0])
    v1 = dist.sample(np.random.default_rng(0))
    v2 = dist.sample(np.random.default_rng(0))
    assert np.array_equal(v1, v2)
    assert np.all(v1 >= dist.low) and np.all(v1 <= dist.high)
    assert v1[2] == 0.0
    with pytest.raises(ValueError):
        UniformVec([1, 0, 0], [0, 0, 0])


def test_axis_angle_is_unit_and_bounded():
    dist = AxisAngle(axis=[0, 0, 1], max_angle=np.pi / 4)
    rng = np.random.default_rng(42)
    for _ in range(50):
        q = dist.sample(rng)
        assert q.shape == (4,)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert abs(q[1]) < 1e-9 and abs(q[2]) < 1e-9


def test_categorical_preserves_native_type_and_is_seeded():
    dist = Categorical(("red", "yellow", "green", "blue"))
    picks1 = [dist.sample(np.random.default_rng(7)) for _ in range(1)]
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = [dist.sample(rng1) for _ in range(20)]
    b = [dist.sample(rng2) for _ in range(20)]
    assert a == b
    assert all(isinstance(p, str) for p in a)
    with pytest.raises(ValueError):
        Categorical(())


def test_pose_dist_shape():
    dist = PoseDist(UniformVec([-1, -1, 0], [1, 1, 0]), AxisAngle([0, 0, 1], 0.5))
    out = dist.sample(np.random.default_rng(1))
    assert out.shape == (7,)
    assert np.isclose(np.linalg.norm(out[3:]), 1.0)


def test_spec_roundtrip():
    dists = [
        Constant([1.0, 2.0, 3.0]),
        UniformVec([-1, -1, -1], [1, 1, 1]),
        AxisAngle([0, 0, 1], 0.3),
        Categorical(("a", "b", "c")),
        PoseDist(UniformVec([0, 0, 0], [1, 1, 1]), AxisAngle([1, 0, 0], 0.2)),
    ]
    for d in dists:
        rebuilt = from_spec(d.to_spec())
        a = d.sample(np.random.default_rng(99))
        b = rebuilt.sample(np.random.default_rng(99))
        assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        from_spec({"type": "nope"})


def test_pose_from_yaml_matches_block_bin_schema():
    spec = {
        "position": {"value": [0.25, -0.4, 0.09],
                     "random": {"low": [-0.05, -0.05, 0.0], "high": [0.05, 0.05, 0.0]}},
        "orientation": {"random": {"axis": [0.0, 0.0, 1.0], "angle": 180}},
    }
    dist = pose_from_yaml(spec)
    out = dist.sample(np.random.default_rng(3))
    assert out.shape == (7,)
    assert 0.20 <= out[0] <= 0.30 and -0.45 <= out[1] <= -0.35
    assert np.isclose(np.linalg.norm(out[3:]), 1.0)


# --------------------------------------------------------------------------- #
# Engine: capture + injection
# --------------------------------------------------------------------------- #
def _scene_dists():
    return {
        "/blocks/red_block": PoseDist(
            UniformVec([-0.25, 0.0, 0.025], [0.25, 0.25, 0.025]),
            AxisAngle([0, 0, 1], np.pi),
        ),
        "color": Categorical(("red", "yellow", "green", "blue")),
        "side": Categorical(("left", "right")),
    }


def _run(rng, *, inject=None):
    rec = RandomizationRecord(seed=0)
    rzr = Randomizer(rng, rec, inject=inject)
    drawn = {name: rzr.draw(name, dist) for name, dist in _scene_dists().items()}
    return drawn, rec


def test_randomizer_captures_every_draw():
    drawn, rec = _run(np.random.default_rng(5))
    assert set(rec.values) == set(_scene_dists())
    assert isinstance(rec.values["/blocks/red_block"], list)
    assert len(rec.values["/blocks/red_block"]) == 7
    assert rec.values["color"] in ("red", "yellow", "green", "blue")


def test_duplicate_draw_name_raises():
    rzr = Randomizer(np.random.default_rng(0), RandomizationRecord(seed=0))
    rzr.draw("x", Categorical(("a", "b")))
    with pytest.raises(KeyError):
        rzr.draw("x", Categorical(("a", "b")))


def test_inject_roundtrip_bit_for_bit():
    drawn1, rec1 = _run(np.random.default_rng(123))
    inject = RandomizationRecord.from_json(rec1.to_json())
    drawn2, rec2 = _run(np.random.default_rng(999), inject=inject)
    for name in _scene_dists():
        assert np.array_equal(np.asarray(drawn1[name]), np.asarray(drawn2[name])), name
    assert rec1.to_json() == rec2.to_json()


def test_record_json_roundtrip():
    _, rec = _run(np.random.default_rng(11))
    again = RandomizationRecord.from_json(rec.to_json())
    assert again.to_json() == rec.to_json()
    assert again.seed == rec.seed


# --------------------------------------------------------------------------- #
# Quaternion ordering helpers + SciPy convention
# --------------------------------------------------------------------------- #
def test_quat_ordering_roundtrip():
    wxyz = [0.5, 0.5, 0.5, 0.5]
    assert np.array_equal(_quat.xyzw_to_wxyz(_quat.wxyz_to_xyzw(wxyz)), wxyz)
    assert np.array_equal(_quat.wxyz_to_xyzw([1, 0, 0, 0]), [0, 0, 0, 1])


def test_axis_angle_against_scipy():
    from scipy.spatial.transform import Rotation as R

    dist = AxisAngle(axis=[0, 0, 1], max_angle=np.pi)
    q_wxyz = dist.sample(np.random.default_rng(0))
    rotvec = R.from_quat(_quat.wxyz_to_xyzw(q_wxyz)).as_rotvec()
    assert abs(rotvec[0]) < 1e-9 and abs(rotvec[1]) < 1e-9


# --------------------------------------------------------------------------- #
# Lifecycle glue: draw_instructions + SceneContext
# --------------------------------------------------------------------------- #
from guide_core.types.randomization.apply import draw_instructions  # noqa: E402
from guide_core.types.scene_context import SceneContext  # noqa: E402


def _pose_dist():
    return PoseDist(UniformVec([0, 0, 0], [1, 1, 1]), AxisAngle([0, 0, 1], 1.0))


def test_draw_instructions_capture_and_skip():
    instrs = [
        {"kwargs": {"prim_path": "/Scene_0/blocks/red", "pose": "BASE"}, "pose_dist": _pose_dist()},
        {"kwargs": {"prim_path": "/Scene_0/bin"}},  # no pose_dist -> untouched
    ]
    rec = RandomizationRecord(seed=0)
    draw_instructions(instrs, Randomizer(np.random.default_rng(0), rec),
                      pose_builder=lambda v: ("POSE", np.asarray(v)))
    assert instrs[0]["kwargs"]["pose"][0] == "POSE"
    assert len(instrs[0]["kwargs"]["pose"][1]) == 7
    assert "/Scene_0/blocks/red" in rec.values
    assert instrs[1]["kwargs"].get("pose") is None


def test_draw_instructions_inject_roundtrip():
    pd = _pose_dist()
    a = [{"kwargs": {"prim_path": "/Scene_0/blocks/red"}, "pose_dist": pd}]
    rec1 = RandomizationRecord(seed=0)
    draw_instructions(a, Randomizer(np.random.default_rng(7), rec1), pose_builder=np.asarray)
    b = [{"kwargs": {"prim_path": "/Scene_0/blocks/red"}, "pose_dist": pd}]
    rec2 = RandomizationRecord(seed=0)
    draw_instructions(b, Randomizer(np.random.default_rng(999), rec2,
                                    inject=RandomizationRecord.from_json(rec1.to_json())),
                      pose_builder=np.asarray)
    assert rec1.values["/Scene_0/blocks/red"] == rec2.values["/Scene_0/blocks/red"]


def test_scene_context_json_roundtrip():
    rec = RandomizationRecord(seed=42, values={"color": "red", "/b": [1.0, 2.0]})
    ctx = SceneContext(scene_id=1, episode_index=3, record=rec)
    again = SceneContext.from_json(ctx.to_json())
    assert again.scene_id == 1 and again.episode_index == 3
    assert again.record.seed == 42 and again.record.values["color"] == "red"
    assert SceneContext.from_json(SceneContext(2).to_json()).record is None


# --------------------------------------------------------------------------- #
# 3x2 [[min, max], ...] ranges: init.yaml `limits` and the pose `random` field
# --------------------------------------------------------------------------- #
def test_as_range_splits_min_max_columns():
    low, high = _quat.as_range([[-1, 2], [-3, 4], [0, 0]])
    assert low.tolist() == [-1, -3, 0] and high.tolist() == [2, 4, 0]
    low, high = _quat.as_range([])
    assert low.tolist() == [0, 0, 0] and high.tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        _quat.as_range([[-1, 2], [-3, 4]])


def test_scene_limits_matrix_becomes_bounding_box():
    from types import SimpleNamespace

    from guide_core.scene.scene_orchestrator import SceneOrchestrator

    scene = SimpleNamespace(_config={"limits": [[-1.0, 2.0], [-3.0, 4.0], [-0.5, 0.5]]})
    SceneOrchestrator._get_limits(scene)
    assert scene.bounding_box == {"xp": 2.0, "xn": 1.0, "yp": 4.0, "yn": 3.0, "zp": 0.5, "zn": 0.5}
