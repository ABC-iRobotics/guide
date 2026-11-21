import logging
from pathlib import Path

import numpy as np
import pytest
from peg_transfer.scene import (
    BLOCK_HALF_H,
    PEG_H,
    SIDE_PEGS,
    Scene,
    draw_plan,
    on_peg,
    peg,
    seated,
)

from guide_core.scene.scene_orchestrator import SceneOrchestrator
from guide_core.types.geometry import Point, Pose
from guide_core.types.randomization import RandomizationRecord, Randomizer, SeedTree

USD = Path(__file__).resolve().parents[1] / "assets/peg_transfer.usd"
PEG = Pose(position=Point([-0.043, 0.69, 0.80 + PEG_H / 2]))  # peg 0's cylinder centre, board top 0.80


def plan(seed):
    rng, _ = SeedTree.create(1).generator(seed)
    return draw_plan(Randomizer(rng, RandomizationRecord(seed=seed)))


def test_the_plan_is_deterministic_per_seed():
    assert plan(3) == plan(3)
    assert any(plan(3) != plan(s) for s in range(4, 10))


def test_the_block_starts_on_the_giver_side_and_goes_to_the_receiver_side():
    for s in range(50):
        p = plan(s)
        source, target = (int(p[k][-2:]) for k in ("source_peg", "target_peg"))
        assert source in SIDE_PEGS[p["giver"]] and target in SIDE_PEGS[p["receiver"]]
        assert {p["giver"], p["receiver"]} == {"psm1", "psm2"}
        # all six blocks on the giver's six pegs; the drawn one on the source peg
        assert sorted(p["layout"]) == sorted(["red", "green", "blue", "yellow", "orange", "purple"])
        assert p["source_pegs"][p["layout"].index(p["colour"])] == p["source_peg"]
        assert all(0 <= y <= 120 for y in p["yaw"])
        # image sides: PSM1's side is the right of the ECM image
        assert p["side_from"] == {"psm1": "right", "psm2": "left"}[p["giver"]] != p["side_to"]


def test_the_direction_is_fifty_fifty():
    assert 80 <= sum(plan(s)["giver"] == "psm1" for s in range(200)) <= 120


def test_success_is_the_block_seated_on_the_target_peg_and_let_go():
    scene = Scene.__new__(Scene)
    scene._logger = logging.getLogger("test")
    on_board = 0.80 + BLOCK_HALF_H

    assert scene.is_success_postprocess([Pose(position=Point([-0.043, 0.69, on_board])), PEG, False, False])
    assert scene.is_success_postprocess([Pose(position=Point([-0.040, 0.69, on_board])), PEG, False])  # peg on the hole's wall
    assert not scene.is_success_postprocess([Pose(position=Point([-0.039, 0.69, on_board])), PEG, False])  # 4 mm off
    assert not scene.is_success_postprocess([Pose(position=Point([-0.043, 0.69, 0.80 + PEG_H + BLOCK_HALF_H])), PEG])
    assert not scene.is_success_postprocess([Pose(position=Point([-0.043, 0.69, on_board])), PEG, False, True])  # held


def test_randomize_puts_the_blocks_on_the_giver_pegs_where_the_board_left_them(monkeypatch):
    def randomize(self, *, seed=None, inject=None, zone=None, arrangement=None, zones=None):
        rng, _ = SeedTree.create(1).generator(seed)
        self.randomize_preprocess(Randomizer(rng, RandomizationRecord(seed=seed)))
        return "context"

    monkeypatch.setattr(SceneOrchestrator, "randomize", randomize)
    pegs = np.array([[0.01 * i, 0.7, 0.80 + PEG_H / 2] for i in range(6)])
    monkeypatch.setattr(Scene, "_peg_positions", lambda self, names: pegs)
    scene = Scene.__new__(Scene)
    scene._scene_id = 1

    assert scene.randomize(seed=5) == "context"
    (place,) = scene.randomize_instructions
    k = scene.plan["layout"].index(scene.plan["colour"])
    assert place["cmd"] == "set_world_poses"
    assert place["kwargs"]["prim_path"][k] == f"/Scene_1/PegTransfer/Blocks/{scene.plan['colour']}"
    block = place["kwargs"]["pose"][k]
    assert np.allclose(block.position.to_numpy(), pegs[k] + [0, 0, BLOCK_HALF_H - PEG_H / 2])
    assert seated(block, Pose(position=Point(pegs[k])))


def test_success_queries_the_block_the_target_peg_and_each_jaw():
    scene = Scene.__new__(Scene)
    scene._scene_id = 0
    scene._config = {"robots": {"psm1": {"path": "/PSM1/psm"}, "psm2": {"path": "/PSM2/psm"}}}
    scene.plan = {"block": "/PegTransfer/Blocks/red", "target_peg": peg(7)}
    templates = [{"cmd": "get_world_poses", "kwargs": {}}, {"cmd": "is_prim_clashing", "kwargs": {"scope": "x"}}]

    out = scene.is_success_preprocess(templates)

    assert [i["kwargs"]["prim_path"] for i in out[:2]] == ["/Scene_0/PegTransfer/Blocks/red", "/Scene_0/PegTransfer/Board/peg_07"]
    assert [i["kwargs"]["scope"] for i in out[2:]] == [
        f"/Scene_0/PSM{n}/psm/psm_tool_gripper{k}_link" for n in (1, 2) for k in (1, 2)]


def test_peg_sides_match_the_scene_usd():
    """Pegs 0-5 are on PSM1's side of the board (world x < 0, like PSM1's base), 6-11 on PSM2's."""
    pytest.importorskip("pxr")
    if not USD.exists():
        pytest.skip(f"{USD} absent")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(USD))
    x = lambda path: UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0).ExtractTranslation()[0]  # noqa: E731
    for arm, pegs in SIDE_PEGS.items():
        base = x(f"/World/{arm.upper()}/psm/psm_base_link")
        assert all(np.sign(x(f"/World{peg(i)}")) == np.sign(base) for i in pegs), arm
    assert x("/World/PSM1/psm/psm_base_link") < 0
    assert on_peg([0, 0, 0.809], 0).position.to_numpy()[2] == pytest.approx(0.80 + BLOCK_HALF_H)
