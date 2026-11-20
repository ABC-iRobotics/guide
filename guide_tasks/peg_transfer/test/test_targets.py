"""peg_transfer's target maths (peg_transfer.targets), the misses and the prompts, offline."""
import itertools
import re

import numpy as np
import pytest
import yaml
from peg_transfer import solve_task as st
from peg_transfer import targets as tg
from peg_transfer.scene import BLOCK_HALF_H, PEG_H, PROCEDURE, draw_plan
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import RandomizationRecord, Randomizer, SeedTree
from guide_ex.utility.pose import MissPose

UP = np.array([0.0, 0.0, 1.0])
RCM = {"psm1": np.array([-0.13, 0.82, 0.95]), "psm2": np.array([0.13, 0.82, 0.95])}  # davinci_sim, scene frame
BLOCK = Pose(Point([-0.043, 0.69, 0.80 + BLOCK_HALF_H]), Rotation(R.from_euler("z", 40, degrees=True)))
HANDOVER = yaml.safe_load((st.recipes_dir("peg_transfer", st.__file__).parent / "randomize.yaml").read_text())[
    "place"]["randomizer.register"]["place"]["with.handover"]["modify.pose"]


def rot(pose):
    return pose.orientation.to_numpy()


def test_tool_down_points_the_jaws_down_with_the_bend_free_closing_axis():
    near = Pose(Point([0, 0, 0]), Rotation(R.identity()))
    m = tg.tool_down(BLOCK.position.to_numpy() - RCM["psm1"], UP, near)
    assert np.allclose(m[:, 2], UP) and abs(m[:, 0] @ UP) < 1e-9
    assert tg.axis_deviation(m[:, 0], tg.bend_free_x(BLOCK.position.to_numpy() - RCM["psm1"], -UP)) < 1e-9
    assert m[:, 0] @ near.orientation.to_numpy()[:, 0] >= 0  # the sign nearer the current wrist


def test_the_grasp_is_tool_down_across_a_wall_and_near_the_bend_free_axis():
    block = Pose(Point(BLOCK.position.to_numpy() - RCM["psm1"]), BLOCK.orientation)  # RCM at the origin
    pose = tg.grasp_pose(block, UP)
    m, c = rot(pose), block.position.to_numpy()
    assert np.allclose(m[:, 2], UP)
    assert np.allclose(pose.position.to_numpy(), c + m @ tg.GRIP)  # the tip GRIP off the block's centre
    assert tg.axis_deviation(m[:, 0], tg.bend_free_x(c, -UP)) <= tg.MAX_DEVIATION + 1e-9
    walls = [rot(block) @ [np.cos(a), np.sin(a), 0] for a in tg.WALLS]
    # within MAX_DEVIATION of the wall normal it grips (exactly on it when that is in reach)
    assert min(tg.axis_deviation(m[:, 0], w) for w in walls) <= np.radians(60) - tg.MAX_DEVIATION + 1e-9


def test_the_lift_clears_the_peg_by_a_block_height():
    assert tg.LIFT == pytest.approx(PEG_H + 2 * BLOCK_HALF_H)
    bottom_on_peg = 0.80 + BLOCK_HALF_H - BLOCK_HALF_H  # the block's bottom while on the peg: the board
    assert bottom_on_peg + tg.LIFT - (0.80 + PEG_H) == pytest.approx(2 * BLOCK_HALF_H)


def test_the_level_turn_points_the_jaws_horizontally_at_the_receiver():
    at = BLOCK.position.to_numpy() + [0, 0, tg.LIFT]
    m = tg.level_turn(at, RCM["psm1"], RCM["psm2"] - RCM["psm1"])
    pointing = -m[:, 2]
    assert abs(pointing @ UP) < 1e-9 and pointing @ (RCM["psm2"] - RCM["psm1"]) > 0
    assert abs(m[:, 0] @ np.cross(at - RCM["psm1"], pointing)) == pytest.approx(np.linalg.norm(np.cross(at - RCM["psm1"], pointing)))


@pytest.mark.parametrize("giver,receiver", [("psm1", "psm2"), ("psm2", "psm1")])
def test_handover_poses_over_the_whole_jitter_box(giver, receiver):
    lo, hi = HANDOVER["position"]["distribution.uniform"]["lower"], HANDOVER["position"]["distribution.uniform"]["upper"]
    rlo, rhi = HANDOVER["rotation"]["distribution.uniform"]["lower"][0], HANDOVER["rotation"]["distribution.uniform"]["upper"][0]
    p = np.sign(RCM[receiver][0] - RCM[giver][0]) * np.array([1.0, 0, 0])
    for corner in itertools.product(*zip(lo, hi)):
        for roll in (rlo, 0.0, rhi):
            marker = Pose(Point(corner), Rotation(R.from_euler("x", roll, degrees=True)))
            give, receive = tg.handover_poses(marker, RCM[giver], RCM[receiver], st.ROLL_GAIN)
            g, r = rot(give), rot(receive)
            # the block's centre at the marker, from either tip
            assert np.allclose(give.position.to_numpy() - g @ tg.GRIP, corner)
            assert np.allclose(receive.position.to_numpy() - r @ tg.GRIP, corner)
            # jaws level, pointing at each other; the same closing axis from the two faces
            assert np.allclose(-g[:, 2], p) and np.allclose(-r[:, 2], -p) and np.allclose(g[:, 0], r[:, 0])
            # the closing axis turned off the arms' bisector by the scaled marker roll only
            free = tg.bend_free_x(np.array(corner) - RCM[giver], p)
            assert tg.axis_deviation(g[:, 0], free) <= abs(np.radians(roll)) * st.ROLL_GAIN + np.radians(10)


def test_place_poses_seat_the_block_1_mm_over_the_board_and_come_down_the_drop():
    peg = Pose(Point([0.043, 0.75, 0.80 + PEG_H / 2]))
    near = Pose(Point([0, 0, 0]), Rotation(R.identity()))
    over, seat = tg.place_poses(Pose(Point(peg.position.to_numpy() - RCM["psm2"])), UP, near)
    m = rot(seat)
    centre = seat.position.to_numpy() - m @ tg.GRIP + RCM["psm2"]
    assert np.allclose(centre, [0.043, 0.75, 0.80 + tg.SEAT_GAP + BLOCK_HALF_H])
    assert np.allclose(over.position.to_numpy() - seat.position.to_numpy(), UP * tg.DROP)
    assert np.allclose(m[:, 2], UP) and np.allclose(rot(over), m)
    assert tg.DROP - PEG_H == pytest.approx(0.010)  # over the peg: the block's bottom 1 cm over its top


@pytest.mark.parametrize("stage,axis", [("grasp", 1), ("handover", 1), ("place", None)])
def test_miss_offsets_lie_in_the_tip_frame_within_their_range(stage, axis):
    v = st.variant("light_fail")
    (target, spec), = v["stages"][stage].items()
    lo, hi = spec["r"]
    tip = tg.grasp_pose(Pose(Point(BLOCK.position.to_numpy() - RCM["psm1"]), BLOCK.orientation), UP)
    for seed in range(20):
        out = MissPose().run(tip, miss={"at": {"red": spec}}, seed=seed, key="red").outputs["pose"]
        local = rot(tip).T @ (out.position.to_numpy() - tip.position.to_numpy())
        assert lo - 1e-9 <= np.linalg.norm(local) <= hi + 1e-9
        assert local[2] == pytest.approx(0, abs=1e-12)  # never along the shaft
        if axis is not None:
            assert np.linalg.norm(local) == pytest.approx(abs(local[axis]))
    # a grasp / receive miss is off the wall's end; a place miss puts the hole's wall on the peg
    if stage == "place":
        assert lo > 0.00416 / np.cos(np.radians(30)) - 0.0015 and hi < 0.00485 + 0.0015
    else:
        assert lo > 0.0097 + 0.005


def test_the_prompts_at_three_levels():
    assert PROCEDURE == "Transfer the block to the other side."
    assert list(st.SUBTASKS.values()) == ["Pick up the block.", "Lift it off the peg.", "Hand it over.",
                                          "Place it on the peg.", "Return to start."]
    for seed in range(20):
        rng, _ = SeedTree.create(1).generator(seed)
        plan = draw_plan(Randomizer(rng, RandomizationRecord(seed=seed)))
        assert plan["task"] == PROCEDURE
        side = {"psm1": "right", "psm2": "left"}  # image sides: PSM1 is image right
        assert plan["tasks"] == [f"Move the {plan['colour']} block from the {side[plan['giver']]} to the "
                                 f"{side[plan['receiver']]} side."]
        assert re.fullmatch(r"Move the \w+ block from the (left|right) to the (left|right) side\.", plan["tasks"][0])


def test_held_only_with_the_block_where_the_grip_puts_it():
    tip = tg.grasp_pose(BLOCK, UP)
    assert tg.holds(tip, BLOCK)
    along_wall = tip * Pose(Point([0.0, 0.02, 0.0]))  # a planned miss: closed 20 mm along the wall, on air
    on_top = tip * Pose(Point([0.0, 0.0, 0.0065]))  # jaws stopped on the block's top (T10, wide open)
    assert not tg.holds(along_wall, BLOCK) and not tg.holds(on_top, BLOCK)
