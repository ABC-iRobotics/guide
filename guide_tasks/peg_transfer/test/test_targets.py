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
BASE = {"psm1": Pose(Point(RCM["psm1"]), Rotation(R.from_quat([-0.1297, -0.3817, -0.2945, 0.8664]))),  # measured
        "psm2": Pose(Point(RCM["psm2"]), Rotation(R.from_quat([-0.1297, 0.3817, 0.2945, 0.8664])))}
BLOCK = Pose(Point([-0.043, 0.69, 0.80 + BLOCK_HALF_H]), Rotation(R.from_euler("z", 40, degrees=True)))
HANDOVER = yaml.safe_load((st.recipes_dir("peg_transfer", st.__file__).parent / "randomize.yaml").read_text())[
    "place"]["randomizer.register"]["place"]["with.handover"]["modify.pose"]


def rot(pose):
    return pose.orientation.to_numpy()


def test_near_home_keeps_the_grip_and_turns_the_closing_axis_to_base_x():
    tip = tg.tip_for(BLOCK.position.to_numpy(), tg.frame([-1.0, 0.2, 0.0], UP))
    flipped = tg.near_home(tip)
    assert rot(flipped)[0, 0] > 0 and np.allclose(rot(flipped)[:, 2], UP)
    assert np.allclose(flipped.position.to_numpy(), tip.position.to_numpy())  # the same grip, the jaws swapped
    assert tg.holds(flipped, BLOCK) and tg.near_home(flipped) is flipped


def test_the_grasp_is_tool_down_across_a_wall_and_near_the_bend_free_axis():
    block = Pose(Point(BLOCK.position.to_numpy() - RCM["psm1"]), BLOCK.orientation)  # RCM at the origin
    pose = tg.grasp_pose(block, UP)
    m, c = rot(pose), block.position.to_numpy()
    assert np.allclose(m[:, 2], UP)
    assert np.linalg.norm(tg.grip_centres(pose) - c, axis=1).min() < 1e-12  # the tip GRIP off the block's centre
    assert m[0, 0] >= 0  # near_home: tool_roll near home
    assert tg.axis_deviation(m[:, 0], tg.bend_free_x(c, -UP)) <= tg.MAX_DEVIATION + 1e-9
    walls = [rot(block) @ [np.cos(a), np.sin(a), 0] for a in tg.WALLS]
    # within MAX_DEVIATION of the wall normal it grips (exactly on it when that is in reach)
    assert min(tg.axis_deviation(m[:, 0], w) for w in walls) <= tg.MAX_DEVIATION + 1e-9
    # given the arm's base: the wall its wrist reaches most easily, still tool down and near home
    base = Pose(Point([0, 0, 0]), BASE["psm1"].orientation)  # RCM at the origin
    eased = tg.grasp_pose(block, UP, base)
    loads = [tg.wrist_load(base, t.position.to_numpy(), rot(t)) for t in (pose, eased)]
    assert np.allclose(rot(eased)[:, 2], UP) and rot(eased)[0, 0] >= 0 and loads[1] <= loads[0] < np.inf
    # the jaws across a wall: the tip on it, not WALL_R along jaws turned far off its normal (closed on air)
    assert min(tg.axis_deviation(rot(eased)[:, 0], w) for w in walls) <= tg.MAX_DEVIATION + 1e-9
    for yaw in range(0, 120, 10):  # every turn of the block on its peg
        b = Pose(block.position, Rotation(R.from_euler("z", yaw, degrees=True)))
        walls = [rot(b) @ [np.cos(a), np.sin(a), 0] for a in tg.WALLS]
        t = tg.grasp_pose(b, UP, base)
        assert min(tg.axis_deviation(rot(t)[:, 0], w) for w in walls) <= tg.MAX_DEVIATION + 1e-9, yaw


def test_the_lift_clears_the_peg_by_a_block_height():
    assert tg.LIFT == pytest.approx(PEG_H + 2 * BLOCK_HALF_H)
    bottom_on_peg = 0.80 + BLOCK_HALF_H - BLOCK_HALF_H  # the block's bottom while on the peg: the board
    assert bottom_on_peg + tg.LIFT - (0.80 + PEG_H) == pytest.approx(2 * BLOCK_HALF_H)


@pytest.mark.parametrize("giver,receiver", [("psm1", "psm2"), ("psm2", "psm1")])
def test_handover_is_level_jaw_to_jaw_within_the_joint_limits(giver, receiver):
    lo, hi = HANDOVER["position"]["distribution.uniform"]["lower"], HANDOVER["position"]["distribution.uniform"]["upper"]
    for yaw, corner in ((0, lo), (40, hi), (100, [lo[0], hi[1], lo[2]])):
        block = Pose(Point([0.043 * np.sign(RCM[giver][0]), 0.69, 0.80 + BLOCK_HALF_H + tg.LIFT]),
                     Rotation(R.from_euler("z", yaw, degrees=True)))
        held = tg.grasp_pose(block, UP)
        give, receive = tg.handover_poses(block, held, Pose(Point(corner)), BASE[giver], BASE[receiver], UP)
        g, r = rot(give), rot(receive)
        # both level, the jaws pointing at each other along one line, each gripped wall on top
        for tip in (give, receive):
            assert (tip.position.to_numpy() - corner) @ UP > 0.5 * tg.WALL_R
        assert abs(g[:, 2] @ UP) < 1e-9 and abs(r[:, 2] @ UP) < 1e-9 and np.allclose(g[:, 2], -r[:, 2])
        # the receiver's jaws point back at the giver: from the receiver's tip, the giver's lies along -z
        assert (give.position.to_numpy() - receive.position.to_numpy()) @ r[:, 2] < -tg.WALL_R
        # the giver keeps its grip (the block turned with it) at the marker; the receiver grips a wall there
        moved = Pose(Point(corner), Rotation(R.from_matrix(g @ rot(held).T @ rot(block))))
        assert np.allclose((block.inv() * held).position.to_numpy(), (moved.inv() * give).position.to_numpy())
        assert np.allclose(receive.position.to_numpy() - r @ tg.GRIP, corner)
        # measured where the giver holds it out, the receiver's grip is the planned one
        assert np.allclose(tg.receive_pose(moved, -g[:, 2], BASE[receiver], UP).position.to_numpy(), receive.position.to_numpy())
        for tip, arm in ((give, giver), (receive, receiver), (tg.back_off(give, tg.GIVE_PRE), giver), (tg.back_off(give, tg.RETREAT), giver)):
            assert tg.wrist_load(BASE[arm], tip.position.to_numpy(), rot(tip)) <= 1.0
        # from that grip the receiver turns tool down where it is and places on its pegs (RCM at the origin)
        at_rcm = lambda pose: Pose(Point(pose.position.to_numpy() - RCM[receiver]), pose.orientation)
        base = Pose(Point([0.0, 0.0, 0.0]), BASE[receiver].orientation)
        for x, y in itertools.product((0.007, 0.05), (0.685, 0.735)):  # peg_transfer.usd: the pegs, board jitter
            peg = at_rcm(Pose(Point([x * np.sign(RCM[receiver][0]), y, 0.80 + PEG_H / 2])))
            for tip in tg.place_poses(peg, UP, at_rcm(receive), at_rcm(moved), base):
                assert np.isfinite(tg.wrist_load(base, tip.position.to_numpy(), rot(tip))), (x, y)


def test_back_off_moves_up_the_tip_z():
    tip = tg.tip_for(BLOCK.position.to_numpy(), tg.frame([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]))
    assert np.allclose(tg.back_off(tip, 0.025).position.to_numpy() - tip.position.to_numpy(), [0.025, 0, 0])


def test_place_poses_seat_the_block_as_held_1_mm_over_the_board_and_come_down_the_drop():
    peg = np.array([0.043, 0.75, 0.80 + PEG_H / 2]) - RCM["psm2"]  # RCM at the origin
    block = Pose(Point([0.0, 0.72, 0.85] - RCM["psm2"]), Rotation(R.from_euler("z", 25, degrees=True)))
    tip = tg.grasp_pose(block, UP) * Pose(Point([0.0, 0.004, 0.001]))  # slid 4 mm along the wall, 1 mm down
    turn, over, seat = tg.place_poses(Pose(Point(peg)), UP, tip, block)
    m, held = rot(seat), (tip.inv() * block).position.to_numpy()
    assert np.allclose(seat.position.to_numpy() + m @ held, peg + UP * (tg.SEAT_GAP + BLOCK_HALF_H - PEG_H / 2))
    assert np.allclose(over.position.to_numpy() - seat.position.to_numpy(), UP * tg.DROP)
    assert np.allclose(m[:, 2], UP) and np.allclose(rot(over), m) and np.allclose(rot(turn), m)
    assert np.allclose(turn.position.to_numpy(), tip.position.to_numpy() + 0.01 * UP)  # turned 1 cm over where it is
    assert tg.axis_deviation(m[:, 0], tg.bend_free_x(peg, -UP)) <= tg.MAX_DEVIATION + 1e-9
    assert m[:, 0] @ rot(tip)[:, 0] > 0  # the short way round: the jaws not swapped
    assert tg.DROP - PEG_H == pytest.approx(0.010)  # over the peg: the block's bottom 1 cm over its top
    # from the level handover grip, the block drooping 10 deg in the jaws: its axis turned up onto the peg
    on_side = Rotation(R.from_euler("y", 90, degrees=True))  # the block's axis along x
    level = tg.tip_for(block.position.to_numpy(), tg.frame(rot(Pose(Point([0, 0, 0]), on_side)) @ [0.5, 0.866, 0], [1, 0, 0]))
    for droop in (0, 10):
        drooped = Pose(block.position, Rotation(R.from_rotvec(rot(level)[:, 0] * np.radians(droop)) * on_side.rot))
        turn, over, seat = tg.place_poses(Pose(Point(peg)), UP, level, drooped)
        in_tip = level.inv() * drooped
        assert np.allclose(rot(seat) @ rot(in_tip)[:, 2], UP) and np.allclose(rot(turn), rot(seat))
        assert np.allclose(seat.position.to_numpy() + rot(seat) @ in_tip.position.to_numpy(),
                           peg + UP * (tg.SEAT_GAP + BLOCK_HALF_H - PEG_H / 2))


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
                                          "Place it on the peg.", "Return to start.", "Open and grasp it again.",
                                          "Open and take it again.", "Lift it and place it again."]
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
