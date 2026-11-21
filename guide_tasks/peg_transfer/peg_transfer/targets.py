"""peg_transfer's target maths: every tool pose the PSMs move to, from the scene as measured.

The PSM tool tip frame (i4h, guide_motion robots/psm): z points back up the shaft, the jaws
point along -z and close along x (they open about y). At home it equals the base frame,
whose origin is the remote centre of motion (RCM). "Tool down" is z_tip = world up.

The wrist bends by tool_pitch (about x_tip, +-80 deg) and tool_yaw (about y_tip, only
+-30 deg). Bending the jaws away from the shaft's line by pitch alone needs x_tip
perpendicular to both the shaft (RCM to tip) and the jaws' pointing direction: bend_free_x.
Every orientation here stays near that axis, so tool_yaw is left a small share.

A block is held at one wall of its triangular ring, the jaws across it (one in the hole,
one outside): the tip sits at GRIP from the block centre, in the tip frame, whichever arm
holds it. The giver picks it tool down, lifts it, turns its jaws level, pointing at the
receiver, and brings it to the handover marker held out sideways (its axis along the jaws);
the receiver, level too, faces it and takes a wall at the block's other end face, pointing
back at the giver (handover_poses: the wrist bends by tool_pitch and tool_roll, checked by IK
on robots/psm). The receiver turns it tool down again over the target peg.

Poses in and out are in the acting robot's base frame (GetPrimPose's frame, S2b Ruling 4);
`scene` is the /PegTransfer prim's pose in that frame (identity in the scene, so its z is up).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Pose
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import node
from peg_transfer.scene import BLOCK_HALF_H, PEG_H, seated

# block.obj (SurRoL): corners at 0, +-120 deg (r 9.7 mm), inner hexagon (inradius 4.16 mm), so the
# walls are thinnest midway between corners, outer face 4.85 mm from the centre.
WALLS = np.radians([60.0, 180.0, 300.0])  # wall normals, block frame
WALL_R = 0.0045  # m, block centre to the middle of a wall
GRIP_DEPTH = 0.004  # m, jaw tip under the face it enters
GRIP = np.array([WALL_R, 0.0, BLOCK_HALF_H - GRIP_DEPTH])  # tip from the block centre, tip frame
APPROACH = 0.015  # m over the grasp before descending to it
LIFT = PEG_H + 2 * BLOCK_HALF_H  # m straight up: the block's bottom a block height over the peg top
SEAT_GAP = 0.001  # m left under the block when it is let go: the jaws never press it into the board
DROP = PEG_H + 0.010  # m from over the peg (the block's bottom 1 cm over its top) down to the seat
# Largest turn of the closing axis off the bend-free one: tool_yaw grows with it and with the bend
# (IK on robots/psm: 15 deg needs <= 0.35 rad of its 0.52 at the lift height).
MAX_DEVIATION = np.radians(15.0)
FLIP = np.diag([-1.0, -1.0, 1.0])  # half a turn about the tip's z (the shaft)
RETREAT = 0.025  # m the giver backs out along its jaws after letting go: past the block's far wall, + 5 mm
GIVE_PRE = 0.02  # m short of the handover, back along the level jaws: where the turn to level ends
HANDOVER_YAWS = np.radians([0.0, 15.0, -15.0, 30.0, -30.0])  # the jaws' turns off the line to the receiver tried
# wrist-load share per rad of a level grip's closing axis off up, the axis being the gripped wall's
# outward normal: with that wall on top the block hangs from the jaws; under it, it balanced on them and
# toppled 90 deg in the grip (and gravity twists a block about a horizontal closing axis, the weak one)
VERTICAL_COST = 1.0


def off_up(x, up) -> float:
    """Angle (rad, 0..pi) of the direction x from up."""
    return float(np.arccos(np.clip(_unit(x) @ _unit(up), -1.0, 1.0)))
HOLD_TOL = 0.003  # m: a held block's centre lies within this of where the tip's grip puts it


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def frame(x, z) -> np.ndarray:
    """Rotation matrix with columns x (made perpendicular to z), z cross x, z."""
    z = _unit(z)
    x = _unit(np.asarray(x, float) - (np.asarray(x, float) @ z) * z)
    return np.column_stack([x, np.cross(z, x), z])


def bend_free_x(tip, pointing) -> np.ndarray:
    """The closing axis whose jaws point along `pointing` at `tip` with tool_yaw 0 (RCM at the origin)."""
    return _unit(np.cross(tip, pointing))


def axis_deviation(a, b) -> float:
    """Angle (rad) between the lines along a and b: 0..pi/2, sign-free (the roll joint turns 180 deg)."""
    return float(np.arccos(min(1.0, abs(_unit(a) @ _unit(b)))))


def up_of(scene: Pose) -> np.ndarray:
    return scene.orientation.to_numpy()[:, 2]


def tip_for(centre, rot) -> Pose:
    """The tip pose holding a block centred at `centre`, the tip turned `rot`."""
    return Pose.from_numpy(np.asarray(centre, float) + rot @ GRIP, R.from_matrix(rot))


def near_home(tip: Pose) -> Pose:
    """`tip`, or the same grip turned half a turn about the shaft (FLIP: the jaws swap sides), whichever
    keeps the closing axis on base x's side: tool_roll then stays within about 90 deg of home. (The IK
    took the far branch, roll 232 deg, and the handover's turn ran into the 260 deg limit.)"""
    rot = tip.orientation.to_numpy()
    return tip if rot[0, 0] >= 0 else Pose.from_numpy(tip.position.to_numpy(), R.from_matrix(rot @ FLIP))


def grip_centres(tip: Pose) -> np.ndarray:
    """The two block centres a grip at `tip` may hold: GRIP off it, as is or FLIPped."""
    p, rot = tip.position.to_numpy(), tip.orientation.to_numpy()
    return np.array([p - rot @ GRIP, p - rot @ FLIP @ GRIP])


def holds(tip: Pose, block: Pose) -> bool:
    """Whether the jaws hold the block: its centre within HOLD_TOL of a centre the grip at `tip` holds.
    Jaws closed on air, beside the block or on its top leave it farther (a bbox clash does not tell)."""
    return bool(np.linalg.norm(grip_centres(tip) - block.position.to_numpy(), axis=1).min() <= HOLD_TOL)


def grasp_pose(block: Pose, up, base: Pose | None = None) -> Pose:
    """Tool down on one of the block's walls, the jaws turned at most MAX_DEVIATION off the bend-free
    closing axis (across the wall within 15 deg): without `base`, the wall whose normal lies nearest
    that axis; given the arm's `base`, the wall that loads its wrist least (wrist_load: a wrist driven
    to its limits stalled short of the grasp). near_home either way."""
    c, b = block.position.to_numpy(), block.orientation.to_numpy()
    free = bend_free_x(c, -np.asarray(up, float))
    normals = [np.cos(a) * b[:, 0] + np.sin(a) * b[:, 1] for a in WALLS]
    if base is None:
        x = min(normals, key=lambda n: axis_deviation(n, free))
        return near_home(tip_for(c, frame(toward_free(free, x, up), up)))
    # only walls the jaws cross within MAX_DEVIATION of their normal: farther off, the tip (WALL_R along
    # the turned jaws) leaves the wall and they close on air
    tips = [near_home(tip_for(c, frame(toward_free(free, n, up), up))) for n in normals
            if axis_deviation(n, free) <= 2 * MAX_DEVIATION + 1e-9]
    return min(tips, key=lambda t: wrist_load(base, t.position.to_numpy(), t.orientation.to_numpy()))


def _about(axis, angle) -> np.ndarray:
    return R.from_rotvec(_unit(axis) * angle).as_matrix()


def toward_free(free, wanted, up, limit=MAX_DEVIATION) -> np.ndarray:
    """`wanted` turned about `up` toward the `free` axis until at most `limit` (rad) off its line."""
    free = free if free @ wanted >= 0 else -free
    angle = np.arctan2(np.cross(free, wanted) @ up, free @ wanted)
    return _about(up, float(np.clip(angle, -limit, limit))) @ free


_MODEL = []


def wrist_ik(base: Pose, position, rot) -> list:
    """The joints (robots/psm, every joint within its limits) of the PSM whose base frame sits at `base`
    (scene frame) that put its tip at `position` turned `rot` (scene frame) to 0.1 mm and 0.5 deg: one
    per IK seed that got there, as (wrist load, q) -- the load the larger of |tool_pitch| and |tool_yaw|
    as shares of their limits."""
    from scipy.optimize import least_squares
    if not _MODEL:
        from guide_motion.model import Model
        _MODEL.append(Model.load("psm"))
    m = _MODEL[0]
    goal = base.inv() * Pose.from_numpy(np.asarray(position, float), R.from_matrix(rot))
    p, g = goal.position.to_numpy(), goal.orientation.to_numpy()

    def res(q):
        tp, tr = m.ee_pose(q)
        return np.r_[tp - p, 0.02 * R.from_matrix(tr @ g.T).as_rotvec()]
    found = []
    for roll in (-2.5, 0.0, 2.5):  # ponytail: 6 seeds of a local solver; a branch they all miss reads as unreachable
        for pitch in (-0.8, 0.8):
            s = least_squares(res, np.array([0.0, 0.0, 0.15, roll, pitch, 0.0]), bounds=(m.lower, m.upper))
            if np.linalg.norm(s.fun[:3]) < 1e-4 and np.linalg.norm(s.fun[3:]) < 0.02 * np.radians(0.5):
                found.append((max(abs(s.x[4]) / m.upper[4], abs(s.x[5]) / m.upper[5]), s.x))
    return found


def wrist_load(base: Pose, position, rot) -> float:
    """How hard the PSM's wrist works to hold its tip there (wrist_ik's least load); inf where no IK
    reaches it."""
    return min((load for load, _ in wrist_ik(base, position, rot)), default=np.inf)


def joints_for(tip: Pose, home: dict) -> dict:
    """An arm-joint goal ("<joint>.pos", robots/psm's arm joints) putting the tip at `tip` (base frame):
    of wrist_ik's solutions, the one nearest the arm's `home` joints -- the roll branch it keeps."""
    found = wrist_ik(Pose(), tip.position.to_numpy(), tip.orientation.to_numpy())
    if not found:
        raise ValueError("no joints reach the turn within the PSM's limits")
    names = [f"{j}.pos" for j in _MODEL[0].arm_joints]
    q_home = np.array([home[n] for n in names])
    q = min((q for _, q in found), key=lambda q: np.abs(q - q_home).sum())
    return {n: float(v) for n, v in zip(names, q)}


def handover_poses(block: Pose, held: Pose, marker: Pose, giver_base: Pose, receiver_base: Pose, up):
    """(giver tip, receiver tip) at the handover, scene frame. The giver holds `block` at the tip pose
    `held`; it brings the block's centre to the marker with its jaws level, pointing at the receiver (the
    line between the RCMs, else turned by the first of HANDOVER_YAWS that works), its closing axis turned
    about the jaws so that its wrist and the receiver's work least (wrist_load) with both closing axes
    near up (VERTICAL_COST: the gripped walls on top), the marker's z (randomize.yaml's roll) breaking
    ties. The receiver faces it, level, pointing back, and takes the wall that suits its wrist best at
    the block's other end face, like a grasp. The block's axis lies along the jaws."""
    up = _unit(up)
    c = marker.position.to_numpy()
    pref = marker.orientation.to_numpy()[:, 2]
    in_tip = held.inv() * block  # the block in the giver's tip frame
    o, rb = in_tip.position.to_numpy(), in_tip.orientation.to_numpy()
    line = receiver_base.position.to_numpy() - giver_base.position.to_numpy()
    h0 = _unit(line - (line @ up) * up)
    normals = [np.array([np.cos(a), np.sin(a), 0.0]) for a in WALLS]
    best = None
    for yaw in HANDOVER_YAWS:
        h = _about(up, yaw) @ h0
        side = np.cross(h, up)
        receiver = {}  # the receiver's wrist load per closing-axis angle about h (15 deg grid)
        for k in range(24):
            n = np.cos(k * np.pi / 12) * up + np.sin(k * np.pi / 12) * side
            receiver[k] = wrist_load(receiver_base, c + frame(n, h) @ GRIP, frame(n, h))
        for phi in np.radians(np.arange(0.0, 360.0, 30.0)):
            rot = frame(np.cos(phi) * up + np.sin(phi) * side, -h)  # z back up the jaws: they point along h
            # reached at the handover and backed off along the jaws (HandoverPoses' give_pre, give_back)
            load = max(wrist_load(giver_base, c - rot @ o + d * rot[:, 2], rot) for d in (0.0, GIVE_PRE, RETREAT))
            if not np.isfinite(load):
                continue
            b = rot @ rb  # the block, turned with the jaws
            wall = -(rot @ o)  # block centre to the gripped wall (not rot x: near_home may have flipped the tip)
            wall = wall - (wall @ h) * h
            face = b[:, 2] if b[:, 2] @ h > 0 else -b[:, 2]  # its end face toward the receiver
            for n in (b @ w for w in normals):
                angle = np.arctan2(n @ side, n @ up)
                k = int(round(angle / (np.pi / 12))) % 24
                key = (max(load, receiver[k]) + 0.3 * abs(yaw) + 0.05 * axis_deviation(rot[:, 0], pref)
                       + VERTICAL_COST * (off_up(wall, up) + off_up(n, up)))
                if np.isfinite(key) and (best is None or key < best[0]):
                    best = (key, Pose.from_numpy(c - rot @ o, R.from_matrix(rot)), tip_for(c, frame(n, face)))
        if best is not None:
            break  # pointing straight at the receiver, or the least turn off it that both wrists reach
    if best is None:
        raise ValueError("no level handover within the PSMs' joint limits")
    return best[1], best[2]


def receive_pose(block: Pose, toward, base: Pose, up) -> Pose:
    """The receiver's grip of `block` as it sits in the giver's jaws (measured, not as planned): its jaws
    in along the block's axis onto the end face that `toward` (the giver's jaws) points at (level, the
    axis's horizontal part, where that is out of reach), on the wall nearest the top
    for the least wrist load (receiver base at `base`)."""
    up = _unit(up)
    c, b = block.position.to_numpy(), block.orientation.to_numpy()
    face = b[:, 2] if b[:, 2] @ toward > 0 else -b[:, 2]  # the end face toward the receiver
    level = _unit(face - (face @ up) * up)
    tips, cost = [], []
    # along the block's axis; where a block drooping in the giver's jaws puts that out of reach, level
    for f, extra in ((face, 0.0), (level, 0.5)):
        for a in WALLS:
            t = tip_for(c, frame(b @ [np.cos(a), np.sin(a), 0.0], f))
            tips.append(t)
            cost.append(wrist_load(base, t.position.to_numpy(), t.orientation.to_numpy()) + extra
                        + VERTICAL_COST * off_up(t.orientation.to_numpy()[:, 0], up))
    if not np.isfinite(min(cost)):
        raise ValueError(f"no receiving grip within the PSM's joint limits (the block's axis "
                         f"{np.degrees(axis_deviation(face, toward)):.0f} deg off the giver's jaws)")
    return tips[int(np.argmin(cost))]


def back_off(tip: Pose, distance: float) -> Pose:
    """`tip` moved `distance` back along its own z (up its shaft, away from what the jaws point at)."""
    return tip * Pose.from_numpy(np.array([0.0, 0.0, distance]), R.identity())


def place_poses(peg: Pose, up, tip: Pose, block: Pose, base: Pose | None = None):
    """(turn, over, seat), tool down, from the block as it sits in the jaws at `tip` (it may have slid
    along the wall, or turned in them): the jaws turned down where they are, 1 cm up (the least turn of
    the block's axis to up, so a block drooping in the jaws comes down straight onto the peg, then a
    turn about up bringing the closing axis within MAX_DEVIATION of the bend-free one at the peg),
    then the block's centre over the peg's axis, its bottom 1 cm over the peg top, and down at the seat
    (SEAT_GAP over the board). The block's turn about its axis does not matter to the peg: given the
    arm's `base`, the jaws turn half a turn more about up (FLIP) where that eases the wrist, and the
    turn, at the handover, far from the peg, keeps the least turn about up the wrist reaches there."""
    up, p = _unit(up), peg.position.to_numpy()
    t, axis = tip.orientation.to_numpy(), block.orientation.to_numpy()[:, 2]
    axis = axis if axis @ t[:, 2] >= 0 else -axis  # the block's axis on the shaft's side
    m = R.align_vectors([up], [axis])[0].as_matrix() @ t
    x = _unit(m[:, 0] - (m[:, 0] @ up) * up)
    want = toward_free(bend_free_x(p, -up), x, up)
    rot = _about(up, np.arctan2(np.cross(x, want) @ up, x @ want)) @ m
    held = (tip.inv() * block).position.to_numpy()  # the block's centre in the tip frame
    seat = p + up * (SEAT_GAP + BLOCK_HALF_H - PEG_H / 2)
    turn = tip.position.to_numpy() + 0.01 * up

    def turned(rot):  # the turn's own: the least turn about up from `rot` that the wrist reaches there
        for k in sorted(range(-12, 13), key=abs):
            r = _about(up, k * np.pi / 12) @ rot
            if base is None or np.isfinite(wrist_load(base, turn, r)):
                return r
        return rot

    def poses(rot):
        at = lambda centre: Pose.from_numpy(centre - rot @ held, R.from_matrix(rot))
        return Pose.from_numpy(turn, R.from_matrix(turned(rot))), at(seat + up * DROP), at(seat)
    if base is None:
        return poses(rot)
    load = lambda ps: max(wrist_load(base, q.position.to_numpy(), q.orientation.to_numpy()) for q in ps)
    return min((poses(rot), poses(rot @ FLIP)), key=load)


def _in_scene(scene: Pose, pose: Pose) -> Pose:
    return scene.inv() * pose


def _in_base(scene: Pose, pose: Pose) -> Pose:
    return scene * pose


@node()
class GraspPose(BaseNode):
    """The giver's tool-down grasp of the block (grasp_pose), in its base frame."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GraspPose", alias, dynamic_map, static_args, output_map)

    def run(self, block: Pose, scene: Pose) -> ExecutionResult:
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": grasp_pose(block, up_of(scene), base=Pose())})


@node()
class HandoverPoses(BaseNode):
    """The handover (handover_poses), from the marker, the held block and the giver's tip as the giver
    measures them and the scene as each arm does, and the giver's `home` joints. The giver's frame:
    `give_pre` (2 cm short of `give`, back along its level jaws) and `give_pre_joints` (joints_for it:
    the turn from tool down to level is a joint move, the planner's IK following the straight turn ran
    the wrist into its limits), `give`,
    `give_back` (RETREAT back out along its jaws) and `give_rise` (2 cm over that). The receiver's grip
    comes from ReceivePoses, once the block is there."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("HandoverPoses", alias, dynamic_map, static_args, output_map)

    def run(self, marker: Pose, block: Pose, held: Pose, giver_scene: Pose, receiver_scene: Pose, home: dict) -> ExecutionResult:
        give, _ = handover_poses(_in_scene(giver_scene, block), _in_scene(giver_scene, held),
                                       _in_scene(giver_scene, marker), giver_scene.inv(), receiver_scene.inv(),
                                       [0.0, 0.0, 1.0])  # scene z is up
        give = _in_base(giver_scene, give)
        back, pre = back_off(give, RETREAT), back_off(give, GIVE_PRE)
        up = up_of(giver_scene)
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "give_pre_joints": joints_for(pre, home), "give_pre": pre, "give": give, "give_back": back,
            "give_rise": Pose.from_numpy(back.position.to_numpy() + 0.02 * up, R.from_matrix(back.orientation.to_numpy()))})


@node()
class ReceivePoses(BaseNode):
    """The receiver's grip (receive_pose) of the block held out at the marker, from the block and the
    giver's tip as the receiver measures them (its base frame): `receive` and `receive_pre` (APPROACH
    back along its jaws)."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ReceivePoses", alias, dynamic_map, static_args, output_map)

    def run(self, block: Pose, giver_tip: Pose, scene: Pose) -> ExecutionResult:
        if not holds(giver_tip, block):  # dropped, or left on its peg: no grip to plan from
            raise ValueError(f"the block is not in the giver's jaws "
                             f"({np.linalg.norm(block.position.to_numpy() - giver_tip.position.to_numpy()) * 1e3:.0f} mm from its tip)")
        receive = receive_pose(block, -giver_tip.orientation.to_numpy()[:, 2], Pose(), up_of(scene))
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"receive": receive, "receive_pre": back_off(receive, APPROACH)})


@node()
class PlacePoses(BaseNode):
    """The receiver's place over `peg` (place_poses) of the block it holds at `tip`: `turn_down` (turned
    where it is) and `turn_down_joints` (joints_for it, nearest the arm's `home` joints: the turn from
    level to tool down is a joint move, like the giver's to level), `over` and `seat`."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PlacePoses", alias, dynamic_map, static_args, output_map)

    def run(self, peg: Pose, scene: Pose, tip: Pose, block: Pose, home: dict) -> ExecutionResult:
        turn, over, seat = place_poses(peg, up_of(scene), tip, block, Pose())  # all in its base frame
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "turn_down": turn, "turn_down_joints": joints_for(turn, home), "over": over, "seat": seat})


@node()
class IsSeated(BaseNode):
    """Whether the block sits down on the peg (scene.seated, the success check's), still held or not."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("IsSeated", alias, dynamic_map, static_args, output_map)

    def run(self, block: Pose, peg: Pose, scene: Pose) -> ExecutionResult:
        ok = seated(_in_scene(scene, block), _in_scene(scene, peg))
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"seated": ok})


@node()
class IsHeld(BaseNode):
    """Whether the jaws at `tip` hold `block` (holds), both in the same frame."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("IsHeld", alias, dynamic_map, static_args, output_map)

    def run(self, tip: Pose, block: Pose) -> ExecutionResult:
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"held": holds(tip, block)})
