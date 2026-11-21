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
holds it. The handover keeps the block upright, tool down: the receiver takes another wall
from above while the giver still holds its own.

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
TURN_COST = 0.1  # rad of deviation a rad of handover turn costs: the least deviation, unless a far smaller turn
FLIP = np.diag([-1.0, -1.0, 1.0])  # half a turn about the tip's z (the shaft)
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


def grasp_pose(block: Pose, up) -> Pose:
    """Tool down on the block's wall whose normal lies nearest the bend-free closing axis, the jaws
    turned at most MAX_DEVIATION off that axis (across the wall within 15 deg)."""
    c, b = block.position.to_numpy(), block.orientation.to_numpy()
    free = bend_free_x(c, -np.asarray(up, float))
    normals = [np.cos(a) * b[:, 0] + np.sin(a) * b[:, 1] for a in WALLS]
    x = min(normals, key=lambda n: axis_deviation(n, free))
    return near_home(tip_for(c, frame(toward_free(free, x, up), up)))


def _about(axis, angle) -> np.ndarray:
    return R.from_rotvec(_unit(axis) * angle).as_matrix()


def toward_free(free, wanted, up, limit=MAX_DEVIATION) -> np.ndarray:
    """`wanted` turned about `up` toward the `free` axis until at most `limit` (rad) off its line."""
    free = free if free @ wanted >= 0 else -free
    angle = np.arctan2(np.cross(free, wanted) @ up, free @ wanted)
    return _about(up, float(np.clip(angle, -limit, limit))) @ free


def handover_poses(block: Pose, held: Pose, marker: Pose, giver_rcm, receiver_rcm, up):
    """(giver tip, receiver tip) at the handover, scene frame, both tool down: the giver brings the block
    it holds (`block`, held at the tip pose `held`) to the marker, turned about `up` so that its own
    jaws and one of the block's two free walls lie nearest the two arms' bend-free axes (the larger
    deviation least: tool_yaw needs about twice the deviation and stops at 30 deg), a large turn
    costing TURN_COST (it swung the tool far off its line and shook the block loose); the receiver
    takes that wall from above, like a grasp. The block stays upright:
    turned level, it slid out of the jaws (T12)."""
    up = np.asarray(up, float)
    c = marker.position.to_numpy()
    in_block = block.inv() * held  # the giver's tip in the block frame
    t, tr = in_block.position.to_numpy(), in_block.orientation.to_numpy()
    normals = [np.array([np.cos(a), np.sin(a), 0.0]) for a in WALLS]
    held_wall = int(np.argmax([t @ n for n in normals]))  # the wall the tip sits at
    free_g, free_r = bend_free_x(c - giver_rcm, -up), bend_free_x(c - receiver_rcm, -up)
    best = None
    for yaw in np.radians(np.arange(-180.0, 180.0, 1.0)):  # ponytail: 1 deg grid search, cheap (720 dot products)
        b = _about(up, yaw) @ block.orientation.to_numpy()
        dev_g = axis_deviation(b @ tr[:, 0], free_g)
        for k, n in enumerate(normals):
            key = max(dev_g, axis_deviation(b @ n, free_r)) + TURN_COST * abs(yaw)
            if k != held_wall and (best is None or key < best[0]):
                best = (key, b, b @ n)
    _, b, n = best
    giver = Pose.from_numpy(c + b @ t, R.from_matrix(b @ tr))
    return giver, tip_for(c, frame(toward_free(free_r, n, up), up))


def back_off(tip: Pose, distance: float) -> Pose:
    """`tip` moved `distance` back along its own z (up its shaft, away from what the jaws point at)."""
    return tip * Pose.from_numpy(np.array([0.0, 0.0, distance]), R.identity())


def place_poses(peg: Pose, up, tip: Pose, block: Pose):
    """(turn, over, seat), tool down, from the block as it sits in the jaws at `tip` (it may have slid
    along the wall): turned about up at the tip until its closing axis lies within 10 deg of the
    bend-free one at the peg (tool_yaw needs about twice that), then the block's centre over the peg's
    axis, its bottom 1 cm over the peg top, and down at the seat (SEAT_GAP over the board)."""
    up, p = np.asarray(up, float), peg.position.to_numpy()
    rot = frame(toward_free(bend_free_x(p, -up), tip.orientation.to_numpy()[:, 0], up, np.radians(10.0)), up)
    held = (tip.inv() * block).position.to_numpy()  # the block's centre in the tip frame
    seat = p + up * (SEAT_GAP + BLOCK_HALF_H - PEG_H / 2)
    at = lambda centre: Pose.from_numpy(centre - rot @ held, R.from_matrix(rot))
    return Pose.from_numpy(tip.position.to_numpy(), R.from_matrix(rot)), at(seat + up * DROP), at(seat)


def _in_scene(scene: Pose, pose: Pose) -> Pose:
    return scene.inv() * pose


def _in_base(scene: Pose, pose: Pose) -> Pose:
    return scene * pose


@node()
class GraspPose(BaseNode):
    """The giver's tool-down grasp of the block (grasp_pose)."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GraspPose", alias, dynamic_map, static_args, output_map)

    def run(self, block: Pose, scene: Pose) -> ExecutionResult:
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": grasp_pose(block, up_of(scene))})


@node()
class HandoverPoses(BaseNode):
    """The handover (handover_poses), from the marker, the held block and the giver's tip as the giver
    measures them and the scene as each arm does: `give` and `give_back` (20 mm up) in the giver's
    frame, `receive`, `receive_pre` (APPROACH over it) and `receive_rise` in the receiver's.
    `receive_rise`: the receiver's tip (`receiver_tip`, where it waits, under the marker) straight up
    to receive_pre's height, turned like it, so it comes over the block from the side, not through it."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("HandoverPoses", alias, dynamic_map, static_args, output_map)

    def run(self, marker: Pose, block: Pose, held: Pose, giver_scene: Pose, receiver_scene: Pose,
            receiver_tip: Pose) -> ExecutionResult:
        rcm_g, rcm_r = (s.inv().position.to_numpy() for s in (giver_scene, receiver_scene))
        give, receive = handover_poses(_in_scene(giver_scene, block), _in_scene(giver_scene, held),
                                       _in_scene(giver_scene, marker), rcm_g, rcm_r, [0.0, 0.0, 1.0])  # scene z is up
        give, receive = _in_base(giver_scene, give), near_home(_in_base(receiver_scene, receive))
        pre, up, start = back_off(receive, APPROACH), up_of(receiver_scene), receiver_tip.position.to_numpy()
        rise = Pose.from_numpy(start + up * ((pre.position.to_numpy() - start) @ up), R.from_matrix(pre.orientation.to_numpy()))
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "give": give, "give_back": back_off(give, 0.02),
            "receive": receive, "receive_pre": pre, "receive_rise": rise})


@node()
class PlacePoses(BaseNode):
    """The receiver's place over `peg` (place_poses) of the block it holds at `tip`: `turn_down` (turned
    where it is), `over` and `seat`."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PlacePoses", alias, dynamic_map, static_args, output_map)

    def run(self, peg: Pose, scene: Pose, tip: Pose, block: Pose) -> ExecutionResult:
        turn, over, seat = place_poses(peg, up_of(scene), tip, block)
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"turn_down": turn, "over": over, "seat": seat})


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
