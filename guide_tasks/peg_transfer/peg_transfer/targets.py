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
holds it. The ring is a prism, so it may land upside down: the receiver, which takes it
jaw-to-jaw from the other face, seats it flipped.

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


def grasp_pose(block: Pose, up) -> Pose:
    """Tool down on the block's wall whose normal lies nearest the bend-free closing axis, the jaws
    turned at most MAX_DEVIATION off that axis (across the wall within 15 deg)."""
    c, b = block.position.to_numpy(), block.orientation.to_numpy()
    free = bend_free_x(c, -np.asarray(up, float))
    normals = [np.cos(a) * b[:, 0] + np.sin(a) * b[:, 1] for a in WALLS]
    x = min(normals, key=lambda n: axis_deviation(n, free))
    return tip_for(c, frame(toward_free(free, x, up), up))


def _about(axis, angle) -> np.ndarray:
    return R.from_rotvec(_unit(axis) * angle).as_matrix()


def toward_free(free, wanted, up, limit=MAX_DEVIATION) -> np.ndarray:
    """`wanted` turned about `up` toward the `free` axis until at most `limit` (rad) off its line."""
    free = free if free @ wanted >= 0 else -free
    angle = np.arctan2(np.cross(free, wanted) @ up, free @ wanted)
    return _about(up, float(np.clip(angle, -limit, limit))) @ free


def handover_poses(marker: Pose, giver_rcm, receiver_rcm, roll_gain: float = 1.0):
    """(giver tip, receiver tip) at the handover, scene frame. The block's centre is at the marker;
    the giver's jaws point level at the receiver, the receiver's back at the giver, both across the
    giver's wall, from the block's two faces. Their closing axis bisects the two arms' bend-free
    axes; the marker's roll about the pointing line (times roll_gain) turns the pair."""
    c = marker.position.to_numpy()
    between = np.asarray(receiver_rcm, float) - np.asarray(giver_rcm, float)
    p = _unit([between[0], between[1], 0.0])
    free_g, free_r = bend_free_x(c - giver_rcm, p), bend_free_x(c - receiver_rcm, -p)
    x = _unit(free_g + (free_r if free_r @ free_g >= 0 else -free_r))
    roll = roll_gain * float(marker.orientation.to_scipy().as_rotvec() @ p)
    giver = frame(_about(p, roll) @ x, -p)
    receiver = giver @ R.from_euler("x", np.pi).as_matrix()
    return tip_for(c, giver), tip_for(c, receiver)


def level_turn(at, rcm, toward) -> np.ndarray:
    """The giver's wrist turned level at `at` (scene frame), jaws pointing horizontally `toward`."""
    p = _unit([toward[0], toward[1], 0.0])
    return frame(bend_free_x(np.asarray(at, float) - rcm, p), -p)


def back_off(tip: Pose, distance: float) -> Pose:
    """`tip` moved `distance` back along its own z (up its shaft, away from what the jaws point at)."""
    return tip * Pose.from_numpy(np.array([0.0, 0.0, distance]), R.identity())


def tool_down(at, up, near: Pose) -> np.ndarray:
    """Tool down at `at` with the bend-free closing axis (a held block on a peg may face any way),
    its sign the one nearer `near`'s, so the wrist turns the least."""
    x = bend_free_x(at, -np.asarray(up, float))
    return frame(x if x @ near.orientation.to_numpy()[:, 0] >= 0 else -x, up)


def place_poses(peg: Pose, up, near: Pose):
    """(over, seat), tool down: the block's centre over the peg's axis with its bottom 1 cm over the
    peg top, and down at the seat (SEAT_GAP over the board)."""
    up = np.asarray(up, float)
    p = peg.position.to_numpy()
    rot = tool_down(p, up, near)
    seat = p + up * (SEAT_GAP + BLOCK_HALF_H - PEG_H / 2)
    return tip_for(seat + up * DROP, rot), tip_for(seat, rot)


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
    """The handover, from the marker as the giver measures it and the scene as each arm does: `turn`
    (the giver's wrist turned level where it is, `at`), `give` and `give_back` (20 mm back) in the
    giver's frame, `receive` and `receive_pre` (10 mm back) in the receiver's."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("HandoverPoses", alias, dynamic_map, static_args, output_map)

    def run(self, marker: Pose, giver_scene: Pose, receiver_scene: Pose, at: Pose,
            roll_gain: float = 1.0) -> ExecutionResult:
        rcm_g, rcm_r = (s.inv().position.to_numpy() for s in (giver_scene, receiver_scene))
        give, receive = handover_poses(_in_scene(giver_scene, marker), rcm_g, rcm_r, roll_gain)
        here = _in_scene(giver_scene, at).position.to_numpy()
        turn = _in_base(giver_scene, Pose.from_numpy(here, R.from_matrix(level_turn(here, rcm_g, rcm_r - rcm_g))))
        give, receive = _in_base(giver_scene, give), _in_base(receiver_scene, receive)
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={
            "turn": turn, "give": give, "give_back": back_off(give, 0.02),
            "receive": receive, "receive_pre": back_off(receive, 0.01)})


@node()
class PlacePoses(BaseNode):
    """The receiver's place over `peg` (place_poses): `turn_down` (tool down where it holds the block,
    at `at`), `over` and `seat`."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PlacePoses", alias, dynamic_map, static_args, output_map)

    def run(self, peg: Pose, scene: Pose, at: Pose) -> ExecutionResult:
        over, seat = place_poses(peg, up_of(scene), at)
        turn = Pose.from_numpy(at.position.to_numpy(), R.from_matrix(tool_down(at.position.to_numpy(), up_of(scene), at)))
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
