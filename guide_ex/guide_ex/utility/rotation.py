import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform.rotation import Rotation as R

from guide_core.types.geometry import Pose, Rotation, Transform
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import node

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _decompose(pose: Pose | Transform):
    """
    Split a Pose/Transform into its scipy rotation plus a rebuild closure that
    restores the *same* container type and preserves the original position.
    """
    if isinstance(pose, Pose):
        rot = pose.orientation.to_scipy()

        def rebuild(new_rot: R) -> Pose:
            return Pose(position=pose.position, orientation=Rotation.from_scipy(new_rot))

        return rot, rebuild

    if isinstance(pose, Transform):
        rot = pose.rotation.to_scipy()

        def rebuild(new_rot: R) -> Transform:
            return Transform(translation=pose.translation, rotation=Rotation.from_scipy(new_rot))

        return rot, rebuild

    raise TypeError(f"Expected Pose or Transform, got {type(pose).__name__}")


def _base_yaw(rot: R, heading_axis: str) -> float:
    """
    Yaw of ``heading_axis`` around the base Z axis: the chosen body axis is
    projected onto the base XY plane and its heading is read with atan2.

    This ignores any tilt out of the XY plane, which is exactly what "assume the
    transform is a single rotation about Z" means. If the chosen axis points
    (near) straight up/down the projection is degenerate and yaw collapses to 0.
    ``"auto"`` avoids that: it takes the body axis lying flattest in the XY plane,
    so a cube that fell over onto another face still yields a usable heading.
    """
    matrix = rot.as_matrix()
    if heading_axis == "auto":
        axis = matrix[:, int(np.argmin(np.abs(matrix[2])))]
    else:
        axis = matrix[:, _AXIS_INDEX[heading_axis]]
    return float(np.arctan2(axis[1], axis[0]))


@node()
class ProjectRotationToBaseZ(BaseNode):
    """
    Project an object's full 3D orientation onto the pure Z-rotation of the base
    frame, assuming the object is meant to be reached from straight above.

    The position is left untouched; only the orientation is replaced by
    ``Rz(yaw)`` where ``yaw`` is the heading of ``heading_axis`` in the base XY
    plane. This strips any pitch/roll so a top-down grasp never ends up tilted
    sideways.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ProjectRotationToBaseZ", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose | Transform, heading_axis: str = "x") -> ExecutionResult:
        """
        Args:
            pose: The object's pose/transform expressed in the base frame.
            heading_axis: Which body axis ("x", "y" or "z") defines the yaw
                heading that is preserved. Defaults to the body X axis; "auto"
                picks the flattest one (for objects that may have toppled).
        Returns:
            ExecutionResult with ``pose`` (same container, Z-only orientation)
            and the extracted ``yaw`` in radians.
        """
        if heading_axis not in (*_AXIS_INDEX, "auto"):
            raise ValueError(
                f"heading_axis must be one of {[*_AXIS_INDEX, 'auto']}, got {heading_axis!r}"
            )

        rot, rebuild = _decompose(pose)
        yaw = _base_yaw(rot, heading_axis)
        z_only = R.from_euler("z", yaw)

        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={"pose": rebuild(z_only), "yaw": yaw},
        )


@node()
class ReduceRotationToSymmetry(BaseNode):
    """
    Collapse a Z rotation to the smallest-magnitude equivalent one under an
    assumed n-fold polygonal symmetry.

    An object with ``symmetry`` identical faces looks the same under any multiple
    of ``2*pi / symmetry`` about Z, so the wrist never needs to turn more than
    half a period. The yaw is wrapped into ``[-pi/n, pi/n]``, minimising the wrist
    travel while landing on a symmetry-equivalent orientation. ``symmetry=4`` is a
    cube/square, ``symmetry=1`` disables reduction (plain angle normalisation).
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ReduceRotationToSymmetry", alias, dynamic_map, static_args, output_map)

    def run(
        self, pose: Pose | Transform, symmetry: int = 1, heading_axis: str = "x"
    ) -> ExecutionResult:
        """
        Args:
            pose: A (Z-projected) pose/transform in the base frame.
            symmetry: Order of the rotational symmetry about Z (identical faces).
            heading_axis: Body axis used to read the current yaw.
        Returns:
            ExecutionResult with ``pose`` carrying the minimal Z rotation and the
            reduced ``yaw`` in radians.
        """
        if heading_axis not in _AXIS_INDEX:
            raise ValueError(f"heading_axis must be one of {list(_AXIS_INDEX)}, got {heading_axis!r}")
        if symmetry < 1:
            raise ValueError(f"symmetry must be a positive integer, got {symmetry}")

        rot, rebuild = _decompose(pose)
        yaw = _base_yaw(rot, heading_axis)

        period = 2.0 * np.pi / symmetry
        reduced = float(yaw - period * np.round(yaw / period))
        z_only = R.from_euler("z", reduced)

        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={"pose": rebuild(z_only), "yaw": reduced},
        )


def _rect(centre, yaw: float, half_x: float, half_y: float):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(centre, float), np.array([[c, s], [-s, c]]), (half_x, half_y)


def _separation(a, b) -> float:
    """Signed gap between two rectangles (separating-axis test): the largest gap along any
    edge normal, negative when they overlap (minus the shallowest overlap)."""
    (ca, axa, ha), (cb, axb, hb) = a, b
    d = cb - ca
    gaps = [
        abs(u @ d) - sum(h * abs(u @ v) for v, h in zip(axa, ha)) - sum(h * abs(u @ v) for v, h in zip(axb, hb))
        for u in (*axa, *axb)
    ]
    return max(gaps)


@node()
class ChooseClearanceYaw(BaseNode):
    """
    Pick, for a top-down grasp of a square-footprint block, the one of its two distinct finger
    axes (the pose's yaw psi, or psi + 90 deg) whose finger pads stay farther from the other
    blocks.

    The pad centres sit at +-(finger_gap + pad_depth) / 2 along the finger axis (the hand's y
    axis), each ``pad_depth`` deep and ``pad_width`` wide, centred on the grasp position. A pad
    over a neighbouring block lands on it instead of beside the target. The turn to psi + 90 is
    kept only if the arm can make it: joint 7 follows ``q7 = q7_home + q1 - yaw`` for a
    down-pointing hand, q1 being the heading of the grasp from the base; |q7| must stay
    within ``q7_limit``. block_bin's pick recovery then turns the hand another 90 deg in base
    yaw (q7 - pi/2), so that must be reachable too. Otherwise (and on a tie) the pose is
    returned unchanged.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ChooseClearanceYaw", alias, dynamic_map, static_args, output_map)

    def run(
        self,
        pose: Pose,
        others: list,
        block_size: float = 0.0515,
        finger_gap: float = 0.08,
        pad_depth: float = 0.018,
        pad_width: float = 0.02,
        q7_home: float = np.pi / 4,
        q7_limit: float = 2.8,
        same_block: float = 0.01,
        enough: float = 0.02,
    ) -> ExecutionResult:
        """
        Args:
            pose: The grasp pose (a top-down pose in the base frame, e.g. Rz(psi) Rx(pi)).
            others: Poses of the blocks to keep clear of, in the same frame; any within
                ``same_block`` metres of the grasp position is the target itself and ignored.
            block_size: Edge of the (square) blocks, metres.
            finger_gap: Opening between the pads, metres.
            pad_depth: Pad thickness along the finger axis, metres.
            pad_width: Pad width across the finger axis, metres.
            q7_home: Joint 7 when the hand's yaw is 0 with q1 = 0 (FR3 home: pi/4).
            q7_limit: Largest |q7| allowed, radians (FR3: 2.8973, less a margin).
            same_block: Distance under which a pose in ``others`` is the target.
            enough: Clearance (metres) past which more room is no better, so that distant
                blocks never turn the hand.
        Returns:
            ExecutionResult with ``pose`` (same position, yaw psi or psi + 90 deg) and the
            ``clearance`` (metres, signed) of the pads at the chosen yaw.
        """
        rot, rebuild = _decompose(pose)
        yaw = _base_yaw(rot, "x")
        xy = np.asarray(pose.position.to_numpy(), float)[:2]
        blocks = []
        for other in others:
            centre = np.asarray(other.position.to_numpy(), float)[:2]
            if np.linalg.norm(centre - xy) > same_block:
                blocks.append(_rect(centre, _base_yaw(other.orientation.to_scipy(), "auto"),
                                    block_size / 2, block_size / 2))

        def clearance(theta: float) -> float:
            ey = np.array([-np.sin(theta), np.cos(theta)])
            reach = (finger_gap + pad_depth) / 2
            pads = [_rect(xy + side * reach * ey, theta + np.pi / 2, pad_depth / 2, pad_width / 2)
                    for side in (1, -1)]
            return min([enough, *(_separation(p, b) for p in pads for b in blocks)])

        # Wrap only the angle difference; wrapping q7 itself could fold an out-of-range q7 into range.
        d = np.arctan2(xy[1], xy[0]) - (yaw + np.pi / 2)
        q7 = q7_home + (d + np.pi) % (2 * np.pi) - np.pi
        keep, turn = clearance(yaw), clearance(yaw + np.pi / 2)
        # The recovery turn (+90 deg base yaw, q7 - pi/2) must be reachable as well as the turn itself.
        if abs(q7) <= q7_limit and abs(q7 - np.pi / 2) <= q7_limit and turn > keep:
            rot, keep = R.from_euler("z", np.pi / 2) * rot, turn
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": rebuild(rot), "clearance": float(keep)})
