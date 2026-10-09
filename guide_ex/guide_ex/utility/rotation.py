import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform.rotation import Rotation as R

from guide_core.types.geometry import Pose, Rotation, Transform
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer

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
