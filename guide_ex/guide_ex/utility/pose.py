import numpy as np

from guide_core.types.geometry import Pose, Transform
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class TransformPose(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("TransformPose", alias, dynamic_map, static_args, output_map)

    def run(self, l_pose: Pose | Transform, r_pose: Pose | Transform) -> ExecutionResult:
        """
        Transforms the input pose by applying the given transform.

        Args:
            input_pose (Pose | Transform): The original pose or transform to be transformed.
            transform (Pose | Transform): The transformation to apply to the input pose or transform.
        Returns:
            ExecutionResult: The result containing the transformed pose.
        """
        # Perform the pose transformation (this is a placeholder for actual transformation logic)
        pose = l_pose * r_pose  # Assuming Pose and Transform have __mul__ defined for composition

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})


class InvertPose(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("InvertPose", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose | Transform) -> ExecutionResult:
        """
        Inverts the given pose or transform.

        Args:
            pose (Pose | Transform): The pose or transform to be inverted.
        Returns:
            ExecutionResult: The result containing the inverted pose.
        """
        # Perform the pose inversion (this is a placeholder for actual inversion logic)
        inverted_pose = pose.inv()  # Assuming Pose and Transform have an inv() method

        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": inverted_pose})


def is_at_offset(pose: Pose, reference: Pose, offset, tolerance) -> bool:
    """True if `pose` sits at `reference` + `offset`, within `tolerance` on each axis.

    Offset and tolerance are 3-vectors in the frame both poses are in (world: z up), so
    "rests on top of" is an offset of (0, 0, height) and "next in a row" (spacing, 0, 0).
    """
    d = pose.position.to_numpy() - reference.position.to_numpy() - np.asarray(offset, float)
    return bool(np.all(np.abs(d) <= np.asarray(tolerance, float)))


class ChainLength(BaseNode):
    """
    How many of `poses`, from the first, follow one another at a fixed offset.

    poses[i + 1] must sit at poses[i] + `offset` (see is_at_offset): a stack is
    (0, 0, height), a row (spacing, 0, 0). Counting stops at the first gap, so whatever
    lies beyond a gap counts as not in place, even if it rests on something.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("ChainLength", alias, dynamic_map, static_args, output_map)

    def run(self, poses: list, offset, tolerance=(0.02, 0.02, 0.01)) -> ExecutionResult:
        """
        Args:
            poses: The chain, first element first.
            offset: Where each element sits relative to the one before it.
            tolerance: Largest deviation from `offset` per axis.
        Returns:
            ExecutionResult with `length` (elements in place, from the first) and
            `complete` (all of them).
        """
        length = min(1, len(poses))
        while length < len(poses) and is_at_offset(
            poses[length], poses[length - 1], offset, tolerance
        ):
            length += 1
        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={"length": length, "complete": length == len(poses)},
        )
