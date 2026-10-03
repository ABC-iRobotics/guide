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


class IsAtPose(BaseNode):
    """
    A check that fails: FAILURE unless `pose` is within `tolerance` of `target`.

    Put it after a move to confirm the robot got there. A Cartesian move whose path MoveIt
    could only partly compute still executes the part it has and reports success; this
    turns that into a failure the composite's fallback can handle.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("IsAtPose", alias, dynamic_map, static_args, output_map)

    def run(self, pose: Pose, target: Pose, tolerance: float = 0.015) -> ExecutionResult:
        """
        Args:
            pose: Where it is (e.g. the measured TCP pose).
            target: Where it was sent, in the same frame.
            tolerance: Largest position error that still counts as arrived, metres.
        Returns:
            ExecutionResult: PERFECT if arrived, FAILURE with the distance otherwise.
        """
        distance = float(np.linalg.norm(pose.position.to_numpy() - target.position.to_numpy()))
        if distance > tolerance:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"[{self.name}] {distance:.3f} m from the target",
            )
        return ExecutionResult(status=DemoStatus.PERFECT)
