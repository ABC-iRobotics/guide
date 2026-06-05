from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

from guide_msgs.srv import Pose as PoseSrv
from guide_core.types.geometry import Pose
from guide_core.types.geometry import Transform

class TransformPose(BaseNode):
    level = Layer.UTILITY
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('TransformPose', alias, dynamic_map, static_args, output_map)
        
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
        pose =  l_pose * r_pose  # Assuming Pose and Transform have __mul__ defined for composition
        
        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={'pose': pose}
        )
        

class InvertPose(BaseNode):
    level = Layer.UTILITY
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('InvertPose', alias, dynamic_map, static_args, output_map)
        
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
        
        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={'pose': inverted_pose}
        )