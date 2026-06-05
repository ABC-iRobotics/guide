from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

import rclpy
from rclpy.node import Node

from guide_msgs.srv import Pose as PoseSrv
from guide_msgs.srv import Collision
from guide_core.types.geometry import Pose

class GetPrimPose(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('GetPrimPose', alias, dynamic_map, static_args, output_map)
    
    def run(self, robot: Node, sim_namespace: str, scene_namespace: str, prim_path: str) -> ExecutionResult:
        """
        Retrieves the current pose of the specified primitive.
        
        Args:
            robot (Node): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            prim_name (str): The name of the primitive to get the pose of.
        Returns:
            ExecutionResult: The result containing the current pose of the primitive.
        """
        
        if not robot.node:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="ROS2 node is not available for GetPrimPose.",
            )
            
        if getattr(robot.node, 'pose', None) is None:
            robot.node.pose = robot.node.create_client(
                PoseSrv, 
                f'{sim_namespace}/PoseRequest', 
                qos_profile=rclpy.qos.QoSProfile(depth=10), 
                callback_group=rclpy.callback_groups.ReentrantCallbackGroup()
            )
            
        request = PoseSrv.Request()
        request.path = scene_namespace + prim_path
        
        pose_response = robot.callService(robot.pose, request, f'Getting pose for {request.path}')
        
        if pose_response is not None:
            pose = Pose.from_ros_pose(pose_response.pose)
            return ExecutionResult(
                status=DemoStatus.PERFECT,
                outputs={'pose': pose}
            )
        else:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"Service call to get pose for primitive {request.path} failed.",
            )

class IsPrimClashing(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('IsPrimClashing', alias, dynamic_map, static_args, output_map)
    
    def run(self, robot: Node, sim_namespace: str, scene_namespace: str, prim1_path: str, prim2_path: str) -> ExecutionResult:
        """
        Checks if the specified primitive is clashing with any other primitives.
        
        Args:
            robot (Node): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_namespace (str): The scene namespace.
            prim_name (str): The name of the primitive to check for clashes.
        Returns:
            ExecutionResult: The result containing whether the primitive is clashing.
        """
        
        if not robot.node:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="ROS2 node is not available for IsPrimClashing.",
            )
            
        if getattr(robot.node, 'collision', None) is None:
            robot.node.collision = robot.node.create_client(
                Collision, 
                f'{sim_namespace}/CollisionRequest', 
                qos_profile=rclpy.qos.QoSProfile(depth=10), 
                callback_group=rclpy.callback_groups.ReentrantCallbackGroup()
            )
            
        request = Collision.Request()
        request.prim1 = scene_namespace + prim1_path
        request.prim2 = scene_namespace + prim2_path
        
        collision_response = robot.callService(robot.collision, request, f'Checking for clashes between {request.prim1} and {request.prim2}')
        
        if collision_response is not None:
            return ExecutionResult(
                status=DemoStatus.PERFECT,
                outputs={'has_collided': collision_response.collision}
            )
        else:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"Service call to check for clashes between {request.prim1} and {request.prim2} failed.",
            )
            
            
        
        