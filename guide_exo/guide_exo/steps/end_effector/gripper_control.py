from typing import Dict

from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

from lerobot.robots import Robot
from irob_lerobot_ros.ros2robot import ROS2Robot

from irob_lerobot_ros.config import ActionType

class SetGripperState(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('GripperControl', alias, dynamic_map, static_args)
        
    def run(self, robot: Robot, gripper_goal_pos: Dict[str, float]) -> ExecutionResult:
        """
        Executes a gripper control action to the specified target position.
        
        Args:
            gripper_goal_pos (Dict[str, float]): A dictionary mapping gripper joint names to target positions.
        Returns:
            ExecutionResult: The result of the gripper control execution.
        """
        
        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()
            if robot.config.gripper_action_type != ActionType.JOINT_POSITION:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot does not support gripper control with current gripper_action_type: {robot.config.gripper_action_type}"
                )
            
            if len(gripper_goal_pos) == 1:
                joint_name = next(iter(gripper_goal_pos))
                target_position = gripper_goal_pos[joint_name]
                self.logger.info(f"Sending gripper control action for single joint: {joint_name} to position {target_position}")
                success = robot.send_action(
                    action={f'{joint_name}.pos': target_position},
                    wait_for_execution=False
                )
                # TODO: Add wait_for_execution logic and proper success checking based on robot response
                success = True 
                
        else:
            robot.send_action(gripper_goal_pos)
            success = True  # Assume success for non-ROS2Robot implementations
                
        if not success:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="Failed to execute gripper control action.",
            )
        return ExecutionResult(
            status=DemoStatus.PERFECT,
        )
        
class OpenGripper(SetGripperState):
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('OpenGripper', alias, dynamic_map, static_args, output_map)
        
    def run(self, robot: Robot | ROS2Robot, **kwargs) -> ExecutionResult:
        if isinstance(robot, ROS2Robot):
            self.static_args = {'gripper_goal_pos': {f'{joint}.pos': robot.config.gripper_open_position[idx] for idx, joint in enumerate(robot.config.gripper_joint_names)}}
        else:
            self.static_args = {'gripper_goal_pos': {joint: 1.0 for joint in robot.config.gripper_joint_names}}  # Assuming 1.0 is the open position for non-ROS2Robot implementations
        return super().run(robot=robot, **kwargs)
        
class CloseGripper(SetGripperState):
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('CloseGripper', alias, dynamic_map, static_args, output_map)
        
    def run(self, robot: Robot | ROS2Robot, **kwargs) -> ExecutionResult:
        if isinstance(robot, ROS2Robot):
            self.static_args = {'gripper_goal_pos': {f'{joint}.pos': robot.config.gripper_closed_position[idx] for idx, joint in enumerate(robot.config.gripper_joint_names)}}
        else:
            self.static_args = {'gripper_goal_pos': {joint: 0.0 for joint in robot.config.gripper_joint_names}}  # Assuming 0.0 is the closed position for non-ROS2Robot implementations
        return super().run(robot=robot, **kwargs)