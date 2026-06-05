from typing import Dict

from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

from lerobot.robots import Robot
from irob_lerobot_ros.ros2robot import ROS2Robot

from irob_lerobot_ros.config import ActionType


class MoveToJointConfiguration(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('JointMove', alias, dynamic_map, static_args, output_map)

    def run(self, robot: Robot, target_configuration: Dict[str, float], speed: float = 1.0) -> ExecutionResult:
        """
        Executes a joint move to the specified target configuration at the given speed.
        
        Args:
            target_configuration (Dict[str, float]): A dictionary mapping joint names to target positions.
            speed (float): The speed at which to execute the move (default: 1.0).
        Returns:
            ExecutionResult: The result of the move execution.
        """
        
        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()
            if robot.config.arm_action_type not in [ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY, ActionType.CARTESIAN_POSE]:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot does not support joint moves with current arm_action_type: {robot.config.arm_action_type}"
                )
            robot.config.arm_action_type = ActionType.JOINT_POSITION  # Ensure the robot is in Joint mode
            robot._moveit2.max_velocity = speed  # Set the speed for the move
            
            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(
                    action=target_configuration,
                    wait_for_execution=True
                )
                if success:
                    break
        else:
            robot.send_action(target_configuration)
            success = True  # Assume success for non-ROS2Robot implementations

        if not success:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="Failed to execute joint move to target configuration.",
            )

        return ExecutionResult(
            status=DemoStatus.PERFECT,
        )
        
        
# TODO: Implement MoveToJointTrajectory with similar structure, but handling trajectory format and execution logic
class MoveToJointTrajectory(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('JointTrajectoryMove', alias, dynamic_map, static_args, output_map)

    def run(self, robot: Robot, trajectory: Dict[str, list], speed: float = 1.0) -> ExecutionResult:
        """
        Executes a joint trajectory move following the specified trajectory at the given speed.
        
        Args:
            trajectory (Dict[str, list]): A dictionary mapping joint names to lists of target positions over time.
            speed (float): The speed at which to execute the move (default: 1.0).
        Returns:
            ExecutionResult: The result of the move execution.
        """
        # Implementation would be similar to MoveToJointConfiguration but would need to handle the trajectory format
        # and ensure that the robot's action type is set to JOINT_TRAJECTORY. This is a placeholder for now.
        return ExecutionResult(
            status=DemoStatus.FAILURE,
            error_message="Joint trajectory moves are not yet implemented."
        )