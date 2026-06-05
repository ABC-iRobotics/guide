from typing import Dict

from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

from lerobot.robots import Robot
from irob_lerobot_ros.ros2robot import ROS2Robot

from irob_lerobot_ros.config import ActionType
from guide_core.types.geometry import Pose
from geometry_msgs.msg import Twist

class MoveToCartesianPose(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('CartesianMove', alias, dynamic_map, static_args, output_map)
    
    def run(self, robot: Robot, target_pose: Pose, speed: float = 1.0, cartesian: bool = False) -> ExecutionResult:
        """
        Executes a Cartesian move to the specified target pose at the given speed.
        
        Args:
            target_pose (Pose): The target pose to move to.
            speed (float): The speed at which to execute the move (default: 1.0).
            cartesian (bool): Whether to execute the move in Cartesian space (default: False).
        Returns:
            ExecutionResult: The result of the move execution.
        """
        
        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()  # Use the robot's logger for consistent logging
            # Convert target_pose to the format expected by the ROS2Robot
            if robot.config.arm_action_type not in [ActionType.CARTESIAN_POSE, ActionType.JOINT_POSITION, ActionType.JOINT_TRAJECTORY]:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot does not support Cartesian moves with current arm_action_type: {robot.config.arm_action_type}"
                )
            robot.config.arm_action_type = ActionType.CARTESIAN_POSE  # Ensure the robot is in Cartesian mode
            robot._moveit2.max_velocity = speed  # Set the speed for the move
            
            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(
                    action = target_pose.to_ros(),
                    cartesian=cartesian,
                    wait_for_execution=True
                )
                if success:
                    break
        else:
            robot.send_action(target_pose.toDict())
            success = True  # Assume success for non-ROS2Robot implementations

        if not success:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="Failed to execute Cartesian move to target pose.",
            )

        return ExecutionResult(
            status=DemoStatus.PERFECT,
        )
        
class MoveWithCartesianVelocity(BaseNode):
    level = Layer.STEP
    
    def __init__(self, alias = None, dynamic_map = None, static_args = None, output_map = None):
        super().__init__('CartesianVelocityMove', alias, dynamic_map, static_args, output_map)
        
    def run(self, robot: Robot, velocity_command: Dict[str, float]) -> ExecutionResult:
        """
        Executes a Cartesian velocity command for a specified duration.
        
        Args:
            velocity_command (Dict[str, float]): A dictionary containing velocity components (e.g., {'vx': 0.1, 'vy': 0.0, 'vz': 0.0}).
        Returns:
            ExecutionResult: The result of the velocity command execution.
        """
        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()
            if robot.config.arm_action_type != ActionType.CARTESIAN_VELOCITY:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot is not configured for Cartesian velocity control. Current arm_action_type: {robot.config.arm_action_type}"
                )
            
            twist_msg = Twist()
            twist_msg.linear.x = float(velocity_command.get("vx", 0.0))
            twist_msg.linear.y = float(velocity_command.get("vy", 0.0))
            twist_msg.linear.z = float(velocity_command.get("vz", 0.0))
            twist_msg.angular.x = float(velocity_command.get("vroll", 0.0))
            twist_msg.angular.y = float(velocity_command.get("vpitch", 0.0))
            twist_msg.angular.z = float(velocity_command.get("vyaw", 0.0))

            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(
                    action = twist_msg,
                    cartesian=True,
                    wait_for_execution=False
                )
                if success:
                    break
            success = robot.send_action(
                action = twist_msg,
                cartesian=True,
                wait_for_execution=False
            )
            # In a real implementation, you would need to handle the timing and stopping of the velocity command after the specified duration.
        else:
            robot.send_action(velocity_command)
            success = True  # Assume success for non-ROS2Robot implementations

        if not success:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="Failed to execute Cartesian velocity command.",
            )

        return ExecutionResult(
            status=DemoStatus.PERFECT,
        )