from typing import Dict

from geometry_msgs.msg import Twist
from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot
from lerobot.robots import Robot

from guide_core.types.geometry import Pose
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class MoveToCartesianPose(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("CartesianMove", alias, dynamic_map, static_args, output_map)

    def run(
        self,
        robot: Robot,
        target_pose: Pose,
        speed: float = 1.0,
        cartesian: bool = False,
        min_fraction: float = 0.95,
    ) -> ExecutionResult:
        """
        Executes a Cartesian move to the specified target pose at the given speed.

        Args:
            robot (Robot): The robot to move.
            target_pose (Pose): The target pose to move to.
            speed (float): 0 to 1 (default: 1.0): MoveIt's velocity scaling factor, or
                controller: oscbf's cruise speed, speed x 0.5 m/s.
            cartesian (bool): Whether to execute the move in Cartesian space (default: False).
            min_fraction (float): A straight-line move MoveIt can compute for less than
                this share of the way is refused before the arm moves (controller: moveit;
                oscbf plans straight lines or refuses).
        Returns:
            ExecutionResult: The result of the move execution.
        """

        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()  # Use the robot's logger for consistent logging
            if robot.config.arm_action_type not in [
                ActionType.CARTESIAN_POSE,
                ActionType.JOINT_POSITION,
                ActionType.JOINT_TRAJECTORY,
            ]:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot does not support Cartesian moves with current arm_action_type: {robot.config.arm_action_type}",
                )
            robot.config.arm_action_type = (
                ActionType.CARTESIAN_POSE
            )  # Ensure the robot is in Cartesian mode
            # speed and min_fraction go to the robot: MoveIt's velocity scaling and straight-line
            # check (controller: moveit), or the planner's cruise speed (controller: oscbf).
            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(
                    action=target_pose.to_ros(),
                    cartesian=cartesian,
                    wait_for_execution=True,
                    speed=speed,
                    min_fraction=min_fraction,
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

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("CartesianVelocityMove", alias, dynamic_map, static_args, output_map)

    def run(self, robot: Robot, velocity_command: Dict[str, float]) -> ExecutionResult:
        """
        Sends one Cartesian velocity command; the node neither times nor stops it.

        Args:
            robot (Robot): The robot to move.
            velocity_command (Dict[str, float]): Velocity components 'vx', 'vy', 'vz', 'vroll',
                'vpitch', 'vyaw'; a missing one is 0 (e.g., {'vx': 0.1, 'vy': 0.0, 'vz': 0.0}).
        Returns:
            ExecutionResult: The result of the velocity command execution.
        """
        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()
            if robot.config.arm_action_type != ActionType.CARTESIAN_VELOCITY:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot is not configured for Cartesian velocity control. Current arm_action_type: {robot.config.arm_action_type}",
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
                    action=twist_msg, cartesian=True, wait_for_execution=False
                )
                if success:
                    break
            # Nothing here times the motion or stops it after the command is sent.
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
