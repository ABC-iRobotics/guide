from typing import Dict

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot
from lerobot.robots import Robot

from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class MoveToJointConfiguration(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("JointMove", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot: Robot, target_configuration: Dict[str, float], speed: float = 1.0
    ) -> ExecutionResult:
        """
        Executes a joint move to the specified target configuration at the given speed.

        Args:
            robot (Robot): The robot to move.
            target_configuration (Dict[str, float]): Target position per arm joint, keyed
                "<joint>.pos" (e.g. "fr3_joint1.pos"); a ROS2Robot needs every arm joint.
            speed (float): MoveIt's velocity scaling factor for the move, 0 to 1 (default: 1.0).
        Returns:
            ExecutionResult: The result of the move execution.
        """

        if isinstance(robot, ROS2Robot):
            self.logger = robot.node.get_logger()
            if robot.config.arm_action_type not in [
                ActionType.JOINT_POSITION,
                ActionType.JOINT_TRAJECTORY,
                ActionType.CARTESIAN_POSE,
            ]:
                return ExecutionResult(
                    status=DemoStatus.FAILURE,
                    error_message=f"ROS2Robot does not support joint moves with current arm_action_type: {robot.config.arm_action_type}",
                )
            robot.config.arm_action_type = (
                ActionType.JOINT_POSITION
            )  # Ensure the robot is in Joint mode
            robot._moveit2.max_velocity = speed  # Set the speed for the move

            for _ in range(3):  # Retry logic for robustness
                success = robot.send_action(action=target_configuration, wait_for_execution=True)
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
