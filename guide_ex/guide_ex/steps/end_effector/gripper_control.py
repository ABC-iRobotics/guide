import time
from typing import Dict

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot
from lerobot.robots import Robot

from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class SetGripperState(BaseNode):
    level = Layer.STEP

    # Seconds to let the gripper/arm settle after a grasp or release before the
    # next step (e.g. planning) runs. Closing onto the cube injects contact forces
    # that shift the arm joints; planning immediately snapshots a still-moving arm,
    # so move_group later rejects the trajectory ("start point deviates from
    # current robot state"). Waiting for the gripper action to finish and then
    # settling makes the next plan start from a stable, grasped state.
    SETTLE_SECONDS = 1.0

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GripperControl", alias, dynamic_map, static_args, output_map)

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
                    error_message=f"ROS2Robot does not support gripper control with current gripper_action_type: {robot.config.gripper_action_type}",
                )

            if len(gripper_goal_pos) == 1:
                joint_name = next(iter(gripper_goal_pos))
                target_position = gripper_goal_pos[joint_name]
                self.logger.info(
                    f"Sending gripper control action for single joint: {joint_name} to position {target_position}"
                )
                success = robot.send_action(
                    action={f"{joint_name}.pos": target_position}, wait_for_execution=False
                )
                # Wait for the grasp/release to physically settle before the next
                # step (planning) starts, so it snapshots a stable arm rather than
                # one still reacting to the new contact forces. (wait_for_execution
                # is left False: awaiting the gripper action deadlocked the node's
                # executor for ~23s, so a fixed settle is used instead.)
                self.logger.info(
                    f"Waiting {self.SETTLE_SECONDS:.2f}s for gripper/arm to settle "
                    f"before continuing."
                )
                time.sleep(self.SETTLE_SECONDS)
                # TODO: Proper success checking based on robot response
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

