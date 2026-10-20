from collections import deque
from typing import Dict

from irob_lerobot_ros.config import ActionType
from irob_lerobot_ros.ros2robot import ROS2Robot
from lerobot.robots import Robot

from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class SetGripperState(BaseNode):
    level = Layer.STEP

    # A ROS2Robot's step returns once the fingers stop and the arm has taken the contact forces, all on
    # sim time: MoveIt rejects a plan that starts from a still-moving arm.
    POLL = 0.05  # s between reads of the finger positions
    STILL_SAMPLES = 3  # polls over which the fingers must stay still
    STILL_M = 2e-4  # m: a smaller finger change over those polls is still (at the target, or stalled on the cube)
    RESIDUAL = 0.2  # s after the fingers stop, for the arm to absorb the contact forces
    TIMEOUT = 2.0  # s to wait for still fingers before warning and going on

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GripperControl", alias, dynamic_map, static_args, output_map)

    def run(self, robot: Robot, gripper_goal_pos: Dict[str, float]) -> ExecutionResult:
        """
        Executes a gripper control action to the specified target position.

        Args:
            robot (Robot): The robot whose gripper to command.
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

            action = {f"{joint}.pos": position for joint, position in gripper_goal_pos.items()}
            self.logger.info(f"Sending gripper control action: {action}")
            robot.send_action(action=action, wait_for_execution=False)
            # wait_for_execution stays False: awaiting the gripper action deadlocked the node's executor for ~23 s
            joints, clock = robot.config.gripper_joint_names, robot.node.get_clock()
            start = clock.now().nanoseconds

            def fingers():
                observation = robot.get_observation()
                return [observation[f"{joint}.pos"] for joint in joints]

            window = deque([fingers()], maxlen=self.STILL_SAMPLES + 1)  # STILL_SAMPLES polls apart, end to end
            while True:
                elapsed = (clock.now().nanoseconds - start) / 1e9
                last = {joint: round(p, 5) for joint, p in zip(joints, window[-1])}
                if len(window) == window.maxlen and max(max(p) - min(p) for p in zip(*window)) < self.STILL_M:
                    self.logger.info(f"Gripper settled after {elapsed:.2f} s of sim time at {last}")
                    break
                if elapsed >= self.TIMEOUT:
                    self.logger.warning(f"Gripper not settled after {elapsed:.2f} s of sim time, at {last}: going on")
                    break
                robot.sleep(self.POLL)
                window.append(fingers())
            robot.sleep(self.RESIDUAL)
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

