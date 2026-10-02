from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_msgs.srv import CheckSuccess


class IsTaskSuccessful(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("IsTaskSuccessful", alias, dynamic_map, static_args, output_map)

    def run(self, robot, sim_namespace: str, scene_id: int) -> ExecutionResult:
        """
        Asks the scene whether its success criterion holds right now.

        A criterion that does not hold is an answer, not a failure: the node is PERFECT
        and outputs `success` False, so a following StopRecording can discard on it.

        Args:
            robot (Node): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_id (int): The scene to check.
        Returns:
            ExecutionResult: outputs `success` (bool) and `reason` (str); FAILURE only if
            the scene could not be asked.
        """
        # Reuse main()'s client: a second client on one service breaks reply routing.
        if getattr(robot, "is_success", None) is None:
            robot.is_success = robot.node.create_client(
                CheckSuccess,
                f"{sim_namespace}/IsSuccess",
                callback_group=robot._reentrant_callback_group,
            )

        try:
            response = robot.callService(
                robot.is_success, CheckSuccess.Request(id=scene_id), f"Checking scene {scene_id}"
            )
        except TimeoutError as e:
            return ExecutionResult(status=DemoStatus.FAILURE, error_message=str(e))

        return ExecutionResult(
            status=DemoStatus.PERFECT,
            outputs={"success": bool(response.success), "reason": response.message},
        )
