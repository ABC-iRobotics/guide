import time

from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class WaitForSeconds(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("WaitForSeconds", alias, dynamic_map, static_args, output_map)

    def run(self, seconds: float, timer=time) -> ExecutionResult:
        """
        Executes a wait action for the specified number of seconds.

        Args:
            timer: Module or object with the sleep(seconds: float) method.
            seconds (float): The number of seconds to wait.
        Returns:
            ExecutionResult: The result of the wait execution.
        """
        timer.sleep(seconds)
        return ExecutionResult(
            status=DemoStatus.PERFECT,
        )
