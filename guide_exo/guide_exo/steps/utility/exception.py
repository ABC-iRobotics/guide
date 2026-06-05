from guide_exo.core.states import Layer
from guide_exo.core.states import DemoStatus
from guide_exo.core.states import ExecutionResult

from guide_exo.core.base_node import BaseNode

class NodeException(BaseNode):
    level = Layer.UTILITY

    def __init__(self, name="NodeException", alias=None, dynamic_map=None, static_args=None, output_map=None, **kwargs):
        super().__init__(name, alias, dynamic_map, static_args, output_map, **kwargs)

    def run(self, **kwargs) -> ExecutionResult:
        self.logger.warning(f"[{self.name}] NodeException triggered! Returning FAILURE to trigger fallback.")
        return ExecutionResult(status=DemoStatus.FAILURE, error_message=f"Exception triggered at {self.name}")