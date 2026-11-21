from contextvars import ContextVar
from logging import getLogger
from typing import Any, Dict, Optional

from guide_ex.core.states import DemoStatus, ExecutionResult, Layer

# (run, branch index) per enclosing ParallelExecuter branch, outermost first; () outside one
BRANCH: ContextVar[tuple] = ContextVar("guide_ex_branch", default=())


class BaseNode:
    level: Layer
    claims_robots = True  # a leaf claims the robots in its inputs inside a ParallelExecuter branch

    def __init__(
        self,
        name: str,
        alias: Optional[str] = None,
        dynamic_map: Dict[str, str] = None,
        static_args: Dict[str, Any] = None,
        output_map: Dict[str, str] = None,
        logger=getLogger(__name__),
    ):

        self.node_type = name
        self.name = alias or name
        self.dynamic_map = dynamic_map or {}
        self.static_args = static_args or {}
        self.logger = logger
        self.output_map = output_map or {}

    def map_inputs(self, context: dict) -> dict:
        """Map the parent context to the node's expected inputs using dynamic_map."""
        inputs = {**self.static_args}
        for arg_name, context_key in self.dynamic_map.items():
            if context_key not in context:
                raise KeyError(
                    f"[{self.name}] expects '{context_key}' from parent context, but it is missing."
                )
            inputs[arg_name] = context[context_key]
        return inputs

    def map_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Map outputs through output_map: utility nodes return ONLY the keys it maps, other
        nodes rename those keys and keep the rest. Without an output_map, outputs pass as-is."""
        if not (outputs and self.output_map):
            return outputs

        if self.level == Layer.UTILITY:
            return {self.output_map[k]: v for k, v in outputs.items() if k in self.output_map}

        return {self.output_map.get(k, k): v for k, v in outputs.items()}

    def execute(self, context: dict) -> ExecutionResult:
        """Handles dependency injection before calling the specific node logic."""
        inputs = self.map_inputs(context)
        refused = self._claim(inputs)
        if refused:
            self.logger.error(f"[{self.name}] {refused}")  # a composite parent drops error_message
            return ExecutionResult(DemoStatus.FAILURE, error_message=f"[{self.name}] {refused}")

        # Run the specific node logic with resolved inputs
        result = self.run(**inputs)

        # Map outputs to the parent context if output_map is defined
        result.outputs = self.map_outputs(result.outputs)

        return result

    def _claim(self, inputs: dict) -> str:
        """Inside ParallelExecuter branches, claim every robot (anything with send_action) in `inputs`
        for this branch at each enclosing executer. Returns why not if another branch holds one, else
        "". A no-op outside an executer."""
        frames = BRANCH.get()
        if not (frames and self.claims_robots):
            return ""
        for value in inputs.values():
            if callable(getattr(value, "send_action", None)):
                for run, branch in frames:
                    other = run.claim(branch, value)
                    if other is not None:
                        return f"robot {value!r} belongs to parallel branch {other}, not {branch}"
        return ""

    def _apply_override(self, overrides: Dict[str, "BaseNode"]) -> None:
        """A leaf has nothing to replace; CompositeNode overrides this."""

    def run(self, **kwargs) -> ExecutionResult:
        """To be implemented by subclasses."""
        raise NotImplementedError
