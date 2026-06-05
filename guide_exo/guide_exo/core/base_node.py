from typing import Any
from typing import Optional
from typing import Dict

from logging import getLogger

from guide_exo.core.states import Layer
from guide_exo.core.states import ExecutionResult


class BaseNode:
    level: Layer
    
    def __init__(   self, 
                    name: str, 
                    alias: Optional[str] = None,
                    dynamic_map: Dict[str, str] = None,
                    static_args: Dict[str, Any] = None,
                    output_map: Dict[str, str] = None,
                    logger = getLogger(__name__)
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
                raise KeyError(f"[{self.name}] expects '{context_key}' from parent context, but it is missing.")
            inputs[arg_name] = context[context_key]
        return inputs
       
    def map_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Map outputs. Utility nodes strictly return ONLY explicitly mapped keys."""
        if not (outputs and self.output_map):
            return outputs
            
        if self.level == Layer.UTILITY:
            return {self.output_map[k]: v for k, v in outputs.items() if k in self.output_map}
            
        return {self.output_map.get(k, k): v for k, v in outputs.items()}
        
    def execute(self, context: dict) -> ExecutionResult:
        """Handles dependency injection before calling the specific node logic."""
        inputs = self.map_inputs(context)
            
        # Run the specific node logic with resolved inputs
        result = self.run(**inputs)
        
        # Map outputs to the parent context if output_map is defined
        result.outputs = self.map_outputs(result.outputs)

        return result

    def run(self, **kwargs) -> ExecutionResult:
        """To be implemented by subclasses in the Registry."""
        raise NotImplementedError