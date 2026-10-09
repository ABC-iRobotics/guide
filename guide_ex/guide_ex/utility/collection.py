from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


class GetItem(BaseNode):
    """``items[index + shift]``: pick from a list by an index another node computed."""

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GetItem", alias, dynamic_map, static_args, output_map)

    def run(self, items: list, index: int, shift: int = 0) -> ExecutionResult:
        """
        Args:
            items: The list.
            index: The position, e.g. a count or a loop iteration.
            shift: Added to `index` (-1: the item before it).
        Returns:
            ExecutionResult: outputs `item`; FAILURE if the position is outside the list.
        """
        position = index + shift
        if not 0 <= position < len(items):
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"[{self.name}] no item {position} in a list of {len(items)}",
            )
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"item": items[position]})
