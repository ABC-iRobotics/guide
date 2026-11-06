from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.recipe import node


@node()
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


@node()
class NoDrop(BaseNode):
    """
    Fails once `value` falls below the highest it reached since the node was built (one episode's
    tree): a count that may only grow, e.g. the cubes standing in a tower, which drops only when the
    tower is knocked over.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("NoDrop", alias, dynamic_map, static_args, output_map)
        self._peak = None

    def run(self, value) -> ExecutionResult:
        """
        Args:
            value: The count.
        Returns:
            ExecutionResult: outputs `peak`; FAILURE if `value` is below it.
        """
        if self._peak is not None and value < self._peak:
            return ExecutionResult(status=DemoStatus.FAILURE,
                                   error_message=f"[{self.name}] {value} dropped below {self._peak}")
        self._peak = value
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"peak": value})
