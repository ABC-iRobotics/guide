import numpy as np

from guide_core.types.geometry import Pose
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer


def is_on_top(upper: Pose, lower: Pose, height: float, xy_tolerance: float, z_tolerance: float):
    """True if `upper` rests on `lower`: centred over it and `height` above it (world z up)."""
    d = upper.position.to_numpy() - lower.position.to_numpy()
    return bool(np.hypot(d[0], d[1]) <= xy_tolerance and abs(d[2] - height) <= z_tolerance)


class TowerProgress(BaseNode):
    """
    How much of a tower stands, and what goes on it next.

    The tower is `order`, bottom first. It counts the objects resting on each other
    from the bottom up and stops at the first gap, so anything above a fallen layer
    is rebuilt too. Measured fresh every time, it makes the build self-correcting: run
    it in a loop and a cube that slid off, or a knocked tower, is simply the next step.
    """

    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("TowerProgress", alias, dynamic_map, static_args, output_map)

    def run(
        self,
        poses: list,
        order: list,
        prompts: list,
        height: float,
        xy_tolerance: float = 0.02,
        z_tolerance: float = 0.01,
    ) -> ExecutionResult:
        """
        Args:
            poses: World poses of the objects in `order` (z up).
            order: The objects, bottom of the tower first.
            prompts: prompts[i] describes putting order[i + 1] on order[i].
            height: Centre-to-centre height of one layer (the cube size).
            xy_tolerance: Largest horizontal offset that still counts as stacked.
            z_tolerance: Largest deviation from `height` that still counts as stacked.
        Returns:
            ExecutionResult with `built` (objects standing in order, >= 1), `done`, and
            while not done the next `top`, its `support` and the step's `prompt`.
        """
        built = 1
        while built < len(order) and is_on_top(
            poses[built], poses[built - 1], height, xy_tolerance, z_tolerance
        ):
            built += 1

        outputs = {"built": built, "done": built == len(order)}
        if not outputs["done"]:
            outputs.update(top=order[built], support=order[built - 1], prompt=prompts[built - 1])
        return ExecutionResult(status=DemoStatus.PERFECT, outputs=outputs)
