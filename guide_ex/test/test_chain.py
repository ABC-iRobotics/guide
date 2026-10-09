import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_ex.core.base_node import BaseNode
from guide_ex.core.composite_node import CompositeNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.utility.collection import GetItem
from guide_ex.utility.pose import ChainLength, is_at_offset
from guide_ex.utility.rotation import ProjectRotationToBaseZ

UP = (0.0, 0.0, 0.05)


def at(x, y, z):
    return Pose(position=Point([x, y, z]))


def chain(poses, offset=UP):
    return ChainLength().run(poses, offset).outputs


def test_is_at_offset_checks_every_axis():
    assert is_at_offset(at(0.01, 0, 0.055), at(0, 0, 0), UP, (0.02, 0.02, 0.01))
    assert not is_at_offset(at(0.03, 0, 0.05), at(0, 0, 0), UP, (0.02, 0.02, 0.01))
    assert not is_at_offset(at(0, 0, 0.0), at(0, 0, 0), UP, (0.02, 0.02, 0.01))


def test_a_chain_counts_from_the_first_and_stops_at_the_first_gap():
    stacked = [at(0, 0, 0.025 + 0.05 * i) for i in range(4)]
    # The third rests on the second, but the second slid off the first.
    broken = [at(0, 0, 0.025), at(0.1, 0, 0.025), at(0.1, 0, 0.075), at(0.5, 0, 0.025)]

    assert chain(stacked) == {"length": 4, "complete": True}
    assert chain(broken) == {"length": 1, "complete": False}
    assert chain([]) == {"length": 0, "complete": True}


def test_a_row_is_a_chain_too():
    row = [at(0.1 * i, 0, 0.025) for i in range(3)]

    assert chain(row, offset=(0.1, 0, 0)) == {"length": 3, "complete": True}


def test_get_item_shifts_and_refuses_out_of_range():
    items = ["a", "b", "c"]

    assert GetItem().run(items, 2).outputs == {"item": "c"}
    assert GetItem().run(items, 2, shift=-1).outputs == {"item": "b"}
    assert GetItem().run(items, 3).status == DemoStatus.FAILURE


def test_auto_heading_survives_a_cube_rolled_onto_its_side():
    # Rolled 90 deg about y, then turned 30 deg about z: body x points straight up.
    rot = R.from_euler("z", 30, degrees=True) * R.from_euler("y", 90, degrees=True)
    pose = Pose(position=Point([0, 0, 0.025]), orientation=Rotation.from_scipy(rot))

    yaw = ProjectRotationToBaseZ().run(pose, heading_axis="auto").outputs["yaw"]

    assert np.isclose(np.degrees(yaw) % 90, 30)


# --- the loop + condition composition a stacking task builds from these nodes --------

ORDER = ["a", "b", "c", "d"]


class World:
    """Four cubes on a table; Place stacks, except on the attempts listed in `slips`."""

    def __init__(self, slips=()):
        self.poses = {name: at(0.2 * i, 0, 0.025) for i, name in enumerate(ORDER)}
        self.slips, self.attempts = set(slips), 0


class Measure(BaseNode):
    level = Layer.STEP

    def __init__(self, world):
        super().__init__("Measure", dynamic_map={"order": "order"}, output_map={"poses": "poses"})
        self.world = world

    def run(self, order):
        poses = [self.world.poses[n] for n in order]
        return ExecutionResult(DemoStatus.PERFECT, outputs={"poses": poses})


class Place(BaseNode):
    level = Layer.STEP

    def __init__(self, world):
        super().__init__("Place", dynamic_map={"top": "top", "support": "support"})
        self.world = world

    def run(self, top, support):
        self.world.attempts += 1
        if self.world.attempts not in self.world.slips:
            base = self.world.poses[support].position.to_numpy()
            self.world.poses[top] = at(base[0], base[1], base[2] + 0.05)
        return ExecutionResult(DemoStatus.PERFECT)


def build(world, max_loops):
    """Loop: measure the chain; unless complete, put order[length] on order[length - 1]."""
    keys = {"order": "order"}
    put_on = CompositeNode(
        "PutOn",
        Layer.SUBTASK,
        dynamic_map={"order": "order", "length": "length"},
        children=[
            GetItem(
                alias="Top",
                dynamic_map={"items": "order", "index": "length"},
                output_map={"item": "top"},
            ),
            GetItem(
                alias="Support",
                dynamic_map={"items": "order", "index": "length"},
                static_args={"shift": -1},
                output_map={"item": "support"},
            ),
            Place(world),
        ],
    )
    next_one = CompositeNode(
        "Next",
        Layer.SUBTASK,
        dynamic_map=keys,
        children=[
            Measure(world),
            ChainLength(
                dynamic_map={"poses": "poses"},
                static_args={"offset": UP},
                output_map={"length": "length", "complete": "complete"},
            ),
        ],
        mode="condition",
        condition_expr="complete",
        false_branch=put_on,
    )
    loop = CompositeNode(
        "Build",
        Layer.TASK,
        dynamic_map=keys,
        children=[next_one],
        mode="loop",
        condition_expr="complete",
        max_loops=max_loops,
    )
    return loop.execute({"order": ORDER})


def test_the_loop_builds_the_chain_and_stops():
    world = World()

    assert build(world, max_loops=4).status == DemoStatus.PERFECT
    assert world.attempts == 3


def test_a_slipped_placement_is_simply_the_next_iteration():
    world = World(slips={2})

    assert build(world, max_loops=5).status == DemoStatus.PERFECT
    assert world.attempts == 4


def test_the_loop_budget_bounds_the_retries():
    world = World(slips={1, 2, 3, 4, 5})

    assert build(world, max_loops=5).status == DemoStatus.FAILURE
