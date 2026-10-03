import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation
from guide_ex.core.base_node import BaseNode
from guide_ex.core.composite_node import CompositeNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_ex.utility.rotation import ProjectRotationToBaseZ
from guide_ex.utility.stacking import TowerProgress

ORDER = ["a", "b", "c", "d"]
PROMPTS = ["b on a", "c on b", "d on c"]


def at(x, y, z):
    return Pose(position=Point([x, y, z]))


def progress(poses):
    return TowerProgress().run(poses, ORDER, PROMPTS, height=0.05).outputs


def test_counts_from_the_bottom_and_names_the_next_step():
    out = progress([at(0, 0, 0.025), at(0.005, 0, 0.075), at(0.3, 0, 0.025), at(0.5, 0, 0.025)])

    assert out == {"built": 2, "done": False, "top": "c", "support": "b", "prompt": "c on b"}


def test_a_fallen_layer_rebuilds_everything_above_it():
    # c still sits on b, but b slid off a: b is next, c comes after it again.
    out = progress([at(0, 0, 0.025), at(0.1, 0, 0.025), at(0.1, 0, 0.075), at(0.5, 0, 0.025)])

    assert (out["built"], out["top"], out["support"]) == (1, "b", "a")


def test_a_complete_tower_is_done():
    out = progress([at(0, 0, 0.025 + 0.05 * i) for i in range(4)])

    assert out == {"built": 4, "done": True}


def test_auto_heading_survives_a_cube_rolled_onto_its_side():
    # Rolled 90 deg about y, then turned 30 deg about z: body x points straight up.
    rot = R.from_euler("z", 30, degrees=True) * R.from_euler("y", 90, degrees=True)
    pose = Pose(position=Point([0, 0, 0.025]), orientation=Rotation.from_scipy(rot))

    yaw = ProjectRotationToBaseZ().run(pose, heading_axis="auto").outputs["yaw"]

    assert np.isclose(np.degrees(yaw) % 90, 30)


class World:
    """Four cubes on a table; put_on stacks, except on the attempts listed in `slips`."""

    def __init__(self, slips=()):
        self.poses = {name: at(0.2 * i, 0, 0.025) for i, name in enumerate(ORDER)}
        self.slips, self.attempts = set(slips), 0


class Measure(BaseNode):
    level = Layer.STEP

    def __init__(self, world):
        super().__init__("Measure", dynamic_map={"order": "order"}, output_map={"poses": "poses"})
        self.world = world

    def run(self, order):
        return ExecutionResult(
            DemoStatus.PERFECT, outputs={"poses": [self.world.poses[n] for n in order]}
        )


class PutOn(BaseNode):
    level = Layer.STEP

    def __init__(self, world):
        super().__init__("PutOnStep", dynamic_map={"top": "top", "support": "support"})
        self.world = world

    def run(self, top, support):
        self.world.attempts += 1
        if self.world.attempts not in self.world.slips:
            base = self.world.poses[support].position.to_numpy()
            self.world.poses[top] = at(base[0], base[1], base[2] + 0.05)
        return ExecutionResult(DemoStatus.PERFECT)


def build_tower(world, max_loops):
    """The loop + condition composition cube_stack's solver uses, on a fake world."""
    keys = {"order": "order", "prompts": "prompts"}
    put_on = CompositeNode(
        "PutOn",
        Layer.SUBTASK,
        dynamic_map={"top": "top", "support": "support"},
        children=[PutOn(world)],
    )
    next_cube = CompositeNode(
        "NextCube",
        Layer.SUBTASK,
        dynamic_map=keys,
        children=[
            Measure(world),
            TowerProgress(
                dynamic_map={"poses": "poses", **keys},
                static_args={"height": 0.05},
                output_map={k: k for k in ("done", "top", "support", "prompt")},
            ),
        ],
        mode="condition",
        condition_expr="done",
        false_branch=put_on,
    )
    tower = CompositeNode(
        "StackCubes",
        Layer.TASK,
        dynamic_map=keys,
        children=[next_cube],
        mode="loop",
        condition_expr="done",
        max_loops=max_loops,
    )
    return tower.execute({"order": ORDER, "prompts": PROMPTS})


def test_the_loop_stacks_every_cube_and_stops():
    world = World()

    assert build_tower(world, max_loops=4).status == DemoStatus.PERFECT
    assert world.attempts == 3


def test_a_slipped_placement_is_simply_the_next_iteration():
    world = World(slips={2})

    assert build_tower(world, max_loops=5).status == DemoStatus.PERFECT
    assert world.attempts == 4


def test_the_loop_budget_bounds_the_retries():
    world = World(slips={1, 2, 3, 4, 5})

    assert build_tower(world, max_loops=5).status == DemoStatus.FAILURE
