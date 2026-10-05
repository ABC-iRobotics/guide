"""The subtask the scene is at, read off the scene: cube_stack's subtask oracle.

While a policy rolls out in simulation, this measures the tower with the demonstration
tree's own nodes -- ``measure_tower`` (the cubes' poses, the chain standing on each
other) and the tree's ``holding`` check -- and names the subtask the demonstration would
be in, ``prompts[built - 1]``: put ``order[built]`` on ``order[built - 1]``. A cube on the
tower counts once the fingers have let go of it: the demonstrations announce the next
subtask after the release. Once the tower is done the last subtask stays, as it does in
the recordings while the arm goes home. Nothing here moves the arm.

It is the ``oracle`` source of the evaluation's InstructionHolder: it stands in for a
learned planner, or overrides one, and a cube knocked off the tower sends the subtask
back to it -- the recovery the demonstrations' loop makes.
"""

from guide_ex.core.composite_node import CompositeNode
from guide_ex.core.states import DemoStatus, Layer

from cube_stack import solve_task as st


def oracle_tree() -> CompositeNode:
    return CompositeNode(
        name="SubtaskOracle",
        level=Layer.SUBTASK,
        dynamic_map=st.keys(*st.SIM, "robot_prim", "order"),
        children=[
            *st.measure_tower(),
            # The highest cube standing, and whether the fingers are still on it.
            st.item("TopOfTower", "order", "top", shift=-1),
            st.holding("StillHeld"),
        ],
    )


class SubtaskOracle:
    """``oracle()`` -> the subtask prompt the scene is at."""

    def __init__(self, robot, sim_namespace, scene_id, order, prompts):
        self.tree = oracle_tree()
        self.context = st.episode_context(robot, sim_namespace, scene_id, order, prompts)
        self.prompts = prompts
        self.done = False

    def __call__(self) -> str:
        result = self.tree.execute(dict(self.context))
        if result.status != DemoStatus.PERFECT:
            raise RuntimeError(f"measuring the tower failed: {result.error_message}")
        built, held = result.outputs["built"], result.outputs["holding"]
        standing = built - 1 if held and built > 1 else built  # a held cube is not placed yet
        self.done = standing == len(self.prompts) + 1
        return self.prompts[min(standing, len(self.prompts)) - 1]


def make_oracle(robot, sim_namespace, scene_id, plan) -> SubtaskOracle:
    """The evaluation's hook: ``plan`` is the scene's Randomize reply ({task, order, subtasks})."""
    return SubtaskOracle(robot, sim_namespace, scene_id, plan["order"], plan["subtasks"])
