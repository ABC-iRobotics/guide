"""The task and subtask the scene is at, read off the scene: cube_stack's oracle.

While a policy rolls out in simulation, this measures the scene with the demonstration
tree's own nodes -- ``measure_tower`` (the cubes' poses, the chain standing on each other)
and the fingers-on-the-cube check (IsPrimClashing) -- and names what the demonstration
would be doing, at both layers below the procedure:

* the TASK: put ``order[k]`` on ``order[k - 1]``, where ``k`` cubes stand. A cube on the
  tower counts once the fingers have let go of it, as in the demonstrations.
* the SUBTASK: Place once that cube is held and lifted off the table (the demonstration's
  Pick ends with the lift), Pick before. A cube knocked off sends both back.

Once the tower stands, the procedure takes the task level back and the subtask is going
home -- what the recordings carry. The recovery subtask (setting an undeliverable cube
down) is a decision, not a state, so the oracle never names it. Nothing here moves the arm.

It is the ``oracle`` source of the evaluation's InstructionHolder: it stands in for a
learned planner or overrides one.
"""

from guide_ex.core.composite_node import CompositeNode
from guide_ex.core.states import DemoStatus, Layer
from guide_ex.steps.simulation.isaac.prim import IsPrimClashing

from cube_stack import solve_task as st
from cube_stack.scene import CUBE


def oracle_tree() -> CompositeNode:
    return CompositeNode(
        name="SceneOracle",
        level=Layer.SUBTASK,
        dynamic_map=st.keys(*st.SIM, "robot_prim", "order"),
        children=[
            *st.measure_tower(),
            # The highest cube standing, and whether the fingers are still on it.
            st.item("TopOfTower", "order", "top", shift=-1),
            st.holding("TopHeld"),
        ],
        mode="condition",
        condition_expr="done",
        # Not done: is the next cube in the fingers?
        false_branch=CompositeNode(
            name="NextCubeHeld",
            level=Layer.SEQUENCE,
            dynamic_map=st.keys(*st.SIM, "robot_prim", "order", "built"),
            children=[
                st.item("NextFree", "order", "next"),
                IsPrimClashing(
                    alias="NextHeld",
                    dynamic_map={**st.keys(*st.SIM), "prim1_path": "robot_prim", "prim2_path": "next"},
                    output_map={"has_collided": "next_held"},
                ),
            ],
        ),
    )


class SceneOracle:
    """``oracle()`` -> {"task": ..., "subtask": ...} the scene is at."""

    def __init__(self, robot, sim_namespace, scene_id, plan):
        self.tree = oracle_tree()
        self.context = st.episode_context(robot, sim_namespace, scene_id, plan)
        self.plan = plan
        self.done = False

    def __call__(self) -> dict:
        result = self.tree.execute(dict(self.context))
        if result.status != DemoStatus.PERFECT:
            raise RuntimeError(f"measuring the scene failed: {result.error_message}")
        out = result.outputs
        built, poses = out["built"], out["tower_poses"]
        if out["holding"] and built > 1:  # still in the fingers: not placed, being placed
            standing, next_held = built - 1, True
        else:
            standing, next_held = built, out.get("next_held", False)
        self.done = standing == len(self.plan["order"])
        if self.done:
            return {"task": self.plan["task"], "subtask": self.plan["subtasks"]["finish"]}
        k = standing - 1  # the placement under way: order[standing] onto order[k]
        lifted = poses[standing].position.to_numpy()[2] - poses[0].position.to_numpy()[2] > CUBE / 2
        role = "place" if next_held and lifted else "pick"
        return {"task": self.plan["tasks"][k], "subtask": self.plan["subtasks"][role][k]}


def make_oracle(robot, sim_namespace, scene_id, plan) -> SceneOracle:
    """The evaluation's hook: ``plan`` is the scene's Randomize reply (scene.plan)."""
    return SceneOracle(robot, sim_namespace, scene_id, plan)
