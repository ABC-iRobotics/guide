"""The task and subtask the scene is at, read off the scene: cube_stack's oracle.

While a policy rolls out in simulation, this measures the scene with the demonstration
tree's own recipes -- MeasureTower and TowerHeight (the cubes' poses, the chain standing on each other)
and the fingers-on-the-cube check (IsPrimClashing) -- and names what the demonstration
would be doing, at both layers below the procedure:

* the TASK: put ``order[k]`` on ``order[k - 1]``, where ``k`` cubes stand. A cube on the
  tower counts once the fingers have let go of it, as in the demonstrations.
* the SUBTASK: Place once that cube is held and lifted off the table (the demonstration's
  Pick ends with the lift), Pick before. A cube knocked off sends both back.

Once the tower stands, the closing task and its subtask are both going home -- what the
recordings carry. The recovery subtask (setting an undeliverable cube
down) is a decision, not a state, so the oracle never names it. Nothing here moves the arm.

It is the ``oracle`` source of the InstructionHolder in block_bin_eval's ``eval_policy``
(a separate package, outside this repository): it stands in for a learned planner or
overrides one.
"""

from guide_ex.core.states import DemoStatus

from cube_stack import solve_task as st
from cube_stack.scene import CUBE


class SceneOracle:
    """``oracle()`` -> {"task": ..., "subtask": ...} the scene is at."""

    def __init__(self, robot, sim_namespace, scene_id, plan):
        self.tree = st.load_recipe("SceneOracle", robot, plan)  # config/recipes/SceneOracle.yaml
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
            return {"task": self.plan["subtasks"]["finish"], "subtask": self.plan["subtasks"]["finish"]}
        k = standing - 1  # the placement under way: order[standing] onto order[k]
        lifted = poses[standing].position.to_numpy()[2] - poses[0].position.to_numpy()[2] > CUBE / 2
        role = "place" if next_held and lifted else "pick"
        return {"task": self.plan["tasks"][k], "subtask": self.plan["subtasks"][role][k]}


def make_oracle(robot, sim_namespace, scene_id, plan) -> SceneOracle:
    """block_bin_eval's hook (``eval_policy.load_oracle``): ``plan`` is the scene's
    Randomize reply (scene.plan)."""
    return SceneOracle(robot, sim_namespace, scene_id, plan)
