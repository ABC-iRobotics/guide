"""block_bin's hooks for guide_ex.recipe.solve: the drawn block goes into the drawn bin.

The episode loop, CLI and Demonstration service are the shared solver's (serve); the tree is
config/recipes/Episode.yaml, variation lives in its recipe values (config/variants/: flawless,
light_fail, retry, retry_mixed), never in send_action's params.
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from guide_core.scene.robot_profile import profile
from guide_core.types.geometry import Pose
from guide_ex.recipe import solve
from guide_ex.recipe.solve import serve

NOISE = {}  # guide_motion's NoiseParams defaults are FR3 scale; --motion-noise overrides
SETTLE_S = 5.0  # sim s after Randomize: the scene-level draw settles


def slots(plan) -> list:
    """Where a planned miss lands: every MissPose target keyed on the drawn block."""
    return [{"*": plan["target"]}]


def values(init, plan, miss, v, rng) -> dict:
    """What config/recipes/ interpolate: the gripper joint, home and camera hold, and the misses."""
    entry = {**profile("fr3"), **init["robots"]["franka"]}  # an in-scene robot: no profile fill-in by resolve
    return {**solve.recipe_values(entry, init), "miss": miss}


def context(plan) -> dict:
    """The drawn block and bin, the arm's finger prim and its rest pose (robot base frame, S2b Ruling 4)."""
    return {"target": plan["target"], "goal": plan["goal"], "robot_prim": "/fr3/fr3_rightfinger",
            "rest_pose": Pose.from_numpy(np.array(solve.REST[0]), R.from_euler("xyz", solve.REST[1]))}


main = lambda: serve(sys.modules[__name__])  # noqa: E731
