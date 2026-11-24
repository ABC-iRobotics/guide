"""Stack the cubes: the GUIDE-EX tree that builds the tower, recorded as one episode.

The scene draws the tower's order; the tree is built for it each episode from config/recipes/, its
layers as GUIDE-EX means them -- the procedure is the long-horizon workflow, each placement one TASK
with a success criterion, the pick and the place its SUBTASKs. Every TASK and SUBTASK announces its
prompt at its level, so the dataset carries all three. The procedure is never a frame's task: after
the last placement a closing TASK ("Return home.") takes over.

    StackingDemonstration (PROCEDURE "Stack the cubes.", condition: not done)
      Unclutch, LocateScene, AnnounceFirst (task + subtask), StartEpisode, MeasureTower, TowerHeight
      -> BuildTower (PROCEDURE, loop until the tower is done)
           NextCube (TASK, condition: done?)
             done -> Finish (TASK "Return home."): AnnounceFinish, GoHome, CheckSuccess, StopEpisode
             else -> PutOn (TASK "Put the red cube on the blue cube.")
                       Pick (SUBTASK "Pick up the red cube."), Place (SUBTASK "Place it on the blue cube.")

The recovery routes (Regrasp, *ViaHome, RepickDropped, the loop's rebuild passes) are in the README.
Variation lives in recipe targets only: the unclutch offset, the grasp tilt and MissPose's misses
(config/variants/); never in send_action's params. The episode loop, the CLI and the Demonstration
service are guide_ex.recipe.solve's (serve); this module is its hooks.
"""

import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from cube_stack.scene import CUBE, STACKED
from guide_core.scene.robot_profile import resolve
from guide_core.types.geometry import Pose
from guide_ex.recipe.solve import REST, home_pose, recipe_values, serve

REBUILDS = 3  # loop passes beyond one per layer, for placements that did not hold
UNCLUTCH = (0.03, 0.03, 0.01, np.radians(10.0))  # +- x, y, z (m) and yaw (rad) of the unclutch pose, uniform
SETTLE_S = 5.0  # sim s after Randomize: the cubes settle, the bins drop onto the floor


def slots(plan) -> list:
    """Where a planned miss lands: placement i puts order[i] on order[i - 1], the pick targets keyed on
    the cube picked, the rest on its support (cells "stage@i")."""
    o = plan["order"]
    return [{"pre_grasp": o[i], "grasp": o[i], "*": o[i - 1]} for i in range(1, len(o))]


def values(init, plan, miss, v, rng) -> dict:
    """What config/recipes/ interpolate: the robot (gripper joint, home joints and their tool pose), the
    plan, the loop's passes (one more per placement a miss may undo), the tower tolerance (the variant's,
    else the scene's), the cube edge, the misses, the grasp tilt (rad, uniform in +- the variant's
    grasp.tilt_deg), the free-yaw place and the unclutch offset (rng None: no draws)."""
    entry = resolve({"profile": "fr3", **init["robots"]["franka"]})  # the scene USD's own FR3: init.yaml names no profile
    deg = (v.get("grasp") or {}).get("tilt_deg", 0.0)
    tilt = np.radians(rng.uniform(-deg, deg)) if deg and rng is not None else 0.0
    j = np.zeros(4) if rng is None else rng.uniform(-1.0, 1.0, 4) * UNCLUTCH
    place = miss.get("place") or {}
    undone = len(place["at"]) if "at" in place else (len(plan["order"]) - 1) * bool(place)  # placements a miss undoes
    base = recipe_values(entry, init)
    return {
        **base,
        "robot": {**base["robot"], "home_pose": home_pose(entry)},
        "plan": plan,
        "tower": {"max_loops": len(plan["order"]) + REBUILDS + undone,
                  "stacked": list((v.get("tower") or {}).get("stacked") or STACKED)},
        "cube": {"edge": CUBE},
        "miss": miss,
        "grasp": {"tilt": float(tilt)},
        "place": {"free_yaw": True},  # a cube only has to face the one below: any yaw, no wrist turn
        "unclutch": {"offset": [float(j[0]), float(j[1]), float(j[2]) - 0.1], "yaw": float(j[3])},  # ~10 cm under rest
    }


def context(plan) -> dict:
    """The tree's own context keys: the frame it works in (the robot's base, as every pose GUIDE and the
    robot exchange is: S2b Ruling 4), the fingers, the rest pose and the drawn plan's prompts."""
    return {
        "scene_path": "/fr3/fr3_link0",
        "robot_prim": "/fr3/fr3_rightfinger",
        "rest_pose": Pose.from_numpy(np.array(REST[0]), R.from_euler("xyz", REST[1])),
        "order": plan["order"],
        "tasks": plan["tasks"],
        "picks": plan["subtasks"]["pick"],
        "places": plan["subtasks"]["place"],
        "set_downs": plan["subtasks"]["set_down"],
        "finish": plan["subtasks"]["finish"],
    }


main = lambda: serve(sys.modules[__name__])  # noqa: E731
