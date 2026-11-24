"""Peg transfer: the GUIDE-EX tree that moves the drawn block across the board, one episode each.

The scene draws the plan (scene.draw_plan, the Randomize reply): which PSM gives, which block, the
target peg. The tree is built for it per episode from config/recipes/, every robot node naming its
arm by role, the giver or the receiver (${hand.*}, psm1 or psm2), the ECM holding the camera:

    PegTransfer (PROCEDURE "Transfer the block to the other side.")
      AnnounceFirst (task + subtask), HoldCamera (ECM, jittered joints), StartEpisode
      -> Transfer (TASK "Move the red block from the left to the right side.")
           LocateScene (each arm's /PegTransfer pose, the handover marker, the block, the grasp)
           Reach (SUBTASK, ParallelExecuter "Right: Reach for the block. Left: Move to the exchange point.")
             GiverApproach (over the block, wrist turning, down to the grasp)  ||  ReceiverToExchange
           PickBlock (SUBTASK "Pick up the block.", condition: holding)  fallback Regrasp
           LiftOff (SUBTASK "Lift it off the peg.")
           HandOver (SUBTASK "Hand it over.")  Receive (condition: received), fallback Rereceive
           Finish (SUBTASK, ParallelExecuter "Right: Place it on the peg. Left: Return to start.", planned in HandOver)
             GiverReturns  ||  PlaceOnPeg (OnPeg fails NotSeated: Replace)
           ReturnToStart (SUBTASK "Return to start.": the receiver home)

A parallel block's branch prompts join with the image side of its arm (sides: PSM1 is image right).
      CheckSuccess, StopEpisode (saved only on success)

Every pose target comes from peg_transfer.targets (the tilted PSM bases leave the library's base-z-up
poses no use). Variation lives in those targets only: the ECM's hold, the handover marker, MissPose's
planned misses (config/variants/: flawless, light_fail, retry_mixed); never in send_action's params.
A flawless episode runs without retries (contact closes kept): a failing step discards it and the layout is drawn again.
The episode loop, the CLI and the Demonstration service are guide_ex.recipe.solve's (serve); this module is its hooks.
"""

import json
import sys

import numpy as np

from guide_ex.recipe.solve import home_pose, serve
from peg_transfer.scene import IMAGE_SIDE
from peg_transfer.targets import APPROACH, DROP, LIFT, TOUCH  # importing registers its nodes (GraspPose, ...)

SUBTASKS = {"reach": "Reach for the block.", "exchange": "Move to the exchange point.", "pick": "Pick up the block.", "lift": "Lift it off the peg.", "handover": "Hand it over.",
            "place": "Place it on the peg.", "return": "Return to start.",
            # a recovery announces itself: the subtask timeline shows the miss and its retry
            "regrasp": "Open and grasp it again.", "rereceive": "Open and take it again.",
            "replace": "Lift it and place it again."}
# Jaw opening (rad between the jaws, 2x each jaw's angle; 0 closes). The grasp straddles the 0.7 mm wall
# with 0.082 rad per jaw (0.77 mm at the tips); wider jaws land on the peg (T10: tip 6.5-7 mm high).
OPEN_RAD = 0.164
# Wide, where no peg is near a jaw's outside (the handover) or the peg leaves the inner jaw 2.6 mm (the release):
# 0.2 rad per jaw (1.9 mm at the tips). At OPEN_RAD the receiver's tips, +-0.4 mm about the wall, landed on its
# end and closed on air (NotReceived x3); opening there, the jaws stay on the wall.
WIDE_RAD = 0.4
# Closing aims past shut, like the dVRK's negative jaw angles: on a wall the drive (0.5 N m/rad) then
# squeezes with 0.05 rad of error, not the 0.003 left at 0, and the block no longer droops out of level
SQUEEZE_RAD = 0.1
# +- per ECM joint about its home (init.yaml): yaw, pitch (rad), insertion (m; home 0.005, its limit 0), tool roll
ECM_JITTER = (0.02, 0.02, 0.004, 0.0)
# cube_stack's motion noise (guide_motion NoiseParams, FR3 scale) x 0.4 in every length (x 0.1, sigma 1.2 mm
# and blend 4 mm, was some 3 px in the ECM image: no visible noise or blending). The aim miss fades within the
# next move (aim_late off): Noise as the robot runs it leaves a precise descent after an aimed hover 0.30 mm
# off its line (p99) 4.5 mm before the grasp, inside the jaws' 0.42 mm each side of the wall. p_perturb 0.8:
# most moves asked to be perturbed are.
NOISE = {"sigma": 0.0048, "decay": 0.012, "ell": 0.06, "spacing": 0.02, "blend": 0.015, "aim": 0.012,
         "aim_ramp": 0.04, "p_perturb": 0.8}
SETTLE_S = 2.0  # sim s after Randomize: the blocks drop onto their pegs


def ecm_hold(entry, rng) -> dict:
    """The ECM's joint target: its default joints (init.yaml), each uniformly within ECM_JITTER (rng None: none)."""
    q = np.asarray(entry["default_joint_states"], float)
    if rng is not None:
        q = q + rng.uniform(-1, 1, len(q)) * ECM_JITTER
    return {f"{j}.pos": float(v) for j, v in zip(entry["joint_names"], q)}


def slots(plan) -> list:
    """Where a planned miss lands: every MissPose target keyed on the block."""
    return [{"*": plan["colour"]}]


def values(init, plan, miss, v, rng) -> dict:
    """What config/recipes/ interpolate: the roles (hand), the plan, the subtask prompts, each role's tool
    tip prim, the PSM home and jaw (description.json's gripper scale), the ECM hold (jittered by rng) and the misses."""
    from guide_core.scene.robot_profile import resolve
    from guide_motion.description import ROBOTS_DIR

    robots = {k: resolve(e) for k, e in init["robots"].items()}
    gripper = json.loads((ROBOTS_DIR / "psm" / "description.json").read_text())["gripper"]
    roles = {"giver": plan["giver"], "receiver": plan["receiver"]}
    sides = {r: side.capitalize() for r, side in IMAGE_SIDE.items()}  # ParallelExecuter's prefix, in prompt order
    first = {plan["giver"]: SUBTASKS["reach"], plan["receiver"]: SUBTASKS["exchange"]}
    return {
        "hand": roles,
        "plan": plan,
        "sides": sides,
        # the first parallel block's joined subtask, announced before the first frame
        "prompts": {**SUBTASKS, "first": " ".join(f"{sides[r]}: {first[r]}" for r in sides)},
        "tip_prim": {role: f"{robots[r]['path']}/psm_tool_tip_link" for role, r in roles.items()},
        "psm": {"home": {role: {f"{j}.pos": q for j, q in zip(robots[r]["joint_names"], robots[r]["default_joint_states"])}
                         for role, r in roles.items()},  # each arm's own (init.yaml)
                "home_pose": {role: home_pose(robots[r]) for role, r in roles.items()}},
        "jaw": {"joint": gripper["joints"][0], "open": gripper["scale"][0] * OPEN_RAD, "wide": gripper["scale"][0] * WIDE_RAD, "closed": -gripper["scale"][0] * SQUEEZE_RAD},
        "camera": {"hold": ecm_hold(robots["ecm"], rng)},
        "offsets": {"approach": APPROACH, "lift": LIFT, "down": -DROP, "touch": -TOUCH},  # m along the tool's z
        "miss": miss,
    }


def info(plan) -> dict:
    """The provenance's plan keys."""
    return {"giver": plan["giver"], "colour": plan["colour"], "target_peg": plan["target_peg"]}


main = lambda: serve(sys.modules[__name__])  # noqa: E731
