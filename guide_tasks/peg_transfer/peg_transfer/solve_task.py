"""Peg transfer: the GUIDE-EX tree that moves the drawn block across the board, one episode each.

The scene draws the plan (scene.draw_plan, the Randomize reply): which PSM gives, which block, the
target peg. The tree is built for it per episode from config/recipes/, every robot node naming its
arm by role, the giver or the receiver (${hand.*}, psm1 or psm2), the ECM holding the camera:

    PegTransfer (PROCEDURE "Transfer the block to the other side.")
      AnnounceFirst (task + subtask), HoldCamera (ECM, jittered joints), StartEpisode
      -> Transfer (TASK "Move the red block from the left to the right side.")
           LocateScene (each arm's /PegTransfer pose)
           PickBlock (SUBTASK "Pick up the block.", condition: holding)  fallback Regrasp
           LiftOff (SUBTASK "Lift it off the peg.")
           HandOver (SUBTASK "Hand it over.")  Receive (condition: received), fallback Rereceive
           PlaceOnPeg (SUBTASK "Place it on the peg.")  Seat (condition: seated), fallback Replace
           ReturnToStart (SUBTASK "Return to start.")
      CheckSuccess, StopEpisode (saved only on success)

Every pose target comes from peg_transfer.targets (the tilted PSM bases leave the library's base-z-up
poses no use). Variation lives in those targets only: the ECM's hold, the handover marker, MissPose's
planned misses (config/variants/: flawless, light_fail, retry_mixed); never in send_action's params.
A flawless episode runs without recoveries: a failing step discards it and the layout is drawn again.
"""

import argparse
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from guide_ex.core.states import DemoStatus
from guide_ex.recipe import load, recipes_dir
from guide_ex.recipe.solve import SERVICES, make_robots
from peg_transfer.targets import APPROACH, DROP, LIFT  # importing registers its nodes (GraspPose, ...)

SUBTASKS = {"pick": "Pick up the block.", "lift": "Lift it off the peg.", "handover": "Hand it over.",
            "place": "Place it on the peg.", "return": "Return to start."}
# Jaw opening (rad between the jaws, 2x each jaw's angle; 0 closes). The grasp straddles the 0.7 mm wall
# with 0.082 rad per jaw (0.77 mm at the tips); wider jaws land on the peg (T10: tip 6.5-7 mm high).
OPEN_RAD = 0.164
RECEIVE_RAD = 0.164  # the receiver's, onto the held block (no peg in the hole)
ECM_JITTER = (0.02, 0.02, 0.005, 0.0)  # +- per ECM joint: yaw, pitch (rad), insertion (m), tool roll
ROLL_GAIN = 0.5  # share of the marker's +-30 deg roll the handover takes (tool_yaw saturates beyond ~15)
# 0.1x cube_stack's motion noise (guide_motion NoiseParams): the PSM works at millimetres
NOISE = {"sigma": 0.0012, "aim": 0.003, "blend": 0.004, "aim_late": True}
SETTLE_S = 2.0  # sim s after Randomize: the blocks drop onto their pegs
MAX_ATTEMPTS = 8  # attempts at one episode before the run stops
SEED_BLOCK = 1000  # episode i draws its layouts from scene seeds seed_base + i * SEED_BLOCK, ... + 1, ...
_generating = threading.Lock()


def init_yaml() -> dict:
    return yaml.safe_load((recipes_dir("peg_transfer", __file__).parent / "init.yaml").read_text())


def variant(name) -> dict:
    """config/variants/<name>.yaml: `miss` (fixed misses) or `failures` + `stages` (planned ones)."""
    return yaml.safe_load((recipes_dir("peg_transfer", __file__).parent / "variants" / f"{name}.yaml").read_text())


def planned_misses(v, key, rng) -> tuple[dict, list[str]]:
    """One episode's misses: n distinct stages (n drawn from failures.counts with failures.weights), each
    moving its MissPose targets once, keyed on the block (`key`). Returns MissPose's `miss`
    ({target: {at: {key: {axes, r}}}}) and the stages, e.g. ["handover"]."""
    if "stages" not in v:
        return dict(v.get("miss") or {}), []
    n = int(rng.choice(v["failures"]["counts"], p=v["failures"]["weights"]))
    chosen = sorted(str(s) for s in rng.choice(list(v["stages"]), n, replace=False))
    return {t: {"at": {key: dict(spec)}} for s in chosen for t, spec in v["stages"][s].items()}, chosen


def ecm_hold(entry, rng) -> dict:
    """The ECM's joint target: its default joints, each moved uniformly within ECM_JITTER (rng None: none)."""
    q = np.asarray(entry["default_joint_states"], float)
    if rng is not None:
        q = q + rng.uniform(-1, 1, len(q)) * ECM_JITTER
    return {f"{j}.pos": float(v) for j, v in zip(entry["joint_names"], q)}


def recipe_values(init, plan, miss=None, hold=None) -> dict:
    """What config/recipes/ interpolate: the roles (hand), the plan, the subtask prompts, each role's tool
    tip prim, the PSM home and jaw (description.json's gripper scale), the ECM hold and the misses."""
    from guide_core.scene.robot_profile import resolve
    from guide_motion.description import ROBOTS_DIR

    robots = {k: resolve(v) for k, v in init["robots"].items()}
    psm = robots[plan["giver"]]
    gripper = json.loads((ROBOTS_DIR / "psm" / "description.json").read_text())["gripper"]
    roles = {"giver": plan["giver"], "receiver": plan["receiver"]}
    return {
        "hand": roles,
        "plan": plan,
        "prompts": SUBTASKS,
        "tip_prim": {role: f"{robots[r]['path']}/psm_tool_tip_link" for role, r in roles.items()},
        "psm": {"home": {f"{j}.pos": q for j, q in zip(psm["joint_names"], psm["default_joint_states"])}},
        "jaw": {"joint": gripper["joints"][0], "open": gripper["scale"][0] * OPEN_RAD,
                "receive": gripper["scale"][0] * RECEIVE_RAD, "closed": 0.0},
        "camera": {"hold": hold or ecm_hold(robots["ecm"], None)},
        "handover": {"roll_gain": ROLL_GAIN},
        "offsets": {"approach": APPROACH, "lift": LIFT, "down": -DROP},  # m along the tool's z
        "miss": miss or {},
    }


def build_tree(init, plan, **variation):
    """config/recipes/Episode.yaml as a GUIDE-EX tree for this plan (variation: recipe_values')."""
    recipes = recipes_dir("peg_transfer", __file__)
    return load(recipes / "Episode.yaml", values=recipe_values(init, plan, **variation), search_paths=[recipes])


def without_recoveries(node):
    """Drop every recovery (fallback) in the tree: a failing step then ends the episode."""
    if getattr(node, "fallbacks", None):
        node.fallbacks = {}
    for child in [*getattr(node, "children", []), getattr(node, "true_branch", None), getattr(node, "false_branch", None)]:
        if child is not None:
            without_recoveries(child)
    return node


def episode_context(robots, sim_namespace, scene_id, path="") -> dict:
    """What the tree starts from: every robot under its init.yaml key, and where things are."""
    return {**robots, "sim_namespace": sim_namespace, "scene_namespace": f"/Scene_{scene_id}",
            "scene_id": scene_id, "dataset_path": path}


def solve_task(scene_id, robots, sim_namespace, init, path="", draw=0, scene_seed=None, clean=False,
               recipe="flawless"):
    """Randomize the scene, build the episode's tree and run it. Returns (saved, provenance)."""
    from guide_msgs import srv

    from guide_core.scene.robot_profile import resolve

    robot = next(iter(robots.values()))
    reply = robot.callService(robot.randomize, srv.Randomize.Request(
        id=scene_id, use_seed=scene_seed is not None, seed=int(scene_seed or 0)), timeout_sec=600.0)
    if not reply.success:
        raise RuntimeError(f"Randomize failed: {reply.message}")
    plan = json.loads(reply.message)
    log = robot.node.get_logger()
    # each arm's routes from its own key; clean = none perturbed (the ECM only moves joints)
    masters = [r.motion_episode(scene_id, draw, i, clean=clean) for i, r in enumerate(robots.values())]
    robot.sleep(SETTLE_S)

    rng = np.random.default_rng([masters[0] or 0, scene_id, draw, 7])  # the episode's own draws
    v = variant(recipe)
    miss, stages = planned_misses(v, plan["colour"], rng)
    hold = ecm_hold(resolve(init["robots"]["ecm"]), rng)
    log.info(f"{plan['tasks'][0]} ({plan['giver']} -> {plan['receiver']}, {plan['source_peg']} -> "
             f"{plan['target_peg']}); recipe {recipe}: misses {stages}")
    tree = build_tree(init, plan, miss={**miss, "seed": int(rng.integers(2**31))}, hold=hold)
    if not stages and not v.get("miss"):
        without_recoveries(tree)
    result = tree.execute(episode_context(robots, sim_namespace, scene_id, path))
    saved = result.status == DemoStatus.PERFECT and bool(result.outputs.get("task_success"))
    if not saved:
        log.error(f"[FAILURE] Scene {scene_id}: {result.outputs.get('task_reason') or result.error_message or 'aborted'}")
    return saved, {"recipe": recipe, "misses": stages, "master": masters[0], "giver": plan["giver"],
                   "colour": plan["colour"], "target_peg": plan["target_peg"]}


def cycle(recipes, i) -> str:
    return recipes[i % len(recipes)]


def generate(n, scene_id, robots, sim_namespace, init, path, recipes, seed_base=0, perturb_all=False):
    """Record n episodes; a failed attempt is discarded and the next scene seed drawn. Without perturb_all
    every other episode runs unperturbed (the motion noise only; the recipe's misses stay). Every attempt
    goes to <path>.provenance.jsonl."""
    from guide_msgs import srv

    robot = next(iter(robots.values()))
    log = robot.node.get_logger()
    record = Path(f"{path.rstrip('/')}.provenance.jsonl")
    draws = errors = 0
    try:
        for done in range(n):
            for attempt in range(MAX_ATTEMPTS):
                draws += 1
                seed = seed_base + done * SEED_BLOCK + attempt
                key = {"episode": done, "attempt": attempt, "seed": seed, "draw": draws, "scene_id": scene_id,
                       "clean": not perturb_all and done % 2 == 0}
                log.info(f"--- Episode {done + 1}/{n} | attempt {attempt + 1} | seed {seed} ---")
                try:
                    saved, info = solve_task(scene_id, robots, sim_namespace, init, path, draws, seed,
                                             key["clean"], cycle(recipes, done))
                    errors = 0
                except Exception as e:  # a timed-out sim call costs this attempt; five in a row, the run
                    errors += 1
                    log.error(f"Attempt failed ({errors} in a row): {e}")
                    if errors >= 5:
                        raise
                    saved, info = False, {"error": repr(e)}
                try:  # an attempt that broke off leaves its episode open: drop it (a no-op when none is)
                    robot.callService(robot.stop_recording, srv.StopRecording.Request(id=scene_id, save_episode=False),
                                      timeout_sec=300.0)
                except Exception as e:
                    log.error(f"Cleanup stop_recording failed: {e}")
                with record.open("a") as f:
                    f.write(json.dumps({**key, **info, "saved": saved}) + "\n")
                if saved:
                    break
            else:
                raise RuntimeError(f"episode {done + 1} failed {MAX_ATTEMPTS} times in a row")
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        try:  # saved episodes stay unreadable until finalized
            robot.callService(robot.finalize_recording, srv.FinalizeRecording.Request(id=scene_id), timeout_sec=600.0)
        except Exception as e:
            log.error(f"Finalize failed: {e}")
        _generating.release()


def main():
    from guide_motion.noise import NoiseParams
    from guide_msgs import srv

    from guide_core.types.randomization import zone_plan

    parser = argparse.ArgumentParser(description="peg_transfer demonstration generator.")
    parser.add_argument("--namespace", type=str, default="/Sim_0/Scene_0")
    parser.add_argument("--controller", choices=["oscbf"], default="oscbf")
    parser.add_argument("--motion-noise", nargs="?", const="{}", default=None, metavar="JSON",
                        help=f"perturbed demonstrations: {NOISE} (0.1x cube_stack's), or a JSON object of overrides")
    parser.add_argument("--seed-base", type=int, default=0, help="first scene seed; give parallel solvers disjoint bases")
    parser.add_argument("--perturb-all", action="store_true", help="every episode perturbed, not every other one")
    parser.add_argument("--recipe", default="flawless", metavar="NAME[,NAME...]",
                        help="variant(s) from config/variants/ (flawless, light_fail, retry_mixed), taken in turn")
    args, _ = parser.parse_known_args()
    recipes = args.recipe.split(",")
    for name in recipes:
        variant(name)  # an unknown recipe stops here, not mid-run
    namespace = args.namespace
    sim_namespace = "/" + namespace.strip("/").split("/")[0]
    match = re.search(r"\d+$", namespace.rstrip("/"))
    scene_id = int(match.group()) if match else 0

    init = init_yaml()
    noise = None if args.motion_noise is None else NoiseParams(**{**NOISE, **json.loads(args.motion_noise)})
    robots = make_robots(init, namespace, motion_noise=noise)
    for r in robots.values():
        r.connect()
    time.sleep(5)  # let the connections come up
    # every client on every robot, one per service per node (rmw_cyclonedds reply routing)
    for r in robots.values():
        for attr, (service, kind) in {**SERVICES, "set_prompt": ("set_prompt", "SetPrompt")}.items():
            setattr(r, attr, r.node.create_client(getattr(srv, kind), f"{sim_namespace}/{service}",
                                                  callback_group=r._reentrant_callback_group))
    robot = next(iter(robots.values()))

    def handle(request, response):
        if not _generating.acquire(blocking=False):
            response.success, response.message = False, "Generation is already in progress."
            return response
        n = len(zone_plan(request.zones, request.counts, 1))  # an ungridded scene: free draws
        threading.Thread(target=generate, args=(n, scene_id, robots, sim_namespace, init, request.path, recipes,
                                                args.seed_base, args.perturb_all)).start()
        response.success, response.message = True, f"Started generating {n} demonstrations."
        return response

    robot.generate_demo_service = robot.node.create_service(
        srv.Demonstration, f"{namespace}/generate_demonstration", handle, callback_group=robot._reentrant_callback_group)
    robot.node.get_logger().info("Ready to receive demonstration generation requests.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for r in robots.values():
            r.disconnect()
