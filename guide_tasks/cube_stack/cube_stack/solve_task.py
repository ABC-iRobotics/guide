"""Stack the cubes: the GUIDE-EX tree that builds the tower, recorded as one episode.

The scene draws the tower's order; the tree is built for it each episode, its layers as
GUIDE-EX means them -- the procedure is the long-horizon workflow, each placement one
TASK with a success criterion, the pick and the place its SUBTASKs. Every TASK and
SUBTASK announces its prompt at its level, so the dataset carries all three: the
procedure beside every frame, then the task and the subtask each frame was part of. The
procedure is never a frame's task: after the last placement a closing TASK takes over.

    StackingDemonstration (PROCEDURE "Stack the cubes.", condition: not done)
      Unclutch, LocateScene, AnnounceFirst (task + subtask), StartEpisode,
      MeasureTower, TowerHeight
      -> BuildTower (PROCEDURE, loop until the tower is done)
           NextCube (TASK, condition: done?)
             MeasureTower, TowerHeight (ChainLength)   -> done, built
             done -> Finish (TASK "Return home.")
                       AnnounceFinish (task + subtask)
                       GoHome (SUBTASK "Return home."): MoveHome, WaitForTower
                       CheckSuccess, StopEpisode (saved only on success)
             else -> PutOn (TASK "Put the red cube on the blue cube.")
                       NextTop, NextSupport, NextTask, ... (GetItem: order[built], ...)
                       AnnounceTask, LocateCube (SEQUENCE)
                       Pick (SUBTASK "Pick up the red cube.")
                       Place (SUBTASK "Place it on the blue cube.")
                         CarryToSupport, Release (SEQUENCE)

A loop of TASKs is the procedure's own work, so BuildTower is a PROCEDURE-level branch of
the root: a composite's children sit strictly below it, branches up to its level.

Recovery routes, for the failures most likely in this task (most likely first):

1. Grasp misses (fingers empty after the lift; block_bin's 5 failed attempts in 300
   were all this). Pick checks the fingers; on a miss PutOn's Regrasp opens, measures
   the cube again (the attempt may have nudged it) and grasps the other pair of faces.
   The hard_fail recipe (--recipe, config/variants/) causes this on purpose, and the place's
   twin (a release off the support, the cube falls, route 4 picks it again); scenario mode,
   which runs without recoveries, then keeps this route.
2. An arm motion fails: no path or an aborted trajectory (under OSCBF, the default, only
   these); under controller: moveit also a straight-line path MoveIt can only partly
   compute (~1% of block_bin's moves, most near the robot base). MoveToCartesianPose
   refuses that last one before moving -- executed, it stopped grasps
   short and left the arm in a self-collision nothing could plan out of. Every move has
   a detour through the home joints before it is tried again: from there a straight
   descent is feasible over the whole spawn region (mapped with MoveIt), whereas a
   rest *pose* lets the planner leave the arm in any posture.
3. The Place subtask fails: the cube slips out (CarryToSupport checks the fingers above
   the tower), or the arm finds no way there or down onto it. RepickDropped -- its own
   subtask, "Set it down." -- sets it down where it was picked (opening
   wherever the arm is would drop it from height onto the tower) and resumes at Pick,
   from wherever the cube is.
4. A placed cube does not stay, or the tower is knocked. Nothing in PutOn trusts a
   placement: each loop pass measures the whole tower and rebuilds from the lowest
   layer that is out of place, within REBUILDS extra passes.
5. Anything else -- a cube knocked off the table, retries used up: the tree fails, the
   episode is discarded, and generation draws a new layout and tries again, up to
   MAX_ATTEMPTS times before it stops the run.
(Cubes spawned too close to grasp, or too close to the robot to reach down to, are
ruled out up front: see Scene.randomize and config/randomize.yaml.)
"""

import argparse
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from cube_stack.scene import CUBE, STACKED
from irob_lerobot_ros.config import ActionType, FR3RobotConfig
from irob_lerobot_ros.ros2robot import ROS2Robot

# pyrefly: ignore [missing-import]
from scipy.spatial.transform import Rotation as R
import yaml

from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import zone_plan
from guide_core.types.randomization.replicator_guide import zone_grid
from guide_ex.core.states import DemoStatus
from guide_ex.recipe import load, recipes_dir, robot_values
from guide_motion.noise import NoiseParams
from guide_msgs import srv

REBUILDS = 3  # loop passes beyond one per layer, for placements that did not hold
# Attempts at one episode before the run stops: past this something is stuck (e.g. an
# arm MoveIt will not plan from), and every further attempt would fail the same way.
MAX_ATTEMPTS = 8


def variant(name) -> dict:
    """config/variants/<name>.yaml: a stacking recipe's misses (guide_ex MissPose's `miss` per target:
    pre_grasp, grasp, pre_place, place) and tower tolerance -- flawless, light_fail, hard_fail."""
    return yaml.safe_load((recipes_dir("cube_stack", __file__).parent / "variants" / f"{name}.yaml").read_text())


PICK_TARGETS = ("pre_grasp", "grasp")  # MissPose targets keyed on the cube picked; the rest on its support


def planned_misses(v, order, rng) -> tuple[dict, list[str]]:
    """A retry variant's misses for one episode (config/variants/retry.yaml): n failures, n drawn from
    failures.counts with failures.weights, at distinct (stage, placement) cells. Placement i puts
    order[i] on order[i - 1]; a stage's pick targets are keyed on order[i], its place targets on
    order[i - 1]. A cell whose target is already missed at that key is skipped. Returns MissPose's
    `miss` ({target: {at: {key: {axes, r}}}}) and the cells, e.g. ["fall@2"]."""
    cells = [(stage, i) for stage in v["stages"] for i in range(1, len(order))]
    n = int(rng.choice(v["failures"]["counts"], p=v["failures"]["weights"]))
    miss, chosen = {}, []
    for j in rng.permutation(len(cells)):
        if len(chosen) == n:
            break
        stage, i = cells[j]
        targets = v["stages"][stage]
        keys = {t: order[i] if t in PICK_TARGETS else order[i - 1] for t in targets}
        if any(keys[t] in miss.get(t, {}).get("at", {}) for t in targets):
            continue
        for t, spec in targets.items():
            miss.setdefault(t, {"at": {}})["at"][keys[t]] = dict(spec)
        chosen.append(f"{stage}@{i}")
    return miss, chosen


def recipe_values(robot, plan, miss=None, stacked=None, tilt=0.0) -> dict:
    """What config/recipes/ interpolate: the robot's gripper joint and home, the drawn plan's
    prompts, the loop's passes (one more per placement a miss may undo) and the tower tolerance (default
    the scene's success check's), the cube edge, the misses (a variant's, with their seed) and the grasp
    tilt (rad)."""
    miss = miss or {}
    place = miss.get("place") or {}
    undone = len(place["at"]) if "at" in place else (len(plan["order"]) - 1) * bool(place)  # placements a miss undoes
    return {
        "robot": robot_values(robot.config, recipes_dir("cube_stack", __file__).parent / "init.yaml"),
        "plan": plan,
        "tower": {"max_loops": len(plan["order"]) + REBUILDS + undone,
                  "stacked": list(stacked or STACKED)},
        "cube": {"edge": CUBE},
        "miss": miss,
        "grasp": {"tilt": float(tilt)},
    }


def load_recipe(name, robot, plan, **variation):
    """config/recipes/<name>.yaml as a GUIDE-EX tree, for this robot and plan (variation: recipe_values')."""
    recipes = recipes_dir("cube_stack", __file__)
    return load(recipes / f"{name}.yaml", values=recipe_values(robot, plan, **variation), search_paths=[recipes])


def build_tree(robot, plan, **variation):
    """The episode's tree, layered as GUIDE-EX means the layers (see the module docstring)."""
    return load_recipe("Episode", robot, plan, **variation)


def without_recoveries(node, keep=()):
    """Drop every recovery (fallback) in the tree but those of the children named in `keep`: a failing
    step then ends the episode. The tower's rebuild passes (route 4) are the loop's own work and stay."""
    if getattr(node, "fallbacks", None):
        node.fallbacks = {k: v for k, v in node.fallbacks.items() if k in keep}
    for child in [*getattr(node, "children", []), getattr(node, "true_branch", None), getattr(node, "false_branch", None)]:
        if child is not None:
            without_recoveries(child, keep)
    return node


def episode_context(robot, sim_namespace, scene_id, plan, path=""):
    """What the tree starts from: the robot, where things are, and the drawn plan with its
    prompts at every layer (scene.plan)."""
    return {
        "robot": robot,
        "sim_namespace": sim_namespace,
        "scene_namespace": f"/Scene_{scene_id}",
        "scene_id": scene_id,
        # The frame the tree works in: the robot's base, as every pose GUIDE and the robot
        # exchange is (S2b Ruling 4), so LocateScene measures the identity.
        "scene_path": "/fr3/fr3_link0",
        "robot_prim": "/fr3/fr3_rightfinger",
        # Where HOME puts the gripper: pointing down, 50 cm over the scene origin, which is
        # 0.3 m in front of the base (fr3_link0 sits at (-0.3, 0, 0) of the scene).
        "rest_pose": Pose(
            position=Point([0.3, 0.0, 0.5]),
            orientation=Rotation(R.from_euler("xyz", [np.pi, 0.0, 0.0])),
        ),
        "dataset_path": path,
        "order": plan["order"],
        "tasks": plan["tasks"],
        "picks": plan["subtasks"]["pick"],
        "places": plan["subtasks"]["place"],
        "set_downs": plan["subtasks"]["set_down"],
        "finish": plan["subtasks"]["finish"],
    }


def solve_task(scene_id, robot, sim_namespace, zone=None, path="", draw=0, scene_seed=None, zones=None,
               clean=False, no_recovery=False, recipe="flawless", tilt_deg=0.0):
    """Randomize the scene, build the episode's tree and run it. True if saved. `scene_seed` pins the
    layout (Randomize use_seed), so draws of the motion noise can be compared on one scene; `zones`
    ({cube prim name: zone}) places each cube in its own cell; clean = no route perturbed; no_recovery =
    a failing step ends the episode instead of running its recovery (Regrasp stays with misses on);
    recipe: the stacking variant (config/variants/; a clean run is flawless); tilt_deg: the episode's
    grasp tilt, uniform in +-tilt_deg, drawn from the trajectory seed and key."""
    zones = zones or {}
    reply = robot.callService(
        robot.randomize,
        srv.Randomize.Request(id=scene_id, use_zone=zone is not None, zone=int(zone or 0),
                              use_seed=scene_seed is not None, seed=int(scene_seed or 0),
                              zone_objects=list(zones), object_zones=[int(z) for z in zones.values()]),
    )
    if not reply.success:
        raise RuntimeError(f"Randomize failed: {reply.message}")
    plan = json.loads(reply.message)
    log = robot.node.get_logger()
    log.info(f"Task: {plan['task']} Order (bottom first): {plan['order']}")
    master = robot.motion_episode(scene_id, draw, clean=clean)  # the trajectories' own seed, apart from the scene's
    if master is not None:
        log.info(f"Motion noise: trajectory master seed {master}, key (scene {scene_id}, draw {draw})")

    # Let the layout settle (the bins drop onto the floor) before solving.
    time.sleep(5)

    # The episode's own draws, from the trajectory seed and key: the grasp tilt and the misses' seed.
    rng = np.random.default_rng(None if master is None else [master, scene_id, draw, 7])
    tilt = np.radians(rng.uniform(-tilt_deg, tilt_deg)) if tilt_deg else 0.0
    v = variant("flawless" if clean else recipe)
    cells = []
    if "stages" in v:  # a retry variant: this episode's failures, drawn per (stage, placement)
        planned, cells = planned_misses(v, plan["order"], rng)
        v = {**v, "miss": planned}
    miss = {**v["miss"], "seed": int(rng.integers(2**31))}
    log.info(f"Recipe {'flawless' if clean else recipe}: misses {v['miss']}{f' (failures {cells})' if cells else ''}, "
             f"grasp tilt {np.degrees(tilt):+.1f} deg")
    tree = build_tree(robot, plan, miss=miss, stacked=v.get("tower", {}).get("stacked"), tilt=tilt)
    if no_recovery:
        without_recoveries(tree, keep=("Pick",) if v["miss"] else ())
    result = tree.execute(episode_context(robot, sim_namespace, scene_id, plan, path))

    if result.status == DemoStatus.PERFECT and result.outputs.get("task_success"):
        log.info(f"\033[92m[SUCCESS] Scene {scene_id}: tower built.\033[0m")
        return True
    reason = result.outputs.get("task_reason") or result.error_message or "sequence aborted"
    log.error(f"\033[91m[FAILURE] Scene {scene_id}: {reason}\033[0m")
    return False


_generating = threading.Lock()
SEED_BLOCK = 1000  # scenario i draws its layouts from scene seeds seed_base + i * SEED_BLOCK, ... + 1, ...


def discard_open_episode(robot, scene_id):
    """Route 5: an attempt that broke off leaves its episode open; drop it (a no-op when none is)."""
    try:
        robot.callService(robot.stop_recording, srv.StopRecording.Request(id=scene_id, save_episode=False),
                          timeout_sec=300.0)
    except TimeoutError as e:
        robot.node.get_logger().error(f"Cleanup stop_recording timed out: {e}")


def finalize(robot, scene_id):
    """Saved episodes are unreadable until the dataset is finalized."""
    try:
        robot.callService(robot.finalize_recording, srv.FinalizeRecording.Request(id=scene_id), timeout_sec=600.0)
        robot.node.get_logger().info("Recording dataset finalized.")
    except Exception as e:
        robot.node.get_logger().error(f"Finalize failed: {e}")


def generate_scenarios(scenarios, layouts, trajectories, scene_id, robot, sim_namespace, path="", seed_base=0,
                       variation=None):
    """Scenario mode of the Demonstration service: per scenario ({cube: zone}), `layouts` scene seeds,
    each recorded `trajectories` times -- first unperturbed, then perturbed. A seed whose unperturbed
    run fails is skipped for the scenario's next one, a perturbed run that fails is drawn again; no run
    recovers. Every attempt is logged to <path>.provenance.jsonl ("saved" ones in dataset order)."""
    log = robot.node.get_logger()
    record = Path(f"{path.rstrip('/')}.provenance.jsonl")
    draws = errors = 0

    def attempt(**key):
        nonlocal draws, errors
        draws += 1
        try:
            saved = solve_task(scene_id, robot, sim_namespace, path=path, draw=draws, scene_seed=key["seed"],
                               zones=scenarios[key["scenario"]], clean=key["trajectory"] == 0, no_recovery=True,
                               **cycle(variation, draws))
            errors = 0
        except Exception as e:  # a timed-out sim call costs this attempt; five in a row, the run
            errors += 1
            log.error(f"Attempt failed ({errors} in a row): {e}")
            if errors >= 5:
                raise
            saved = False
        discard_open_episode(robot, scene_id)
        with record.open("a") as f:
            f.write(json.dumps({**key, "zones": scenarios[key["scenario"]], "draw": draws, "scene_id": scene_id,
                                "master": robot._noise.master if robot._noise else None, "saved": saved}) + "\n")
        return saved

    try:
        for s in range(len(scenarios)):
            seed = seed_base + s * SEED_BLOCK
            for layout in range(layouts):
                for _ in range(MAX_ATTEMPTS):
                    log.info(f"--- scenario {s} layout {layout}: seed {seed}, unperturbed ---")
                    ok = attempt(scenario=s, layout=layout, seed=seed, trajectory=0)
                    seed += 1
                    if ok:
                        break
                else:
                    raise RuntimeError(f"scenario {s}: {MAX_ATTEMPTS} seeds in a row failed unperturbed")
                for t in range(1, trajectories):
                    for n in range(MAX_ATTEMPTS):
                        log.info(f"--- scenario {s} layout {layout} (seed {seed - 1}) trajectory {t}, try {n + 1} ---")
                        if attempt(scenario=s, layout=layout, seed=seed - 1, trajectory=t):
                            break
                    else:
                        log.error(f"scenario {s} layout {layout} trajectory {t}: {MAX_ATTEMPTS} failures, skipped")
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        finalize(robot, scene_id)
        _generating.release()


def cycle(variation, i) -> dict:
    """solve_task's keywords from the CLI's variation for episode (or draw) i: its recipes in turn."""
    v = dict(variation or {})
    recipes = v.pop("recipes", None)
    return {**v, **({"recipe": recipes[i % len(recipes)]} if recipes else {})}


def generate(plan, scene_id, robot, sim_namespace, path="", variation=None):
    """Record one episode per `plan` entry; a failed attempt is discarded and redrawn."""
    log = robot.node.get_logger()
    try:
        done = attempts = errors = draws = 0
        while done < len(plan):
            if attempts == MAX_ATTEMPTS:
                raise RuntimeError(f"episode {done + 1} failed {attempts} times in a row")
            attempts += 1
            draws += 1
            log.info(f"--- Episode {done + 1}/{len(plan)} | attempt {attempts} ---")
            try:
                saved = solve_task(scene_id, robot, sim_namespace, zone=plan[done], path=path, draw=draws,
                                   **cycle(variation, done))
                errors = 0
            except Exception as e:
                # A timed-out sim call costs this attempt; five in a row, the run.
                errors += 1
                log.error(f"Attempt failed ({errors} in a row): {e}")
                if errors >= 5:
                    raise
                saved = False

            discard_open_episode(robot, scene_id)

            if saved:
                done, attempts = done + 1, 0
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        finalize(robot, scene_id)
        _generating.release()


def main():
    parser = argparse.ArgumentParser(description="cube_stack demonstration generator.")
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--controller", choices=["moveit", "oscbf"], default="oscbf")
    parser.add_argument(
        "--motion-noise", nargs="?", const="{}", default=None, metavar="JSON",
        help="perturbed demonstrations (controller: oscbf): guide_motion.noise.NoiseParams' defaults, "
             'or a JSON object of overrides, e.g. \'{"sigma": 0.008, "seed": 7}\'',
    )
    parser.add_argument("--seed-base", type=int, default=0,
                        help="first scene seed of scenario mode; give parallel solvers disjoint bases")
    parser.add_argument("--recipe", default="flawless", metavar="NAME[,NAME...]",
                        help="stacking recipe(s) from config/variants/ (flawless, light_fail, hard_fail), "
                             "taken in turn episode by episode")
    parser.add_argument("--grasp-tilt", type=float, default=0.0, metavar="DEG",
                        help="each episode's grasp tilt about the fingers' free axis, uniform in +-DEG")
    args, _ = parser.parse_known_args()
    variation = {"recipes": args.recipe.split(","), "tilt_deg": args.grasp_tilt}
    for name in variation["recipes"]:
        variant(name)  # an unknown recipe stops here, not mid-run
    namespace = args.namespace  # e.g. /Sim_0/Scene_0
    sim_namespace = "/" + namespace.split("/")[1]
    match = re.search(r"\d+$", namespace.split("/")[-1])
    scene_id = int(match.group()) if match else 0

    config = FR3RobotConfig(  # poses in its base frame, fr3_link0 (the default; S2b Ruling 4)
        namespace=f"{namespace}/franka",
        planner_id="BiESTkConfigDefault",
        fallback_planner_id="PRMstarkConfigDefault",
        max_velocity=1.0,
        max_acceleration=1.0,
        gripper_action_type=ActionType.JOINT_POSITION,
        controller=args.controller,
        motion_noise=None if args.motion_noise is None else NoiseParams(**json.loads(args.motion_noise)),
    )
    config.arm_action_type = ActionType.CARTESIAN_POSE
    config.cameras = {}  # the recorder captures images sim-side; the solver needs none
    robot = ROS2Robot(config=config)
    robot.connect()
    time.sleep(5)  # let the connections come up

    # Every client up front, one per service, under the attribute the nodes look for:
    # a second client on one service on one node breaks rmw_cyclonedds reply routing.
    for attr, srv_type, name in [
        ("randomize", srv.Randomize, "Randomize"),
        ("pose", srv.Pose, "PoseRequest"),
        ("collision", srv.Collision, "CollisionRequest"),
        ("is_success", srv.CheckSuccess, "IsSuccess"),
        ("start_recording", srv.StartRecording, "start_recording"),
        ("stop_recording", srv.StopRecording, "stop_recording"),
        ("set_prompt", srv.SetPrompt, "set_prompt"),
        ("finalize_recording", srv.FinalizeRecording, "finalize_recording"),
    ]:
        client = robot.node.create_client(
            srv_type, f"{sim_namespace}/{name}", callback_group=robot._reentrant_callback_group
        )
        setattr(robot, attr, client)

    def handle(request, response):
        if not _generating.acquire(blocking=False):
            response.success, response.message = False, "Generation is already in progress."
            return response
        if request.scenarios:
            refusal = ("Scenario mode needs layouts >= 1 and trajectories >= 1." if min(request.layouts, request.trajectories) < 1
                       else "Perturbed trajectories need the solver's --motion-noise." if request.trajectories > 1
                       and robot.config.motion_noise is None else None)
            if refusal:
                _generating.release()
                response.success, response.message = False, refusal
                return response
            scenarios = [dict(zip(s.zone_objects, s.object_zones)) for s in request.scenarios]
            threading.Thread(target=generate_scenarios, args=(
                scenarios, request.layouts, request.trajectories, scene_id, robot, sim_namespace, request.path,
                args.seed_base, variation)).start()
            n = len(scenarios) * request.layouts * request.trajectories
            response.success, response.message = True, f"Started generating {n} demonstrations."
            return response
        # zones: [] = free draws; [-1] = every zone of the starting cube's grid.
        grid = zone_grid(
            Path(get_package_share_directory("cube_stack")) / "config" / "randomize.yaml"
        )
        plan = zone_plan(request.zones, request.counts, grid.num_zones if grid else 1)
        threading.Thread(
            target=generate, args=(plan, scene_id, robot, sim_namespace, request.path, variation)
        ).start()
        response.success, response.message = True, f"Started generating {len(plan)} demonstrations."
        return response

    robot.generate_demo_service = robot.node.create_service(
        srv.Demonstration,
        f"{namespace}/generate_demonstration",
        handle,
        callback_group=robot._reentrant_callback_group,
    )
    robot.node.get_logger().info("Ready to receive demonstration generation requests.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
