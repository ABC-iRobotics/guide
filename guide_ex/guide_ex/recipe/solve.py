"""Run a recipe-built task on its registered scene: per attempt, Randomize (a free draw, or one
arrangement), settle, then load and run the task's config/recipes/Episode.yaml with the context and
values the GUIDE-EX library reads (Plan A contract addition 7, ruling 20). Task-agnostic: compiled tasks
ship no solver. Every init.yaml robot is built (make_robots); the library recipes drive the first
as `robot`, each is also in the context under its key. No Panda: irob_lerobot_ros has no Panda
config (Plan C ruling R18).

    python -m guide_ex.recipe.solve --task <task dir> [--namespace /Sim_0/Scene_0] [--episodes 1]
        [--attempts 3] [--dataset ~/dataset/<task>] [--arrangement demo_0 | --arrangements demo_0,demo_1,...]

--arrangements records one episode per listed arrangement (--attempts each); one that fails them all is
recorded from a free draw instead (--attempts more, log "fallback <id> -> free draw").

A task with its own solver (peg_transfer, cube_stack, block_bin) is a module of hooks served by serve():
its package names the task (config/ beside it), NOISE (the NoiseParams --motion-noise is merged over),
optional SETTLE_S, and
    slots(plan) -> [{target: key, "*": default}, ...]   where a planned miss may land (planned_misses)
    values(init, plan, miss, v, rng) -> dict           what its config/recipes/ interpolate (rng None: no draws)
    context(plan) -> dict                              optional: more context keys beside the robots
    info(plan) -> dict                                 optional: more provenance keys
with `main = lambda: serve(sys.modules[__name__])`. Its episode rules: every other episode clean (no
motion noise, the misses kept), recoveries stripped but the Contact closes when it plans no miss.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import logging
import re
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from guide_ex.recipe.loader import load, recipes_dir

SETTLE_S = 2.0  # sim s after Randomize: LIBERO's free draws drop movables from 7 cm
REST = ([0.3, 0.0, 0.5], [np.pi, 0.0, 0.0])  # block_bin's rest pose, robot base frame (S2b Ruling 4)
SERVICES = {  # attribute -> (service under /Sim_<n>, guide_msgs.srv type): block_bin's clients
    "randomize": ("Randomize", "Randomize"), "reset": ("Reset", "Randomize"), "pose": ("PoseRequest", "Pose"),
    "collision": ("CollisionRequest", "Collision"), "is_success": ("IsSuccess", "CheckSuccess"),
    "start_recording": ("start_recording", "StartRecording"), "stop_recording": ("stop_recording", "StopRecording"),
    "finalize_recording": ("finalize_recording", "FinalizeRecording"),
}


def context(namespace: str, robot, entry: dict, dataset: str, reply: str) -> dict:
    """The library's context keys for a RecipeScene episode (Plan A contract addition 7), from the
    Randomize reply {task, target, goal, subtasks} and the resolved robot entry. Per goal kind: PutOn / PutIn read
    top, support (a body or a region prim), pick, place; Pull / Turn read joint_body (the part they move) and subtask;
    Push reads top, support (the zone's prim) and subtask."""
    from guide_core.types.geometry import Pose

    try:
        answer = json.loads(reply) if reply else {}
    except ValueError:
        answer = {}
    answer = answer if isinstance(answer, dict) else {}
    goals = answer.get("goal") or []
    top, support = None, None
    for g in goals:  # on / in: [obj, support or region]; on_region / on_zone (a table zone, pushed into): {obj, region}
        if "on" in g or "in" in g:
            top, support = g.get("on") or g["in"]
        for k in ("on_region", "on_zone"):
            if k in g:
                top, support = g[k]["obj"], g[k]["region"]
    # The part a Pull or Turn moves: open / close / turn_on / turn_off {joint, body, value, direction}. A task whose
    # success is "in the drawer" lists the drawer's open goal too, so Pull runs before PutIn.
    joint = next((v for g in goals for k, v in g.items() if k in ("open", "close", "turn_on", "turn_off")), None) or {}
    subtasks = list(answer.get("subtasks") or [])
    # Pull and Turn announce the first label, PutOn and PutIn the rest; Push (an on_zone goal) the only one
    lead = subtasks.pop(0) if (joint or any("on_zone" in g for g in goals)) and subtasks else ""
    scene_id = int(re.search(r"(\d+)$", namespace.rstrip("/")).group(1))
    return {"sim_namespace": "/" + namespace.strip("/").split("/")[0], "scene_namespace": f"/Scene_{scene_id}",
            "scene_id": scene_id, "robot": robot, "robot_prim": f"{entry['path']}/{entry['profile']}_rightfinger",
            "target": answer.get("target"), "top": top, "support": support,
            "pick": subtasks[0] if subtasks else "",
            "goal": next((g["in"][1] for g in goals if "in" in g), None),  # Place's container (block_bin-style tasks)
            "place": subtasks[1] if len(subtasks) > 1 else "", "subtask": lead, "dataset_path": dataset,
            "joint_body": joint.get("body"),
            "rest_pose": Pose.from_numpy(np.array(REST[0]), R.from_euler("xyz", REST[1])), "scene_pose_inv": Pose()}


def recipe_values(entry: dict, init: dict) -> dict:
    """The library's values (Plan A ruling 20): robot.gripper_joint and robot.home from the resolved profile
    entry, episode.camera_hold / hold_seconds from dataset.camera_transition (block_bin's rule)."""
    dataset = init.get("dataset") or {}
    transition = dataset.get("camera_transition") or {}
    hold = transition.get("enabled") is True
    return {"robot": {"gripper_joint": entry["gripper_joint"],
                      "home": {f"{j}.pos": q for j, q in zip(entry["joint_names"], entry["default_joint_states"])}},
            "episode": {"camera_hold": hold,
                        "hold_seconds": float(transition.get("duration", 0.0)) + 1.0 / dataset.get("fps", 10) if hold else 0.0}}


def dataset_path(task: Path, given: str) -> str:
    """--dataset, defaulting to ~/dataset/<task dir name> (the generate-locally rule)."""
    return str(Path(given or f"~/dataset/{task.name}").expanduser())


def episodes_saved(dataset: str) -> int:
    """Episodes on disk under the dataset dir: the recorder nests dataset_<sim>_<scene>_<ts> there and
    LeRobot rewrites meta/info.json on every saved episode. 0 when nothing was ever written."""
    total = 0
    for info in Path(dataset).glob("dataset_*/meta/info.json"):
        try:
            total += int(json.loads(info.read_text()).get("total_episodes", 0))
        except (OSError, ValueError):
            pass
    return total


def draw(robot, request, timeout_sec: float = 600.0) -> tuple[str, str]:
    """Call Randomize: (reply message, why). ``why`` is empty on success, else the failure message
    or the exception (a timeout), so the caller counts a failed attempt instead of ending the run."""
    try:
        reply = robot.callService(robot.randomize, request, timeout_sec=timeout_sec)
    except Exception as e:
        return "", f"Randomize failed: {e!r}"
    return reply.message, "" if reply.success else f"Randomize failed: {reply.message}"


def make_robots(init: dict, namespace: str, **config) -> dict:
    """One unconnected ROS2Robot per init.yaml `robots` entry: the irob_lerobot_ros config class registered
    for its profile (fr3, psm, ecm), id = the key, namespace <namespace>/<key>, the entry's controller.
    A joints-only arm (the ECM) keeps joint actions, the others take Cartesian poses. `config`: more
    fields for every config (e.g. motion_noise)."""
    from irob_lerobot_ros.config import ActionType
    from irob_lerobot_ros.ros2robot import ROS2Robot
    from lerobot.robots import RobotConfig

    from guide_core.scene.robot_profile import resolve

    robots = {}
    for name, raw in init["robots"].items():
        entry = resolve(raw)
        cfg = RobotConfig.get_choice_class(entry.get("profile", "fr3"))(
            id=name, namespace=f"{namespace}/{name}", controller=entry.get("controller", "oscbf"),
            gripper_action_type=ActionType.JOINT_POSITION, **config)
        if not cfg.joints_only:
            cfg.arm_action_type = ActionType.CARTESIAN_POSE
        cfg.cameras = {}  # ROS2Robot reads config.cameras; the recorder takes its images inside GUIDE (init.yaml cameras)
        # ponytail: the PSM/ECM configs lack FR3's directly_publish, which ROS2Robot reads; move it to ROS2RobotConfig
        cfg.directly_publish = getattr(cfg, "directly_publish", False)
        robots[name] = ROS2Robot(config=cfg)
    return robots


def run_episodes(arrangements, attempts: int, try_once, log=print) -> tuple[int, int, int]:
    """One episode per arrangement, ``attempts`` tries each, then ``attempts`` tries from a free draw ("").
    try_once(arrangement) -> bool. Returns (episodes done, tries made, fallbacks used)."""
    done = tries = fallbacks = 0
    for arr in arrangements:
        for start in ([arr, ""] if arr else [""]):
            if start != arr:
                fallbacks += 1
                log(f"[solve] fallback {arr} -> free draw")
            ok = False
            for _ in range(attempts):
                tries += 1
                if ok := try_once(start):
                    break
            if ok:
                break
        done += ok
    return done, tries, fallbacks


MAX_ATTEMPTS = 8  # attempts at one episode before the run stops
SEED_BLOCK = 1000  # episode i draws its layouts from scene seeds seed_base + i * SEED_BLOCK, ... + 1, ...
MOVES = []  # the attempt's arm goals in order, for its provenance (peg_transfer's tools/path_stats.py): log_moves
_generating = threading.Lock()


def log_moves(name, robot):
    """Wrap robot.send_action: each arm goal goes to MOVES (robot, a pose goal's position or a joint goal's
    joints, the move's flags); a jaw-only action does not."""
    send = robot.send_action

    def logged(action, *args, **kwargs):
        if isinstance(action, dict):
            joints = {k: float(v) for k, v in action.items() if k.removesuffix(".pos") in robot.config.arm_joint_names}
            goal = {"joints": joints} if len(joints) == len(robot.config.arm_joint_names) else None
        else:
            goal = {"position": [action.position.x, action.position.y, action.position.z]}
        if goal:
            MOVES.append({"robot": name, **goal, **{k: kwargs[k] for k in ("speed", "perturb", "precise_end", "aim") if k in kwargs}})
        return send(action, *args, **kwargs)
    robot.send_action = logged


def config_dir(task) -> Path:
    """The task package's config/ (task: its solver module)."""
    return recipes_dir(task.__package__, task.__file__).parent


def init_yaml(task) -> dict:
    return yaml.safe_load((config_dir(task) / "init.yaml").read_text())


def variant(task, name) -> dict:
    """config/variants/<name>.yaml: `miss` (fixed misses) or `failures` + `stages` (planned ones)."""
    return yaml.safe_load((config_dir(task) / "variants" / f"{name}.yaml").read_text())


def cycle(recipes, i) -> str:
    return recipes[i % len(recipes)]


def planned_misses(v, slots, rng) -> tuple[dict, list[str]]:
    """One episode's misses: n distinct (stage, slot) cells, n drawn from failures.counts with failures.weights,
    each moving its MissPose targets once. A slot keys each target ({target: key, "*": default}, e.g. the
    cube picked and the one placed on); a cell whose target is already missed at its key is skipped.
    Returns MissPose's `miss` ({target: {at: {key: {axes, r}}}}) and the cells: "stage" with one slot,
    "stage@i" (i from 1) otherwise. A variant without stages: its fixed `miss`, no cells."""
    if "stages" not in v:
        return dict(v.get("miss") or {}), []
    cells = [(stage, i) for stage in v["stages"] for i in range(len(slots))]
    n = int(rng.choice(v["failures"]["counts"], p=v["failures"]["weights"]))
    miss, chosen = {}, []
    for j in rng.permutation(len(cells)):
        if len(chosen) == n:
            break
        stage, i = cells[j]
        targets = v["stages"][stage]
        keys = {t: slots[i].get(t, slots[i].get("*")) for t in targets}
        if any(keys[t] in miss.get(t, {}).get("at", {}) for t in targets):
            continue
        for t, spec in targets.items():
            miss.setdefault(t, {"at": {}})["at"][keys[t]] = dict(spec)
        chosen.append(stage if len(slots) == 1 else f"{stage}@{i + 1}")
    return miss, chosen


def without_recoveries(node, keep=("Contact",)):
    """Drop every recovery (fallback) in the tree but those whose name ends in one of `keep` (a descent
    stalled on contact a millimetre short closes where it stopped: no retry in the data): a failing step
    then ends the episode. A loop's own passes are not recoveries and stay."""
    if getattr(node, "fallbacks", None):
        node.fallbacks = {k: r for k, r in node.fallbacks.items() if r.name.endswith(keep)}
    for child in [*getattr(node, "children", []), *getattr(node, "branches", []), getattr(node, "true_branch", None),
                  getattr(node, "false_branch", None)]:
        if child is not None:
            without_recoveries(child, keep)
    return node


@functools.lru_cache
def _model(profile):
    from guide_motion.model import Model

    return Model.load(profile)


def home_pose(entry) -> dict:
    """The tool tip's pose (base frame, a static_args pose) at a resolved init.yaml robot entry's default
    joints (arm first): a return home as a pose move blends out of the move before and takes the motion noise."""
    m = _model(entry["profile"])
    p, rot = m.ee_pose(np.asarray(entry["default_joint_states"][:len(m.arm_joints)], float))
    return {"position": [float(v) for v in p], "orientation": {"quat": [float(v) for v in R.from_matrix(rot).as_quat()]}}


def build_tree(task, init, plan, miss=None, v=None, rng=None):
    """The task's config/recipes/Episode.yaml as a GUIDE-EX tree, with task.values(init, plan, miss, v, rng)."""
    recipes = recipes_dir(task.__package__, task.__file__)
    return load(recipes / "Episode.yaml", values=task.values(init, plan, miss or {}, v or {}, rng), search_paths=[recipes])


def episode_context(task, robots, sim_namespace, scene_id, plan, path="") -> dict:
    """What the tree starts from: every robot under its init.yaml key, the first also as `robot`, where
    things are, and task.context(plan)."""
    return {**robots, "robot": next(iter(robots.values())), "sim_namespace": sim_namespace,
            "scene_namespace": f"/Scene_{scene_id}", "scene_id": scene_id, "dataset_path": path,
            **(task.context(plan) if hasattr(task, "context") else {})}


def solve_episode(task, scene_id, robots, sim_namespace, init, path="", draw=0, scene_seed=None, clean=False,
                  recipe="flawless", zone=None, zones=None, no_recovery=False):
    """Randomize the scene (`zone`: the target's cell; `zones`: {prim name: cell}), build the episode's tree
    and run it. clean = no route perturbed (the misses stay); no_recovery = every recovery stripped but the
    Contact closes (else so only when the episode plans no miss). Returns (saved, provenance)."""
    from guide_msgs import srv

    from guide_ex.core.states import DemoStatus

    zones = zones or {}
    robot = next(iter(robots.values()))
    reply = robot.callService(robot.randomize, srv.Randomize.Request(
        id=scene_id, use_zone=zone is not None, zone=int(zone or 0), use_seed=scene_seed is not None,
        seed=int(scene_seed or 0), zone_objects=list(zones), object_zones=[int(z) for z in zones.values()]),
        timeout_sec=600.0)
    if not reply.success:
        raise RuntimeError(f"Randomize failed: {reply.message}")
    plan = json.loads(reply.message)
    log = robot.node.get_logger()
    masters = [r.motion_episode(scene_id, draw, i, clean=clean) for i, r in enumerate(robots.values())]  # each arm's own key
    robot.sleep(getattr(task, "SETTLE_S", SETTLE_S))

    rng = np.random.default_rng([masters[0] or 0, scene_id, draw, 7])  # the episode's own draws
    v = variant(task, recipe)
    miss, stages = planned_misses(v, task.slots(plan), rng)
    log.info(f"{plan.get('task', '')} recipe {recipe}: misses {stages or list(miss)}")
    tree = build_tree(task, init, plan, {**miss, "seed": int(rng.integers(2**31))}, v, rng)
    if no_recovery or not miss:
        without_recoveries(tree)
    fatal = logging.Handler()  # the failed node and its composites, innermost first, for the provenance
    fatal.failed = []
    fatal.emit = lambda r: fatal.failed.extend(re.findall(r"\[FATAL\] '(\w+)' failed", r.getMessage()))
    logging.getLogger("guide_ex").addHandler(fatal)
    try:
        result = tree.execute(episode_context(task, robots, sim_namespace, scene_id, plan, path))
    finally:
        logging.getLogger("guide_ex").removeHandler(fatal)
    saved = result.status == DemoStatus.PERFECT and bool(result.outputs.get("task_success"))
    if not saved:
        log.error(f"[FAILURE] Scene {scene_id}: {result.outputs.get('task_reason') or result.error_message or 'aborted'}")
    return saved, {"recipe": recipe, "misses": stages, "master": masters[0],
                   **(task.info(plan) if hasattr(task, "info") else {}), "failed": fatal.failed}


def discard(robot, scene_id):
    """An attempt that broke off leaves its episode open: drop it (a no-op when none is)."""
    from guide_msgs import srv

    try:
        robot.callService(robot.stop_recording, srv.StopRecording.Request(id=scene_id, save_episode=False), timeout_sec=300.0)
    except Exception as e:
        robot.node.get_logger().error(f"Cleanup stop_recording failed: {e}")


def finalize(robot, scene_id):
    """Saved episodes stay unreadable until the dataset is finalized; then the generation lock is released."""
    from guide_msgs import srv

    try:
        robot.callService(robot.finalize_recording, srv.FinalizeRecording.Request(id=scene_id), timeout_sec=600.0)
    except Exception as e:
        robot.node.get_logger().error(f"Finalize failed: {e}")
    finally:
        _generating.release()


def _attempts(task, scene_id, robots, sim_namespace, init, path):
    """attempt(key, **solve_episode keywords) -> saved: one attempt at scene seed key["seed"], its open
    episode discarded and its line ({**key, draw, scene_id, provenance, saved, moves}) added to
    <path>.provenance.jsonl. Five attempts raising in a row (a timed-out sim call each) end the run."""
    robot = next(iter(robots.values()))
    log = robot.node.get_logger()
    record = Path(f"{path.rstrip('/')}.provenance.jsonl").expanduser()  # the recorder expands a request's ~ too
    count = {"draws": 0, "errors": 0}

    def attempt(key, **kw):
        count["draws"] += 1
        MOVES.clear()
        try:
            saved, info = solve_episode(task, scene_id, robots, sim_namespace, init, path, count["draws"], key["seed"], **kw)
            count["errors"] = 0
        except Exception as e:
            count["errors"] += 1
            log.error(f"Attempt failed ({count['errors']} in a row): {e}")
            if count["errors"] >= 5:
                raise
            saved, info = False, {"error": repr(e)}
        discard(robot, scene_id)
        with record.open("a") as f:
            f.write(json.dumps({**key, "draw": count["draws"], "scene_id": scene_id, **info, "saved": saved,
                                "moves": MOVES}) + "\n")
        return saved
    return attempt


def generate(task, zones, scene_id, robots, sim_namespace, init, path, recipes, seed_base=0, perturb_all=False):
    """Record one episode per `zones` entry (zone_plan's: a cell, or None for a free draw); a failed attempt
    is discarded and the next scene seed (seed_base + episode * SEED_BLOCK + attempt) drawn. Without
    perturb_all every other episode runs unperturbed (the motion noise only; the recipe's misses stay)."""
    robot = next(iter(robots.values()))
    log = robot.node.get_logger()
    attempt = _attempts(task, scene_id, robots, sim_namespace, init, path)
    discard(robot, scene_id)  # so does a solver killed mid-episode: its frames would open this run's first episode
    try:
        for done, zone in enumerate(zones):
            for n in range(MAX_ATTEMPTS):
                key = {"episode": done, "attempt": n, "seed": seed_base + done * SEED_BLOCK + n,
                       "clean": not perturb_all and done % 2 == 0, **({} if zone is None else {"zone": zone})}
                log.info(f"--- Episode {done + 1}/{len(zones)} | attempt {n + 1} | seed {key['seed']} ---")
                if attempt(key, clean=key["clean"], recipe=cycle(recipes, done), zone=zone):
                    break
            else:
                raise RuntimeError(f"episode {done + 1} failed {MAX_ATTEMPTS} times in a row")
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        finalize(robot, scene_id)


def generate_scenarios(task, scenarios, layouts, trajectories, scene_id, robots, sim_namespace, init, path, recipes,
                       seed_base=0, perturb_all=False):
    """Scenario mode of the Demonstration service: per scenario ({prim name: zone}), `layouts` scene seeds,
    each recorded `trajectories` times -- first unperturbed (perturb_all: perturbed too), then perturbed.
    A seed whose first run fails is skipped for the scenario's next one, a later run that fails is drawn
    again; no run recovers but by its Contact closes. Every attempt goes to <path>.provenance.jsonl."""
    robot = next(iter(robots.values()))
    log = robot.node.get_logger()
    attempt = _attempts(task, scene_id, robots, sim_namespace, init, path)
    tries = itertools.count()

    def run(s, layout, seed, t):
        return attempt({"scenario": s, "layout": layout, "seed": seed, "trajectory": t, "zones": scenarios[s]},
                       clean=t == 0 and not perturb_all, recipe=cycle(recipes, next(tries)), zones=scenarios[s],
                       no_recovery=True)

    try:
        for s in range(len(scenarios)):
            seed = seed_base + s * SEED_BLOCK
            for layout in range(layouts):
                for _ in range(MAX_ATTEMPTS):
                    log.info(f"--- scenario {s} layout {layout}: seed {seed}, unperturbed ---")
                    seed += 1
                    if run(s, layout, seed - 1, 0):
                        break
                else:
                    raise RuntimeError(f"scenario {s}: {MAX_ATTEMPTS} seeds in a row failed unperturbed")
                for t in range(1, trajectories):
                    for n in range(MAX_ATTEMPTS):
                        log.info(f"--- scenario {s} layout {layout} (seed {seed - 1}) trajectory {t}, try {n + 1} ---")
                        if run(s, layout, seed - 1, t):
                            break
                    else:
                        log.error(f"scenario {s} layout {layout} trajectory {t}: {MAX_ATTEMPTS} failures, skipped")
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        finalize(robot, scene_id)


def serve(task):
    """A task solver's main: the robots of its init.yaml and the Demonstration service on
    <namespace>/generate_demonstration (zone mode: zone_plan over the zone grid of its config/randomize.yaml;
    scenario mode: request.scenarios)."""
    from guide_motion.noise import NoiseParams
    from guide_msgs import srv

    from guide_core.types.randomization import zone_plan
    from guide_core.types.randomization.replicator_guide import zone_grid

    noise_defaults = getattr(task, "NOISE", {})
    parser = argparse.ArgumentParser(description=f"{task.__package__} demonstration generator.")
    parser.add_argument("--namespace", type=str, default="/Sim_0/Scene_0")
    parser.add_argument("--controller", choices=["oscbf"], default="oscbf")
    parser.add_argument("--motion-noise", nargs="?", const="{}", default=None, metavar="JSON",
                        help=f"perturbed demonstrations: {noise_defaults}, or a JSON object of overrides")
    parser.add_argument("--seed-base", type=int, default=0, help="first scene seed; give parallel solvers disjoint bases")
    parser.add_argument("--perturb-all", action="store_true", help="every episode perturbed, not every other one")
    parser.add_argument("--recipe", default="flawless", metavar="NAME[,NAME...]",
                        help="variant(s) from config/variants/, taken in turn")
    args, _ = parser.parse_known_args()
    recipes = args.recipe.split(",")
    for name in recipes:
        variant(task, name)  # an unknown recipe stops here, not mid-run
    namespace = args.namespace
    sim_namespace = "/" + namespace.strip("/").split("/")[0]
    match = re.search(r"\d+$", namespace.rstrip("/"))
    scene_id = int(match.group()) if match else 0

    init = init_yaml(task)
    noise = None if args.motion_noise is None else NoiseParams(**{**noise_defaults, **json.loads(args.motion_noise)})
    robots = make_robots(init, namespace, motion_noise=noise)
    for name, r in robots.items():
        r.connect()
        log_moves(name, r)
    time.sleep(5)  # let the connections come up
    # every client on every robot, one per service per node (rmw_cyclonedds reply routing)
    for r in robots.values():
        for attr, (service, kind) in {**SERVICES, "set_prompt": ("set_prompt", "SetPrompt")}.items():
            setattr(r, attr, r.node.create_client(getattr(srv, kind), f"{sim_namespace}/{service}",
                                                  callback_group=r._reentrant_callback_group))
    robot = next(iter(robots.values()))
    grid = zone_grid(config_dir(task) / "randomize.yaml")

    def handle(request, response):
        if not _generating.acquire(blocking=False):
            response.success, response.message = False, "Generation is already in progress."
            return response
        common = (scene_id, robots, sim_namespace, init, request.path, recipes, args.seed_base, args.perturb_all)
        if request.scenarios:
            refusal = ("Scenario mode needs layouts >= 1 and trajectories >= 1." if min(request.layouts, request.trajectories) < 1
                       else "Perturbed trajectories need the solver's --motion-noise."
                       if (request.trajectories > 1 or args.perturb_all) and noise is None else None)
            if refusal:
                _generating.release()
                response.success, response.message = False, refusal
                return response
            scenarios = [dict(zip(s.zone_objects, s.object_zones)) for s in request.scenarios]
            n = len(scenarios) * request.layouts * request.trajectories
            run = (generate_scenarios, (task, scenarios, request.layouts, request.trajectories, *common))
        else:  # zones: [] = free draws; [-1] = every zone of the grid
            zones = zone_plan(request.zones, request.counts, grid.num_zones if grid else 1)
            n, run = len(zones), (generate, (task, zones, *common))
        threading.Thread(target=run[0], args=run[1]).start()
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--namespace", default="/Sim_0/Scene_0")
    ap.add_argument("--episodes", type=int, default=1, help="successful episodes wanted")
    ap.add_argument("--attempts", type=int, default=3, help="attempts allowed per episode")
    ap.add_argument("--dataset", default="", help="dataset directory (default: ~/dataset/<task dir name>)")
    ap.add_argument("--arrangement", default="", help="config/arrangements/<id> to start from (default: a free draw)")
    ap.add_argument("--arrangements", default="", help="comma list of arrangement ids, one episode each (overrides --episodes)")
    args = ap.parse_args(argv)

    from guide_core.scene.robot_profile import resolve
    from guide_ex.core.states import DemoStatus
    from guide_msgs import srv

    task = args.task.expanduser().resolve()
    init = yaml.safe_load((task / "config/init.yaml").read_text())
    robots = make_robots(init, args.namespace)
    entry = resolve(next(iter(init["robots"].values())))
    robot = next(iter(robots.values()))
    for r in robots.values():
        r.connect()
    time.sleep(5)  # block_bin's wait for the connections
    sim = "/" + args.namespace.strip("/").split("/")[0]
    for r in robots.values():  # one client per service per node (rmw_cyclonedds reply routing)
        for attr, (service, kind) in SERVICES.items():
            setattr(r, attr, r.node.create_client(getattr(srv, kind), f"{sim}/{service}",
                                                  callback_group=r._reentrant_callback_group))
    recipes = task / "config" / "recipes"
    values = recipe_values(entry, init)
    dataset = dataset_path(task, args.dataset)
    scene_id = context(args.namespace, robot, entry, dataset, "")["scene_id"]  # the one parse of the namespace
    log = robot.node.get_logger().info
    count = 0

    def try_once(arrangement: str) -> bool:
        nonlocal count
        count += 1
        # 600 s: a Randomize on a recipe scene runs Replicator draws and settles the physics
        reply, why = draw(robot, srv.Randomize.Request(id=scene_id, arrangement=arrangement))
        if why:
            log(f"[solve] attempt {count} ({arrangement or 'free'}): FAILURE {why}")
            return False
        robot.sleep(SETTLE_S)
        ctx = {**robots, **context(args.namespace, robot, entry, dataset, reply)}
        saved = episodes_saved(dataset)
        try:
            result = load(recipes / "Episode.yaml", values=values, search_paths=[recipes]).execute(ctx)
            ok = result.status == DemoStatus.PERFECT and bool(result.outputs.get("task_success"))  # as block_bin's solver
            why = result.outputs.get("task_reason") or result.error_message
        except Exception as e:  # a failed attempt, not a failed run (block_bin's generator)
            ok, why = False, repr(e)
        if ok and episodes_saved(dataset) <= saved:  # Plan C Task 18: a solved episode the recorder never wrote
            ok, why = False, "solved, but no episode was written to the dataset"
        log(f"[solve] attempt {count} ({arrangement or 'free'}): {'SUCCESS' if ok else f'FAILURE {why}'}")
        # an attempt that broke off leaves its episode open: drop it (a no-op when nothing is open)
        try:  # a stalled recorder costs this attempt, not the run (block_bin's solver)
            robot.callService(robot.stop_recording, srv.StopRecording.Request(id=scene_id, save_episode=False), timeout_sec=300.0)
        except Exception as e:
            robot.node.get_logger().error(f"Cleanup stop_recording failed: {e}")
        return ok

    wanted = [a for a in args.arrangements.split(",") if a] or [args.arrangement] * args.episodes
    try:
        done, attempts, fallbacks = run_episodes(wanted, args.attempts, try_once, log)
    finally:
        try:  # saved episodes stay unreadable until finalized
            robot.callService(robot.finalize_recording, srv.FinalizeRecording.Request(id=scene_id), timeout_sec=300.0)
        except Exception as e:
            robot.node.get_logger().error(f"Finalize failed: {e}")
        finally:
            for r in robots.values():
                r.disconnect()
    print(f"[solve] {done}/{len(wanted)} episodes in {attempts} attempts, {fallbacks} fallbacks", flush=True)
    return 0 if done == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
