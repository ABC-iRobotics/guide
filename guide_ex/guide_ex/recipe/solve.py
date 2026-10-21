"""Run a recipe-built task on its registered scene: per attempt, Randomize (a free draw, or one
arrangement), settle, then load and run the task's config/recipes/Episode.yaml with the context and
values the GUIDE-EX library reads (Plan A contract addition 7, ruling 20). Task-agnostic: compiled tasks
ship no solver. FR3 only: irob_lerobot_ros has no Panda config (Plan C ruling R18).

    python -m guide_ex.recipe.solve --task <task dir> [--namespace /Sim_0/Scene_0] [--episodes 1]
        [--attempts 3] [--dataset ~/dataset/<task>] [--arrangement demo_0 | --arrangements demo_0,demo_1,...]

--arrangements records one episode per listed arrangement (--attempts each); one that fails them all is
recorded from a free draw instead (--attempts more, log "fallback <id> -> free draw").
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

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

    from irob_lerobot_ros.config import ActionType, FR3RobotConfig
    from irob_lerobot_ros.ros2robot import ROS2Robot

    from guide_core.scene.robot_profile import resolve
    from guide_ex.core.states import DemoStatus
    from guide_ex.recipe import load
    from guide_msgs import srv

    task = args.task.expanduser().resolve()
    init = yaml.safe_load((task / "config/init.yaml").read_text())
    ((name, raw),) = init["robots"].items()
    entry = resolve(raw)
    if entry.get("profile") != "fr3":
        raise SystemExit(f"{task}: robot profile {entry.get('profile')!r}: this runner drives the FR3 only (FR3RobotConfig)")
    config = FR3RobotConfig(namespace=f"{args.namespace}/{name}", controller=entry.get("controller", "oscbf"),
                            gripper_action_type=ActionType.JOINT_POSITION)
    config.arm_action_type = ActionType.CARTESIAN_POSE
    config.cameras = {}  # ROS2Robot reads config.cameras; the recorder takes its images inside GUIDE (init.yaml cameras, publish_camera_topics: false)
    robot = ROS2Robot(config=config)
    robot.connect()
    time.sleep(5)  # block_bin's wait for the connections
    sim = "/" + args.namespace.strip("/").split("/")[0]
    for attr, (service, kind) in SERVICES.items():
        setattr(robot, attr, robot.node.create_client(getattr(srv, kind), f"{sim}/{service}",
                                                      callback_group=robot._reentrant_callback_group))
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
        ctx = context(args.namespace, robot, entry, dataset, reply)
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
            robot.disconnect()
    print(f"[solve] {done}/{len(wanted)} episodes in {attempts} attempts, {fallbacks} fallbacks", flush=True)
    return 0 if done == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
