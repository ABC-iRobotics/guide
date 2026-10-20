"""Run a recipe-built task on its registered scene: per attempt, Randomize (a free draw, or one
arrangement), settle, then load and run the task's config/recipes/Episode.yaml with the context and
values the GUIDE-EX library reads (Plan A contract addition 7, ruling 20). Task-agnostic: compiled tasks
ship no solver. FR3 only: irob_lerobot_ros has no Panda config (Plan C ruling R18).

    python -m guide_ex.recipe.solve --task <task dir> [--namespace /Sim_0/Scene_0] [--episodes 1]
        [--attempts 3] [--dataset ~/dataset/<task>] [--arrangement demo_0]
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
    Randomize reply {task, target, goal, subtasks} and the resolved robot entry."""
    from guide_core.types.geometry import Pose

    try:
        answer = json.loads(reply) if reply else {}
    except ValueError:
        answer = {}
    answer = answer if isinstance(answer, dict) else {}
    goals = answer.get("goal") or []
    top, support = next((g["on"] for g in goals if "on" in g), (None, None))
    subtasks = answer.get("subtasks") or []
    scene_id = int(re.search(r"(\d+)$", namespace.rstrip("/")).group(1))
    return {"sim_namespace": "/" + namespace.strip("/").split("/")[0], "scene_namespace": f"/Scene_{scene_id}",
            "scene_id": scene_id, "robot": robot, "robot_prim": f"{entry['path']}/{entry['profile']}_rightfinger",
            "target": answer.get("target"), "top": top, "support": support,
            "pick": subtasks[0] if subtasks else "",
            "goal": next((g["in"][1] for g in goals if "in" in g), None),  # Place's container (block_bin-style tasks)
            "place": subtasks[1] if len(subtasks) > 1 else "", "dataset_path": dataset,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--namespace", default="/Sim_0/Scene_0")
    ap.add_argument("--episodes", type=int, default=1, help="successful episodes wanted")
    ap.add_argument("--attempts", type=int, default=0, help="attempts allowed (default: 3 x episodes)")
    ap.add_argument("--dataset", default="", help="dataset directory (default: ~/dataset/<task dir name>)")
    ap.add_argument("--arrangement", default="", help="config/arrangements/<id> to start from (default: a free draw)")
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
    done = attempts = 0
    try:
        while done < args.episodes and attempts < (args.attempts or 3 * args.episodes):
            attempts += 1
            # 600 s: a Randomize on a recipe scene runs Replicator draws and settles the physics
            reply = robot.callService(robot.randomize, srv.Randomize.Request(id=scene_id, arrangement=args.arrangement),
                                      timeout_sec=600.0)
            robot.sleep(SETTLE_S)
            ctx = context(args.namespace, robot, entry, dataset, reply.message)
            saved = episodes_saved(dataset)
            try:
                result = load(recipes / "Episode.yaml", values=values, search_paths=[recipes]).execute(ctx)
                ok = result.status == DemoStatus.PERFECT and bool(result.outputs.get("task_success"))  # as block_bin's solver
                why = result.outputs.get("task_reason") or result.error_message
            except Exception as e:  # a failed attempt, not a failed run (block_bin's generator)
                ok, why = False, repr(e)
            if ok and episodes_saved(dataset) <= saved:  # Plan C Task 18: a solved episode the recorder never wrote
                ok, why = False, "solved, but no episode was written to the dataset"
            done += ok
            robot.node.get_logger().info(f"[solve] attempt {attempts}: {'SUCCESS' if ok else f'FAILURE {why}'}")
            # an attempt that broke off leaves its episode open: drop it (a no-op when nothing is open)
            try:  # a stalled recorder costs this attempt, not the run (block_bin's solver)
                robot.callService(robot.stop_recording, srv.StopRecording.Request(id=scene_id, save_episode=False), timeout_sec=300.0)
            except Exception as e:
                robot.node.get_logger().error(f"Cleanup stop_recording failed: {e}")
    finally:
        try:  # saved episodes stay unreadable until finalized
            robot.callService(robot.finalize_recording, srv.FinalizeRecording.Request(id=scene_id), timeout_sec=300.0)
        except Exception as e:
            robot.node.get_logger().error(f"Finalize failed: {e}")
        finally:
            robot.disconnect()
    print(f"[solve] {done}/{args.episodes} episodes in {attempts} attempts", flush=True)
    return 0 if done == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
