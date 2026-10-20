"""Stack the cubes: the GUIDE-EX tree that builds the tower, recorded as one episode.

The scene draws the tower's order; the tree is built for it each episode, its layers as
GUIDE-EX means them -- the procedure is the long-horizon workflow, each placement one
TASK with a success criterion, the pick and the place its SUBTASKs. Every TASK and
SUBTASK announces its prompt at its level, so the dataset carries all three: the
procedure beside every frame, then the task and the subtask each frame was part of. The
procedure is never a frame's task: after the last placement a closing TASK takes over.

    StackingDemonstration (PROCEDURE "Stack the cubes.", condition: not done)
      Unclutch, LocateScene, AnnounceFirst (task + subtask), StartRecording,
      MeasureTower, TowerHeight
      -> BuildTower (PROCEDURE, loop until the tower is done)
           NextCube (TASK, condition: done?)
             MeasureTower, TowerHeight (ChainLength)   -> done, built
             done -> Finish (TASK "Return home.")
                       AnnounceFinish (task + subtask)
                       GoHome (SUBTASK "Return home."): MoveHome, WaitForTower
                       CheckSuccess, StopRecording (saved only on success)
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
2. An arm motion fails: no path, an aborted trajectory, or a straight-line path MoveIt
   can only partly compute (~1% of block_bin's moves, most near the robot base).
   MoveToCartesianPose refuses the last before moving -- executed, it stopped grasps
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

from guide_core.types.geometry import Point, Pose, Rotation
from guide_core.types.randomization import zone_plan
from guide_core.types.randomization.replicator_guide import zone_grid
from guide_ex.core.states import DemoStatus
from guide_ex.recipe import load
from guide_msgs import srv

# The FR3's home joints (init.yaml default_joint_states): gripper down over the scene
# origin, every joint mid-range. A 7-DoF arm reaches a pose in many ways and each
# Cartesian move keeps the current one, so moves drift towards joint limits; a pose goal
# lets the planner pick any of them. Home, as joints, is the one well-posed start.
HOME = {
    f"fr3_joint{i}.pos": q
    for i, q in enumerate([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398], start=1)
}
REBUILDS = 3  # loop passes beyond one per layer, for placements that did not hold
# Attempts at one episode before the run stops: past this something is stuck (e.g. an
# arm MoveIt will not plan from), and every further attempt would fail the same way.
MAX_ATTEMPTS = 8


def recipes_dir() -> Path:
    """config/recipes: beside the package in a source tree, else in the installed share."""
    source = Path(__file__).resolve().parents[1] / "config" / "recipes"
    if source.is_dir():
        return source
    return Path(get_package_share_directory("cube_stack")) / "config" / "recipes"


def recipe_values(robot, plan) -> dict:
    """What config/recipes/ interpolate: the robot's gripper joint and home, the drawn plan's
    prompts, the loop's passes, and the tower geometry the scene's success check uses too."""
    return {
        "robot": {"gripper_joint": robot.config.gripper_joint_names[0], "home": HOME},
        "plan": plan,
        "tower": {"max_loops": len(plan["order"]) + REBUILDS},
        "cube": {"edge": CUBE, "stacked": list(STACKED)},
    }


def load_recipe(name, robot, plan):
    """config/recipes/<name>.yaml as a GUIDE-EX tree, for this robot and plan."""
    recipes = recipes_dir()
    return load(recipes / f"{name}.yaml", values=recipe_values(robot, plan), search_paths=[recipes])


def build_tree(robot, plan):
    """The episode's tree, layered as GUIDE-EX means the layers (see the module docstring)."""
    return load_recipe("Episode", robot, plan)


def episode_context(robot, sim_namespace, scene_id, plan, path=""):
    """What the tree starts from: the robot, where things are, and the drawn plan with its
    prompts at every layer (scene.plan)."""
    return {
        "robot": robot,
        "sim_namespace": sim_namespace,
        "scene_namespace": f"/Scene_{scene_id}",
        "scene_id": scene_id,
        "scene_path": "",
        "robot_prim": "/fr3/fr3_rightfinger",
        # Where HOME puts the gripper: pointing down, 50 cm over the scene origin.
        "rest_pose": Pose(
            position=Point([0.0, 0.0, 0.5]),
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


def solve_task(scene_id, robot, sim_namespace, zone=None, path=""):
    """Randomize the scene, build the episode's tree and run it. True if saved."""
    reply = robot.callService(
        robot.randomize,
        srv.Randomize.Request(id=scene_id, use_zone=zone is not None, zone=int(zone or 0)),
    )
    plan = json.loads(reply.message)
    log = robot.node.get_logger()
    log.info(f"Task: {plan['task']} Order (bottom first): {plan['order']}")

    # Let the layout settle (the bins drop onto the floor) before solving.
    time.sleep(5)

    tree = build_tree(robot, plan)
    result = tree.execute(episode_context(robot, sim_namespace, scene_id, plan, path))

    if result.status == DemoStatus.PERFECT and result.outputs.get("task_success"):
        log.info(f"\033[92m[SUCCESS] Scene {scene_id}: tower built.\033[0m")
        return True
    reason = result.outputs.get("task_reason") or result.error_message or "sequence aborted"
    log.error(f"\033[91m[FAILURE] Scene {scene_id}: {reason}\033[0m")
    return False


_generating = threading.Lock()


def generate(plan, scene_id, robot, sim_namespace, path=""):
    """Record one episode per `plan` entry; a failed attempt is discarded and redrawn."""
    log = robot.node.get_logger()
    try:
        done = attempts = errors = 0
        while done < len(plan):
            if attempts == MAX_ATTEMPTS:
                raise RuntimeError(f"episode {done + 1} failed {attempts} times in a row")
            attempts += 1
            log.info(f"--- Episode {done + 1}/{len(plan)} | attempt {attempts} ---")
            try:
                saved = solve_task(scene_id, robot, sim_namespace, zone=plan[done], path=path)
                errors = 0
            except Exception as e:
                # A timed-out sim call costs this attempt; five in a row, the run.
                errors += 1
                log.error(f"Attempt failed ({errors} in a row): {e}")
                if errors >= 5:
                    raise
                saved = False

            # Route 5: an attempt that broke off leaves its episode open; drop it.
            # stop_recording is a no-op when nothing is open.
            try:
                robot.callService(
                    robot.stop_recording,
                    srv.StopRecording.Request(id=scene_id, save_episode=False),
                    timeout_sec=300.0,
                )
            except TimeoutError as e:
                log.error(f"Cleanup stop_recording timed out: {e}")

            if saved:
                done, attempts = done + 1, 0
    except Exception as e:
        log.error(f"Error during generation: {e}")
    finally:
        # Saved episodes are unreadable until the dataset is finalized.
        try:
            robot.callService(
                robot.finalize_recording,
                srv.FinalizeRecording.Request(id=scene_id),
                timeout_sec=600.0,
            )
            log.info("Recording dataset finalized.")
        except Exception as e:
            log.error(f"Finalize failed: {e}")
        _generating.release()


def main():
    parser = argparse.ArgumentParser(description="cube_stack demonstration generator.")
    parser.add_argument("--namespace", type=str, default="")
    args, _ = parser.parse_known_args()
    namespace = args.namespace  # e.g. /Sim_0/Scene_0
    sim_namespace = "/" + namespace.split("/")[1]
    match = re.search(r"\d+$", namespace.split("/")[-1])
    scene_id = int(match.group()) if match else 0

    config = FR3RobotConfig(
        frame_id=namespace.split("/")[-1] or "world",
        namespace=f"{namespace}/franka",
        planner_id="BiESTkConfigDefault",
        fallback_planner_id="PRMstarkConfigDefault",
        max_velocity=1.0,
        max_acceleration=1.0,
        gripper_action_type=ActionType.JOINT_POSITION,
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
        # zones: [] = free draws; [-1] = every zone of the starting cube's grid.
        grid = zone_grid(
            Path(get_package_share_directory("cube_stack")) / "config" / "randomize.yaml"
        )
        plan = zone_plan(request.zones, request.counts, grid.num_zones if grid else 1)
        threading.Thread(
            target=generate, args=(plan, scene_id, robot, sim_namespace, request.path)
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
