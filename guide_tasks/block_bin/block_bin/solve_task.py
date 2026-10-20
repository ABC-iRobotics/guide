import argparse
import json
import re
import threading
import time
from typing import Any

import numpy as np
from irob_lerobot_ros.config import ActionType, FR3RobotConfig, ROS2CameraConfig
from irob_lerobot_ros.ros2robot import ROS2Robot

# pyrefly: ignore [missing-import]
from scipy.spatial.transform.rotation import Rotation as R

from guide_core.types.geometry import Point as PointType
from guide_core.types.geometry import Pose as PoseType
from guide_core.types.geometry import Rotation as RotationType
from guide_core.types.randomization import zone_plan
from guide_ex.core.states import DemoStatus
from guide_ex.recipe import load, recipes_dir, robot_values
from guide_msgs.srv import (
    CheckSuccess,
    Collision,
    Demonstration,
    FinalizeRecording,
    Pose,
    Randomize,
    StartRecording,
    StopRecording,
)

namespace_base = ""
# dataset.camera_transition's duration in config/init.yaml (the launch passes it): s of sim time
# the arm holds still right after StartRecording while the cameras move onto their mounts. 0: off.
startup_padding = 0.0
# dataset.fps (the launch passes it). The hold lasts one frame period longer: it starts at the
# StartRecording reply, the transition at the first recorded frame, up to 1/fps later.
record_fps = 10.0


def episode_tree(robot):
    """The episode's GUIDE-EX tree: config/recipes/Episode.yaml, for this robot's gripper and the
    camera transition's hold (WaitForCameras, only while it is on)."""
    recipes = recipes_dir("block_bin", __file__)
    values = {
        "robot": robot_values(robot.config, recipes.parent / "init.yaml"),
        "episode": {"camera_hold": bool(startup_padding), "hold_seconds": startup_padding + 1.0 / record_fps},
    }
    return load(recipes / "Episode.yaml", values=values, search_paths=[recipes])


def solveTask(scene_id, robot, zone=None, path=""):
    global namespace_base

    task = robot.callService(
        robot.randomize,
        Randomize.Request(
            id=scene_id, use_zone=zone is not None, zone=int(zone) if zone is not None else 0
        ),
    )

    task_dict = json.loads(task.message)
    target = task_dict["target"]
    goal = task_dict["goal"]

    # Allow the (now scene-level, seeded) randomization to settle before solving (sim time).
    robot.sleep(5)

    # rest_pose is the only pose fed into the GUIDE-EX graph as context; every
    # other pose (cube/bin/approach/retreat) is computed inside the node tree
    # (GetPrimPose / TransformPose), so no imperative pose math lives here.
    # All of them are in the robot's base frame, fr3_link0 (S2b Ruling 4), the frame
    # GetPrimPose answers in. The rest pose stays where the arm has always had it:
    # (0, 0, 0.5) of Scene_0 with the base at (-0.3, 0, 0) there, as MoveIt mounts it
    # (launch/bringup.launch.py) and as the base-frame datasets of 2026-10-10 end.
    rest_pose = PoseType(
        position=PointType([0.3, 0.0, 0.5]),
        orientation=RotationType(R.from_euler("xyz", [np.pi, 0.0, 0.0])),
    )

    global_context: dict[str, Any] = {
        "sim_namespace": f'/{namespace_base.split("/")[1]}',
        "scene_namespace": f"/Scene_{scene_id}",
        "scene_id": scene_id,
        "robot": robot,
        "robot_prim": "/fr3/fr3_rightfinger",
        "target": target,
        "goal": goal,
        "rest_pose": rest_pose,
        "dataset_path": path,
    }

    robot.node.get_logger().info(f"Context {global_context}")
    pick_and_place = episode_tree(robot)

    robot.node.get_logger().info("Executing pick and place motion sequence")
    try:
        result = pick_and_place.execute(context=global_context)
        robot.node.get_logger().info("Executed pick and place motion sequence")

        # The tree checked success itself, right before StopRecording saved or dropped
        # the episode. A run that broke off earlier never got that far: not a success.
        if result.status == DemoStatus.PERFECT and result.outputs.get("task_success"):
            robot.node.get_logger().info("\033[92m" + "=" * 50 + "\033[0m")
            robot.node.get_logger().info(
                f"\033[92m[SUCCESS] Scene {scene_id}: Task completed successfully!\033[0m"
            )
            robot.node.get_logger().info("\033[92m" + "=" * 50 + "\033[0m")
            return True
        else:
            reason = result.outputs.get("task_reason") or result.error_message or "Sequence aborted"
            robot.node.get_logger().error("\033[91m" + "=" * 50 + "\033[0m")
            robot.node.get_logger().error(
                f"\033[91m[FAILURE] Scene {scene_id}: Task failed! Reason: {reason}\033[0m"
            )
            robot.node.get_logger().error("\033[91m" + "=" * 50 + "\033[0m")
            return False
    except Exception as e:
        robot.node.get_logger().error(
            f"\033[91mFailed to execute pick and place motion sequence or verify success: {e}\033[0m"
        )
        return False


is_generating = False
generation_lock = threading.Lock()


# block_bin has a zone grid; read it from this package's randomize.yaml so
# "all zones" expands to the right count.
def scene_num_zones() -> int:
    try:
        import os

        from ament_index_python.packages import get_package_share_directory

        from guide_core.types.randomization.replicator_guide import zone_grid

        share = get_package_share_directory("block_bin")
        grid = zone_grid(os.path.join(share, "config", "randomize.yaml"))
        if grid is not None:
            return grid.num_zones
    except Exception:
        pass
    return 1


def zoned_request(zone_counts: dict, path: str = "") -> Demonstration.Request:
    """Build a Demonstration request: ``zoned_request({2: 4, 10: 10})`` -> 4 demos
    with the cube in zone 2 and 10 in zone 10."""
    zones = [int(z) for z in zone_counts]
    counts = [int(zone_counts[z]) for z in zones]
    return Demonstration.Request(path=path, zones=zones, counts=counts)


def all_zones_request(count: int, path: str = "") -> Demonstration.Request:
    """`count` demos in EVERY zone (``zones=[-1]``); ``zones=[]`` would be free draws."""
    return Demonstration.Request(path=path, zones=[-1], counts=[int(count)])


def generate_demos_thread(plan, scene_id, robot, path=""):
    global is_generating
    try:
        total = len(plan)
        idx = 0
        attempts = 0
        errors = 0  # consecutive attempts that raised
        while idx < total:
            zone = plan[idx]
            attempts += 1
            robot.node.get_logger().info(
                f"--- Episode {idx + 1}/{total} (zone={zone}) | attempt {attempts} ---"
            )

            # The tree records its own episode: StartRecording after Unclutch,
            # StopRecording after homing, saving only on success.
            try:
                success = solveTask(scene_id, robot, zone=zone, path=path)
                errors = 0
            except Exception as e:
                # A timed-out sim call (Randomize, ...) costs this attempt, not the run;
                # five in a row means the simulator is gone.
                errors += 1
                robot.node.get_logger().error(f"Attempt failed ({errors} in a row): {e}")
                if errors >= 5:
                    raise
                success = False

            # A run that broke off between the two leaves its episode open: drop it.
            # stop_recording is a no-op when nothing is open, and a stalled recorder
            # must cost this one attempt, not the whole generation run.
            try:
                robot.callService(
                    robot.stop_recording,
                    StopRecording.Request(id=scene_id, save_episode=False),
                    timeout_sec=300.0,
                )
            except TimeoutError as e:
                robot.node.get_logger().error(f"Cleanup stop_recording timed out: {e}")

            if success:
                idx += 1
                attempts = 0

        robot.node.get_logger().info("Generation finished. Finalizing recording dataset...")
        robot.callService(robot.finalize_recording, FinalizeRecording.Request(id=scene_id))
        robot.node.get_logger().info("Recording dataset finalized.")
    except Exception as e:
        robot.node.get_logger().error(f"Error during generation: {e}")
        # Episodes already saved stay unreadable until the dataset is finalized.
        try:
            robot.callService(robot.finalize_recording, FinalizeRecording.Request(id=scene_id))
        except Exception as finalize_error:
            robot.node.get_logger().error(f"Finalize after abort failed: {finalize_error}")
    finally:
        with generation_lock:
            is_generating = False


def handle_generate_demonstration(request, response, scene_id, robot):
    global is_generating

    with generation_lock:
        if is_generating:
            response.success = False
            response.message = "Generation is already in progress."
            return response

        is_generating = True

    plan = zone_plan(request.zones, request.counts, scene_num_zones())
    robot.node.get_logger().info(
        f"Received request: zones={list(request.zones)} counts={list(request.counts)} "
        f"-> {len(plan)} demonstrations."
    )

    t = threading.Thread(target=generate_demos_thread, args=(plan, scene_id, robot, request.path))
    t.start()

    response.success = True
    response.message = f"Started generating {len(plan)} demonstrations."
    return response


def main():
    # Create a robot configuration
    argparser = argparse.ArgumentParser(
        description="block_bin demonstration generator."
    )
    argparser.add_argument("--namespace", type=str, default="")
    argparser.add_argument("--controller", choices=["moveit", "oscbf"], default="oscbf")
    argparser.add_argument("--startup-padding", type=float, default=0.0)
    argparser.add_argument("--fps", type=float, default=10.0)
    args = argparser.parse_known_args()
    global namespace_base, startup_padding, record_fps
    namespace_base = args[0].namespace if args[0].namespace else ""
    startup_padding = args[0].startup_padding
    record_fps = args[0].fps

    match = re.search(r"\d+$", namespace_base.split("/")[-1].strip())
    scene_id = int(match.group()) if match else 0

    config = FR3RobotConfig(
        namespace=f"{namespace_base}/franka",
        planner_id="BiESTkConfigDefault",
        fallback_planner_id="PRMstarkConfigDefault",
        max_velocity=1.0,
        max_acceleration=1.0,
        gripper_action_type=ActionType.JOINT_POSITION,
        controller=args[0].controller,
    )
    config.cameras = {
        "cam_base": ROS2CameraConfig(
            namespace=namespace_base, frame_id="cam_base", topic="cam_base"
        ),
        "cam_top": ROS2CameraConfig(namespace=namespace_base, frame_id="cam_top", topic="cam_top"),
        "cam_wrist": ROS2CameraConfig(
            namespace=namespace_base, frame_id="cam_wrist", topic="cam_wrist"
        ),
    }
    config.arm_action_type = ActionType.CARTESIAN_POSE
    # Initialize the robot with the configuration
    robot = ROS2Robot(config=config)
    robot.connect()
    time.sleep(5)  # Wait for connections to establish
    print("Connected to robot and cameras.")

    robot.node.get_logger().info(
        f'Service path is: {"/" + namespace_base.split("/")[1] + "/Randomize",}'
    )

    robot.randomize = robot.node.create_client(
        srv_type=Randomize,
        srv_name="/" + namespace_base.split("/")[1] + "/Randomize",
        callback_group=robot._reentrant_callback_group,
    )

    robot.reset = robot.node.create_client(
        srv_type=Randomize,
        srv_name="/" + namespace_base.split("/")[1] + "/Reset",
        callback_group=robot._reentrant_callback_group,
    )

    robot.pose = robot.node.create_client(
        srv_type=Pose,
        srv_name="/" + namespace_base.split("/")[1] + "/PoseRequest",
        callback_group=robot._reentrant_callback_group,
    )

    robot.collision = robot.node.create_client(
        srv_type=Collision,
        srv_name="/" + namespace_base.split("/")[1] + "/CollisionRequest",
        callback_group=robot._reentrant_callback_group,
    )

    robot.is_success = robot.node.create_client(
        srv_type=CheckSuccess,
        srv_name="/" + namespace_base.split("/")[1] + "/IsSuccess",
        callback_group=robot._reentrant_callback_group,
    )

    robot.start_recording = robot.node.create_client(
        srv_type=StartRecording,
        srv_name="/" + namespace_base.split("/")[1] + "/start_recording",
        callback_group=robot._reentrant_callback_group,
    )

    robot.stop_recording = robot.node.create_client(
        srv_type=StopRecording,
        srv_name="/" + namespace_base.split("/")[1] + "/stop_recording",
        callback_group=robot._reentrant_callback_group,
    )

    robot.finalize_recording = robot.node.create_client(
        srv_type=FinalizeRecording,
        srv_name="/" + namespace_base.split("/")[1] + "/finalize_recording",
        callback_group=robot._reentrant_callback_group,
    )

    robot.generate_demo_service = robot.node.create_service(
        srv_type=Demonstration,
        srv_name=namespace_base + "/generate_demonstration",
        callback=lambda req, res: handle_generate_demonstration(req, res, scene_id, robot),
        callback_group=robot._reentrant_callback_group,
    )

    robot.node.get_logger().info("Ready to receive demonstration generation requests.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        robot.node.get_logger().info("Keyboard interrupt received. Exiting...")
    finally:
        robot.disconnect()
        exit(0)
