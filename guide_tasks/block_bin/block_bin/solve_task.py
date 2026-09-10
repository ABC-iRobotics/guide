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
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.utility.exception import NodeException
from guide_ex.steps.utility.pose import InvertPose, TransformPose
from guide_ex.steps.utility.rotation import ProjectRotationToBaseZ, ReduceRotationToSymmetry
from guide_ex.steps.utility.wait import WaitForSeconds
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


def solveTask(scene_id, robot, zone=None):
    global namespace_base

    task = robot.callService(
        robot.randomize,
        Randomize.Request(
            id=scene_id, use_zone=zone is not None, zone=int(zone) if zone is not None else 0
        ),
    )
    # robot.node.get_logger().info(f'Result is: {task}')

    task_dict = json.loads(task.message)
    target = task_dict["target"]
    goal = task_dict["goal"]

    # Allow the (now scene-level, seeded) randomization to settle before solving.
    time.sleep(5)

    # rest_pose is the only pose fed into the GUIDE-EX graph as context; every
    # other pose (scene/cube/bin/approach/retreat) is computed inside the node
    # tree (GetPrimPose / InvertPose / TransformPose), so no imperative pose
    # math lives here.
    rest_pose = PoseType(
        position=PointType([0.0, 0.0, 0.5]),
        orientation=RotationType(R.from_euler("xyz", [np.pi, 0.0, 0.0])),
    )

    # ================================================================================================== #
    # Scene solving nodes: MotionSequence
    robot.node.get_logger().info("Creating pick and place motion sequence")
    global_context: dict[str, Any] = {
        "sim_namespace": f'/{namespace_base.split("/")[1]}',
        "scene_namespace": f"/Scene_{scene_id}",
        "robot": robot,
        "robot_prim": "/fr3/fr3_rightfinger",
        "target": target,
        "goal": goal,
        "scene_path": "",
        "rest_pose": rest_pose,
    }

    robot.node.get_logger().info(f"Context {global_context}")

    robot.node.get_logger().info("Initializing GetTargetPose sequence...")
    get_target_pose_sequence = CompositeNode(
        name="GetTargetPose",
        level=Layer.SEQUENCE,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "target": "target",
            "scene_path": "scene_path",
        },
        children=[
            GetPrimPose(
                alias="GetScenePose",
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "prim_path": "scene_path",
                },
                output_map={"pose": "scene_pose"},
            ),
            InvertPose(
                alias="InvertScenePose",
                dynamic_map={"pose": "scene_pose"},
                output_map={"pose": "scene_pose_inv"},
            ),
            GetPrimPose(
                alias="GetCubePose",
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "prim_path": "target",
                },
                output_map={"pose": "cube_pose"},
            ),
            TransformPose(
                alias="TransformCubePose",
                dynamic_map={"l_pose": "scene_pose_inv", "r_pose": "cube_pose"},
                output_map={"pose": "cube_pose"},
            ),
            # Flatten the cube's orientation to a pure base-Z rotation so the grasp
            # comes straight down instead of tilting sideways with the cube.
            ProjectRotationToBaseZ(
                alias="ProjectCubeToBaseZ",
                dynamic_map={"pose": "cube_pose"},
                output_map={"pose": "cube_pose"},
            ),
            # A cube has 4 identical sides, so reduce the yaw to the smallest
            # equivalent wrist rotation.
            ReduceRotationToSymmetry(
                alias="MinimizeCubeRotation",
                dynamic_map={"pose": "cube_pose"},
                static_args={"symmetry": 4},
                output_map={"pose": "cube_pose"},
            ),
            TransformPose(
                alias="TransformCubePoseForGripper",
                dynamic_map={"l_pose": "cube_pose"},
                static_args={
                    "r_pose": PoseType(orientation=RotationType(R.from_euler("x", np.pi)))
                },
                output_map={"pose": "cube_pose"},
            ),
        ],
        output_map={
            "scene_pose": "scene_pose",
            "scene_pose_inv": "scene_pose_inv",
            "cube_pose": "cube_pose",
        },
    )

    robot.node.get_logger().info("Initializing Pick subtask...")
    pick_subtask = CompositeNode(
        name="Pick",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "robot_prim": "robot_prim",
            "target": "target",
            "scene_path": "scene_path",
            "cube_pose": "cube_pose",
            "rest_pose": "rest_pose",
        },
        children=[
            # ~~~~~~~~~~~~~~~~~~ Pick ~~~~~~~~~~~~~~~~~ #
            TransformPose(
                alias="TransformCubeApproachPose",
                dynamic_map={"r_pose": "cube_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, 0.1]))},
                output_map={"pose": "cube_approach_pose"},
            ),
            MoveToCartesianPose(
                alias="MoveToCubeApproach",
                dynamic_map={"robot": "robot", "target_pose": "cube_approach_pose"},
                static_args={"speed": 0.5, "cartesian": True},
            ),
            TransformPose(
                alias="TransformCubeGraspPose",
                dynamic_map={"r_pose": "cube_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, 0.01]))},
                output_map={"pose": "cube_grasp_pose"},
            ),
            MoveToCartesianPose(
                alias="MoveToCube",
                dynamic_map={"robot": "robot", "target_pose": "cube_grasp_pose"},
                static_args={"speed": 0.2, "cartesian": True},
            ),
            SetGripperState(
                # Over-close onto the cube: command fully closed (0.0) and let the cube
                # stop the fingers. The GripperActionController latches the *stalled*
                # position on success (set_hold_position), so the fingers must be pushing
                # PAST the cube for that held position to keep a squeeze force. Commanding
                # the cube's surface (~0.02) held ~zero force and the grip loosened.
                # Tune upward (e.g. 0.01) if 0.0 squeezes too hard for the cube.
                alias="CloseGripper",
                dynamic_map={"robot": "robot"},
                static_args={"gripper_goal_pos": {robot.config.gripper_joint_names[0]: 0.01}},
            ),
            WaitForSeconds(alias="WaitAfterClose", static_args={"seconds": 2.0}),
            TransformPose(
                alias="TransformCubeRetreatPose",
                dynamic_map={"r_pose": "cube_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, 0.2]))},
                output_map={"pose": "cube_retreat_pose"},
            ),
            MoveToCartesianPose(
                alias="MoveToCubeRetreat",
                dynamic_map={"robot": "robot", "target_pose": "cube_retreat_pose"},
                static_args={"speed": 0.5, "cartesian": True},
            ),
            IsPrimClashing(
                alias="CheckClash",
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "prim1_path": "robot_prim",
                    "prim2_path": "target",
                },
                output_map={"has_collided": "grasp_success"},
            ),
        ],
        output_map={
            "grasp_success": "grasp_success",
        },
        mode="condition",
        condition_expr="grasp_success",
        true_branch=None,
        false_branch=NodeException(name="NoGrasp"),
        fallbacks={
            # If the straight-up retreat can't execute (its Cartesian move aborts),
            # lift via the (high) rest pose instead, then resume the pick at the
            # grasp check -- so the arm reaches the bin THROUGH the rest pose rather
            # than sweeping across at grasp height and clipping the bin.
            "MoveToCubeRetreat": RecoveryNode(
                name="RetreatViaRest",
                level=Layer.SEQUENCE,
                children=[
                    MoveToCartesianPose(
                        alias="MoveToRestRetreat",
                        dynamic_map={"robot": "robot", "target_pose": "rest_pose"},
                        static_args={"speed": 1.0},
                    ),
                ],
                dynamic_map={"robot": "robot", "rest_pose": "rest_pose"},
                resume_target="CheckClash",
            ),
        },
    )

    robot.node.get_logger().info("Initializing Unclutch subtask...")
    unclutch_subtask = CompositeNode(
        name="Unclutch",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "rest_pose": "rest_pose",
        },
        children=[
            # Move to rest position
            MoveToCartesianPose(
                alias="MoveToRest",
                dynamic_map={"robot": "robot", "target_pose": "rest_pose"},
                static_args={"speed": 1.0},
            ),
            # Unclutch the robot if previous movement did not move the robot
            TransformPose(
                alias="TransformUncluchPose",
                dynamic_map={"r_pose": "rest_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, -0.1]))},
                output_map={"pose": "uncluch_pose"},
            ),
            MoveToCartesianPose(
                alias="MoveToUncluch",
                dynamic_map={"robot": "robot", "target_pose": "uncluch_pose"},
                static_args={"speed": 1.0},
            ),
            # Open gripper
            SetGripperState(
                alias="OpenGripper",
                dynamic_map={"robot": "robot"},
                static_args={"gripper_goal_pos": {robot.config.gripper_joint_names[0]: 0.04}},
            ),
            WaitForSeconds(alias="WaitAfterOpen", static_args={"seconds": 2.0}),
        ],
    )

    robot.node.get_logger().info("Initializing Place subtask...")
    place_subtask = CompositeNode(
        name="Place",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "goal": "goal",
            "scene_pose_inv": "scene_pose_inv",
        },
        children=[
            # Get bin approach pose
            GetPrimPose(
                alias="GetBinPose",
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "prim_path": "goal",
                },
                output_map={"pose": "bin_pose"},
            ),
            TransformPose(
                alias="TransformBinPose",
                dynamic_map={"l_pose": "scene_pose_inv", "r_pose": "bin_pose"},
                output_map={"pose": "bin_pose"},
            ),
            # Same treatment for the bin: drop straight in, minimal wrist rotation.
            ProjectRotationToBaseZ(
                alias="ProjectBinToBaseZ",
                dynamic_map={"pose": "bin_pose"},
                output_map={"pose": "bin_pose"},
            ),
            ReduceRotationToSymmetry(
                alias="MinimizeBinRotation",
                dynamic_map={"pose": "bin_pose"},
                static_args={"symmetry": 4},
                output_map={"pose": "bin_pose"},
            ),
            TransformPose(
                alias="TransformBinPoseForGripper",
                dynamic_map={"l_pose": "bin_pose"},
                static_args={
                    "r_pose": PoseType(orientation=RotationType(R.from_euler("x", np.pi)))
                },
                output_map={"pose": "bin_pose"},
            ),
            TransformPose(
                alias="TransformBinApproachPose",
                dynamic_map={"r_pose": "bin_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, 0.15]))},
                output_map={"pose": "bin_approach_pose"},
            ),
            # Move to bin and release the cube
            MoveToCartesianPose(
                alias="MoveToBinApproach",
                dynamic_map={"robot": "robot", "target_pose": "bin_approach_pose"},
                static_args={"speed": 0.5},
            ),
            SetGripperState(
                alias="OpenGripper",
                dynamic_map={"robot": "robot"},
                static_args={"gripper_goal_pos": {robot.config.gripper_joint_names[0]: 0.04}},
            ),
            WaitForSeconds(alias="WaitAfterOpen", static_args={"seconds": 2.0}),
        ],
    )

    robot.node.get_logger().info("Initializing Return subtask...")
    return_subtask = CompositeNode(
        name="Return",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "rest_pose": "rest_pose",
        },
        children=[
            # Move back to rest position
            MoveToCartesianPose(
                alias="MoveToRest",
                dynamic_map={"robot": "robot", "target_pose": "rest_pose"},
                static_args={"speed": 1.0},
            ),
            # Wait for scene to settle
            WaitForSeconds(alias="WaitAfterTask", static_args={"seconds": 5.0}),
        ],
    )

    robot.node.get_logger().info("Initializing PickAndPlace task...")
    pick_and_place = CompositeNode(
        name="PickAndPlace",
        level=Layer.TASK,
        dynamic_map={key: key for key, _ in global_context.items()},
        children=[
            unclutch_subtask,
            get_target_pose_sequence,
            pick_subtask,
            place_subtask,
            return_subtask,
        ],
        fallbacks={
            "Pick": RecoveryNode(
                name="PickRecovery",
                level=Layer.SUBTASK,
                children=[
                    get_target_pose_sequence,
                    SetGripperState(
                        alias="OpenGripperRecovery",
                        dynamic_map={"robot": "robot"},
                        static_args={
                            "gripper_goal_pos": {robot.config.gripper_joint_names[0]: 0.04}
                        },
                    ),
                    WaitForSeconds(alias="WaitAfterOpenRecovery", static_args={"seconds": 2.0}),
                    TransformPose(
                        alias="RotateCubePose",
                        dynamic_map={"l_pose": "cube_pose"},
                        static_args={
                            "r_pose": PoseType(
                                orientation=RotationType(R.from_euler("z", np.pi / 2))
                            )
                        },
                        output_map={"pose": "cube_pose"},
                    ),
                ],
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "scene_path": "scene_path",
                    "target": "target",
                },
                output_map={
                    "cube_pose": "cube_pose",
                },
                resume_target="Pick",
            ),
        },
    )

    robot.node.get_logger().info("Executing pick and place motion sequence")
    try:
        pick_and_place.execute(context=global_context)
        robot.node.get_logger().info("Executed pick and place motion sequence")

        # Evaluate success of the task
        time.sleep(1.0)  # Wait a second for physics to settle completely before evaluating
        response = robot.callService(robot.is_success, CheckSuccess.Request(id=scene_id))
        if response.success:
            robot.node.get_logger().info("\033[92m" + "=" * 50 + "\033[0m")
            robot.node.get_logger().info(
                f"\033[92m[SUCCESS] Scene {scene_id}: Task completed successfully!\033[0m"
            )
            robot.node.get_logger().info("\033[92m" + "=" * 50 + "\033[0m")
            return True
        else:
            reason = response.message if response.message else "Conditions not met"
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
    # ================================================================================================== #


is_generating = False
generation_lock = threading.Lock()


# block_bin has a zone grid; read it from this package's randomize.yaml so
# "all zones" expands to the right count.
def scene_num_zones() -> int:
    try:
        import os

        import yaml
        from ament_index_python.packages import get_package_share_directory

        from guide_core.types.randomization import grid_from_yaml

        share = get_package_share_directory("block_bin")
        with open(os.path.join(share, "config", "randomize.yaml")) as f:
            spec = yaml.safe_load(f)
        for instr in spec.get("instructions", []):
            pos = (instr.get("kwargs", {}).get("pose") or {}).get("position")
            g = grid_from_yaml(pos)
            if g is not None:
                return g.num_zones
    except Exception:
        pass
    return 1


def zoned_request(zone_counts: dict, path: str = "") -> Demonstration.Request:
    """Build a Demonstration request: ``zoned_request({2: 4, 16: 10})`` -> 4 demos
    with the cube in zone 2 and 10 in zone 16."""
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
        while idx < total:
            zone = plan[idx]
            attempts += 1
            robot.node.get_logger().info(
                f"--- Episode {idx + 1}/{total} (zone={zone}) | attempt {attempts} ---"
            )

            robot.callService(
                robot.start_recording, StartRecording.Request(id=scene_id, path=path)
            )

            success = solveTask(scene_id, robot, zone=zone)

            if success:
                robot.node.get_logger().info("Task succeeded. Saving episode...")
                robot.callService(
                    robot.stop_recording, StopRecording.Request(id=scene_id, save_episode=True)
                )
                idx += 1
                attempts = 0
            else:
                robot.node.get_logger().info("Task failed. Discarding episode...")
                robot.callService(
                    robot.stop_recording, StopRecording.Request(id=scene_id, save_episode=False)
                )

        robot.node.get_logger().info("Test completed. Finalizing recording dataset...")
        robot.callService(robot.finalize_recording, FinalizeRecording.Request(id=scene_id))
        robot.node.get_logger().info("Recording dataset finalized.")
    except Exception as e:
        robot.node.get_logger().error(f"Error during generation: {e}")
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
        description="Test ROS2Robot connection and action execution."
    )
    argparser.add_argument("--namespace", type=str, default="")
    args = argparser.parse_known_args()
    global namespace_base
    namespace_base = args[0].namespace if args[0].namespace else ""

    match = re.search(r"\d+$", namespace_base.split("/")[-1].strip())
    scene_id = int(match.group()) if match else 0

    # SPARStwokConfigDefault
    # BiESTkConfigDefault
    config = FR3RobotConfig(
        frame_id=namespace_base.split("/")[-1] if namespace_base else "world",
        namespace=f"{namespace_base}/franka",
        planner_id="BiESTkConfigDefault",
        fallback_planner_id="PRMstarkConfigDefault",
        max_velocity=1.0,
        max_acceleration=1.0,
        gripper_action_type=ActionType.JOINT_POSITION,
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
