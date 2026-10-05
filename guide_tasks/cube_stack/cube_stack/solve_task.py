"""Stack the cubes: the GUIDE-EX tree that builds the tower, recorded as one episode.

The scene draws the tower's order; the tree is built for it each episode:

    StackingDemonstration (PROCEDURE)
      Unclutch, LocateScene, AnnounceFirstSubtask, StartRecording
      StackCubes (TASK, loop until the tower is done)
        NextCube (SUBTASK, condition: done?)
          MeasureTower, TowerHeight (ChainLength)   -> done, built
          else PutOn (SUBTASK)    "Put the red cube on the blue cube."
            NextTop, NextSupport, NextPrompt (GetItem: order[built], ...)
            AnnounceSubtask, LocateCube, Pick, CarryToSupport, Release
      GoHome, CheckSuccess, StopRecording (saved only on success)

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
3. The cube does not get to the tower: it slips out (CarryToSupport checks the fingers
   above the tower), or the arm finds no way there. RepickDropped sets it down where it
   was picked -- opening wherever the arm is would drop it from height onto the tower --
   and picks it up again from wherever it is.
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
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import DemoStatus, Layer
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration
from guide_ex.steps.simulation.isaac.prim import (
    GetPrimPose,
    GetPrimPoses,
    IsPrimClashing,
)
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility.collection import GetItem
from guide_ex.utility.exception import NodeException
from guide_ex.utility.pose import ChainLength, InvertPose, TransformPose
from guide_ex.utility.recording import SetSubtaskPrompt, StartRecording, StopRecording
from guide_ex.utility.rotation import ProjectRotationToBaseZ, ReduceRotationToSymmetry
from guide_ex.utility.wait import WaitForSeconds
from guide_msgs import srv

# The FR3's home joints (init.yaml default_joint_states): gripper down over the scene
# origin, every joint mid-range. A 7-DoF arm reaches a pose in many ways and each
# Cartesian move keeps the current one, so moves drift towards joint limits; a pose goal
# lets the planner pick any of them. Home, as joints, is the one well-posed start.
HOME = {
    f"fr3_joint{i}.pos": q
    for i, q in enumerate([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398], start=1)
}
GRASP = 0.01  # TCP above the cube centre when grasping (block_bin's grasp)
OVER_CUBE = 0.275  # approach above a cube centre: TCP ~30 cm over the table
PLACE = CUBE + GRASP + 0.005  # TCP above the support's centre at release: 5 mm drop
OVER_SUPPORT = PLACE + 0.10  # carried cube clears the tower top by 10 cm
REBUILDS = 3  # loop passes beyond one per layer, for placements that did not hold
# Attempts at one episode before the run stops: past this something is stuck (e.g. an
# arm MoveIt will not plan from), and every further attempt would fail the same way.
MAX_ATTEMPTS = 8
OPEN, CLOSED = 0.04, 0.01  # finger targets; closed over-squeezes, see block_bin

SIM = ("robot", "sim_namespace", "scene_namespace")


def keys(*names):
    return {name: name for name in names}


def move(alias, target, speed=0.5, cartesian=True):
    return MoveToCartesianPose(
        alias=alias,
        dynamic_map={"robot": "robot", "target_pose": target},
        static_args={"speed": speed, "cartesian": cartesian},
    )


def item(alias, items, out, shift=0):
    return GetItem(
        alias=alias,
        dynamic_map={"items": items, "index": "built"},
        static_args={"shift": shift},
        output_map={"item": out},
    )


def above(alias, pose, height, out):
    """`pose` raised by `height` along the world z axis."""
    return TransformPose(
        alias=alias,
        dynamic_map={"r_pose": pose},
        static_args={"l_pose": Pose(position=Point([0.0, 0.0, height]))},
        output_map={"pose": out},
    )


def gripper(alias, robot, width, settle=1.0):
    return [
        SetGripperState(
            alias=alias,
            dynamic_map={"robot": "robot"},
            static_args={"gripper_goal_pos": {robot.config.gripper_joint_names[0]: width}},
        ),
        WaitForSeconds(alias=f"{alias}Settle", static_args={"seconds": settle}),
    ]


def holding(alias):
    """Is the cube `top` between the fingers? (block_bin's grasp check)"""
    return IsPrimClashing(
        alias=alias,
        dynamic_map={**keys(*SIM), "prim1_path": "robot_prim", "prim2_path": "top"},
        output_map={"has_collided": "holding"},
    )


def locate(name, prim, out):
    """`prim`'s pose in the scene frame, turned into a top-down grasp with the least wrist turn."""
    return [
        GetPrimPose(
            alias=f"Get{name}Pose",
            dynamic_map={**keys(*SIM), "prim_path": prim},
            output_map={"pose": out},
        ),
        TransformPose(
            alias=f"{name}InScene",
            dynamic_map={"l_pose": "scene_pose_inv", "r_pose": out},
            output_map={"pose": out},
        ),
        # "auto": a cube that fell may lie on any face; take its flattest axis.
        ProjectRotationToBaseZ(
            alias=f"Flatten{name}",
            dynamic_map={"pose": out},
            static_args={"heading_axis": "auto"},
            output_map={"pose": out},
        ),
        ReduceRotationToSymmetry(
            alias=f"Minimize{name}Turn",
            dynamic_map={"pose": out},
            static_args={"symmetry": 4},
            output_map={"pose": out},
        ),
        TransformPose(
            alias=f"{name}GripperDown",
            dynamic_map={"l_pose": out},
            static_args={"r_pose": Pose(orientation=Rotation(R.from_euler("x", np.pi)))},
            output_map={"pose": out},
        ),
    ]


def measure_tower():
    """The tower as it stands: `built` cubes of `order` on each other from the bottom,
    `done` once all are. The demonstration's loop and the subtask oracle both read it."""
    return [
        GetPrimPoses(
            alias="MeasureTower",
            dynamic_map={**keys(*SIM), "prim_paths": "order"},
            output_map={"poses": "tower_poses"},
        ),
        # The tower is a chain of cubes, each one cube edge above the last.
        ChainLength(
            alias="TowerHeight",
            dynamic_map={"poses": "tower_poses"},
            static_args={"offset": (0.0, 0.0, CUBE), "tolerance": STACKED},
            output_map={"length": "built", "complete": "done"},
        ),
    ]


def home(alias):
    return MoveToJointConfiguration(
        alias=alias,
        dynamic_map={"robot": "robot"},
        static_args={"target_configuration": HOME, "speed": 1.0},
    )


def via_home(name, resume, *then, needs=()):
    """Route 2: detour through the home joints (resets the arm's posture), then retry."""
    return RecoveryNode(
        name=name,
        level=Layer.SEQUENCE,
        children=[home(f"{name}ToHome"), *then],
        dynamic_map=keys("robot", *needs),
        resume_target=resume,
        max_retries=2,
    )


def build_tree(robot, prompts, n_cubes):
    """The episode's tree. `prompts[0]` is announced before recording starts."""
    locate_cube = CompositeNode(
        name="LocateCube",
        level=Layer.SEQUENCE,
        dynamic_map=keys(*SIM, "scene_pose_inv", "top"),
        children=locate("Cube", "top", "cube_pose"),
    )

    pick = CompositeNode(
        name="Pick",
        level=Layer.SEQUENCE,
        dynamic_map=keys(*SIM, "robot_prim", "top", "cube_pose"),
        children=[
            above("OverCubePose", "cube_pose", OVER_CUBE, "over_cube_pose"),
            move("MoveOverCube", "over_cube_pose"),
            above("GraspPose", "cube_pose", GRASP, "grasp_pose"),
            move("MoveToGrasp", "grasp_pose", speed=0.2),
            *gripper("CloseGripper", robot, CLOSED, settle=2.0),
            move("LiftCube", "over_cube_pose"),
            holding("CheckHolding"),
        ],
        mode="condition",
        condition_expr="holding",
        false_branch=NodeException(name="NotHolding"),
        fallbacks={
            "MoveOverCube": via_home("OverCubeViaHome", "MoveOverCube"),
            "MoveToGrasp": via_home("GraspViaHome", "MoveOverCube"),
            "LiftCube": via_home("LiftViaHome", "CheckHolding"),
        },
    )

    carry = CompositeNode(
        name="CarryToSupport",
        level=Layer.SEQUENCE,
        dynamic_map=keys(*SIM, "robot_prim", "scene_pose_inv", "top", "support"),
        children=[
            # Measured now, not at the start: the support may have been nudged since.
            *locate("Support", "support", "support_pose"),
            above("OverSupportPose", "support_pose", OVER_SUPPORT, "over_support_pose"),
            move("MoveOverSupport", "over_support_pose"),
            holding("CheckStillHolding"),
        ],
        mode="condition",
        condition_expr="holding",
        false_branch=NodeException(name="DroppedInTransit"),
        fallbacks={"MoveOverSupport": via_home("CarryViaHome", "MoveOverSupport")},
    )

    release = CompositeNode(
        name="Release",
        level=Layer.SEQUENCE,
        dynamic_map=keys("robot", "support_pose", "over_support_pose"),
        children=[
            above("PlacePose", "support_pose", PLACE, "place_pose"),
            move("LowerOntoSupport", "place_pose", speed=0.2),
            *gripper("OpenGripper", robot, OPEN),
            # Slow and straight up, so the fingers do not drag the cube along.
            move("RetreatFromTower", "over_support_pose", speed=0.2),
        ],
        fallbacks={
            "LowerOntoSupport": via_home(
                "LowerViaHome",
                "LowerOntoSupport",
                move("BackOverSupport", "over_support_pose"),
                needs=("over_support_pose",),
            ),
            "RetreatFromTower": via_home("RetreatViaHome", "RetreatFromTower"),
        },
    )

    put_on = CompositeNode(
        name="PutOn",
        level=Layer.SUBTASK,
        dynamic_map=keys(
            *SIM,
            "scene_id",
            "robot_prim",
            "scene_pose_inv",
            "order",
            "prompts",
            "built",
        ),
        children=[
            # `built` cubes stand: the next one goes on the last of them.
            item("NextTop", "order", "top"),
            item("NextSupport", "order", "support", shift=-1),
            item("NextPrompt", "prompts", "prompt", shift=-1),
            SetSubtaskPrompt(
                alias="AnnounceSubtask",
                dynamic_map=keys("robot", "sim_namespace", "scene_id", "prompt"),
            ),
            locate_cube,
            pick,
            carry,
            release,
        ],
        fallbacks={
            # Route 1: the other pair of faces, on a freshly measured cube.
            "Pick": RecoveryNode(
                name="Regrasp",
                level=Layer.SUBTASK,
                children=[
                    *gripper("OpenForRegrasp", robot, OPEN),
                    locate_cube,
                    TransformPose(
                        alias="TurnGrasp",
                        dynamic_map={"l_pose": "cube_pose"},
                        static_args={
                            "r_pose": Pose(orientation=Rotation(R.from_euler("z", np.pi / 2)))
                        },
                        output_map={"pose": "cube_pose"},
                    ),
                ],
                dynamic_map=keys(*SIM, "scene_pose_inv", "top"),
                resume_target="Pick",
                max_retries=2,
            ),
            # Route 3: the cube did not get to the tower -- it slipped out, or the arm found
            # no way there. Set it down where it was picked (if it is still held: opening
            # anywhere else drops it from height), then pick it up again from wherever it is.
            "CarryToSupport": RecoveryNode(
                name="RepickDropped",
                level=Layer.SUBTASK,
                children=[
                    move("BackOverCube", "over_cube_pose"),
                    move("SetDown", "grasp_pose", speed=0.2),
                    *gripper("OpenForRepick", robot, OPEN),
                    move("LeaveCube", "over_cube_pose", speed=0.2),
                    locate_cube,
                ],
                dynamic_map=keys(*SIM, "scene_pose_inv", "top", "over_cube_pose", "grasp_pose"),
                resume_target="Pick",
                max_retries=2,
            ),
        },
    )

    tower_keys = keys(
        *SIM,
        "scene_id",
        "robot_prim",
        "scene_pose_inv",
        "order",
        "prompts",
    )
    # Route 4: the loop. Each pass measures the tower and stacks the next cube onto
    # the highest one still in place, so a failed placement is just the next pass.
    stack_cubes = CompositeNode(
        name="StackCubes",
        level=Layer.TASK,
        dynamic_map=tower_keys,
        children=[
            CompositeNode(
                name="NextCube",
                level=Layer.SUBTASK,
                dynamic_map=tower_keys,
                children=measure_tower(),
                mode="condition",
                condition_expr="done",
                false_branch=put_on,
            )
        ],
        mode="loop",
        condition_expr="done",
        # One pass per placement, one to see the tower done, and the spare passes.
        max_loops=n_cubes + REBUILDS,
    )

    recording = keys("robot", "sim_namespace", "scene_id")
    return CompositeNode(
        name="StackingDemonstration",
        level=Layer.PROCEDURE,
        dynamic_map=keys(
            *SIM,
            "scene_id",
            "scene_path",
            "robot_prim",
            "rest_pose",
            "dataset_path",
            "order",
            "prompts",
        ),
        children=[
            CompositeNode(
                name="Unclutch",
                level=Layer.SUBTASK,
                dynamic_map=keys("robot", "rest_pose"),
                children=[
                    home("MoveHome"),
                    # Moves the arm even if it already was home (see block_bin); straight
                    # down, so it keeps home's posture for the first pick.
                    above("UnclutchPose", "rest_pose", -0.1, "unclutch_pose"),
                    move("MoveToUnclutch", "unclutch_pose", speed=1.0),
                    *gripper("OpenGripper", robot, OPEN, settle=2.0),
                ],
            ),
            CompositeNode(
                name="LocateScene",
                level=Layer.SEQUENCE,
                dynamic_map=keys(*SIM, "scene_path"),
                children=[
                    GetPrimPose(
                        alias="GetScenePose",
                        dynamic_map={**keys(*SIM), "prim_path": "scene_path"},
                        output_map={"pose": "scene_pose"},
                    ),
                    InvertPose(
                        alias="InvertScenePose",
                        dynamic_map={"pose": "scene_pose"},
                        output_map={"pose": "scene_pose_inv"},
                    ),
                ],
            ),
            # Before the first frame, so the whole episode carries a subtask.
            SetSubtaskPrompt(
                alias="AnnounceFirstSubtask",
                dynamic_map=recording,
                static_args={"prompt": prompts[0]},
            ),
            # The episode starts once the arm is at rest, not at randomization.
            StartRecording(
                dynamic_map={**recording, "path": "dataset_path"},
                static_args={"timeout_sec": 240.0},
            ),
            stack_cubes,
            CompositeNode(
                name="GoHome",
                level=Layer.SUBTASK,
                dynamic_map=keys("robot"),
                children=[
                    home("MoveHome"),
                    WaitForSeconds(alias="WaitForTower", static_args={"seconds": 2.0}),
                ],
            ),
            IsTaskSuccessful(
                alias="CheckSuccess",
                dynamic_map=recording,
                output_map={"success": "task_success", "reason": "task_reason"},
            ),
            # Saved only if the scene sees the tower; anything else is discarded.
            StopRecording(
                dynamic_map={**recording, "save_episode": "task_success"},
                static_args={"timeout_sec": 300.0},
            ),
        ],
    )


def episode_context(robot, sim_namespace, scene_id, order, prompts, path=""):
    """What the tree starts from: the robot, where things are, and the drawn plan."""
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
        "order": order,
        "prompts": prompts,
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

    tree = build_tree(robot, plan["subtasks"], len(plan["order"]))
    result = tree.execute(
        episode_context(robot, sim_namespace, scene_id, plan["order"], plan["subtasks"], path)
    )

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
        ("set_subtask", srv.SetSubtask, "set_subtask"),
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
