"""cube_stack's trees as solve_task.py and oracle.py built them in Python before the recipes, kept
only until the Isaac regression passes: test_solve_task.py's identity tests compare the recipe trees
to them."""

import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform import Rotation as R

from cube_stack.scene import CUBE, STACKED
from guide_core.types.geometry import Point, Pose, Rotation
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.manipulation.joint_move import MoveToJointConfiguration
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, GetPrimPoses, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility.collection import GetItem
from guide_ex.utility.exception import NodeException
from guide_ex.utility.pose import ChainLength, InvertPose, TransformPose
from guide_ex.utility.recording import SetPrompt, StartRecording, StopRecording
from guide_ex.utility.rotation import ProjectRotationToBaseZ, ReduceRotationToSymmetry
from guide_ex.utility.wait import WaitForSeconds

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


def announce(alias, task_key=None, subtask_key=None, task="", subtask=""):
    """SetPrompt: the task and/or subtask read from context keys, or given as text. Both
    change at once, so a new task never meets the last task's subtask on a frame."""
    dynamic = keys("robot", "sim_namespace", "scene_id")
    dynamic.update({k: v for k, v in (("task", task_key), ("subtask", subtask_key)) if v})
    static = {k: v for k, v in (("task", task), ("subtask", subtask)) if v}
    return SetPrompt(alias=alias, dynamic_map=dynamic, static_args=static)


def build_tree(robot, plan):
    """The episode's tree, layered as GUIDE-EX means the layers (see the module docstring):
    the PROCEDURE stacks the cubes, each placement is a TASK, the pick and the place are
    its SUBTASKs. Each TASK and SUBTASK announces its prompt as it starts, at its level."""
    n_cubes = len(plan["order"])
    locate_cube = CompositeNode(
        name="LocateCube",
        level=Layer.SEQUENCE,
        dynamic_map=keys(*SIM, "scene_pose_inv", "top"),
        children=locate("Cube", "top", "cube_pose"),
    )

    pick = CompositeNode(
        name="Pick",
        level=Layer.SUBTASK,
        dynamic_map=keys(*SIM, "scene_id", "robot_prim", "top", "cube_pose", "pick"),
        children=[
            announce("AnnouncePick", subtask_key="pick"),
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

    place = CompositeNode(
        name="Place",
        level=Layer.SUBTASK,
        dynamic_map=keys(*SIM, "scene_id", "robot_prim", "scene_pose_inv", "top", "support", "place"),
        children=[announce("AnnouncePlace", subtask_key="place"), carry, release],
    )

    put_on = CompositeNode(
        name="PutOn",
        level=Layer.TASK,
        dynamic_map=keys(
            *SIM, "scene_id", "robot_prim", "scene_pose_inv", "order", "built",
            "tasks", "picks", "places", "set_downs",
        ),
        children=[
            # `built` cubes stand: the next one goes on the last of them.
            item("NextTop", "order", "top"),
            item("NextSupport", "order", "support", shift=-1),
            item("NextTask", "tasks", "task_prompt", shift=-1),
            item("NextPick", "picks", "pick", shift=-1),
            item("NextPlace", "places", "place", shift=-1),
            item("NextSetDown", "set_downs", "set_down", shift=-1),
            announce("AnnounceTask", task_key="task_prompt", subtask_key="pick"),
            locate_cube,
            pick,
            place,
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
            "Place": RecoveryNode(
                name="RepickDropped",
                level=Layer.SUBTASK,
                children=[
                    announce("AnnounceSetDown", subtask_key="set_down"),
                    move("BackOverCube", "over_cube_pose"),
                    move("SetDown", "grasp_pose", speed=0.2),
                    *gripper("OpenForRepick", robot, OPEN),
                    move("LeaveCube", "over_cube_pose", speed=0.2),
                    locate_cube,
                ],
                dynamic_map=keys(
                    *SIM, "scene_id", "scene_pose_inv", "top", "over_cube_pose", "grasp_pose", "set_down"
                ),
                resume_target="Pick",
                max_retries=2,
            ),
        },
    )

    recording = keys("robot", "sim_namespace", "scene_id")
    # The closing TASK: once the tower stands, go home and let the scene judge it. The
    # procedure is never a task, so this one has its own prompt, as its one subtask does.
    finish = CompositeNode(
        name="Finish",
        level=Layer.TASK,
        dynamic_map=keys("robot", "sim_namespace", "scene_id", "finish"),
        children=[
            announce("AnnounceFinish", task_key="finish", subtask_key="finish"),
            CompositeNode(
                name="GoHome",
                level=Layer.SUBTASK,
                dynamic_map=keys("robot"),
                children=[home("MoveHome"), WaitForSeconds(alias="WaitForTower", static_args={"seconds": 2.0})],
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

    tower_keys = keys(
        *SIM, "scene_id", "robot_prim", "scene_pose_inv", "order",
        "tasks", "picks", "places", "set_downs", "finish",
    )
    # Route 4: the loop. Each pass measures the tower and stacks the next cube onto the
    # highest one still in place, so a failed placement is just the next pass. A loop of
    # TASKs is the procedure's own work, so it sits at PROCEDURE level -- as a branch of
    # the root, since a composite's children sit strictly below it.
    build_tower = CompositeNode(
        name="BuildTower",
        level=Layer.PROCEDURE,
        dynamic_map=tower_keys,
        children=[
            CompositeNode(
                name="NextCube",
                level=Layer.TASK,
                dynamic_map=tower_keys,
                children=measure_tower(),
                mode="condition",
                condition_expr="done",
                true_branch=finish,
                false_branch=put_on,
            )
        ],
        mode="loop",
        condition_expr="done",
        # One pass per placement, one to see the tower done and finish, the spare passes.
        max_loops=n_cubes + REBUILDS,
    )

    return CompositeNode(
        name="StackingDemonstration",
        level=Layer.PROCEDURE,
        # scene_pose_inv is the root's own: LocateScene measures it.
        dynamic_map=keys(*(k for k in tower_keys if k != "scene_pose_inv"),
                         "scene_path", "rest_pose", "dataset_path"),
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
            # Before the first frame, so the whole episode carries a task and a subtask.
            announce("AnnounceFirst", task=plan["tasks"][0], subtask=plan["subtasks"]["pick"][0]),
            # The episode starts once the arm is at rest, not at randomization.
            StartRecording(
                dynamic_map={**recording, "path": "dataset_path"},
                static_args={"timeout_sec": 240.0},
            ),
            *measure_tower(),
        ],
        mode="condition",
        condition_expr="not done",
        true_branch=build_tower,
        false_branch=NodeException(name="AlreadyStacked"),
    )


def oracle_tree() -> CompositeNode:
    return CompositeNode(
        name="SceneOracle",
        level=Layer.SUBTASK,
        dynamic_map=keys(*SIM, "robot_prim", "order"),
        children=[
            *measure_tower(),
            # The highest cube standing, and whether the fingers are still on it.
            item("TopOfTower", "order", "top", shift=-1),
            holding("TopHeld"),
        ],
        mode="condition",
        condition_expr="done",
        # Not done: is the next cube in the fingers?
        false_branch=CompositeNode(
            name="NextCubeHeld",
            level=Layer.SEQUENCE,
            dynamic_map=keys(*SIM, "robot_prim", "order", "built"),
            children=[
                item("NextFree", "order", "next"),
                IsPrimClashing(
                    alias="NextHeld",
                    dynamic_map={**keys(*SIM), "prim1_path": "robot_prim", "prim2_path": "next"},
                    output_map={"has_collided": "next_held"},
                ),
            ],
        ),
    )


def plain(value):
    """A static arg as comparable data: a Pose as its rounded matrix, a tuple as a list, floats rounded."""
    if isinstance(value, Pose):
        return {"pose": np.round(value.to_numpy(), 12).tolist()}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return round(value, 12) if isinstance(value, float) else value


def describe(node):
    """Everything about a node that execution reads (not node_type: it only names log lines)."""
    out = {"class": type(node).__name__, "name": node.name, "level": node.level.name,
           "dynamic_map": node.dynamic_map, "static_args": plain(node.static_args), "output_map": node.output_map}
    if isinstance(node, CompositeNode):
        out.update(mode=node.mode.value, condition_expr=node.condition_expr, max_loops=node.max_loops,
                   children=[describe(c) for c in node.children],
                   fallbacks={k: describe(v) for k, v in node.fallbacks.items()},
                   true_branch=node.true_branch and describe(node.true_branch),
                   false_branch=node.false_branch and describe(node.false_branch))
    if isinstance(node, RecoveryNode):
        out.update(resume_target=node.resume_target, max_retries=node.max_retries)
    return out
