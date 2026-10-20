"""block_bin's tree as solve_task.py built it in Python before the recipes, kept only until the
Isaac regression passes: test_recipe_identity.py compares the recipe tree to it."""

import numpy as np

# pyrefly: ignore [missing-import]
from scipy.spatial.transform.rotation import Rotation as R

from guide_core.types.geometry import Point as PointType
from guide_core.types.geometry import Pose
from guide_core.types.geometry import Pose as PoseType
from guide_core.types.geometry import Rotation as RotationType
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.steps.end_effector.gripper_control import SetGripperState
from guide_ex.steps.manipulation.cartesian_move import MoveToCartesianPose
from guide_ex.steps.simulation.isaac.prim import GetPrimPose, IsPrimClashing
from guide_ex.steps.simulation.success import IsTaskSuccessful
from guide_ex.utility import recording
from guide_ex.utility.exception import NodeException
from guide_ex.utility.pose import TransformPose
from guide_ex.utility.rotation import ProjectRotationToBaseZ, ReduceRotationToSymmetry
from guide_ex.utility.wait import WaitForSeconds


def legacy_tree(robot, global_context, startup_padding, record_fps):
    get_target_pose_sequence = CompositeNode(
        name="GetTargetPose",
        level=Layer.SEQUENCE,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "target": "target",
        },
        children=[
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
            "cube_pose": "cube_pose",
        },
    )

    pick_subtask = CompositeNode(
        name="Pick",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "robot_prim": "robot_prim",
            "target": "target",
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
                # Over-close onto the cube: command 0.01, past the cube's surface (~0.02),
                # and let the cube stop the fingers. The GripperActionController latches the
                # *stalled* position on success (set_hold_position), so the fingers must be
                # pushing PAST the cube for that held position to keep a squeeze force.
                # Commanding the cube's surface (~0.02) held ~zero force and the grip loosened.
                alias="CloseGripper",
                dynamic_map={"robot": "robot"},
                static_args={"gripper_goal_pos": {robot.config.gripper_joint_names[0]: 0.01}},
            ),
            WaitForSeconds(alias="WaitAfterClose", dynamic_map={"timer": "robot"}, static_args={"seconds": 0.5}),
            # Lift 28 cm straight up before heading for the bin. 20 cm taught policies a
            # lift too shallow to clear the bin walls (their main failure mode).
            TransformPose(
                alias="TransformCubeRetreatPose",
                dynamic_map={"r_pose": "cube_pose"},
                static_args={"l_pose": PoseType(position=PointType([0.0, 0.0, 0.28]))},
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
            WaitForSeconds(alias="WaitAfterOpen", dynamic_map={"timer": "robot"}, static_args={"seconds": 0.5}),
        ],
    )

    place_subtask = CompositeNode(
        name="Place",
        level=Layer.SUBTASK,
        dynamic_map={
            "robot": "robot",
            "sim_namespace": "sim_namespace",
            "scene_namespace": "scene_namespace",
            "goal": "goal",
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
            WaitForSeconds(alias="WaitAfterOpen", dynamic_map={"timer": "robot"}, static_args={"seconds": 0.5}),
        ],
    )

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
            WaitForSeconds(alias="WaitAfterTask", dynamic_map={"timer": "robot"}, static_args={"seconds": 1.0}),  # the stationary end of "go home"
        ],
    )

    pick_and_place = CompositeNode(
        name="PickAndPlace",
        level=Layer.TASK,
        dynamic_map={key: key for key, _ in global_context.items()},
        children=[
            unclutch_subtask,
            # The episode starts once the arm is unclutched at rest, not at randomization.
            recording.StartRecording(
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_id": "scene_id",
                    "path": "dataset_path",
                },
                # A busy recorder has held start_recording for ~3 min before.
                static_args={"timeout_sec": 240.0},
            ),
            get_target_pose_sequence,
            *(
                [WaitForSeconds(alias="WaitForCameras", dynamic_map={"timer": "robot"},
                                static_args={"seconds": startup_padding + 1.0 / record_fps})]
                if startup_padding
                else []
            ),
            pick_subtask,
            place_subtask,
            return_subtask,
            # ...and ends after homing: saved if the scene says the task is done, else
            # discarded. Return's WaitAfterTask has already let the scene settle.
            IsTaskSuccessful(
                alias="CheckSuccess",
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_id": "scene_id",
                },
                output_map={"success": "task_success", "reason": "task_reason"},
            ),
            recording.StopRecording(
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_id": "scene_id",
                    "save_episode": "task_success",
                },
                # Saving encodes every camera stream of the episode before it answers.
                static_args={"timeout_sec": 300.0},
            ),
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
                    WaitForSeconds(alias="WaitAfterOpenRecovery", dynamic_map={"timer": "robot"}, static_args={"seconds": 0.5}),
                    TransformPose(
                        alias="RotateCubePose",
                        dynamic_map={"l_pose": "cube_pose"},
                        static_args={
                            "r_pose": PoseType(
                                # -90 deg, not +90: the same pair of faces with the hand turned 180 deg. From the
                                # FR3's home wrist (q7 = +45 deg) the +90 turn pushed joint 7 into its limit for
                                # left-of-centre cubes (offline sweep: 25/25 reached at -90 vs 19/25 at +90).
                                orientation=RotationType(R.from_euler("z", -np.pi / 2))
                            )
                        },
                        output_map={"pose": "cube_pose"},
                    ),
                ],
                dynamic_map={
                    "robot": "robot",
                    "sim_namespace": "sim_namespace",
                    "scene_namespace": "scene_namespace",
                    "target": "target",
                },
                output_map={
                    "cube_pose": "cube_pose",
                },
                resume_target="Pick",
            ),
        },
    )
    return pick_and_place


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
