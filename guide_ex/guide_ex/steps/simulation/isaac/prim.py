import rclpy
from rclpy.node import Node

from guide_core.types.geometry import Pose
from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_msgs.srv import Collision
from guide_msgs.srv import Pose as PoseSrv


class GetPrimPose(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GetPrimPose", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot: Node, sim_namespace: str, scene_namespace: str, prim_path: str
    ) -> ExecutionResult:
        """
        Retrieves the current world pose of the specified primitive.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_namespace (str): The scene namespace.
            prim_path (str): The prim path, relative to the scene.
        Returns:
            ExecutionResult: outputs `pose`, the current world pose of the primitive.
        """

        if not robot.node:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="ROS2 node is not available for GetPrimPose.",
            )

        # Reuse the single PoseRequest client that lives on `robot.pose` (created up
        # front by the solver's main()). The check must target the SAME attribute the
        # call below uses: the old code checked `robot.node.pose` (always None) and so
        # created a SECOND client on the same /PoseRequest service every run. Two
        # clients on one service on one node breaks rmw_cyclonedds reply routing — the
        # server sends the response but the calling client's future never completes
        # (confirmed: server logs "backend RETURNED", client executor idle, future
        # never done). Create it only if truly absent, on the node's registered group.
        if getattr(robot, "pose", None) is None:
            robot.pose = robot.node.create_client(
                PoseSrv,
                f"{sim_namespace}/PoseRequest",
                callback_group=robot._reentrant_callback_group,
            )

        request = PoseSrv.Request()
        request.path = scene_namespace + prim_path

        pose_response = robot.callService(robot.pose, request, f"Getting pose for {request.path}")

        if pose_response is not None:
            pose = Pose.from_ros_pose(pose_response.pose)
            return ExecutionResult(status=DemoStatus.PERFECT, outputs={"pose": pose})
        else:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"Service call to get pose for primitive {request.path} failed.",
            )


class GetPrimPoses(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("GetPrimPoses", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot: Node, sim_namespace: str, scene_namespace: str, prim_paths: list
    ) -> ExecutionResult:
        """
        Retrieves the current world poses of several primitives, in the given order.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_namespace (str): The scene namespace.
            prim_paths (list): Prim paths relative to the scene.
        Returns:
            ExecutionResult: outputs `poses`, one Pose per path; FAILURE if any is missing.
        """
        poses = []
        for prim_path in prim_paths:
            result = GetPrimPose().run(robot, sim_namespace, scene_namespace, prim_path)
            if result.status == DemoStatus.FAILURE:
                return result
            poses.append(result.outputs["pose"])
        return ExecutionResult(status=DemoStatus.PERFECT, outputs={"poses": poses})


class IsPrimClashing(BaseNode):
    level = Layer.STEP

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("IsPrimClashing", alias, dynamic_map, static_args, output_map)

    def run(
        self,
        robot: Node,
        sim_namespace: str,
        scene_namespace: str,
        prim1_path: str,
        prim2_path: str,
    ) -> ExecutionResult:
        """
        Checks whether two primitives are clashing (the simulator compares their bounding boxes).

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_namespace (str): The scene namespace.
            prim1_path (str): The first prim path, relative to the scene.
            prim2_path (str): The second prim path, relative to the scene.
        Returns:
            ExecutionResult: outputs `has_collided`, whether the two primitives are clashing.
        """

        if not robot.node:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message="ROS2 node is not available for IsPrimClashing.",
            )

        # Same fix as GetPrimPose: reuse the single `robot.collision` client (created
        # by main()); the old check on `robot.node.collision` always created a second
        # client on the same /CollisionRequest service and would hang reply routing.
        if getattr(robot, "collision", None) is None:
            robot.collision = robot.node.create_client(
                Collision,
                f"{sim_namespace}/CollisionRequest",
                callback_group=robot._reentrant_callback_group,
            )

        request = Collision.Request()
        request.prim1 = scene_namespace + prim1_path
        request.prim2 = scene_namespace + prim2_path

        collision_response = robot.callService(
            robot.collision,
            request,
            f"Checking for clashes between {request.prim1} and {request.prim2}",
        )

        if collision_response is not None:
            return ExecutionResult(
                status=DemoStatus.PERFECT, outputs={"has_collided": collision_response.collision}
            )
        else:
            return ExecutionResult(
                status=DemoStatus.FAILURE,
                error_message=f"Service call to check for clashes between {request.prim1} and {request.prim2} failed.",
            )
