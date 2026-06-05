from __future__ import annotations

import argparse
import traceback
from threading import Thread
from typing import Optional

import rclpy
import rclpy.logging
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from guide_core.core.guide_simulator import GUIDESimulator
from guide_core.types.geometry import Pose
from guide_msgs.srv import Attribute, CheckSuccess, Collision, FinalizeRecording
from guide_msgs.srv import Pose as PoseSrv
from guide_msgs.srv import Randomize, RegisterScene, StartRecording, StopRecording


class GUIDEROS2Interface(Node):
    def __init__(self, backend: GUIDESimulator, node_name: Optional[str], namespace: Optional[str]):
        super().__init__(node_name=node_name, namespace=namespace)

        self._backend = backend

        self._has_clock = False
        self._reentrant_group = ReentrantCallbackGroup()
        self._mutually_exclusive_group = MutuallyExclusiveCallbackGroup()

        # Randomize Scene
        self._randomize_scene = self.create_service(
            srv_type=Randomize,
            srv_name="Randomize",
            callback=self._randomize_callback,
            callback_group=self._reentrant_group,
        )

        # Reset Scene
        self._reset_scene = self.create_service(
            srv_type=Randomize,
            srv_name="Reset",
            callback=self._reset_callback,
            callback_group=self._reentrant_group,
        )

        # Register Scene
        self._register_scene = self.create_service(
            srv_type=RegisterScene,
            srv_name="Register",
            callback=self._register_callback,
            callback_group=self._mutually_exclusive_group,
        )

        # Pose
        self._pose_request = self.create_service(
            srv_type=PoseSrv,
            srv_name="PoseRequest",
            callback=self._pose_request_callback,
            callback_group=self._reentrant_group,
        )

        # Attribute
        self._attribute_request = self.create_service(
            srv_type=Attribute,
            srv_name="AttributeRequest",
            callback=self._attribute_request_callback,
            callback_group=self._reentrant_group,
        )

        # Collision
        self._collision_request = self.create_service(
            srv_type=Collision,
            srv_name="CollisionRequest",
            callback=self._collision_request_callback,
            callback_group=self._reentrant_group,
        )

        # Success
        self._is_success_request = self.create_service(
            srv_type=CheckSuccess,
            srv_name="IsSuccess",
            callback=self._is_success_callback,
            callback_group=self._reentrant_group,
        )

        # Start Recording
        self._start_recording = self.create_service(
            srv_type=StartRecording,
            srv_name="start_recording",
            callback=self._start_recording_callback,
            callback_group=self._reentrant_group,
        )

        # Stop Recording
        self._stop_recording = self.create_service(
            srv_type=StopRecording,
            srv_name="stop_recording",
            callback=self._stop_recording_callback,
            callback_group=self._reentrant_group,
        )

        # Finalize Recording
        self._finalize_recording = self.create_service(
            srv_type=FinalizeRecording,
            srv_name="finalize_recording",
            callback=self._finalize_recording_callback,
            callback_group=self._reentrant_group,
        )

    def _randomize_callback(
        self, request: Randomize.Request, response: Randomize.Response
    ) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id

            output = self._backend.randomize_scene(scene_id=id)

            response.message = output if isinstance(output, str) else ""
            response.success = output is not None
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response

    def _reset_callback(
        self, request: Randomize.Request, response: Randomize.Response
    ) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id

            success = self._backend.reset_scene(scene_id=id)

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _register_callback(
        self, request: RegisterScene.Request, response: RegisterScene.Response
    ) -> RegisterScene.Response:
        response = RegisterScene.Response()
        try:
            self._backend.stop()

            path = request.path

            id, offset = self._backend.register_scene(path)

            self._logger.info(f"Registered scene with id {id} at offset {offset}")

            if not self._has_clock:
                self._backend.call("create_clock")
                self._has_clock = True

            self._backend.play()
            response.id = id
            response.offset = list(offset)
            response.message = ""
            response.success = True

        except Exception as e:
            err_msg = f"{e}\n{traceback.format_exc()}"
            self._logger.error(f"Failed to register scene: {err_msg}")
            response.id = -1
            response.offset = [0.0, 0.0, 0.0]
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _pose_request_callback(
        self, request: PoseSrv.Request, response: PoseSrv.Response
    ) -> PoseSrv.Response:
        response = PoseSrv.Response()
        try:
            path = request.path

            pose: Pose = self._backend.call("get_world_poses", prim_path=path)
            if isinstance(pose, bool):
                raise Exception("Getting pose failed!")

            response.pose = pose.to_ros()
            response.message = ""
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _attribute_request_callback(
        self, request: Randomize.Request, response: Randomize.Response
    ) -> Randomize.Response:
        response = Randomize.Response()
        try:
            path = request.path
            attribute = request.attribute
            value = request.value

            success = self._backend.call(
                "set_prim_attribute_value", prim_path=path, attribute_name=attribute, value=value
            )

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _collision_request_callback(
        self, request: Collision.Request, response: Collision.Response
    ) -> Collision.Response:
        response = Collision.Response()
        try:
            prim1 = request.prim1
            prim2 = request.prim2

            collision = self._backend.call(
                "check_bounding_box_collision", prim_path=prim1, target_scope=prim2
            )

            response.collision = collision
        except Exception:
            response.collision = False
        finally:
            return response

    def _is_success_callback(
        self, request: CheckSuccess.Request, response: CheckSuccess.Response
    ) -> CheckSuccess.Response:
        try:
            id = request.id
            self.get_logger().info(f"[TRACE] _is_success_callback: ENTER, scene_id={id}")

            success = self._backend.is_success(id)

            self.get_logger().info(f"[TRACE] _is_success_callback: is_success returned {success}")
            response.message = ""
            response.success = success
        except Exception as e:
            self.get_logger().warn(
                f"[TRACE] _is_success_callback: EXCEPTION: {type(e).__name__}: {e}"
            )
            response.message = str(e)
            response.success = False
        self.get_logger().info(
            f"[TRACE] _is_success_callback: EXIT, response.success={response.success}"
        )
        return response

    def _start_recording_callback(
        self, request: StartRecording.Request, response: StartRecording.Response
    ) -> StartRecording.Response:
        response = StartRecording.Response()
        try:
            id = request.id
            self._logger.info(f"Starting recording for scene {id}...")

            self._backend._scene_manager.start_recording(id)

            # Block until warmup is done and state is RECORDING
            self._backend._scene_manager.wait_start_recording_event(id)

            response.message = "Recording started successfully."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _stop_recording_callback(
        self, request: StopRecording.Request, response: StopRecording.Response
    ) -> StopRecording.Response:
        response = StopRecording.Response()
        try:
            id = request.id
            save_episode = request.save_episode
            self._logger.info(f"Stopping recording for scene {id}... (Save: {save_episode})")

            self._backend._scene_manager.clear_idle_event(id)
            self._backend._scene_manager.stop_recording(id, save_episode)

            # Block until Consumer thread processes Poison Pill
            self._backend._scene_manager.wait_stop_recording_event(id)

            # Also block until scene resets and transitions to IDLE (if not already)
            # Actually we can just wait until the recorder is idle.
            # self._backend._scene_manager.wait_idle_event(id)

            response.message = "Recording stopped and scene reset successfully."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _finalize_recording_callback(
        self, request: FinalizeRecording.Request, response: FinalizeRecording.Response
    ) -> FinalizeRecording.Response:
        response = FinalizeRecording.Response()
        try:
            id = request.id
            self._logger.info(f"Finalizing recording for scene {id}...")

            self._backend._scene_manager.finalize_recording(id)

            response.message = "Recording finalized successfully."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response


def launch_ros_interface(node: GUIDEROS2Interface):
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()

        node.destroy_node()
        rclpy.shutdown()

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutting down cleanly... Finalizing all datasets!")
        try:
            if hasattr(node, "_backend") and node._backend is not None:
                if (
                    hasattr(node._backend, "_scene_manager")
                    and node._backend._scene_manager is not None
                ):
                    node._backend._scene_manager.finalize_all_recordings()
        except Exception as e:
            node.get_logger().error(f"Error while finalizing datasets on shutdown: {e}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--id",
        type=int,
        help="Id number of the initialized simulation. Used for namespacing.",
        default=0,
    )
    parser.add_argument("-d", "--debug", type=str2bool, help="Debug simulator.", default=False)


def ros_entry_point():
    parser = argparse.ArgumentParser(
        description="GUIDE synthetic data generator framework ROS 2 entry point."
    )
    create_arguments(parser)
    args = parser.parse_args()

    NAMESPACE = f"Sim_{args.id}"

    # 1. Initialize Isaac Sim BEFORE ROS 2 to prevent deadlocks with ROS 2 bridge
    sim = GUIDESimulator(sim_id=args.id, namespace=NAMESPACE)
    sim.init_runtime(debug=args.debug, logger=None)
    sim.init_scene_manager()

    # 2. Initialize ROS 2
    rclpy.init(args=None)

    ros_interface = GUIDEROS2Interface(sim, node_name="GUIDE", namespace=NAMESPACE)

    # Dynamically set ROS 2 node logger severity based on debug CLI flag
    from rclpy.logging import LoggingSeverity

    severity = LoggingSeverity.DEBUG if args.debug else LoggingSeverity.INFO
    ros_interface.get_logger().set_level(severity)

    # Update loggers to use the ROS 2 logger
    sim._logger = ros_interface.get_logger()
    sim._runtime._logger = ros_interface.get_logger()
    sim._scene_manager._logger = ros_interface.get_logger()

    ros_t = Thread(target=launch_ros_interface, args=(ros_interface,))
    ros_t.start()

    ros_interface.get_logger().info("Running simulation loop...")
    sim.run_runtime_loop()

    ros_t.join()


if __name__ == "__main__":
    ros_entry_point()
