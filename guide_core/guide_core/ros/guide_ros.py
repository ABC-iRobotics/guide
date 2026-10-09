from __future__ import annotations

import argparse
import json
import signal
import traceback
from threading import Lock, Thread
from typing import Optional

import rclpy
import rclpy.logging


class _RosLoggerAdapter:
    """Make GUIDE's Python-logging-style calls safe on ROS 2 Jazzy's rclpy logger.

    Jazzy's rclpy logger requires a **string** message and rejects unknown
    keyword arguments. GUIDE (written for older rclpy) passes exception objects
    (``logger.error(e)``) and Python-logging kwargs (``exc_info=True``), which
    raise ``TypeError`` on Jazzy — turning a simple error into a crash cascade.
    This wrapper coerces the message to ``str`` and drops unsupported kwargs.
    """

    _ALLOWED = {"throttle_duration_sec", "throttle_time_source_type", "skip_first", "once"}

    def __init__(self, logger) -> None:
        self._logger = logger

    def _kw(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k in self._ALLOWED}

    def debug(self, msg, **kw) -> None:
        self._logger.debug(str(msg), **self._kw(kw))

    def info(self, msg, **kw) -> None:
        self._logger.info(str(msg), **self._kw(kw))

    def warning(self, msg, **kw) -> None:
        self._logger.warning(str(msg), **self._kw(kw))

    warn = warning

    def error(self, msg, **kw) -> None:
        self._logger.error(str(msg), **self._kw(kw))

    def fatal(self, msg, **kw) -> None:
        self._logger.fatal(str(msg), **self._kw(kw))

    critical = fatal

    def __getattr__(self, name):
        # Delegate anything else (e.g. set_level, get_child) to the real logger.
        return getattr(self._logger, name)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from std_srvs.srv import Trigger

from guide_core.core.guide_simulator import GUIDESimulator
from guide_core.types.geometry import Pose
from guide_msgs.srv import Attribute, CheckSuccess, Collision, FinalizeRecording
from guide_msgs.srv import Pose as PoseSrv
from guide_msgs.srv import (
    PauseRecording,
    Randomize,
    RegisterScene,
    SetPrompt,
    StartRecording,
    StopRecording,
)

# Writing a long dataset's videos takes minutes; past this, finalize reports a stuck recorder.
FINALIZE_TIMEOUT_S = 600.0


class GUIDEROS2Interface(Node):
    def __init__(
        self,
        backend: GUIDESimulator,
        node_name: Optional[str],
        namespace: Optional[str],
        tasks=None,
    ):
        super().__init__(node_name=node_name, namespace=namespace)

        self._backend = backend
        # Fetches, builds and launches tasks for Register (guide_core.ros.task_bringup).
        self._tasks = tasks
        self._shutdown_lock = Lock()
        self._shutting_down = False

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

        # Pause Recording (resume = start_recording)
        self._pause_recording = self.create_service(
            srv_type=PauseRecording,
            srv_name="pause_recording",
            callback=self._pause_recording_callback,
            callback_group=self._reentrant_group,
        )

        # Task / subtask prompt: stamped on every frame recorded after it
        self._set_prompt = self.create_service(
            srv_type=SetPrompt,
            srv_name="set_prompt",
            callback=self._set_prompt_callback,
            callback_group=self._reentrant_group,
        )

        # Finalize Recording
        self._finalize_recording = self.create_service(
            srv_type=FinalizeRecording,
            srv_name="finalize_recording",
            callback=self._finalize_recording_callback,
            callback_group=self._reentrant_group,
        )

        # Every dataset GUIDE finalizes, for whoever drives this simulator (a master, a
        # script): {"scene": id, "path": dir}; path "" when the scene recorded nothing.
        self._finalized_pub = self.create_publisher(
            String,
            "dataset_finalized",
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # Stop this simulator. Ctrl-C and SIGTERM take the same path (ros_entry_point).
        self._shutdown_service = self.create_service(
            srv_type=Trigger,
            srv_name="shutdown",
            callback=self._shutdown_callback,
            callback_group=self._reentrant_group,
        )

    def shutdown(self) -> None:
        """The one way this simulator stops -- /shutdown, Ctrl-C and SIGTERM alike: finalize
        every scene (announcing what was written), stop the task launches, close Isaac, end ROS."""
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        self._logger.info("Shutting down: finalizing every scene...")
        try:
            for scene_id, path in self._backend._scene_manager.finalize_all_recordings():
                if path:
                    self._announce_finalized(scene_id, path)
            if self._tasks:
                self._tasks.shutdown()
        except Exception as e:  # Isaac must still close and ROS end, or only SIGKILL stops us
            self._logger.error(f"Shutdown: finalizing or stopping tasks failed: {e}")
        try:
            self._backend.call("shutdown", 60.0)  # closes Isaac; run_runtime_loop returns
        finally:
            rclpy.try_shutdown()

    def _shutdown_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        Thread(target=self.shutdown).start()  # answer first: the shutdown ends this node
        response.success = True
        response.message = "Shutting down."
        return response

    def _announce_finalized(self, scene_id: int, path: str) -> None:
        self._finalized_pub.publish(String(data=json.dumps({"scene": scene_id, "path": path})))

    def _randomize_callback(
        self, request: Randomize.Request, response: Randomize.Response
    ) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id

            output = self._backend.randomize_scene(
                scene_id=id,
                use_zone=request.use_zone,
                zone=request.zone,
                seed=request.seed if request.use_seed else None,
            )

            response.message = output if isinstance(output, str) else ""
            response.success = output is not None
        except Exception as e:
            response.message = str(e)
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
                # /Sim_N/clock: every simulator runs at its own speed. Task launches remap
                # their nodes' /clock to it (SetRemap in <task>/launch/bringup.launch.py).
                self._backend.call("create_clock", namespace=self.get_namespace().strip("/"))
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
        self, request: Attribute.Request, response: Attribute.Response
    ) -> Attribute.Response:
        response = Attribute.Response()
        try:
            path = request.path
            attribute = request.attribute
            value = request.value

            success = self._backend.call(
                "set_prim_attribute_value", prim_path=path, attribute_name=attribute, value=value
            )

            response.result = str(success)
        except Exception as e:
            response.result = str(e)
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

            self._backend._scene_manager.start_recording(id, request.path)

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

            if self._backend._scene_manager.stop_recording(id, save_episode):
                # Block until the SceneRecorder thread has saved or discarded the episode
                # (it sets the stop event on FINALIZE_EPISODE / DISCARD_EPISODE)
                self._backend._scene_manager.wait_stop_recording_event(id)
                response.message = "Recording stopped."
            else:
                # Idempotent: a caller cleaning up after a failed run may stop a scene
                # that never started, or already stopped.
                response.message = "Not recording; nothing to stop."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _pause_recording_callback(
        self, request: PauseRecording.Request, response: PauseRecording.Response
    ) -> PauseRecording.Response:
        response = PauseRecording.Response()
        try:
            self._logger.info(f"Pausing recording for scene {request.id}...")
            self._backend._scene_manager.pause_recording(request.id)
            response.message = "Recording paused; start_recording resumes it."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response

    def _set_prompt_callback(
        self, request: SetPrompt.Request, response: SetPrompt.Response
    ) -> SetPrompt.Response:
        response = SetPrompt.Response()
        try:
            self._logger.info(f"Prompts of scene {request.id}: task {request.task!r}, subtask {request.subtask!r}")
            self._backend._scene_manager.set_prompt(request.id, request.task, request.subtask)
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        return response

    def _finalize_recording_callback(
        self, request: FinalizeRecording.Request, response: FinalizeRecording.Response
    ) -> FinalizeRecording.Response:
        response = FinalizeRecording.Response()
        try:
            id = request.id
            self._logger.info(f"Finalizing recording for scene {id}...")

            self._backend._scene_manager.finalize_recording(id)
            # Answer once the recorder has written the dataset: it adds the language columns
            # after LeRobot's finalize, and only then is the dataset complete.
            path = self._backend._scene_manager.wait_finalized(id, FINALIZE_TIMEOUT_S)
            if path is None:
                raise TimeoutError(f"the recorder did not finish scene {id} in {FINALIZE_TIMEOUT_S:.0f} s")
            self._announce_finalized(id, path)

            response.message = path or "Nothing was recorded."
            response.success = True
        except Exception as e:
            response.message = str(e)
            response.success = False
        finally:
            return response


def launch_ros_interface(node: GUIDEROS2Interface):
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except ExternalShutdownException:
        pass  # GUIDEROS2Interface.shutdown ended ROS after finalizing every scene


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

    # 2. Initialize ROS 2. GUIDE handles the signals itself so that Ctrl-C, SIGTERM (docker stop,
    #    ros2 launch) and /shutdown all take ros_interface.shutdown: rclpy's own handler interrupts
    #    Isaac's loop mid-frame on SIGINT and never ends it on SIGTERM.
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)

    ros_interface = GUIDEROS2Interface(sim, node_name="GUIDE", namespace=NAMESPACE)

    # Dynamically set ROS 2 node logger severity based on debug CLI flag
    from rclpy.logging import LoggingSeverity

    severity = LoggingSeverity.DEBUG if args.debug else LoggingSeverity.INFO
    ros_interface.get_logger().set_level(severity)

    # Update loggers to use the ROS 2 logger (wrapped so exception/kwarg-style
    # calls from the backend are Jazzy-rclpy safe).
    backend_logger = _RosLoggerAdapter(ros_interface.get_logger())
    sim._logger = backend_logger
    sim._runtime._logger = backend_logger
    sim._scene_manager._logger = backend_logger

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: Thread(target=ros_interface.shutdown).start())

    ros_t = Thread(target=launch_ros_interface, args=(ros_interface,))
    ros_t.start()

    ros_interface.get_logger().info("Running simulation loop...")
    sim.run_runtime_loop()

    ros_t.join()


if __name__ == "__main__":
    ros_entry_point()
