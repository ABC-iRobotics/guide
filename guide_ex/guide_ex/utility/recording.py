from guide_ex.core.base_node import BaseNode
from guide_ex.core.states import DemoStatus, ExecutionResult, Layer
from guide_msgs import srv


def _call_recording_service(robot, service, srv_type, sim_namespace, request, timeout_sec, what):
    """Call one of the simulator's recording services and turn its answer into a result."""
    if not robot.node:
        return ExecutionResult(
            status=DemoStatus.FAILURE,
            error_message=f"ROS2 node is not available for {what}.",
        )

    # Reuse the client solve_task's main() keeps under the same name as the service
    # (robot.start_recording / robot.stop_recording). A second client on one service on
    # one node breaks rmw_cyclonedds reply routing -- see GetPrimPose.
    if getattr(robot, service, None) is None:
        setattr(
            robot,
            service,
            robot.node.create_client(
                srv_type,
                f"{sim_namespace}/{service}",
                callback_group=robot._reentrant_callback_group,
            ),
        )

    try:
        client = getattr(robot, service)
        response = robot.callService(client, request, what, timeout_sec=timeout_sec)
    except TimeoutError as e:
        return ExecutionResult(status=DemoStatus.FAILURE, error_message=str(e))

    if response is None or not response.success:
        reason = response.message if response is not None else "no response"
        return ExecutionResult(status=DemoStatus.FAILURE, error_message=f"{what} failed: {reason}")
    return ExecutionResult(status=DemoStatus.PERFECT)


class StartRecording(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("StartRecording", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot, sim_namespace: str, scene_id: int, path: str = "", timeout_sec: float = 60.0
    ) -> ExecutionResult:
        """
        Starts recording a LeRobot episode in the scene, or resumes a paused one.

        Returns once the cameras have warmed up and frames are being captured. The
        dataset itself is created on the first frame of the run, under `path`.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_id (int): The scene to record.
            path (str): Dataset base directory; empty means ~/dataset.
            timeout_sec (float): Per-attempt service timeout. A busy recorder has
                stalled start_recording for ~3 min, so raise it for long runs.
        Returns:
            ExecutionResult: PERFECT once recording, FAILURE otherwise.
        """
        return _call_recording_service(
            robot,
            "start_recording",
            srv.StartRecording,
            sim_namespace,
            srv.StartRecording.Request(id=scene_id, path=path),
            timeout_sec,
            f"Start recording scene {scene_id}",
        )


class PauseRecording(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("PauseRecording", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot, sim_namespace: str, scene_id: int, timeout_sec: float = 60.0
    ) -> ExecutionResult:
        """
        Pauses the recording; the episode stays open and StartRecording resumes it.

        The paused stretch is cut out of the episode (LeRobot numbers frames, so no
        timestamp gap). Pause only where the robot is still: the first frame after the
        resume follows the last one before the pause directly.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_id (int): The scene being recorded.
            timeout_sec (float): Per-attempt service timeout.
        Returns:
            ExecutionResult: PERFECT once paused, FAILURE if it was not recording.
        """
        return _call_recording_service(
            robot,
            "pause_recording",
            srv.PauseRecording,
            sim_namespace,
            srv.PauseRecording.Request(id=scene_id),
            timeout_sec,
            f"Pause recording scene {scene_id}",
        )


class StopRecording(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("StopRecording", alias, dynamic_map, static_args, output_map)

    def run(
        self,
        robot,
        sim_namespace: str,
        scene_id: int,
        save_episode: bool = True,
        timeout_sec: float = 60.0,
    ) -> ExecutionResult:
        """
        Stops recording and saves or discards the episode.

        Returns once the recorder has written (or dropped) the episode. The dataset stays
        open for the next episode; finalize_recording closes it.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_id (int): The scene being recorded.
            save_episode (bool): True saves the episode, False discards it.
            timeout_sec (float): Per-attempt service timeout.
        Returns:
            ExecutionResult: PERFECT once stopped, FAILURE otherwise.
        """
        return _call_recording_service(
            robot,
            "stop_recording",
            srv.StopRecording,
            sim_namespace,
            srv.StopRecording.Request(id=scene_id, save_episode=save_episode),
            timeout_sec,
            f"Stop recording scene {scene_id} (save={save_episode})",
        )


class SetPrompt(BaseNode):
    level = Layer.UTILITY

    def __init__(self, alias=None, dynamic_map=None, static_args=None, output_map=None):
        super().__init__("SetPrompt", alias, dynamic_map, static_args, output_map)

    def run(
        self, robot, sim_namespace: str, scene_id: int, task: str = "", subtask: str = "",
        timeout_sec: float = 30.0,
    ) -> ExecutionResult:
        """
        Tells the scene what the robot works on from now on, at the GUIDE-EX layers.

        A TASK composite announces its task (e.g. "Put the red cube on the blue cube.")
        together with its first subtask, each SUBTASK composite its subtask (e.g. "Pick up
        the red cube."). The two change at once, so no frame pairs a new task with the
        last task's subtask. Every frame recorded after this carries them, until the next
        prompt of the same level or the end of the episode: the task as the frames'
        LeRobot task, the subtask as LeRobot's subtask annotation (the procedure is
        recorded beside them as GUIDE's ``procedure`` style). Sending the prompts that are
        already active changes nothing, so a retried node may announce itself again.

        Args:
            robot (ROS2Robot): The ROS2 robot to use for service calls.
            sim_namespace (str): The simulation namespace.
            scene_id (int): The scene being recorded.
            task (str): The task, in natural language; empty leaves it as it is.
            subtask (str): The subtask; empty leaves it as it is.
            timeout_sec (float): Per-attempt service timeout.
        Returns:
            ExecutionResult: PERFECT once the scene has the prompts, FAILURE otherwise.
        """
        return _call_recording_service(
            robot,
            "set_prompt",
            srv.SetPrompt,
            sim_namespace,
            srv.SetPrompt.Request(id=scene_id, task=task, subtask=subtask),
            timeout_sec,
            f"Set the prompts of scene {scene_id} to task {task!r}, subtask {subtask!r}",
        )
