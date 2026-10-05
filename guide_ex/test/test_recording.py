from types import SimpleNamespace

from guide_ex.core.states import DemoStatus, Layer
from guide_ex.utility.recording import (
    PauseRecording,
    SetPrompt,
    StartRecording,
    StopRecording,
)


class FakeRobot:
    """Records service calls; answers with `response` or raises `error`."""

    def __init__(self, response=None, error=None):
        self.node = SimpleNamespace(create_client=self._create_client)
        self._reentrant_callback_group = object()
        self.created, self.calls = [], []
        self.response = response or SimpleNamespace(success=True, message="")
        self.error = error

    def _create_client(self, srv_type, name, callback_group):
        self.created.append(name)
        return name

    def callService(self, client, request, message=None, timeout_sec=60.0):
        self.calls.append((client, request, timeout_sec))
        if self.error:
            raise self.error
        return self.response


def test_start_reuses_the_existing_client_and_sends_scene_and_path():
    robot = FakeRobot()
    robot.start_recording = "existing"

    result = StartRecording().run(robot, "/Sim_0", scene_id=2, path="~/ds", timeout_sec=200)

    assert result.status == DemoStatus.PERFECT
    client, request, timeout = robot.calls[0]
    assert (client, request.id, request.path, timeout) == ("existing", 2, "~/ds", 200)
    assert robot.created == []


def test_stop_creates_its_client_once_and_can_discard():
    robot = FakeRobot()
    node = StopRecording()

    node.run(robot, "/Sim_0", scene_id=0, save_episode=False)
    node.run(robot, "/Sim_0", scene_id=0)

    assert robot.created == ["/Sim_0/stop_recording"]
    assert [r.save_episode for _, r, _ in robot.calls] == [False, True]


def test_pause_calls_its_own_service_and_start_resumes():
    robot = FakeRobot()

    PauseRecording().run(robot, "/Sim_0", scene_id=3)
    StartRecording().run(robot, "/Sim_0", scene_id=3)

    assert robot.created == ["/Sim_0/pause_recording", "/Sim_0/start_recording"]
    assert [r.id for _, r, _ in robot.calls] == [3, 3]


def test_a_refused_call_is_a_failure_with_the_reason():
    robot = FakeRobot(response=SimpleNamespace(success=False, message="no such scene"))

    result = StartRecording().run(robot, "/Sim_0", scene_id=9)

    assert result.status == DemoStatus.FAILURE
    assert "no such scene" in result.error_message


def test_a_timeout_is_a_failure_not_an_exception():
    robot = FakeRobot(error=TimeoutError("Service 'start_recording' timed out after 60s"))

    result = StartRecording().run(robot, "/Sim_0", scene_id=0)

    assert result.status == DemoStatus.FAILURE
    assert "timed out" in result.error_message


def test_both_are_utility_nodes_wired_through_the_context():
    robot = FakeRobot()
    node = StopRecording(
        dynamic_map={"robot": "robot", "sim_namespace": "sim_namespace", "scene_id": "scene_id"},
        static_args={"save_episode": False},
    )

    result = node.execute({"robot": robot, "sim_namespace": "/Sim_0", "scene_id": 1})

    assert StartRecording.level == StopRecording.level == Layer.UTILITY
    assert result.status == DemoStatus.PERFECT
    assert robot.calls[0][1].id == 1


def test_task_and_subtask_go_to_their_service_in_one_request():
    robot = FakeRobot()
    node = SetPrompt(
        dynamic_map={"robot": "robot", "sim_namespace": "sim_namespace", "scene_id": "scene_id"},
        static_args={"task": "Put the red cube on the blue cube.", "subtask": "Pick up the red cube."},
    )

    result = node.execute({"robot": robot, "sim_namespace": "/Sim_0", "scene_id": 2})

    assert SetPrompt.level == Layer.UTILITY
    assert result.status == DemoStatus.PERFECT
    assert robot.created == ["/Sim_0/set_prompt"]
    request = robot.calls[0][1]
    assert (request.id, request.task, request.subtask) == (
        2, "Put the red cube on the blue cube.", "Pick up the red cube.")
