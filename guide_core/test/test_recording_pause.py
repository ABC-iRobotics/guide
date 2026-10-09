"""Pause/resume/stop against the real recorder writer loop (no Isaac, no LeRobot write).

Frames without a "timestamp" key never reach LeRobot, so the writer's control protocol
(start flag, queue, stop event) runs exactly as in production. A paused writer parks on
the start flag; stop must still get its signal through or the StopRecording service
waits forever.
"""
import threading
import time

import pytest

from guide_core.scene.scene_manager import SceneManager
from guide_core.scene.scene_recorder import SceneRecorder
from guide_core.types.scene_state import SceneState


class FakeScene:
    def __init__(self):
        self.state = SceneState.IDLE
        self.recorder = SceneRecorder("pkg", "task", {})
        self.prompts = {"task": "", "subtask": ""}

    def set_render_products_enabled(self, enabled):
        pass

    def clear_recording_history(self):
        pass


@pytest.fixture
def manager():
    try:  # the writer imports lerobot before its loop; warm it so the loop starts at once
        import lerobot.datasets.lerobot_dataset  # noqa: F401
    except ImportError:
        pass
    m = SceneManager()
    scene = FakeScene()
    m._scenes, m._locks = [scene], {0: threading.Lock()}
    writer = threading.Thread(target=scene.recorder.run, daemon=True)
    writer.start()
    yield m, scene
    scene.recorder.stop_flag.set()
    scene.recorder.set_start_recording()
    scene.recorder.put_record_data("SHUTDOWN")


def record(manager, frames=3):
    """start_recording + what SceneManager.step does once the warm-up is done."""
    m, scene = manager
    m.start_recording(0)
    scene.state = SceneState.RECORDING
    scene.recorder.set_start_recording()
    for _ in range(frames):
        scene.recorder.put_record_data({"frame": True})


def drained(scene, timeout=2.0):
    end = time.monotonic() + timeout
    while not scene.recorder.record_queue.empty() and time.monotonic() < end:
        time.sleep(0.01)
    time.sleep(0.1)  # let the writer re-check its loop condition and park


def test_stop_after_pause_reaches_the_parked_writer(manager):
    m, scene = manager
    record(manager)
    m.pause_recording(0)
    drained(scene)  # writer has left its loop and waits on the start flag

    assert m.stop_recording(0, save_episode=True)
    assert scene.recorder.wait_stop_recording(2.0), "stop signal never reached the writer"
    assert not scene.recorder.start_recording_event.is_set()


def test_resume_is_start_and_the_episode_stays_open(manager):
    m, scene = manager
    record(manager)
    m.pause_recording(0)
    assert scene.state == SceneState.PAUSED
    drained(scene)

    record(manager)  # resume == start_recording
    assert scene.state == SceneState.RECORDING
    assert m.stop_recording(0, save_episode=False)
    assert scene.recorder.wait_stop_recording(2.0)


def test_only_a_recording_scene_pauses(manager):
    m, _ = manager
    with pytest.raises(RuntimeError, match="IDLE"):
        m.pause_recording(0)


def test_stop_without_an_open_episode_is_a_no_op(manager):
    m, scene = manager
    assert m.stop_recording(0) is False
    assert scene.recorder.record_queue.empty()
