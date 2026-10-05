"""A frame captured while stop_recording closes the episode belongs to that episode.

The capture runs on the physics thread and takes milliseconds; stop_recording runs on a
ROS thread. A frame queued after the episode's end marker became frame 0 of the next
episode (15-20% of block_bin's recorded episodes).
"""
import threading
import time
from types import SimpleNamespace

from guide_core.scene.scene_manager import SceneManager
from guide_core.types.scene_state import SceneState

END = "FINALIZE_EPISODE"


class Recorder:
    def __init__(self):
        self.queue = []

    def put_record_data(self, item):
        self.queue.append(item)

    def clear_stop_recording(self):
        pass

    def set_start_recording(self):
        pass

    def is_idle(self):
        return False


class SlowCaptureScene:
    """record_step blocks until released, so stop_recording can arrive mid-capture."""

    record_frequency = 10

    def __init__(self):
        self.state = SceneState.RECORDING
        self.recorder = Recorder()
        self.capturing, self.release = threading.Event(), threading.Event()
        self.prompts = {"task": "Put the red cube on the blue cube.", "subtask": "Pick up the red cube."}

    def record_step(self, step):
        self.capturing.set()
        self.release.wait(2.0)
        return {"timestamp": step}


def test_a_capture_cut_by_stop_recording_stays_in_its_episode():
    manager = SceneManager()
    scene = SlowCaptureScene()
    manager._scenes, manager._locks = [scene], {0: threading.Lock()}
    step_task = manager.step(SimpleNamespace(_world=SimpleNamespace(current_time_step_index=0)))

    physics = threading.Thread(target=step_task, args=(0.1,))  # 10 Hz: captures this step
    physics.start()
    assert scene.capturing.wait(2.0)
    ros = threading.Thread(target=manager.stop_recording, args=(0,))
    ros.start()
    time.sleep(0.1)  # unguarded, stop_recording would queue its marker now
    scene.release.set()
    physics.join(2.0)
    ros.join(2.0)

    frames = [i for i, item in enumerate(scene.recorder.queue) if isinstance(item, dict)]
    assert frames and frames[-1] < scene.recorder.queue.index(END)


def test_no_capture_once_the_episode_is_closed():
    manager = SceneManager()
    scene = SlowCaptureScene()
    scene.release.set()
    manager._scenes, manager._locks = [scene], {0: threading.Lock()}
    manager.stop_recording(0)
    manager.step(SimpleNamespace(_world=SimpleNamespace(current_time_step_index=0)))(0.1)

    assert scene.recorder.queue == [END]
