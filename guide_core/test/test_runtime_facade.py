import time
from test.mock_backend import MockIsaacRuntime

import pytest

from guide_core.core.runtime_facade import RuntimeFacade, TaskDescriptor
from guide_core.types.isaac_state import IsaacState


@pytest.fixture
def runtime():
    backend = MockIsaacRuntime()
    return RuntimeFacade(backend)


def test_load_task(runtime):
    td = TaskDescriptor(stage_config={}, timeout_s=5.0)

    runtime.load_task(td)

    assert runtime.state == IsaacState.READY


def test_start_pause_stop(runtime):
    runtime.load_task(TaskDescriptor({}, 5.0))

    runtime.start()
    assert runtime.state == IsaacState.RUNNING

    runtime.pause()
    assert runtime.state == IsaacState.PAUSED

    runtime.stop()
    assert runtime.state == IsaacState.STOPPED


def test_scene_registry(runtime):
    runtime.load_task(TaskDescriptor({}, 5.0))

    runtime.register_scene("scene_1")
    runtime.register_scene("scene_2")

    scenes = runtime.list_scenes()

    assert "scene_1" in scenes
    assert "scene_2" in scenes


def test_call_wrapper(runtime):
    runtime.load_task(TaskDescriptor({}, 5.0))

    runtime.call("start")
    assert runtime.state == IsaacState.RUNNING


def test_tick_once(runtime):
    runtime.load_task(TaskDescriptor({}, 5.0))
    runtime.start()

    runtime.tick_once()
    runtime.tick_once()

    assert runtime._backend.get_ticks() == 2


# Not possible to test timeout without async mock backend
# def test_timeout(runtime):
#     runtime.load_task(TaskDescriptor({}, timeout_s=0.1))
#     runtime.start()

#     time.sleep(0.2)

#     runtime.tick_once()

#     assert runtime.state == IsaacState.STOPPED


def test_run_loop(runtime):
    runtime.load_task(TaskDescriptor({}, 1.0))

    runtime.start()

    # futtassuk külön thread-ben
    import threading

    t = threading.Thread(target=runtime.run_loop)
    t.start()

    time.sleep(0.2)

    runtime.stop()
    t.join()

    assert runtime.state == IsaacState.STOPPED
