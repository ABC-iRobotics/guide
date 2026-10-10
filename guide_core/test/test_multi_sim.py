"""Several simulators on one ROS domain: launches, clocks, finalized datasets, shutdown, Register."""

import importlib.util
import json
import logging
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def load_task_launch(package):
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory(package)) / "launch" / "bringup.launch.py"
    spec = importlib.util.spec_from_file_location(f"{package}_bringup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("package", ["block_bin", "cube_stack"])
def test_a_task_launch_follows_its_simulator_and_clock(package):
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node, SetRemap

    context = LaunchContext()
    context.launch_configurations.update({"sim_id": "3", "first_scene": "2", "num_env": "1"})

    (group,) = load_task_launch(package).generate_nodes(context)
    entities = group.get_sub_entities()

    (remap,) = [e for e in entities if isinstance(e, SetRemap)]
    assert perform_substitutions(context, remap.src) == "/clock"
    assert perform_substitutions(context, remap.dst) == "/Sim_3/clock"
    solvers = [e for e in entities if isinstance(e, Node)]
    assert [s._Node__arguments for s in solvers] == [["--namespace", "/Sim_3/Scene_2"]]


def recorder():
    from guide_core.scene.scene_recorder import SceneRecorder

    r = SceneRecorder("pkg", "dataset_0_0", {"dataset": {"fps": 10}})
    r._logger = logging.getLogger("test_multi_sim")
    return r


def test_finalizing_reports_the_written_dataset(tmp_path):
    r = recorder()
    r.dataset = SimpleNamespace(root=tmp_path, finalize=lambda: None)

    assert r._finalize_dataset() == str(tmp_path)
    assert r._finalize_dataset() == ""  # nothing recorded since


def test_finalizing_with_saved_prompts_still_reports_the_dataset(tmp_path, monkeypatch):
    r = recorder()
    r.dataset = SimpleNamespace(root=tmp_path, finalize=lambda: None)
    r._saved_language = {0: {"task": "t"}}
    monkeypatch.setattr("guide_core.scene.scene_recorder.write_language", lambda *a, **k: True)

    assert r._finalize_dataset() == str(tmp_path)


def test_the_recorder_thread_reports_every_finalize(monkeypatch):
    r = recorder()
    monkeypatch.setattr(r, "_attach_file_log", lambda: None)  # no log file in ~/.ros
    r.start()
    assert r.wait_finalized(0) is None

    for control in ("FINALIZE", "SHUTDOWN"):
        r.clear_finalized()
        r.put_record_data(control)
        r.set_start_recording()
        assert r.wait_finalized(10) == ""  # reported: nothing was recorded
    assert r.wait_shutdown(10)


def test_a_shutdown_queued_after_a_finalize_is_not_lost(monkeypatch):
    # The FINALIZE's tail clears the start event a SHUTDOWN queued meanwhile relied on.
    r = recorder()
    monkeypatch.setattr(r, "_attach_file_log", lambda: None)
    r.start()
    r.put_record_data("FINALIZE")
    r.set_start_recording()
    assert r.wait_finalized(10) == ""

    r.put_record_data("SHUTDOWN")  # start_recording_event left clear

    assert r.wait_shutdown(5)


class FakeRecorder:
    def __init__(self, path):
        self.path, self.controls = path, []

    def clear_stop_recording(self):
        pass

    def clear_finalized(self):
        self.controls.append("clear")

    def put_record_data(self, item):
        self.controls.append(item)

    def set_start_recording(self):
        pass

    def wait_shutdown(self, timeout=None):
        return True

    def wait_finalized(self, timeout=None):
        return self.path


def test_shutdown_finalizes_every_scene_and_says_what_it_wrote(isaac_import):
    manager = isaac_import("guide_core.scene.scene_manager").SceneManager
    recorders = [FakeRecorder("/scratch/d0"), FakeRecorder("")]
    me = SimpleNamespace(
        _scenes=[SimpleNamespace(state=None, recorder=r) for r in recorders],
        _locks=[threading.Lock(), threading.Lock()],
        _logger=MagicMock(),
    )
    me.wait_finalized = MethodType(manager.wait_finalized, me)

    assert manager.finalize_all_recordings(me) == [(0, "/scratch/d0"), (1, "")]
    assert recorders[0].controls == ["clear", "SHUTDOWN"]


def test_shutdown_warns_about_a_dataset_it_could_not_wait_for(isaac_import):
    manager = isaac_import("guide_core.scene.scene_manager").SceneManager
    recorders = [FakeRecorder("/scratch/d0"), FakeRecorder(None)]
    me = SimpleNamespace(
        _scenes=[SimpleNamespace(state=None, recorder=r) for r in recorders],
        _locks=[threading.Lock(), threading.Lock()],
        _logger=MagicMock(),
    )
    me.wait_finalized = MethodType(manager.wait_finalized, me)

    assert manager.finalize_all_recordings(me) == [(0, "/scratch/d0")]
    (warning,) = me._logger.warning.call_args_list
    assert "1" in warning.args[0] and "cut off" in warning.args[0]


def test_no_solver_finalizes_once_shutdown_owns_the_datasets(isaac_import):
    manager = isaac_import("guide_core.scene.scene_manager").SceneManager
    recorder = FakeRecorder("/scratch/d0")
    me = SimpleNamespace(
        _scenes=[SimpleNamespace(state=None, recorder=recorder)], _locks=[threading.Lock()], _logger=MagicMock(),
    )
    me.wait_finalized = MethodType(manager.wait_finalized, me)
    manager.finalize_all_recordings(me)

    with pytest.raises(RuntimeError, match="shutting down"):
        manager.finalize_recording(me, 0)
    assert recorder.controls == ["clear", "SHUTDOWN"]  # its finalized event untouched


def ros_class(isaac_import):
    return isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface


def test_the_clock_is_created_in_the_simulator_namespace(isaac_import):
    from guide_msgs.srv import RegisterScene

    backend = SimpleNamespace(
        stop=MagicMock(), play=MagicMock(), call=MagicMock(),
        register_scene=MagicMock(return_value=(0, (0.0, 0.0, 0.0))),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=False, _tasks=None,
        get_namespace=lambda: "/Sim_3",
    )

    reply = ros_class(isaac_import)._register_callback(me, RegisterScene.Request(path="block_bin"), None)

    assert reply.success
    backend.call.assert_called_once_with("create_clock", namespace="Sim_3")


def register(isaac_import, prepare, bringup=True):
    from guide_msgs.srv import RegisterScene

    order = []
    backend = SimpleNamespace(
        stop=lambda: order.append("stop"),
        play=lambda: order.append("play"),
        call=MagicMock(),
        register_scene=lambda path: order.append(("register", path)) or (1, (0.0, 2.0, 0.0)),
    )
    tasks = SimpleNamespace(
        prepare=lambda path: order.append("prepare") or prepare(path),
        launch=lambda pkg, scene_id: order.append(("launch", pkg, scene_id)),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=True, _tasks=tasks,
        get_namespace=lambda: "/Sim_3",
    )
    request = RegisterScene.Request(path="s3://t/my_task.tar.gz", bringup=bringup)
    return ros_class(isaac_import)._register_callback(me, request, None), order


def test_register_builds_first_and_launches_the_scene_last(isaac_import):
    reply, order = register(isaac_import, lambda path: ("my_task", "my_task"))

    assert (reply.success, reply.id, reply.package) == (True, 1, "my_task")
    assert order == ["prepare", "stop", ("register", "my_task"), "play", ("launch", "my_task", 1)]


def test_register_without_bringup_launches_nothing(isaac_import):
    reply, order = register(isaac_import, lambda path: ("my_task", "my_task"), bringup=False)

    assert reply.success
    assert [o for o in order if o[0] == "launch"] == []


def test_a_failed_build_fails_register_and_keeps_the_simulator_running(isaac_import):
    def broken(path):
        raise RuntimeError("rosdep: cannot resolve key 'libfoo'")

    reply, order = register(isaac_import, broken)

    assert not reply.success and "libfoo" in reply.message
    assert order == ["prepare"]  # never stopped: the other scenes kept stepping


def test_bringup_needs_a_bringup_launch(isaac_import):
    reply, order = register(isaac_import, lambda path: ("/scenes/flat", None))

    assert not reply.success and "bringup.launch.py" in reply.message
    assert "stop" not in order


def test_finalize_answers_with_the_written_dataset_and_announces_it(isaac_import):
    from guide_msgs.srv import FinalizeRecording

    scenes = SimpleNamespace(finalize_recording=MagicMock(), wait_finalized=lambda id, timeout=None: "/s/d1")
    announced = []
    me = SimpleNamespace(
        _backend=SimpleNamespace(_scene_manager=scenes), _logger=MagicMock(),
        _announce_finalized=lambda i, p: announced.append((i, p)),
    )

    reply = ros_class(isaac_import)._finalize_recording_callback(me, FinalizeRecording.Request(id=2), None)

    assert (reply.success, reply.message) == (True, "/s/d1")
    assert announced == [(2, "/s/d1")]


def test_a_recorder_that_never_finishes_fails_finalize(isaac_import):
    from guide_msgs.srv import FinalizeRecording

    scenes = SimpleNamespace(finalize_recording=MagicMock(), wait_finalized=lambda id, timeout=None: None)
    announced = []
    me = SimpleNamespace(
        _backend=SimpleNamespace(_scene_manager=scenes), _logger=MagicMock(),
        _announce_finalized=lambda i, p: announced.append((i, p)),
    )

    reply = ros_class(isaac_import)._finalize_recording_callback(me, FinalizeRecording.Request(id=2), None)

    assert not reply.success and "did not finish" in reply.message
    assert announced == []


def test_announcements_are_json_with_scene_and_path(isaac_import):
    published = []
    me = SimpleNamespace(_finalized_pub=SimpleNamespace(publish=published.append), _announced=set())

    ros_class(isaac_import)._announce_finalized(me, 1, "/s/d2")

    assert json.loads(published[0].data) == {"scene": 1, "path": "/s/d2"}


def test_a_dataset_is_announced_once_and_nothing_recorded_every_time(isaac_import):
    published = []
    me = SimpleNamespace(_finalized_pub=SimpleNamespace(publish=published.append), _announced=set())
    announce = ros_class(isaac_import)._announce_finalized

    for path in ("/s/d2", "/s/d2", "", ""):
        announce(me, 1, path)

    assert [json.loads(m.data)["path"] for m in published] == ["/s/d2", "", ""]


def test_shutdown_runs_once_finalize_announce_tasks_isaac_ros(isaac_import, monkeypatch):
    module = isaac_import("guide_core.ros.guide_ros")
    order = []
    monkeypatch.setattr(module.rclpy, "try_shutdown", lambda: order.append("ros"))
    monkeypatch.setattr(module.RecorderServer, "stop", lambda: order.append("recorder"))
    scenes = SimpleNamespace(
        finalize_all_recordings=lambda: order.append("finalize") or [(0, "/s/d0"), (1, "")]
    )
    me = SimpleNamespace(
        _backend=SimpleNamespace(
            _scene_manager=scenes,
            stop=lambda: order.append("stop"),
            call=lambda name, timeout=None: order.append(name),
        ),
        _logger=MagicMock(),
        _tasks=SimpleNamespace(shutdown=lambda: order.append("tasks")),
        _shutdown_lock=threading.Lock(),
        _announce_finalized=lambda i, p: order.append(("announce", i, p)),
    )

    module.GUIDEROS2Interface.shutdown(me)
    module.GUIDEROS2Interface.shutdown(me)  # a second Ctrl-C or request changes nothing

    # The world stops before the recorder so no physics step polls a recorder that is gone.
    assert order == [
        "finalize", ("announce", 0, "/s/d0"), "tasks", "stop", "recorder", "shutdown", "ros"
    ]


def test_the_shutdown_service_answers_before_shutting_down(isaac_import):
    from std_srvs.srv import Trigger

    started = threading.Event()
    me = SimpleNamespace(shutdown=started.set)

    reply = ros_class(isaac_import)._shutdown_callback(me, Trigger.Request(), Trigger.Response())

    assert reply.success
    assert started.wait(2)


def test_the_loop_leaves_as_soon_as_a_command_shut_isaac_down(isaac_import):
    runtime = isaac_import("guide_core.core.runtime")
    me = SimpleNamespace(state=runtime.RUNNING, _gate_render=MagicMock())
    me._process_commands = lambda max_per_cycle: setattr(me, "state", runtime.UNINITIALIZED)

    runtime.IsaacSimRuntime.run_loop(me)

    me._gate_render.assert_not_called()  # Isaac is closed: touch nothing more


def test_shutdown_still_closes_isaac_when_finalizing_fails(isaac_import, monkeypatch):
    module = isaac_import("guide_core.ros.guide_ros")
    order = []
    monkeypatch.setattr(module.rclpy, "try_shutdown", lambda: order.append("ros"))

    def broken():
        raise AttributeError("recorder is None")

    me = SimpleNamespace(
        _backend=SimpleNamespace(
            _scene_manager=SimpleNamespace(finalize_all_recordings=broken),
            call=lambda name, timeout=None: order.append(name),
        ),
        _logger=MagicMock(),
        _tasks=None,
        _shutdown_lock=threading.Lock(),
    )

    module.GUIDEROS2Interface.shutdown(me)

    assert order == ["shutdown", "ros"]
    me._logger.error.assert_called_once()
