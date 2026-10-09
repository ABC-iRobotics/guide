"""Several simulators on one ROS domain: launches, clocks, finalized datasets, shutdown, Register."""

import importlib.util
import logging
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

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
    )
    me.wait_finalized = MethodType(manager.wait_finalized, me)

    assert manager.finalize_all_recordings(me) == [(0, "/scratch/d0"), (1, "")]
    assert recorders[0].controls == ["clear", "SHUTDOWN"]
