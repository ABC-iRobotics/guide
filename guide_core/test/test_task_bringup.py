"""Register a task by name, directory or S3 archive: fetch it, build it, launch its bringup."""

import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from guide_core.ros import task_bringup as tb


def make_bundle(root: Path, name="my_task", reqs=False) -> Path:
    pkg = root / name
    (pkg / name).mkdir(parents=True)
    (pkg / name / "scene.py").write_text("class Scene: pass\n")
    (pkg / "package.xml").write_text(f"<package><name>{name}</name></package>\n")
    if reqs:
        (root / "requirements.txt").write_text("six\n")
    return root


@pytest.fixture
def nothing_installed(monkeypatch):
    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    monkeypatch.setattr(tb, "has_bringup", lambda name: True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("AMENT_PREFIX_PATH", os.environ.get("AMENT_PREFIX_PATH", ""))


def test_an_installed_package_is_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(tb, "is_installed", lambda name: name == "block_bin")
    monkeypatch.setattr(tb, "has_bringup", lambda name: True)
    ran = []
    assert tb.TaskBringup(0, tmp_path, run=ran.append).prepare("block_bin") == ("block_bin", "block_bin")
    assert ran == []


def test_a_bundle_is_built_with_its_dependencies(nothing_installed, monkeypatch, tmp_path):
    bundle = make_bundle(tmp_path / "bundle", reqs=True)
    monkeypatch.setenv("GUIDE_PINS", "/pins.txt")
    ran = []

    assert tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(bundle)) == ("my_task", "my_task")

    assert [cmd[:2] for cmd in ran] == [["rosdep", "install"], ["uv", "pip"], ["colcon", "--log-base"]]
    assert ran[1][-2:] == ["-c", "/pins.txt"]
    assert ran[2][-2:] == ["--packages-up-to", "my_task"]


def test_a_bundle_without_requirements_skips_pip(nothing_installed, tmp_path):
    ran = []
    tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(make_bundle(tmp_path / "b")))
    assert [cmd[0] for cmd in ran] == ["rosdep", "colcon"]


def test_a_bundle_holds_at_most_one_task(tmp_path):
    make_bundle(tmp_path, "a")
    make_bundle(tmp_path, "b")
    with pytest.raises(ValueError, match="more than one task package"):
        tb.task_package(tmp_path)


def test_a_plain_scene_directory_is_registered_without_building(nothing_installed, tmp_path):
    (tmp_path / "scene.py").write_text("class Scene: pass\n")  # like guide_core/dummy_scene
    ran = []
    assert tb.TaskBringup(0, tmp_path / "w", run=ran.append).prepare(str(tmp_path)) == (str(tmp_path), None)
    assert ran == []


def test_an_unknown_path_fails_with_the_reason(nothing_installed, tmp_path):
    with pytest.raises(FileNotFoundError, match="neither an installed package nor a directory"):
        tb.TaskBringup(0, tmp_path).prepare("no_such_task")


def test_an_s3_bundle_is_downloaded_and_unpacked(nothing_installed, tmp_path):
    src = make_bundle(tmp_path / "src")
    archive = tmp_path / "my_task.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=".")

    class FakeS3:
        def download_file(self, bucket, key, dest):
            assert (bucket, key) == ("tasks", "v1/my_task.tar.gz")
            Path(dest).write_bytes(archive.read_bytes())

    work = tmp_path / "work"
    tasks = tb.TaskBringup(0, work, run=lambda cmd: None, s3=FakeS3())

    assert tasks.prepare("s3://tasks/v1/my_task.tar.gz") == ("my_task", "my_task")
    assert (work / "src" / "my_task" / "my_task" / "my_task" / "scene.py").is_file()


def test_activation_puts_the_overlay_first(monkeypatch, tmp_path):
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    monkeypatch.setattr(sys, "path", list(sys.path))

    tb.TaskBringup(0, tmp_path).activate()

    assert os.environ["AMENT_PREFIX_PATH"] == f"{tmp_path / 'install'}{os.pathsep}/opt/ros/jazzy"
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    assert sys.path[0] == str(tmp_path / "install" / "lib" / py / "site-packages")


def test_launch_starts_one_scene_in_this_simulator(tmp_path):
    started = []

    def popen(cmd, **kwargs):
        started.append(cmd)
        return SimpleNamespace(pid=1, poll=lambda: 0)

    tb.TaskBringup(3, tmp_path, popen=popen).launch("my_task", 2)

    assert started == [[
        "bash", "-c",
        "exec ros2 launch my_task bringup.launch.py sim_id:=3 first_scene:=2 num_env:=1",
    ]]
