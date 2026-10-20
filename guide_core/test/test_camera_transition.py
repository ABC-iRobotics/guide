"""dataset.camera_transition: every camera starts an episode on the `from` camera's pose and
eases onto its own mount over `duration` of sim time, under the label `label`.

The camera moves run against block_bin's three cameras on an in-memory USD stage, laid out
as Isaac resolves block_bin.usd (probed 2026-10-20): translate/orient/scale cameras, the
wrist one under a body that its sensor asset turns and scales by 0.5.
"""

import logging
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from guide_core.scene.scene_manager import SceneManager
from guide_core.scene.scene_orchestrator import SceneOrchestrator, eased_pose
from guide_core.scene.scene_recorder import camera_transition
from guide_core.types.scene_state import SceneState

DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"
CAMERAS = {"top": {"path": "/top"}, "base": {"path": "/base"}, "wrist": {"path": "/fr3/rsd455/RSD455/cam"}}
ON = {"enabled": True, "duration": 2.0, "from": "base", "label": "Startup"}
HZ = 120  # physics steps per second; the dataset is 10 fps, so a frame every 12 steps


# ---- the eased move ------------------------------------------------------------------------

def angle(q0, q1):
    return (R.from_quat(q0, scalar_first=True).inv() * R.from_quat(q1, scalar_first=True)).magnitude()


def test_eased_pose_eases_in_and_out_from_one_pose_to_the_other():
    p0, p1 = np.array([1.25, 0.0, 0.75]), np.array([0.2, 0.0, 1.2])
    q0 = R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat(scalar_first=True)
    q1 = R.from_euler("xyz", [-40, 5, 120], degrees=True).as_quat(scalar_first=True)
    pose = lambda u: eased_pose(p0, q0, p1, q1, u)  # noqa: E731

    assert np.allclose(pose(0)[0], p0) and angle(pose(0)[1], q0) < 1e-9
    assert np.allclose(pose(1)[0], p1) and angle(pose(1)[1], q1) < 1e-9
    p, q = pose(0.5)  # smoothstep(0.5) = 0.5: halfway, in position and in angle
    assert np.allclose(p, (p0 + p1) / 2)
    assert np.isclose(angle(q0, q), angle(q0, q1) / 2) and np.isclose(angle(q, q1), angle(q0, q1) / 2)
    h = 1e-4  # zero speed at both ends: a step h moves O(h^2), not O(h)
    for u, v in ((0, h), (1, 1 - h)):
        assert np.linalg.norm(pose(v)[0] - pose(u)[0]) < 1e-6 * np.linalg.norm(p1 - p0)
        assert angle(pose(v)[1], pose(u)[1]) < 1e-6 * angle(q0, q1)
    for u in np.linspace(0, 1, 21):
        assert np.isclose(np.linalg.norm(pose(u)[1]), 1.0, atol=1e-12)


# ---- config ----------------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", [{}, {"camera_transition": None}, {"camera_transition": {"enabled": False}}])
def test_a_missing_or_disabled_section_is_off(dataset):
    assert camera_transition({"cameras": CAMERAS, "dataset": dataset}) is None


def test_the_section_is_read():
    assert camera_transition({"cameras": CAMERAS, "dataset": {"camera_transition": ON}}) == {
        "duration": 2.0, "from": "base", "label": "Startup"}


@pytest.mark.parametrize("bad", [
    {"from": "side"}, {"duration": 0}, {"duration": -1.0}, {"duration": "2"}, {"duration": True},
    {"label": ""}, {"label": None}, {"enabled": "yes"}, {"fps": 10},
])
def test_a_bad_section_raises(bad):
    with pytest.raises(ValueError, match="camera_transition"):
        camera_transition({"cameras": CAMERAS, "dataset": {"camera_transition": {**ON, **bad}}})


def test_a_bad_section_fails_the_registration(tmp_path):
    scene = tmp_path / "scene"
    shutil.copytree(DUMMY, scene)
    init = scene / "config" / "init.yaml"
    init.write_text(init.read_text() + "\ndataset:\n  camera_transition: {enabled: true, duration: 2.0, from: base, label: Startup}\n")

    sid, _offset, result = SceneManager(logger=logging.getLogger("test")).add_scene(str(scene))

    assert sid == -1 and "camera_transition.from" in result["error"]


# ---- GUIDE: StartRecording, the physics steps, the frames ---------------------------------------

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

DOUBLE, FLOAT = UsdGeom.XformOp.PrecisionDouble, UsdGeom.XformOp.PrecisionFloat


def block_bin_stage():
    stage = Usd.Stage.CreateInMemory()

    def prim(path, translate, orient=None, scale=(1.0, 1.0, 1.0), camera=False, rotate_zyx=None):
        x = (UsdGeom.Camera if camera else UsdGeom.Xform).Define(stage, path)
        x.AddTranslateOp(DOUBLE).Set(Gf.Vec3d(*translate))
        if rotate_zyx is not None:
            x.AddRotateZYXOp(FLOAT).Set(Gf.Vec3f(*rotate_zyx))
            x.AddScaleOp(FLOAT).Set(Gf.Vec3f(*scale))
        else:
            x.AddOrientOp(DOUBLE).Set(Gf.Quatd(*orient))
            x.AddScaleOp(DOUBLE).Set(Gf.Vec3d(*scale))

    UsdGeom.Xform.Define(stage, "/Scene_0")
    prim("/Scene_0/top", (0.2, 0.0, 1.2), (0.7071067811865476, 0.0, 0.0, 0.7071067811865475), camera=True)
    prim("/Scene_0/base", (1.25, 0.0, 0.75),
         (0.6123724356957947, 0.3535533905932736, 0.3535533905932738, 0.6123724356957944), camera=True)
    prim("/Scene_0/fr3", (-0.3, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    prim("/Scene_0/fr3/rsd455", (0.12336, 0.03536, 0.873), rotate_zyx=(-90.0, 45.0, 90.0), scale=(0.5, 0.5, 0.5))
    prim("/Scene_0/fr3/rsd455/RSD455", (0.0, 0.0, 0.0), (6.123233995736766e-17, 0.0, 0.0, 1.0))
    prim("/Scene_0/fr3/rsd455/RSD455/cam", (0.0, 0.0115, 0.0), (0.5, 0.5, 0.5, 0.5),
         scale=(1.0, 0.9999989867210388, 1.0), camera=True)
    return stage


def world(stage, name):
    """A camera's world (position, rotation), its parents' scale divided out."""
    transform = Gf.Transform(UsdGeom.Xformable(stage.GetPrimAtPath(f"/Scene_0{CAMERAS[name]['path']}"))
                             .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    q = transform.GetRotation().GetQuat()
    return np.array(transform.GetTranslation()), np.array([q.GetReal(), *q.GetImaginary()])


def same_pose(a, b, tol=1e-9):
    return np.allclose(a[0], b[0], atol=tol) and angle(a[1], b[1]) < tol


def ops(stage, name):
    """The camera's authored op values, as USD holds them."""
    return [op.Get() for op in UsdGeom.Xformable(stage.GetPrimAtPath(f"/Scene_0{CAMERAS[name]['path']}"))
            .GetOrderedXformOps()]


class Recorder:
    def __init__(self):
        self.frames = []

    def put_record_data(self, item):
        if isinstance(item, dict):
            self.frames.append(item)

    def set_output_path(self, path): pass
    def set_start_recording(self): pass
    def clear_start_recording(self): pass
    def clear_stop_recording(self): pass
    def is_idle(self): return False


class Scene(SceneOrchestrator):
    reset_preprocess = reset_postprocess = randomize_preprocess = randomize_postprocess = None
    is_success_preprocess = is_success_postprocess = None

    def check_warmup(self):
        return True


@pytest.fixture
def episode(monkeypatch):
    """`episode(section)`: StartRecording on a block_bin-like scene, then physics steps.

    Returns the stage, the recorder and `step(n)`, which runs n physics steps and gives the
    camera world poses at every frame they recorded."""
    stage = block_bin_stage()
    usd = SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: stage))
    replicator = SimpleNamespace(core=SimpleNamespace())
    for name, module in {"omni": SimpleNamespace(usd=usd, replicator=replicator), "omni.usd": usd,
                         "omni.replicator": replicator, "omni.replicator.core": replicator.core}.items():
        monkeypatch.setitem(sys.modules, name, module)

    def start(section):
        config = {"cameras": CAMERAS, "dataset": {"fps": 10, **({"camera_transition": section} if section else {})}}
        scene = Scene.__new__(Scene)
        scene._config, scene._scene_id, scene._logger = config, 0, logging.getLogger("test")
        scene.camera_transition = camera_transition(config)
        scene.task, scene.prompts = "Put the red block in the left bin.", {"task": "", "subtask": ""}
        scene.state, scene.recorder = SceneState.IDLE, Recorder()
        scene.render_products_ready, scene.warmup_frames = True, 0
        manager = SceneManager()
        manager._scenes, manager._locks = [scene], {0: threading.Lock()}
        clock = SimpleNamespace(current_time_step_index=0)
        step_task = manager.step(SimpleNamespace(_world=clock))
        manager.start_recording(0)
        poses = []

        def step(n):
            for _ in range(n):
                count = len(scene.recorder.frames)
                step_task(1.0 / HZ)
                clock.current_time_step_index += 1
                if len(scene.recorder.frames) > count:
                    poses.append({name: world(stage, name) for name in CAMERAS})
            return poses

        return SimpleNamespace(stage=stage, scene=scene, manager=manager, frames=scene.recorder.frames, step=step)

    return start


def test_start_recording_eases_every_camera_from_base_onto_its_mount_under_startup(episode):
    run = episode(ON)
    mount = {name: world(run.stage, name) for name in CAMERAS}
    authored = {name: ops(run.stage, name) for name in CAMERAS}

    run.step(1)  # the first physics step of the warm-up puts every camera on base
    assert all(same_pose(world(run.stage, name), mount["base"]) for name in CAMERAS)
    poses = run.step(40 * 12)

    assert len(run.frames) == 40
    assert [f["prompts"]["subtask"] for f in run.frames] == ["Startup"] * 20 + [""] * 20
    assert {f["task"] for f in run.frames} == {"Put the red block in the left bin."}
    assert all(same_pose(poses[0][name], mount["base"]) for name in CAMERAS)  # frame 0: one viewpoint
    for name in ("top", "wrist"):  # frame 10, t = 1 s = smoothstep's middle: halfway in position
        assert np.allclose(poses[10][name][0], (mount["base"][0] + mount[name][0]) / 2, atol=1e-9)
        assert not same_pose(poses[19][name], mount[name], tol=1e-4)
    for k in range(20, 40):  # frame duration * fps on, every camera is on its mount, op for op
        assert all(same_pose(poses[k][name], mount[name]) for name in CAMERAS)
    assert {name: ops(run.stage, name) for name in CAMERAS} == authored


def test_until_frame_0_every_camera_holds_on_base_under_its_parent_as_it_stands(episode):
    run = episode(ON)
    run.step(1)
    body = UsdGeom.Xformable(run.stage.GetPrimAtPath("/Scene_0/fr3/rsd455/RSD455"))
    body.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.001, -0.002, 0.003))  # the arm's stale hold drifts
    poses = run.step(2 * 12)

    assert all(same_pose(poses[0][name], poses[0]["base"]) for name in CAMERAS)


def test_without_the_section_no_camera_moves_and_no_frame_is_labelled(episode):
    run = episode(None)
    authored = {name: ops(run.stage, name) for name in CAMERAS}

    run.step(1)
    assert {name: ops(run.stage, name) for name in CAMERAS} == authored
    run.step(30 * 12)

    assert len(run.frames) == 30 and {f["prompts"]["subtask"] for f in run.frames} == {""}
    assert {name: ops(run.stage, name) for name in CAMERAS} == authored


def test_stopping_or_pausing_inside_the_transition_puts_the_cameras_back(episode):
    for leave in ("stop_recording", "pause_recording"):
        run = episode(ON)
        authored = {name: ops(run.stage, name) for name in CAMERAS}
        run.step(10 * 12)
        assert ops(run.stage, "top") != authored["top"]

        getattr(run.manager, leave)(0)
        run.step(1)

        assert {name: ops(run.stage, name) for name in CAMERAS} == authored, leave


def test_a_resumed_episode_does_not_start_another_transition(episode):
    run = episode(ON)
    run.step(30 * 12)
    run.manager.pause_recording(0)
    run.step(1)
    authored, paused_at = {name: ops(run.stage, name) for name in CAMERAS}, len(run.frames)

    run.manager.start_recording(0)
    run.step(1)
    assert {name: ops(run.stage, name) for name in CAMERAS} == authored
    run.step(20 * 12)

    assert len(run.frames) > paused_at and {f["prompts"]["subtask"] for f in run.frames[paused_at:]} == {""}
    assert {name: ops(run.stage, name) for name in CAMERAS} == authored
