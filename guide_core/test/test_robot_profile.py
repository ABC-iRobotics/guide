"""Robot profiles (S3 v2 contract B -> C): resolve() pure, mount() on a pxr stage, build_stage at registration."""
import logging
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from guide_core.scene import robot_profile as rp
from guide_core.scene.scene_manager import SceneManager
from guide_motion.description import TCP_Z

TASKS = Path(__file__).resolve().parents[2] / "guide_tasks"
DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"


@pytest.mark.parametrize("task", ["block_bin", "cube_stack"])
def test_a_robot_inside_the_scene_usd_comes_through_unchanged(task):
    """Review focus 4: no profile, no mount -> exactly as written, and nothing to mount."""
    for entry in yaml.safe_load((TASKS / task / "config" / "init.yaml").read_text())["robots"].values():
        assert rp.resolve(entry) == entry and "profile" not in rp.resolve(entry)


def test_a_profile_fills_the_robot_in():
    entry = rp.resolve({"profile": "panda"})
    assert entry["path"] == "/panda" and entry["profile"] == "panda"
    assert entry["joint_names"] == [f"panda_joint{i}" for i in range(1, 8)]
    assert (entry["gripper_joint"], entry["end_effector_name"], entry["base_name"]) == (
        "panda_finger_joint1", "panda_hand_tcp", "panda_link0")
    assert len(entry["default_joint_states"]) == 9
    assert (entry["controller"], entry["oscbf"]) == ("oscbf", {"robot": "panda"})


def test_a_mount_alone_means_the_fr3_and_explicit_keys_win():
    entry = rp.resolve({"mount": {"position": [-0.66, 0.0, 0.012]}, "base_name": "fr3_link1",
                        "oscbf": {"mode": "velocity"}})
    assert (entry["profile"], entry["path"], entry["base_name"]) == ("fr3", "/fr3", "fr3_link1")
    assert entry["oscbf"] == {"robot": "fr3", "mode": "velocity"}


@pytest.mark.parametrize("entry, message", [
    ({"profile": "ur5"}, "ur5"),
    ({"mount": {"position": [0, 0]}}, "position"),
    ({"mount": {"orientation": {"quat": [0, 0, 0, 1]}}}, "euler"),
    ({"mount": {"origin": [0, 0, 0]}}, "origin"),
    ({"mount": {"orientation": {"euler": ["xyz", 0.1]}}}, "euler"),
])
def test_a_bad_profile_or_mount_fails_the_registration(entry, message):
    with pytest.raises(ValueError, match=message):
        rp.resolve(entry)


def test_mount_pose_reads_euler_radians():
    position, quat = rp.mount_pose({"mount": {"position": [1, 2, 3], "orientation": {"euler": ["z", math.pi / 2]}}})
    assert position.tolist() == [1, 2, 3]
    assert np.allclose(quat, R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True))
    assert np.allclose(rp.mount_pose({})[1], [1, 0, 0, 0])


def test_focal_length_gives_the_field_of_view():
    assert math.isclose(rp.focal_length(45.0), 20.955 / (2 * math.tan(math.radians(22.5))))


def robot_usd(tmp_path, robot):
    """A stand-in for guide_motion's robots/<p>/<p>.usda: link0 and the hand, and an authored
    translate the mount must override."""
    path = tmp_path / f"{robot}.usda"
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "{robot}"
)

def Xform "{robot}"
{{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "{robot}_link0"
    {{
    }}

    def Xform "{robot}_hand"
    {{
    }}
}}
''')
    return str(path)


@pytest.mark.parametrize("robot", ["fr3", "panda"])
def test_mount_references_the_robot_at_its_mount_with_its_wrist_camera(tmp_path, robot):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Scene_0").AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))
    entry = rp.resolve({"profile": robot, "mount": {"position": [-0.66, 0.0, 0.012], "orientation": {"euler": ["z", math.pi]}}})
    path = rp.mount(stage, "/Scene_0", entry, usd=robot_usd(tmp_path, robot))
    assert path == f"/Scene_0/{robot}"
    world = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert np.allclose(world.ExtractTranslation(), [-0.66, 2.0, 0.012])
    assert np.isclose(abs(world.ExtractRotation().GetAngle()), 180.0)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(f"{path}/{robot}_hand/wrist_camera"))
    assert camera.GetHorizontalApertureAttr().Get() == camera.GetVerticalApertureAttr().Get()
    assert math.isclose(camera.GetFocalLengthAttr().Get(), rp.focal_length(75.0), rel_tol=1e-6)
    assert camera.GetPrim().GetAttribute("xformOp:translate").Get() == Gf.Vec3d(0.05, 0, 0)
    orient = camera.GetPrim().GetAttribute("xformOp:orient").Get()  # what a camera transition moves
    assert np.allclose([orient.GetReal(), *orient.GetImaginary()], rp.WRIST_CAMERA["orientation_wxyz"])
    tcp = stage.GetPrimAtPath(f"{path}/{robot}_hand/{robot}_hand_tcp")
    assert tcp.IsValid() == (robot == "panda")  # Isaac's FR3 asset has its own fr3_hand_tcp
    if robot == "panda":
        assert tcp.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(0, 0, TCP_Z)


def test_mount_refuses_a_path_the_scene_usd_already_has(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Scene_0/fr3")
    with pytest.raises(ValueError, match="already exists"):
        rp.mount(stage, "/Scene_0", rp.resolve({"profile": "fr3"}), usd=robot_usd(tmp_path, "fr3"))


def registered(tmp_path, robots):
    task = tmp_path / "task"
    shutil.copytree(DUMMY, task)
    config = yaml.safe_load((task / "config" / "init.yaml").read_text())
    config["robots"] = robots
    (task / "config" / "init.yaml").write_text(yaml.safe_dump(config))
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task))
    assert sid == 0, cfg
    return manager._scenes[0], cfg


def stub_isaac(monkeypatch):
    """omni.usd, omni.kit.app and isaacsim's is_stage_loading, as build_stage imports them."""
    stage = object()
    omni = SimpleNamespace(usd=SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: stage)),
                           kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda: SimpleNamespace(update=lambda: None))))
    for name, module in {"omni": omni, "omni.usd": omni.usd, "omni.kit": omni.kit, "omni.kit.app": omni.kit.app,
                         "isaacsim": SimpleNamespace(), "isaacsim.core": SimpleNamespace(),
                         "isaacsim.core.utils": SimpleNamespace(),
                         "isaacsim.core.utils.stage": SimpleNamespace(is_stage_loading=lambda: False)}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return stage


def test_registration_resolves_the_profile_into_the_scene_config(tmp_path):
    _, cfg = registered(tmp_path, {"franka": {"profile": "panda", "mount": {"position": [-0.66, 0, 0.012]}}})
    assert cfg["robots"]["franka"]["path"] == "/panda" and cfg["robots"]["franka"]["oscbf"]["robot"] == "panda"


def test_build_stage_mounts_the_profile_robots_only(tmp_path, monkeypatch):
    scene, _ = registered(tmp_path, {"franka": {"profile": "fr3"},
                                     "legacy": {"path": "/ur", "default_joint_states": [0.0] * 6}})
    stage = stub_isaac(monkeypatch)
    mounted = []
    monkeypatch.setattr(rp, "mount", lambda s, prefix, entry: mounted.append((s, prefix, entry["path"])))
    scene.build_stage()
    assert mounted == [(stage, "/Scene_0", "/fr3")]


def test_a_legacy_robot_without_a_scene_recipe_never_imports_omni(tmp_path, monkeypatch):
    """PF3: build_stage returns before any omni import (None in sys.modules makes `import omni` raise)."""
    monkeypatch.setitem(sys.modules, "omni", None)
    scene, _ = registered(tmp_path, {"legacy": {"path": "/ur", "default_joint_states": [0.0] * 6}})
    assert scene.scene_recipe is None
    scene.build_stage()
