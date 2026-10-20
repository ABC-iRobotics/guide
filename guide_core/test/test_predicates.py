"""Success predicates with LIBERO's thresholds (S3 v2 spec 3.4, LIBERO.md section 6)."""
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from guide_core.core.commands import _cmd_predicates as cp
from guide_core.scene import predicates as p
from guide_core.scene.scene_manager import SceneManager

DUMMY = Path(__file__).resolve().parents[1] / "dummy_scene"


def test_on_is_the_origin_test_with_a_strict_3_cm_gate():
    assert p.on([0.01, 0.0, 0.05], [0.0, 0.0, 0.0])["success"]
    assert not p.on([0.031, 0.0, 0.05], [0.0, 0.0, 0.0])["success"]
    assert not p.on([0.03, 0.0, 0.05], [0.0, 0.0, 0.0])["success"]  # strict
    assert not p.on([0.0, 0.0, -0.001], [0.0, 0.0, 0.0])["success"]  # the support above the object
    assert "0.0310" in p.on([0.031, 0.0, 0.05], [0.0, 0.0, 0.0])["reason"]


def test_in_box_is_libero_in_with_its_lower_margin():
    center, half = [0.0, 0.0, 0.1], [0.05, 0.05, 0.05]
    assert p.in_box([0.0, 0.0, 0.051], center, half)["success"]
    assert p.in_box([0.0, 0.0, 0.045], center, half)["success"]  # inside the 1 cm drop of the lower face
    assert not p.in_box([0.0, 0.0, 0.039], center, half)["success"]
    assert not p.in_box([0.05, 0.0, 0.1], center, half)["success"]  # strict


def test_in_box_takes_the_abs_of_the_rotated_half_vector_as_libero_does():
    """LIBERO's half = |site_xmat @ size|, not the rotated box's true AABB."""
    half_vec = R.from_euler("z", 30, degrees=True).apply([0.1, 0.02, 0.05])  # (0.0766, 0.0673, 0.05)
    assert p.in_box([0.07, 0.0, 0.0], [0, 0, 0], half_vec)["success"]
    assert not p.in_box([0.08, 0.0, 0.0], [0, 0, 0], half_vec)["success"]


def test_obb_world_places_a_prims_own_box():
    center, half_vec = p.obb_world([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [1, 2, 3],
                                   R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True), [0.1, 0.2, 0.3])
    assert np.allclose(center, [1, 2, 3])
    assert np.allclose(half_vec, [-0.1, 0.05, 0.15])


def test_joint_thresholds_are_strict_and_say_the_value():
    assert p.joint_below(-0.15, -0.14, "top_level")["success"]
    assert not p.joint_below(-0.14, -0.14)["success"]
    assert p.joint_above(0.001, 0.0)["success"] and not p.joint_above(0.0, 0.0)["success"]
    assert p.joint_below(-0.15, -0.14, "top_level")["reason"] == "top_level = -0.1500 (< -0.14)"


def test_the_commands_read_the_scene_and_apply_the_predicates():
    poses = {"/Scene_0/plate": ([0.055, 0.205, 0.0023], [1, 0, 0, 0], [1, 1, 1]),
             "/Scene_0/mark": ([0.05, 0.2, 0.0], [1, 0, 0, 0], [1, 1, 1])}
    host = SimpleNamespace(**{
        "__pose": lambda path: tuple(np.asarray(v, float) for v in poses[path]),
        "__region_box": lambda region: (np.array([0.05, 0.2, 0.03]), np.array([0.03, 0.03, 0.03])),
        "__joint_position": lambda joint: -0.12,
    })
    assert cp._cmd_on(host, "/Scene_0/plate", "/Scene_0/mark")["success"]
    assert cp._cmd_in_region(host, "/Scene_0/plate", "/Scene_0/region")["success"]
    assert cp._cmd_joint_below(host, "/Scene_0/drawer", -0.10)["success"]
    assert not cp._cmd_joint_above(host, "/Scene_0/drawer", 0.0)["success"]


def test_predicate_paths_are_scene_prefixed_like_prim_path(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(DUMMY, task)
    (task / "config" / "success.yaml").write_text(yaml.safe_dump({"instructions": [
        {"cmd": "on", "kwargs": {"a": "/objects/plate", "b": "/marks/home", "xy_tol": 0.03}},
        {"cmd": "in_region", "kwargs": {"obj": "/objects/plate", "region": "/marks/region"}},
        {"cmd": "joint_below", "kwargs": {"joint": "/objects/cab/top_level", "value": -0.1}},
        {"cmd": "some_other", "kwargs": {"a": "/not/a/prim"}},
    ]}))
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task))
    assert sid == 0, cfg
    on, inside, joint, other = (i["kwargs"] for i in manager._scenes[0].success_instructions)
    assert (on["a"], on["b"], on["xy_tol"]) == ("/Scene_0/objects/plate", "/Scene_0/marks/home", 0.03)
    assert (inside["obj"], inside["region"]) == ("/Scene_0/objects/plate", "/Scene_0/marks/region")
    assert joint["joint"] == "/Scene_0/objects/cab/top_level" and other["a"] == "/not/a/prim"
