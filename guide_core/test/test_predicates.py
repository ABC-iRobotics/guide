"""Success predicates with LIBERO's thresholds (S3 v2 spec 3.4, LIBERO.md section 6)."""
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
    assert p.on([0, 0, 0.05], [0, 0, 0], label="on(/p, /m)")["reason"].startswith("on(/p, /m): xy")


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


STOVE_FRONT = ([-0.05, 0.21, -0.005], [1, 0, 0, 0], [0.04, 0.04, 0.007])  # Goal #3's table zone (-0.09, 0.17, -0.01, 0.25), LIBERO's TargetZone
COOK = ([0.15, 0.0, 0.0], [1, 0, 0, 0], [0.075, 0.075, 0.0025])  # the stove's cook_region (top face at z 0.0025)
RACK = ([0.0, -0.08003, 0.21745], R.from_euler("x", -62.5, degrees=True).as_quat(scalar_first=True), [0.10087, 0.022, 0.00251])


def test_on_region_is_libero_under_window_and_gate():
    """SiteObject.under: size_z - 0.005 < z < size_z + 0.10 over the site centre, |xy| < size_xy (all strict)."""
    c, q, h = STOVE_FRONT
    assert h[2] - 0.005 == pytest.approx(0.002) and h[2] + 0.10 == pytest.approx(0.107)  # the plan's window
    at = lambda x, y, z: p.on_region([c[0] + x, c[1] + y, c[2] + z], c, q, h)["success"]  # noqa: E731
    assert at(0, 0, 0.0021) and at(0, 0, 0.1069)
    assert not at(0, 0, 0.0019) and not at(0, 0, 0.1071)  # (the edges themselves are strict, but a float cannot say which side)
    assert at(0.0399, -0.0399, 0.02) and not at(0.04, 0, 0.02) and not at(0, -0.04, 0.02)
    assert "0.0200" not in p.on_region([0, 0, 0], c, q, h)["reason"] and "over its top" in p.on_region([0, 0, 0], c, q, h)["reason"]


def test_on_region_needs_the_contact_only_when_the_geometry_holds_and_it_is_asked():
    c, q, h = COOK
    near = [0.15, 0.0, 0.03]
    assert p.on_region(near, c, q, h)["success"]
    assert p.on_region(near, c, q, h, touching=lambda: True)["success"]
    r = p.on_region(near, c, q, h, touching=lambda: False)
    assert not r["success"] and r["reason"].endswith("but not touching")

    def boom():
        raise AssertionError("contact is queried only when the geometry holds")
    assert not p.on_region([0.3, 0.0, 0.03], c, q, h, touching=boom)["success"]
    assert not p.on([0.0, 0.0, 0.05], [0.0, 0.0, 0.0], touching=lambda: False)["success"]  # obj-obj On too
    assert p.on([0.0, 0.0, 0.05], [0.0, 0.0, 0.0], touching=lambda: True)["success"]
    assert not p.on([0.0, 0.0, -0.01], [0.0, 0.0, 0.0], touching=boom)["success"]


def test_on_region_tests_a_tilted_site_in_its_own_frame_where_libero_multiplies_by_the_matrix():
    """A bottle lying 1 cm above the rack's top face along its normal: inside in the site frame. LIBERO's mat @ delta
    (not mat.T) tilts the point the wrong way and rejects it; at yaw 0 / 180 deg the two agree."""
    c, q, h = RACK
    normal = R.from_quat(q, scalar_first=True).apply([0, 0, 1])
    point = np.asarray(c) + (h[2] + 0.01) * normal
    assert p.on_region(point, c, q, h)["success"]
    libero = R.from_quat(q, scalar_first=True).as_matrix() @ (point - c)  # what LIBERO computes
    assert not (h[2] - 0.005 < libero[2] < h[2] + 0.10 and np.all(np.abs(libero[:2]) < h[:2]))
    for yaw in (0, 180):
        qz = R.from_euler("z", yaw, degrees=True)
        d = np.array([0.03, 0.01, 0.02])
        assert np.allclose(qz.as_matrix() @ d, qz.inv().as_matrix() @ d)


def test_joint_above_inclusive_is_libero_turn_on():
    assert p.joint_above(0.5, 0.5, "button", inclusive=True)["success"]  # qpos >= 0.5
    assert not p.joint_above(0.4999, 0.5, "button", inclusive=True)["success"]
    assert not p.joint_above(0.5, 0.5)["success"]  # the strict default is unchanged
    assert p.joint_above(0.5, 0.5, "button", inclusive=True)["reason"] == "button = 0.5000 (>= 0.5)"
    assert p.joint_below(-0.1401, -0.14, "middle_level")["success"] and not p.joint_below(-0.14, -0.14)["success"]  # Open(drawer)


def test_the_commands_read_the_scene_and_apply_the_predicates():
    poses = {"/Scene_0/plate": ([0.055, 0.205, 0.0023], [1, 0, 0, 0], [1, 1, 1]),
             "/Scene_0/mark": ([0.05, 0.2, 0.0], [1, 0, 0, 0], [1, 1, 1])}
    host = SimpleNamespace(**{
        "__pose": lambda path: tuple(np.asarray(v, float) for v in poses[path]),
        "__region_box": lambda region: (np.array([0.05, 0.2, 0.03]), np.array([0.03, 0.03, 0.03])),
        "__joint_position": lambda joint: -0.12,
        "__region_bound": lambda region: (np.array([0.05, 0.2, 0.0]), np.array([1, 0, 0, 0.0]), np.ones(3),
                                          np.array([-0.03, -0.03, -0.0025]), np.array([0.03, 0.03, 0.0025])),
        "__touching": lambda a, b: (lambda: calls.append((a, b)) or touch),
    })
    calls, touch = [], True
    assert cp._cmd_on(host, "/Scene_0/plate", "/Scene_0/mark")["success"]
    assert cp._cmd_on(host, "/Scene_0/plate", "/Scene_0/mark")["reason"].startswith("on(/Scene_0/plate, /Scene_0/mark): ")
    assert cp._cmd_in_region(host, "/Scene_0/plate", "/Scene_0/region")["reason"].startswith("in_region(/Scene_0/plate, /Scene_0/region): ")
    assert cp._cmd_in_region(host, "/Scene_0/plate", "/Scene_0/region")["success"]
    assert cp._cmd_joint_below(host, "/Scene_0/drawer", -0.10)["success"]
    assert not cp._cmd_joint_above(host, "/Scene_0/drawer", 0.0)["success"]
    assert cp._cmd_joint_above(host, "/Scene_0/drawer", -0.12, inclusive=True)["success"]
    assert calls == [] and cp._cmd_on(host, "/Scene_0/plate", "/Scene_0/mark")["success"]  # no contact asked, none queried
    assert cp._cmd_on(host, "/Scene_0/plate", "/Scene_0/mark", contact=True)["success"] and calls == [("/Scene_0/plate", "/Scene_0/mark")]
    assert cp._cmd_on_region(host, "/Scene_0/plate", "/Scene_0/region", touch="/Scene_0/cab")["success"]  # the box is 0.03 half over 0.03 centre
    assert calls[-1] == ("/Scene_0/plate", "/Scene_0/cab")
    scene = (np.array([1.0, 2.0, 0.0]), np.array([1, 0, 0, 0.0]), np.ones(3))  # the scene root, away from the world origin
    host.__dict__["__pose"] = lambda path: scene if path == "/Scene_0" else poses[path]
    zone = {"center": [0.05, 0.2, 0.0], "half": [0.04, 0.04, 0.007]}  # a table zone: a box in the scene frame, no prim
    assert not cp._cmd_on_region(host, "/Scene_0/plate", "/Scene_0", **zone)["success"]  # the plate is 1 m, 2 m from it
    poses["/Scene_0/plate"] = (np.array([1.055, 2.205, 0.0023]), np.array([1, 0, 0, 0.0]), np.ones(3))
    assert cp._cmd_on_region(host, "/Scene_0/plate", "/Scene_0", **zone)["success"]


def test_predicate_paths_are_scene_prefixed_like_prim_path(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(DUMMY, task)
    (task / "config" / "success.yaml").write_text(yaml.safe_dump({"instructions": [
        {"cmd": "on", "kwargs": {"a": "/objects/plate", "b": "/marks/home", "xy_tol": 0.03}},
        {"cmd": "in_region", "kwargs": {"obj": "/objects/plate", "region": "/marks/region"}},
        {"cmd": "on_region", "kwargs": {"obj": "/objects/plate", "region": "", "touch": "/objects/cab", "center": [0, 0, 0]}},
        {"cmd": "joint_below", "kwargs": {"joint": "/objects/cab/top_level", "value": -0.1}},
        {"cmd": "some_other", "kwargs": {"a": "/not/a/prim"}},
    ]}))
    manager = SceneManager(sim_id=0, logger=logging.getLogger("test"))
    sid, _, cfg = manager.add_scene(str(task))
    assert sid == 0, cfg
    on, inside, zone, joint, other = (i["kwargs"] for i in manager._scenes[0].success_instructions)
    assert (on["a"], on["b"], on["xy_tol"]) == ("/Scene_0/objects/plate", "/Scene_0/marks/home", 0.03)
    assert (inside["obj"], inside["region"]) == ("/Scene_0/objects/plate", "/Scene_0/marks/region")
    assert (zone["obj"], zone["region"], zone["touch"], zone["center"]) == ("/Scene_0/objects/plate", "/Scene_0", "/Scene_0/objects/cab", [0, 0, 0])
    assert joint["joint"] == "/Scene_0/objects/cab/top_level" and other["a"] == "/not/a/prim"
