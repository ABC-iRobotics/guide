"""The pipeline's Kit-free half against the exporter and editor defects the XRDF test found."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest
import yaml

from fr3_fixture import DATA, usd_poses
from guide_motion import description

RAW_URDF = (DATA / "fr3_exporter_raw.urdf").read_text()
RAW_XRDF = yaml.safe_load((DATA / "fr3_editor_raw.xrdf").read_text())


def joints(text):
    return {j.get("name"): j for j in ET.fromstring(text).iter("joint")}


def test_raw_exporter_urdf_is_rejected_and_the_fixed_one_loads():
    with pytest.raises(ValueError):
        pin.buildModelFromXML(RAW_URDF)  # no effort on the limits (defect C)
    assert pin.buildModelFromXML(description.fix_urdf(RAW_URDF, "fr3")).nq == 9


def test_exporter_defects_are_fixed():
    fixed = joints(description.fix_urdf(RAW_URDF, "fr3"))
    assert fixed["fr3_finger_joint2"].find("mimic").get("multiplier") == "1"  # A
    assert float(fixed["fr3_finger_joint1"].find("limit").get("velocity")) == pytest.approx(0.2, abs=1e-6)  # B
    efforts = {n: float(j.find("limit").get("effort")) for n, j in fixed.items() if j.find("limit") is not None}  # C
    assert efforts == {**{f"fr3_joint{i}": 87.0 for i in range(1, 5)}, **{f"fr3_joint{i}": 12.0 for i in range(5, 8)},
                       "fr3_finger_joint1": 100.0, "fr3_finger_joint2": 100.0}
    raw = joints(RAW_URDF)
    assert fixed["fr3_joint1"].find("limit").get("upper") == raw["fr3_joint1"].find("limit").get("upper")  # USD limits kept


def test_a_mimic_multiplier_written_as_a_float_is_fixed_too():
    text = RAW_URDF.replace('multiplier="-1"', 'multiplier="-1.0"')  # a Panda export may write it so
    assert 'multiplier="-1.0"' in text
    assert joints(description.fix_urdf(text, "fr3"))["fr3_finger_joint2"].find("mimic").get("multiplier") == "1"


def test_a_mimic_joint_without_a_drive_takes_its_leaders_effort():
    """The Panda's export: panda_finger_joint2's source_drive comment has no "drive" at all."""
    fr3 = '{"armature": 0.0, "drive": {"damping": 0.0, "max_force": Infinity, "stiffness": 0.0, "target_position": 0.0}, "instance": "linear", "source": "physx"}'
    assert RAW_URDF.count(fr3) == 1
    text = RAW_URDF.replace(fr3, '{"armature": 0.0, "instance": "linear", "source": "physx"}')
    assert float(joints(description.fix_urdf(text, "fr3"))["fr3_finger_joint2"].find("limit").get("effort")) == 100.0


def test_a_mimic_joint_takes_its_leaders_velocity():
    """The Panda's export gives panda_finger_joint2 (the mimic) 0.17453293, its leader 0.00349066."""
    a = '<limit lower="0" upper="0.04" velocity="0.00349066"/>\n    <dynamics friction="0"/>\n    <mimic'
    assert RAW_URDF.count(a) == 1
    text = RAW_URDF.replace(a, a.replace("0.00349066", "0.17453293"))
    fixed = joints(description.fix_urdf(text, "fr3"))
    assert fixed["fr3_finger_joint2"].find("limit").get("velocity") == fixed["fr3_finger_joint1"].find("limit").get("velocity")


def links(text):
    return {link.get("name") for link in ET.fromstring(text).iter("link")}


def with_world(joint_type):
    return RAW_URDF.replace('<robot name="fr3">', f'''<robot name="fr3">
  <link name="world"/>
  <joint name="floating_base" type="{joint_type}">
    <parent link="world"/>
    <child link="fr3_link0"/>
  </joint>''', 1)


def test_a_floating_joint_from_world_is_dropped():
    """The Panda's export: franka.usd has no root FixedJoint, so the exporter adds a world link and a
    floating joint to it, though PhysX runs the arm fixed-base. The root must be link0, as for the FR3."""
    fixed = description.fix_urdf(with_world("floating"), "fr3")
    assert links(fixed) == links(description.fix_urdf(RAW_URDF, "fr3")) and "floating_base" not in joints(fixed)
    assert pin.buildModelFromXML(fixed).nq == 9


def test_a_fixed_root_is_left_alone():
    assert "world" in links(description.fix_urdf(with_world("fixed"), "fr3"))  # only a floating joint is dropped
    fixed = description.fix_urdf(RAW_URDF, "fr3")  # the FR3's own root path: fr3_link0, nothing removed
    assert links(fixed) == links(RAW_URDF) and set(joints(fixed)) == set(joints(RAW_URDF))
    assert description.link_poses(fixed)["fr3_link0"].tolist() == np.eye(4).tolist()


def test_geometry_is_dropped():
    text = description.fix_urdf(RAW_URDF, "fr3")
    assert "<visual" not in text and "<collision" not in text and "<mesh" not in text


def test_a_missing_tool_frame_is_added_on_the_hand():
    text = re.sub(r'\s*<link name="fr3_hand_tcp"/>', "", RAW_URDF)
    text = re.sub(r'\s*<joint name="fr3_hand_tcp_joint".*?</joint>', "", text, flags=re.S)
    assert "fr3_hand_tcp" not in text
    poses = description.link_poses(description.fix_urdf(text, "fr3"))
    offset = np.linalg.inv(poses["fr3_hand"]) @ poses["fr3_hand_tcp"]
    expected = np.eye(4)
    expected[2, 3] = 0.1034
    np.testing.assert_allclose(offset, expected, atol=1e-12)


def test_xrdf_fixes():
    x = description.fix_xrdf(RAW_XRDF, "fr3")
    assert x["tool_frames"] == ["fr3_hand_tcp"]
    assert {"fr3_hand", "fr3_leftfinger", "fr3_rightfinger"} <= set(x["self_collision"]["ignore"]["fr3_link7"])
    assert x["cspace"]["acceleration_limits"] == [10.0] * 7 and x["cspace"]["jerk_limits"] == [5000.0] * 7
    world = x["geometry"][x["world_collision"]["geometry"]]["spheres"]
    assert "fr3_link0" not in world and x["geometry"]["fr3_self_only"]["spheres"]["fr3_link0"]
    assert x["geometry"][x["self_collision"]["geometry"]]["clone"] == [x["world_collision"]["geometry"], "fr3_self_only"]
    assert "fr3_link0" in RAW_XRDF["geometry"][RAW_XRDF["world_collision"]["geometry"]]["spheres"]  # input untouched


def test_spheres_land_where_the_xrdf_test_put_them():
    """Re-expressed centres equal the XRDF test's fr3_fixed_usd_export_mimicfix_frames.xrdf."""
    _, xrdf, _ = description.build(RAW_URDF, RAW_XRDF, usd_poses(), "fr3")
    reference = yaml.safe_load((DATA / "fr3_spheres_reference.xrdf").read_text())
    ref = reference["geometry"][reference["world_collision"]["geometry"]]["spheres"]
    ours = {**xrdf["geometry"][xrdf["world_collision"]["geometry"]]["spheres"], **xrdf["geometry"]["fr3_self_only"]["spheres"]}
    assert set(ours) == set(ref)
    for link in ref:
        np.testing.assert_allclose([s["center"] for s in ours[link]], [s["center"] for s in ref[link]], atol=1e-6, err_msg=link)


def test_inertials_move_with_their_frame():
    X = np.eye(4)
    X[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 deg about z
    X[:3, 3] = [0.1, 0, 0]
    I = np.diag([1.0, 2.0, 3.0])
    (m, c, J), = description.to_urdf_frames({"l": 2.0}, {"l": [1.0, 0, 0]}, {"l": I}, {"l": X}).values()
    assert m == 2.0
    np.testing.assert_allclose(c, [0.1, 1.0, 0.0])
    np.testing.assert_allclose(J, np.diag([2.0, 1.0, 3.0]))


@pytest.mark.skipif(not description.FRANKA_DESCRIPTION.exists(), reason="no ~/ros2_ws/src/franka_description")
def test_franka_inertials_cover_the_arm_and_the_hand():
    fer = description.franka_inertials("fer")
    assert set(fer) == {f"panda_link{i}" for i in range(8)} | {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
    assert fer["panda_hand"][0] == pytest.approx(0.6544)
    np.testing.assert_allclose(fer["panda_rightfinger"][1], fer["panda_leftfinger"][1])


def test_principal_inertia_rebuilds_the_tensor():
    from scipy.spatial.transform import Rotation

    I = np.array([[0.0090, 0.0, 0.0020], [0.0, 0.0115, 0.0], [0.0020, 0.0, 0.0085]])  # franka_description fer link0
    diagonal, (w, x, y, z) = description.principal_inertia(I)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    np.testing.assert_allclose(R @ np.diag(diagonal) @ R.T, I, atol=1e-12)


def test_the_mass_override_keeps_its_sources_up_axis_and_units(tmp_path):
    """A new USD layer defaults to Y-up centimetres, and a reference does not carry its source's: the
    Panda's override, opened alone, gave PhysX gravity torques 100x off (M matched)."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
    UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/link", "Xform"))  # no MassAPI, as fr3_link8
    stage.GetRootLayer().Save()
    out = tmp_path / "override.usda"
    description._write_mass_override(str(out), str(source), {"link": (2.0, np.zeros(3), np.eye(3))})
    result = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(result) == UsdGeom.Tokens.z and UsdGeom.GetStageMetersPerUnit(result) == 1.0
    assert UsdPhysics.MassAPI(result.GetPrimAtPath("/override/link")).GetMassAttr().Get() == 2.0


needs_fd = pytest.mark.skipif(not description.FRANKA_DESCRIPTION.exists(), reason="no ~/ros2_ws/src/franka_description")


@needs_fd
def test_the_fr3_override_corrects_the_hand_and_the_phantom_frames():
    """Ruling 10: fr3_link8 and fr3_hand_tcp (1.0 kg bodies in Isaac 6.0's asset) become 1 g; the hand and
    fingers take franka_description's; link0..7 keep the asset's."""
    o = description.mass_override("fr3")
    assert set(o) == {"fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link8", "fr3_hand_tcp"}
    assert o["fr3_hand"][0] == 0.6544 and o["fr3_leftfinger"][0] == o["fr3_rightfinger"][0] == 0.0291
    for frame in ("fr3_link8", "fr3_hand_tcp"):
        mass, _, inertia = o[frame]
        assert mass == 0.001 and 0 < np.linalg.eigvalsh(inertia).min() and inertia.max() <= 1e-7


@needs_fd
@pytest.mark.parametrize("robot, prefix", [("fr3", "fr3"), ("panda", "panda")])
def test_the_right_finger_is_turned_into_the_usd_frame(robot, prefix):
    """Both assets keep the right finger in the left finger's orientation; franka_description turns it pi about z."""
    o = description.mass_override(robot)
    left, right = o[f"{prefix}_leftfinger"], o[f"{prefix}_rightfinger"]
    np.testing.assert_allclose(right[1], left[1] * [-1, -1, 1], atol=1e-12)
    np.testing.assert_allclose(right[2], np.diag([-1, -1, 1]) @ left[2] @ np.diag([-1, -1, 1]), atol=1e-15)


@needs_fd
def test_the_panda_override_is_franka_description_on_every_link():
    o = description.mass_override("panda")
    assert set(o) == {f"panda_link{i}" for i in range(8)} | {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
    assert o["panda_hand"][0] == 0.6544
