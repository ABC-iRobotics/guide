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
