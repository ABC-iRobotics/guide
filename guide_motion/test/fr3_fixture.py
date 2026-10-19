"""The FR3 test fixture: a description built Kit-free from the XRDF test's raw exporter outputs.

franka_description's FR3 stands in for the USD: the XRDF test measured its link frames
equal to the USD's, except the right finger, whose USD frame equals the left finger's at
q = 0. Its inertials stand in for PhysX's.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

from guide_motion import description

DATA = Path(__file__).parent / "data"
FD_URDF = (DATA / "fr3_franka_description.urdf").read_text()
ARM = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = np.array(description.HOME)


def usd_poses() -> dict:
    poses = description.link_poses(FD_URDF, root="fr3_link0")
    poses["fr3_rightfinger"] = poses["fr3_leftfinger"]
    return poses


def fd_dynamics(q, urdf=FD_URDF):
    """(M, g) of franka_description's FR3, fingers locked, by Pinocchio: the physical reference."""
    full = pin.buildModelFromXML(urdf)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    M = pin.crba(model, data, q)
    return np.triu(M) + np.triu(M, 1).T, pin.computeGeneralizedGravity(model, data, q).copy()


def simulator_mass_properties():
    """franka_description's inertials as PhysX would report them: in the (stand-in) USD frames."""
    raw = {}
    for link in ET.fromstring(FD_URDF).iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        raw[link.get("name")] = (
            float(inertial.find("mass").get("value")),
            np.array(inertial.find("origin").get("xyz").split(), float),
            np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]),
        )
    fd, usd = description.link_poses(FD_URDF, root="fr3_link0"), usd_poses()
    moved = description.to_urdf_frames(
        {n: v[0] for n, v in raw.items()}, {n: v[1] for n, v in raw.items()}, {n: v[2] for n, v in raw.items()},
        {n: np.linalg.inv(usd[n]) @ fd[n] for n in raw},
    )
    return ({n: v[0] for n, v in moved.items()}, {n: v[1] for n, v in moved.items()}, {n: v[2] for n, v in moved.items()})


def fd_with_camera(T, mass):
    """franka_description's FR3 with a point-ish camera fixed on fr3_hand at T: the reference for extra links."""
    root = ET.fromstring(FD_URDF)
    inertial = ET.SubElement(ET.SubElement(root, "link", name="camera"), "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "inertia", ixx="1e-4", iyy="1e-4", izz="1e-4", ixy="0", ixz="0", iyz="0")
    joint = ET.SubElement(root, "joint", name="camera_fixed", type="fixed")
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, T[:3, 3])), rpy="0 0 0")
    ET.SubElement(joint, "parent", link="fr3_hand")
    ET.SubElement(joint, "child", link="camera")
    return ET.tostring(root, encoding="unicode")
