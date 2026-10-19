"""Robot description pipeline (S2b spec 4.2): Isaac USD -> URDF + XRDF for guide_motion.

Run once per robot, in the Isaac venv, with no other Isaac process running:

    OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python -m guide_motion.description --robot fr3

Only main() starts Kit. Everything above it is plain Python, tested without Isaac.
"""

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PIPELINE_VERSION = 1
ROBOTS_DIR = Path(__file__).parent / "robots"
FRANKA_DESCRIPTION = Path.home() / "ros2_ws" / "src" / "franka_description"
HOME = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]  # block_bin init.yaml
TCP_Z = 0.1034  # franka_description franka_hand.xacro tcp_xyz
ROBOTS = {
    # acceleration / jerk: franka_description for the FR3, Isaac's shipped Panda XRDF for the Panda
    "fr3": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        "accel": [10.0] * 7,
        "jerk": [5000.0] * 7,
        "datasheet": "fr3",
    },
    "panda": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "accel": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
        "jerk": [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],
        "datasheet": "fer",
    },
}
_DRIVE = re.compile(r"\s*isaac:source_drive\s+(\{.*\})\s*$", re.S)


def _num(x) -> str:
    return f"{float(x):.10g}"


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def fix_urdf(text: str, robot: str) -> str:
    """The USD exporter's URDF with the XRDF test's defects fixed.

    A: mimic multiplier -1 -> +1. B: prismatic velocity had the degree conversion -> undone.
    C: no effort -> the USD drive's max force (a mimic joint takes its leader's).
    Joint limits stay the USD's. Visual and collision geometry are dropped: collision is
    the XRDF's spheres, and OSCBF's URDF parser would load every mesh. A missing
    <robot>_hand_tcp (the Panda USD has none) is added 0.1034 m along the hand's z.
    """
    root = _parse(text)
    joints = {j.get("name"): j for j in root.iter("joint")}
    effort = {}
    for name, joint in joints.items():
        for child in joint:
            match = child.tag is ET.Comment and _DRIVE.match(child.text or "")
            if match:
                force = json.loads(match.group(1))["drive"]["max_force"]
                if math.isfinite(force) and force > 0:
                    effort[name] = force
    for name, joint in joints.items():
        mimic, limit = joint.find("mimic"), joint.find("limit")
        if mimic is not None and float(mimic.get("multiplier", "1")) == -1.0:
            mimic.set("multiplier", "1")
        if limit is None:
            continue
        if joint.get("type") == "prismatic":
            limit.set("velocity", _num(float(limit.get("velocity")) * 180.0 / math.pi))
        leader = mimic.get("joint") if mimic is not None else name
        limit.set("effort", _num(effort[name] if name in effort else effort[leader]))
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    tcp = f"{robot}_hand_tcp"
    if root.find(f"link[@name='{tcp}']") is None:
        ET.SubElement(root, "link", name=tcp)
        joint = ET.SubElement(root, "joint", name=f"{tcp}_joint", type="fixed")
        ET.SubElement(joint, "origin", xyz=f"0 0 {TCP_Z}", rpy="0 0 0")
        ET.SubElement(joint, "parent", link=f"{robot}_hand")
        ET.SubElement(joint, "child", link=tcp)
    return ET.tostring(root, encoding="unicode")


def fix_xrdf(xrdf: dict, robot: str) -> dict:
    """The editor's XRDF with tool frame, link7 ignore line, real limits, link0 self-only."""
    x = copy.deepcopy(xrdf)
    x["tool_frames"] = [f"{robot}_hand_tcp"]
    ignore = x["self_collision"].setdefault("ignore", {})
    hand = {f"{robot}_hand", f"{robot}_leftfinger", f"{robot}_rightfinger"}
    ignore[f"{robot}_link7"] = sorted(set(ignore.get(f"{robot}_link7", [])) | hand)
    x["cspace"]["acceleration_limits"] = list(ROBOTS[robot]["accel"])
    x["cspace"]["jerk_limits"] = list(ROBOTS[robot]["jerk"])
    # link0's spheres reach below the base plane: self-collision only, as Isaac's Panda XRDF does
    world = x["world_collision"]["geometry"]
    base = f"{robot}_link0"
    x["geometry"][f"{robot}_self_only"] = {"spheres": {base: x["geometry"][world]["spheres"].pop(base)}}
    x["geometry"][f"{robot}_self"] = {"clone": [world, f"{robot}_self_only"]}
    x["self_collision"]["geometry"] = f"{robot}_self"
    return x


def link_poses(urdf_text: str, root: str | None = None) -> dict:
    """Every frame's 4x4 pose at q = 0, relative to `root` (default: the URDF's root link)."""
    import pinocchio as pin

    model = pin.buildModelFromXML(urdf_text)
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    origin = data.oMf[model.getFrameId(root)].inverse() if root else pin.SE3.Identity()
    return {frame.name: (origin * data.oMf[i]).homogeneous for i, frame in enumerate(model.frames)}


def usd_to_urdf(usd_poses: dict, urdf_text: str) -> dict:
    """Per link, the fixed transform taking USD body-frame coordinates into the URDF link frame.

    usd_poses: each link prim's pose at q = 0 relative to the root link prim (timeline stopped).
    """
    urdf = link_poses(urdf_text)
    return {name: np.linalg.inv(urdf[name]) @ np.asarray(T, float) for name, T in usd_poses.items() if name in urdf}


def reframe_spheres(xrdf: dict, frames: dict) -> dict:
    """Sphere centres authored in USD body frames, re-expressed in the URDF's link frames."""
    x = copy.deepcopy(xrdf)
    for group in x["geometry"].values():
        for link, spheres in group.get("spheres", {}).items():
            X = frames[link]
            for sphere in spheres:
                sphere["center"] = [float(v) for v in (X @ np.r_[sphere["center"], 1.0])[:3]]
    return x


def to_urdf_frames(masses: dict, coms: dict, inertias: dict, frames: dict) -> dict:
    """{link: (mass, com, inertia)} with COM and inertia (about the COM) moved by frames[link]."""
    out = {}
    for name, mass in masses.items():
        X = np.asarray(frames.get(name, np.eye(4)), float)
        R = X[:3, :3]
        out[name] = (float(mass), R @ np.asarray(coms[name], float) + X[:3, 3], R @ np.asarray(inertias[name], float) @ R.T)
    return out


def set_inertials(urdf_text: str, inertials: dict, extra_links: dict | None = None) -> str:
    """Each named link's <inertial> replaced; extra_links {name: (parent, T_parent_child)} added as fixed links."""
    root = _parse(urdf_text)
    for name, (parent, T) in (extra_links or {}).items():
        T = np.asarray(T, float)
        ET.SubElement(root, "link", name=name)
        joint = ET.SubElement(root, "joint", name=f"{name}_fixed", type="fixed")
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        ET.SubElement(joint, "origin", xyz=" ".join(map(_num, T[:3, 3])), rpy=" ".join(map(_num, rpy)))
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
    links = {link.get("name"): link for link in root.iter("link")}
    for name, (mass, com, I) in inertials.items():
        link = links[name]
        for element in link.findall("inertial"):
            link.remove(element)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(_num, com)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=_num(mass))
        ET.SubElement(
            inertial, "inertia",
            ixx=_num(I[0, 0]), ixy=_num(I[0, 1]), ixz=_num(I[0, 2]),
            iyy=_num(I[1, 1]), iyz=_num(I[1, 2]), izz=_num(I[2, 2]),
        )
    return ET.tostring(root, encoding="unicode")


def franka_inertials(arm: str, fd: Path = FRANKA_DESCRIPTION) -> dict:
    """franka_description's {'<prefix>_<link>': (mass, com, inertia)}, arm 'fer' (Panda) or 'fr3', plus the hand."""
    prefix = {"fer": "panda", "fr3": "fr3"}[arm]
    out = {}
    for path in (fd / "robots" / arm / "inertials.yaml", fd / "end_effectors" / "franka_hand" / "inertials.yaml"):
        for name, entry in yaml.safe_load(path.read_text()).items():
            assert entry["origin"]["rpy"].split() == ["0", "0", "0"], (path, name)
            i = entry["inertia"]
            out[f"{prefix}_{name}"] = (
                float(entry["mass"]),
                np.array(entry["origin"]["xyz"].split(), float),
                np.array([[i["xx"], i["xy"], i["xz"]], [i["xy"], i["yy"], i["yz"]], [i["xz"], i["yz"], i["zz"]]], float),
            )
    return out


def build(raw_urdf: str, raw_xrdf: dict, usd_poses: dict, robot: str):
    """Kit-free half of the pipeline: (urdf text, xrdf dict, description dict)."""
    urdf = fix_urdf(raw_urdf, robot)
    frames = usd_to_urdf(usd_poses, urdf)
    xrdf = reframe_spheres(fix_xrdf(raw_xrdf, robot), frames)
    description = {
        "robot": robot,
        "pipeline_version": PIPELINE_VERSION,
        "usd_to_urdf": {name: X.tolist() for name, X in frames.items()},
    }
    return urdf, xrdf, description
