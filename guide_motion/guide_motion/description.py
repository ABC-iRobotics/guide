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

PIPELINE_VERSION = 2  # 2: the Panda's export fixes, base facts, mass override (ruling 10)
ROBOTS_DIR = Path(__file__).parent / "robots"
FRANKA_DESCRIPTION = Path.home() / "ros2_ws" / "src" / "franka_description"
HOME = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]  # block_bin init.yaml
TCP_Z = 0.1034  # franka_description franka_hand.xacro tcp_xyz
FRAME_MASS = (0.001, np.zeros(3), 1e-7 * np.eye(3))  # a frame the asset made a body: 1 g (PhysX needs > 0)
ROBOTS = {
    # acceleration / jerk: franka_description for the FR3, Isaac's shipped Panda XRDF for the Panda
    "fr3": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        "accel": [10.0] * 7,
        "jerk": [5000.0] * 7,
        "datasheet": "fr3",
        # Mass override, {USD link: franka_description link to copy | (mass, com, inertia) in the USD body
        # frame}. Ruling 10: the hand and fingers take franka_description's; fr3_link8 and fr3_hand_tcp,
        # frames on the real robot, are 1.0 kg bodies in Isaac 6.0's asset. link0..7 keep the asset's.
        "mass_override": {"fr3_hand": "fr3_hand", "fr3_leftfinger": "fr3_leftfinger", "fr3_rightfinger": "fr3_rightfinger",
                          "fr3_link8": FRAME_MASS, "fr3_hand_tcp": FRAME_MASS},
    },
    "panda": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "accel": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
        "jerk": [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],
        "datasheet": "fer",
        # "fer": every franka_description link, copied (spec 4.2.3: the shipped inertias are ~1e-7 kg m^2,
        # and torque control chattered).
        "mass_override": "fer",
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
    C: no effort -> the USD drive's max force (a mimic joint, or one without a drive, takes its leader's).
    E: a floating joint from a parentless world link -> dropped. F: a mimic joint takes its leader's velocity.
    Joint limits stay the USD's. Visual and collision geometry are dropped: collision is
    the XRDF's spheres, and OSCBF's URDF parser would load every mesh.
    Input must be the raw exporter/editor output; running it on its own output corrupts the finger velocity.
    """
    root = _parse(text)
    # E (the Panda): franka.usd has no root FixedJoint, so the exporter adds a parentless "world" link and a
    # floating joint to it, though PhysX runs the arm fixed-base. Dropped: the root is link0, as for the FR3.
    children = {j.find("child").get("link") for j in root.iter("joint")}
    for joint in root.findall("joint"):
        if joint.get("type") == "floating" and joint.find("parent").get("link") == "world" and "world" not in children:
            root.remove(joint)
            root.remove(root.find("link[@name='world']"))
    joints = {j.get("name"): j for j in root.iter("joint")}
    effort = {}
    for name, joint in joints.items():
        for child in joint:
            match = child.tag is ET.Comment and _DRIVE.match(child.text or "")
            if match:
                force = json.loads(match.group(1)).get("drive", {}).get("max_force", math.inf)  # none: the Panda's mimic
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
    for joint in joints.values():  # F (the Panda): a mimic joint's own velocity is meaningless; its leader's
        mimic, limit = joint.find("mimic"), joint.find("limit")
        if mimic is not None and limit is not None:
            limit.set("velocity", joints[mimic.get("joint")].find("limit").get("velocity"))
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    return ET.tostring(root, encoding="unicode")


def fix_xrdf(xrdf: dict, robot: str) -> dict:
    """The editor's XRDF with tool frame, link7 ignore line, real limits, link0 self-only.

    Input must be the raw editor output; running it on its own output fails (link0 is already moved).
    """
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


def mass_override(robot: str, fd: Path = FRANKA_DESCRIPTION) -> dict:
    """{link: (mass, com, inertia)} in the USD's body frames for the robot's mass override layer ({}: none)."""
    spec = ROBOTS[robot].get("mass_override") or {}
    sheet = franka_inertials(ROBOTS[robot]["datasheet"], fd) if spec else {}
    if isinstance(spec, str):  # shorthand: every link of that franka_description arm
        spec = {name: name for name in sheet}
    out = {}
    for link, value in spec.items():
        if isinstance(value, str):  # a franka_description link, moved into the USD's body frame
            # franka_description's right finger frame is the USD's turned pi about z (both assets keep it in
            # the left finger's orientation; measured for the FR3 by the XRDF test, the Panda against xacro).
            X = np.diag([-1.0, -1.0, 1.0, 1.0]) if value.endswith("rightfinger") else np.eye(4)
            (mass, com, inertia), = to_urdf_frames({link: sheet[value][0]}, {link: sheet[value][1]}, {link: sheet[value][2]}, {link: X}).values()
            value = (mass, com, inertia)
        out[link] = value
    return out


def build(raw_urdf: str, raw_xrdf: dict, usd_poses: dict, robot: str):
    """Kit-free half of the pipeline: (urdf text, xrdf dict, description dict).

    A missing <robot>_hand_tcp (the Panda USD has none) is added 0.1034 m along the USD's hand z
    (franka_description's hand frame): the exporter's Panda hand link is that frame turned 90 deg.
    """
    urdf = fix_urdf(raw_urdf, robot)
    frames = usd_to_urdf(usd_poses, urdf)
    tcp, hand = f"{robot}_hand_tcp", f"{robot}_hand"
    if _parse(urdf).find(f"link[@name='{tcp}']") is None:
        along_z = np.eye(4)
        along_z[2, 3] = TCP_Z
        urdf = set_inertials(urdf, {}, {tcp: (hand, frames[hand] @ along_z)})
    xrdf = reframe_spheres(fix_xrdf(raw_xrdf, robot), frames)
    description = {
        "robot": robot,
        "pipeline_version": PIPELINE_VERSION,
        "usd_to_urdf": {name: X.tolist() for name, X in frames.items()},
    }
    return urdf, xrdf, description


def principal_inertia(inertia):
    """USD's (diagonalInertia, principalAxes wxyz) for a full 3x3 inertia about the COM."""
    diagonal, axes = np.linalg.eigh(np.asarray(inertia, float))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    return diagonal, np.array([w, x, y, z])


def _write_mass_override(path: str, source: str, inertials: dict) -> None:
    """A wrapper USD that references `source` and overrides its links' mass properties (Kit's pxr)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    # A new layer is Y-up centimetres by default, and a reference does not carry its source's metadata.
    original = Usd.Stage.Open(source, load=Usd.Stage.LoadNone)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    root = stage.DefinePrim("/" + Path(path).stem, "Xform")
    root.GetReferences().AddReference(source)
    stage.SetDefaultPrim(root)
    links = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    for name, (mass, com, inertia) in inertials.items():
        diagonal, (w, x, y, z) = principal_inertia(inertia)
        api = UsdPhysics.MassAPI.Apply(links[name])
        api.CreateMassAttr().Set(float(mass))
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*(float(v) for v in com)))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v) for v in diagonal)))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    stage.GetRootLayer().Save()


def main(argv=None) -> None:
    import argparse
    import os
    import shutil
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description="Build guide_motion/robots/<robot> from Isaac's robot USD.")
    parser.add_argument("--robot", choices=sorted(ROBOTS), required=True)
    parser.add_argument("--spheres", type=int, default=6, help="collision spheres per link (the XRDF test used 6)")
    parser.add_argument("--radius-offset", type=float, default=0.01, help="the editor UI's default")
    parser.add_argument("--franka-description", type=Path, default=FRANKA_DESCRIPTION)
    args = parser.parse_args(argv)
    robot = args.robot

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from isaacsim.core.utils.extensions import enable_extension

    for extension in ("isaacsim.robot_setup.xrdf_editor", "isaacsim.asset.exporter.urdf",
                      "isaacsim.robot_motion.cumotion", "isaacsim.storage.native"):
        enable_extension(extension)
    app.update()

    import cumotion
    import isaacsim.core.experimental.utils.xform as xform_utils
    import isaacsim.robot_motion.cumotion as cu
    import omni.timeline
    import omni.usd
    from isaacsim.asset.exporter.urdf.converter import UsdToUrdfConverter
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.robot_setup.xrdf_editor import EditorState, articulation_discovery, sphere_generation
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdPhysics

    from guide_motion.model import Model

    def fail(check: str, detail) -> None:
        print(f"[guide_motion.description] {robot}: check '{check}' FAILED: {detail}; nothing written", flush=True)
        os._exit(1)  # skip Kit's teardown: nothing to save

    tmp = Path(tempfile.mkdtemp(prefix=f"guide_motion_{robot}_"))
    source = get_assets_root_path() + ROBOTS[robot]["usd"]
    stage_path = source
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()

    def open_stage(path):
        context.open_stage(path)
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        return context.get_stage()

    override = mass_override(robot, args.franka_description)
    if override:
        stage = open_stage(source)  # the asset's masses as PhysX simulates them, recorded next to the override's
        timeline.play()
        app.update()
        arm = Articulation(articulation_discovery.find_all_articulation_base_paths(stage)[0])
        app.update()
        asset_kg = dict(zip(arm.link_names, arm.get_link_masses().numpy()[0].tolist()))
        timeline.stop()
        app.update()
        stage_path = str(tmp / f"{robot}.usda")
        _write_mass_override(stage_path, source, override)
    stage = open_stage(stage_path)
    while any(prim.IsInstanceable() for prim in stage.Traverse()):  # instanced meshes cannot feed the sphere generator
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
    app.update()
    base = articulation_discovery.find_all_articulation_base_paths(stage)[0]
    bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(base)) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    link0 = str(bodies[f"{robot}_link0"].GetPath())
    fixed_root = any(p.IsA(UsdPhysics.FixedJoint) and not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
                     for p in Usd.PrimRange(stage.GetPrimAtPath(base)))

    # Timeline stopped: authored link poses, and an export that does not bake the current pose (defect D).
    usd_poses = {name: np.asarray(xform_utils.get_relative_transform(prim, link0)) for name, prim in bodies.items()}
    raw = tmp / "raw"
    raw.mkdir()
    UsdToUrdfConverter(stage=stage, root_prim_path=base, mesh_dir_name="meshes", mesh_path_prefix="./").convert(str(raw / f"{robot}.urdf"))

    timeline.play()
    for _ in range(10):
        app.update()
    editor = EditorState()
    editor.select_articulation(base)
    for link, meshes in editor.link_to_meshes.items():  # one mesh per link, as the UI's "Generate Spheres"
        mesh = "/collisions" if "/collisions" in meshes else meshes[0]
        path = editor.link_path(link)
        points, faces, counts = sphere_generation.compute_link_frame_mesh(path, path + mesh)
        editor.collision_sphere_editor.generate_spheres(path, points, faces, counts, args.spheres, args.radius_offset, False)
    editor.active_joints[:] = [name.startswith(f"{robot}_joint") for name in editor.dof_names]
    editor.joint_positions = np.array((HOME + [0.0, 0.0])[: editor.num_dof], dtype=float)
    editor.export_xrdf(str(raw / f"{robot}.xrdf"), format_version=2.0)
    urdf, xrdf, meta = build((raw / f"{robot}.urdf").read_text(), yaml.safe_load((raw / f"{robot}.xrdf").read_text()), usd_poses, robot)

    # The simulator's mass properties go into the URDF (in its link frames).
    arm = Articulation(base)
    app.update()
    dofs = list(arm.dof_names)
    arm_dofs = [dofs.index(j) for j in xrdf["cspace"]["joint_names"]]

    def settle(q):
        full = np.zeros(arm.num_dofs)
        full[arm_dofs] = q
        arm.set_dof_positions(full[None])
        arm.set_dof_position_targets(full[None])
        arm.set_dof_velocities(np.zeros((1, arm.num_dofs)))
        app.update()
        return arm.get_dof_positions().numpy()[0][arm_dofs].astype(float)

    settle(HOME)
    names = list(arm.link_names)
    unknown = sorted(set(names) - set(meta["usd_to_urdf"]))
    if unknown:
        fail("links", f"PhysX links the URDF lacks: {unknown}")
    # The M/g check indexes DOFs as for a fixed base. The Panda's franka.usd has no root FixedJoint (its
    # export got a floating joint, which fix_urdf drops), yet PhysX runs it fixed-base: recorded below.
    mass_matrix_shape = list(arm.get_mass_matrices().shape)
    if mass_matrix_shape[-1] != arm.num_dofs:
        fail("fixed_base", f"PhysX mass matrix {mass_matrix_shape} for {arm.num_dofs} DOFs: a floating base")
    base_start = arm.get_world_poses()[0].numpy()[0].copy()
    masses = arm.get_link_masses().numpy()[0]
    coms = arm.get_link_coms()[0].numpy()[0]
    inertias = arm.get_link_inertias().numpy()[0].reshape(-1, 3, 3)
    frames = {name: np.array(X) for name, X in meta["usd_to_urdf"].items()}
    urdf = set_inertials(urdf, to_urdf_frames(dict(zip(names, masses)), dict(zip(names, coms)), dict(zip(names, inertias)), frames))
    out = tmp / robot
    out.mkdir()
    (out / f"{robot}.urdf").write_text(urdf)
    (out / f"{robot}.xrdf").write_text(yaml.safe_dump(xrdf, sort_keys=False))
    model = Model.load(out)

    # Self-validation (spec 4.2.5): fail loudly, write nothing.
    checks = {}
    try:
        description_robot = cu.load_cumotion_robot(str(out), f"{robot}.urdf", str(out / f"{robot}.xrdf")).robot_description
    except Exception as error:
        fail("cumotion_load", error)
    checks["cumotion_load"] = "ok"
    inspector = cumotion.create_robot_world_inspector(description_robot, cumotion.create_world().add_world_view())
    home_hits = inspector.frames_in_self_collision(description_robot.default_cspace_configuration())
    if home_hits:
        fail("home_self_collision", home_hits)
    checks["home_self_collision"] = "free"
    kinematics = description_robot.kinematics()
    limits = np.array([[kinematics.cspace_coord_limits(i).lower, kinematics.cspace_coord_limits(i).upper]
                       for i in range(kinematics.num_cspace_coords())])
    samples = np.random.default_rng(0).uniform(limits[:, 0], limits[:, 1], size=(1000, len(limits)))
    fraction = float(np.mean([bool(inspector.frames_in_self_collision(q.reshape(-1, 1))) for q in samples]))
    if fraction > 0.08:
        fail("random_self_collision", f"{fraction:.3f} > 0.08")
    checks["random_self_collision"] = round(fraction, 3)
    worst_M = worst_g = 0.0
    for q in [model.home] + list(np.random.default_rng(1).uniform(model.lower, model.upper, size=(5, len(arm_dofs)))):
        q = settle(q)
        M_physx = arm.get_mass_matrices().numpy()[0][np.ix_(arm_dofs, arm_dofs)]
        g_physx = arm.get_dof_gravity_compensation_forces().numpy()[0][arm_dofs]
        M, g = np.asarray(model.manipulator.mass_matrix(q)), np.asarray(model.manipulator.gravity_vector(q))
        worst_M = max(worst_M, float(np.linalg.norm(M - M_physx) / np.linalg.norm(M_physx)))
        worst_g = max(worst_g, float(np.linalg.norm(g - g_physx) / np.linalg.norm(g_physx)))
    if max(worst_M, worst_g) > 1e-3:
        fail("dynamics", f"M {worst_M:.2e}, g {worst_g:.2e} relative > 1e-3")
    checks["mass_matrix_rel"], checks["gravity_rel"] = worst_M, worst_g
    meta["base"] = {
        "usd_root_fixed_joint": fixed_root,
        "exporter_floating_joint": 'type="floating"' in (raw / f"{robot}.urdf").read_text(),
        "physx_mass_matrix_shape": mass_matrix_shape,
        "physx_base_drift_m": float(np.linalg.norm(arm.get_world_poses()[0].numpy()[0] - base_start)),
    }

    # Spec 9, informational (no tolerance): the FR3's masses come from PhysX's colliders; compare once
    # with franka_description's. Every PhysX link is listed, None where franka_description has no such
    # link: Isaac 6.0's FR3 has 13 PhysX links, fr3_link8 and fr3_hand_tcp being 1.0 kg rigid bodies.
    datasheet = franka_inertials(ROBOTS[robot]["datasheet"], args.franka_description)
    meta["mass_check_kg"] = {n: [float(m), datasheet[n][0] if n in datasheet else None] for n, m in zip(names, masses)}
    for name, (physx, sheet) in meta["mass_check_kg"].items():
        sheet = "none" if sheet is None else f"{sheet:.3f} kg"
        print(f"[guide_motion.description] {name}: PhysX {physx:.3f} kg, franka_description {sheet}")
    meta.update(source_usd=source, spheres_per_link=args.spheres, radius_offset=args.radius_offset, checks=checks)
    if override:
        meta["mass_override_kg"] = {n: {"asset": asset_kg.get(n), "override": float(v[0])} for n, v in override.items()}
    (out / "description.json").write_text(json.dumps(meta, indent=1))
    if stage_path != source:
        shutil.copy(stage_path, out / f"{robot}.usda")
    timeline.stop()
    app.update()
    destination = ROBOTS_DIR / robot
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(out, destination)
    print(f"[guide_motion.description] {robot}: wrote {destination}: {checks}", flush=True)
    app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
