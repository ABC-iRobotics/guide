"""Robot profiles (S3 v2 spec 3.3.2): a task names its robot (robots.<name>.profile: any guide_motion robots/<p>) and
where it stands (robots.<name>.mount), and GUIDE references guide_motion's robots/<p>/<p>.usda there
at registration, with the profile's wrist camera.

resolve() is pure (no pxr, no Isaac), so a solver reading init.yaml gets the same defaults; mount()
writes the stage (pxr, imported inside). guide_motion is imported only for a profile robot: importing
it sets JAX's process environment (CPU, float64), which a legacy task's process must not get from here.
"""
from __future__ import annotations

import copy
import functools
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

APERTURE = 20.955  # mm, USD's default; square, so fovy = fovx on a square (512 x 512) image
#: LIBERO's robot0_eye_in_hand in the hand frame (robosuite's right_hand: the USD <p>_hand, approach +z)
WRIST_CAMERA = {"position": (0.05, 0.0, 0.0), "orientation_wxyz": (0.0, 0.707107, 0.707107, 0.0), "fovy": 75.0}
DEFAULT_KEYS = ("joint_names", "gripper_joint", "gripper_joints", "end_effector_name", "base_name", "default_joint_states")
# ponytail: the Franka camera/TCP until guide_motion's description.json carries wrist_camera and tcp (plan T1)
_FRANKA = {p: {"wrist_camera": f"{p}_hand/wrist_camera", "tcp": f"{p}_hand_tcp"} for p in ("fr3", "panda")}


@functools.lru_cache(maxsize=None)
def _profile(name: str) -> dict:
    import json

    from guide_motion import description
    from guide_motion.model import Model

    names = sorted(d.name for d in description.ROBOTS_DIR.iterdir() if (d / f"{d.name}.xrdf").is_file())
    if name not in names:
        raise ValueError(f"robots.profile {name!r} is not one of {names} (guide_motion robots/*)")
    model = Model.load(name)
    meta_file = model.robot_dir / "description.json"
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    root = ET.fromstring(model.urdf_text)
    hand = [j.get("name") for j in root.iter("joint") if j.get("type") != "fixed" and j.get("name") not in model.arm_joints]
    rest = model.xrdf.get("default_joint_positions", {})
    # the Franka keeps guide_motion's HOME exactly (its XRDF rounds it to 4 decimals)
    home = description.ROBOTS.get(name, {}).get("home") or (description.HOME if name in _FRANKA else model.home)
    gripper_joints = list(getattr(model, "gripper_joints", [model.gripper_joint]))
    camera, tcp = (meta[key] if key in meta else _FRANKA.get(name, {}).get(key) for key in ("wrist_camera", "tcp"))
    parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.iter("joint")}.get(tcp)
    return {
        "joint_names": list(model.arm_joints),
        "gripper_joint": gripper_joints[0] if gripper_joints else None,
        "gripper_joints": gripper_joints,
        "end_effector_name": model.tool_frame,
        "base_name": model.base_link,
        # the arm, then every other moving URDF joint (fingers, jaws) at its XRDF default (0.0 if unlisted)
        "default_joint_states": [*map(float, home), *(float(rest.get(j, 0.0)) for j in hand)],
        "wrist_camera": camera,
        "tcp": f"{parent}/{tcp}" if tcp and parent else tcp,  # the TCP frame under its URDF parent link
    }


def profile(name: str) -> dict:
    """guide_motion's robots/<name>: joint_names, gripper_joint(s), end_effector_name, base_name,
    default_joint_states, and wrist_camera / tcp (prim paths relative to the robot prim; None: none;
    mount() adds the TCP where the asset has no prim of that name).
    Built from Model.load(name) and description.json; unknown names raise ValueError."""
    return copy.deepcopy(_profile(name))


def resolve(entry: dict) -> dict:
    """``robots.<name>`` with its profile filled in. An entry with none of ``profile``, ``mount`` and
    ``in_scene`` is a robot inside the scene USD (block_bin) and comes back as written. Otherwise the
    profile (default fr3) gives path /<profile>, joint_names, gripper_joint(s), end_effector_name,
    base_name, default_joint_states and controller: oscbf with oscbf.robot = <profile>; keys the entry
    sets win (oscbf merged key by key). ``in_scene: true``: the robot is already in the scene USD at
    ``path`` (mount() checks it, adds no reference), so it takes no ``mount``. A bad profile or mount
    raises here, failing the registration."""
    if "profile" not in entry and "mount" not in entry and not entry.get("in_scene"):
        return entry
    if entry.get("in_scene") and "mount" in entry:
        raise ValueError("robots: in_scene means the robot stands where the scene USD has it; it takes no mount")
    name = entry.get("profile", "fr3")
    defaults = profile(name)
    out = {"path": f"/{name}", "controller": "oscbf", **{key: defaults[key] for key in DEFAULT_KEYS}}
    out.update(entry)
    out["profile"] = name
    out["oscbf"] = {"robot": name, **(entry.get("oscbf") or {})}
    mount_pose(out)
    return out


def mount_pose(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """(position, quat w, x, y, z) of ``mount: {position: [x, y, z], orientation: {euler: [axes,
    angles...]}}`` (radians, scipy axes), in the scene frame; the identity where left out."""
    mount = entry.get("mount") or {}
    if set(mount) - {"position", "orientation"}:
        raise ValueError(f"robots.mount takes position and orientation; got {sorted(mount)}")
    position = np.asarray(mount.get("position", [0.0, 0.0, 0.0]), float)
    if position.shape != (3,):
        raise ValueError(f"robots.mount.position must be [x, y, z]; got {mount.get('position')!r}")
    orientation = mount.get("orientation") or {"euler": ["z", 0.0]}
    if set(orientation) != {"euler"}:
        raise ValueError(f"robots.mount.orientation must be {{euler: [axes, angles...]}}; got {orientation!r}")
    axes, *angles = orientation["euler"]
    if len(angles) != len(axes):
        raise ValueError(f"robots.mount.orientation.euler needs one angle per axis of {axes!r}; got {angles}")
    return position, Rotation.from_euler(axes, angles[0] if len(angles) == 1 else angles).as_quat(scalar_first=True)


def focal_length(fovy_deg: float, aperture: float = APERTURE) -> float:
    """The focal length that gives a square-aperture camera ``fovy_deg``."""
    return aperture / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))


def _place(xformable, position, quat_wxyz, suffix: str = "") -> None:
    """Translate + orient as the prim's only ops (``suffix`` keeps them clear of a referenced asset's)."""
    from pxr import Gf, UsdGeom

    translate = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    orient = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    translate.Set(Gf.Vec3d(*(float(v) for v in position)))
    orient.Set(Gf.Quatd(float(quat_wxyz[0]), *(float(v) for v in quat_wxyz[1:])))
    xformable.SetXformOpOrder([translate, orient])


def mount(stage, scene_prefix: str, entry: dict, usd: str | None = None) -> str:
    """Reference the profile robot (a resolve()d entry) at ``scene_prefix + path``, place it at its mount,
    add its wrist camera and, where the asset lacks one, its TCP frame. Returns the robot prim path.
    An ``in_scene`` robot is neither referenced nor placed: its path must exist in the scene USD.
    ``usd`` replaces guide_motion's robots/<p>/<p>.usda (tests)."""
    from guide_motion import description
    from guide_motion.description import ROBOTS_DIR, TCP_Z
    from pxr import Gf, Usd, UsdGeom

    name = entry["profile"]
    defaults = profile(name)
    path = f"{scene_prefix}{entry['path']}"
    if entry.get("in_scene"):
        if not stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"robots: in_scene robot {path} is not in the scene USD")
    elif stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"robots: {path} already exists in the scene USD; a robot inside the scene USD "
                         f"takes no profile or mount (or set in_scene: true)")
    else:
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(usd or str(ROBOTS_DIR / name / f"{name}.usda"))
        _place(UsdGeom.Xformable(prim), *mount_pose(entry), suffix="mount")
    tcp_z = description.ROBOTS.get(name, {}).get("tcp", (None, TCP_Z))[1]  # None: the asset has its TCP
    frame = (defaults["tcp"] or "").rsplit("/", 1)[-1]
    if frame and tcp_z is not None and not any(p.GetName() == frame for p in Usd.PrimRange(stage.GetPrimAtPath(path))):
        # Isaac's FR3 asset carries fr3_hand_tcp (anywhere under the robot); its Panda asset has none
        _place(UsdGeom.Xform.Define(stage, f"{path}/{defaults['tcp']}"), (0.0, 0.0, tcp_z), (1.0, 0.0, 0.0, 0.0))
    if not defaults["wrist_camera"]:
        return path
    camera = UsdGeom.Camera.Define(stage, f"{path}/{defaults['wrist_camera']}")
    _place(camera, WRIST_CAMERA["position"], WRIST_CAMERA["orientation_wxyz"])
    camera.CreateHorizontalApertureAttr(APERTURE)
    camera.CreateVerticalApertureAttr(APERTURE)
    camera.CreateFocalLengthAttr(focal_length(WRIST_CAMERA["fovy"]))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    return path
