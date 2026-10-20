"""Robot profiles (S3 v2 spec 3.3.2): a task names its robot (robots.<name>.profile: fr3 | panda) and
where it stands (robots.<name>.mount), and GUIDE references guide_motion's robots/<p>/<p>.usda there
at registration, with the profile's wrist camera.

resolve() is pure (no pxr, no Isaac), so a solver reading init.yaml gets the same defaults; mount()
writes the stage (pxr, imported inside). guide_motion is imported only for a profile robot: importing
it sets JAX's process environment (CPU, float64), which a legacy task's process must not get from here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

APERTURE = 20.955  # mm, USD's default; square, so fovy = fovx on a square (512 x 512) image
#: LIBERO's robot0_eye_in_hand in the hand frame (robosuite's right_hand: the USD <p>_hand, approach +z)
WRIST_CAMERA = {"position": (0.05, 0.0, 0.0), "orientation_wxyz": (0.0, 0.707107, 0.707107, 0.0), "fovy": 75.0}
TCP_Z = 0.1034  # franka_description's hand -> TCP, the frame guide_motion's TCP is (description.TCP_Z)
DEFAULT_KEYS = ("joint_names", "gripper_joint", "end_effector_name", "base_name")

PROFILES = {
    p: {
        "joint_names": [f"{p}_joint{i}" for i in range(1, 8)],
        "gripper_joint": f"{p}_finger_joint1",
        "end_effector_name": f"{p}_hand_tcp",
        "base_name": f"{p}_link0",
        "wrist_camera": f"{p}_hand/wrist_camera",  # relative to the robot prim
        # Isaac's FR3 asset carries fr3_hand_tcp; its Panda asset has no TCP prim, so the profile adds one
        "tcp": None if p == "fr3" else f"{p}_hand/{p}_hand_tcp",
    }
    for p in ("fr3", "panda")
}


def resolve(entry: dict) -> dict:
    """``robots.<name>`` with its profile filled in. An entry with neither ``profile`` nor ``mount`` is a
    robot inside the scene USD (block_bin) and comes back as written. Otherwise the profile (default fr3)
    gives path /<profile>, joint_names, gripper_joint, end_effector_name, base_name,
    default_joint_states and controller: oscbf with oscbf.robot = <profile>; keys the entry sets win
    (oscbf merged key by key). A bad profile or mount raises here, failing the registration."""
    if "profile" not in entry and "mount" not in entry:
        return entry
    from guide_motion.description import HOME  # block_bin's home joints, guide_motion's q_rest

    name = entry.get("profile", "fr3")
    if name not in PROFILES:
        raise ValueError(f"robots.profile {name!r} is not one of {sorted(PROFILES)}")
    out = {"path": f"/{name}", "controller": "oscbf", "default_joint_states": [*HOME, 0.0, 0.0],
           **{key: (list(PROFILES[name][key]) if isinstance(PROFILES[name][key], list) else PROFILES[name][key])
              for key in DEFAULT_KEYS}}
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
    ``usd`` replaces guide_motion's robots/<p>/<p>.usda (tests)."""
    from guide_motion.description import ROBOTS_DIR
    from pxr import Gf, UsdGeom

    name = entry["profile"]
    profile = PROFILES[name]
    path = f"{scene_prefix}{entry['path']}"
    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"robots: {path} already exists in the scene USD; a robot inside the scene USD "
                         f"takes no profile or mount")
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(usd or str(ROBOTS_DIR / name / f"{name}.usda"))
    _place(UsdGeom.Xformable(prim), *mount_pose(entry), suffix="mount")
    if profile["tcp"] and not stage.GetPrimAtPath(f"{path}/{profile['tcp']}").IsValid():
        _place(UsdGeom.Xform.Define(stage, f"{path}/{profile['tcp']}"), (0.0, 0.0, TCP_Z), (1.0, 0.0, 0.0, 0.0))
    camera = UsdGeom.Camera.Define(stage, f"{path}/{profile['wrist_camera']}")
    _place(camera, WRIST_CAMERA["position"], WRIST_CAMERA["orientation_wxyz"])
    camera.CreateHorizontalApertureAttr(APERTURE)
    camera.CreateVerticalApertureAttr(APERTURE)
    camera.CreateFocalLengthAttr(focal_length(WRIST_CAMERA["fovy"]))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    return path
