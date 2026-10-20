"""Fixture welds and joint states, the way PhysX reads them (spikes K and J, 2026-10-20,
~/libero_assets/s3_inputs/spike_kj/findings.md).

- A fixture (a converted LIBERO cabinet, stove, ...) is an articulation welded to the world by one
  fixed joint whose other end is not a rigid body. PhysX takes a run-time write of that weld's
  localPos0/localRot0 as a WORLD pose; GUIDE empties body0 at build so parse time reads it so too.
- A drawer or knob is set through PhysX's JointStateAPI (state:<linear|angular>:physics:position),
  exact in two frames with no drive holding it.

Plain pxr (usd-core in the tests). replicator_guide's guide.fixture_pose and guide.joint_state call
these at build (registration) and in fire().
"""
from __future__ import annotations

import math
import re

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

KINDS = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}


def match_prims(stage, pattern: str) -> list:
    """Prims whose path ``pattern`` finds (re.search, as Replicator's get.prims), sorted by path."""
    found = re.compile(pattern).search
    return sorted((p for p in stage.Traverse() if found(str(p.GetPath()))), key=lambda p: str(p.GetPath()))


def only(stage, pattern: str, what: str):
    """The one prim ``pattern`` matches; a ValueError naming ``what`` and the matches otherwise."""
    prims = match_prims(stage, pattern)
    if len(prims) != 1:
        raise ValueError(f"{what}: {pattern!r} must match exactly one prim; it matches "
                         f"{[str(p.GetPath()) for p in prims]}")
    return prims[0]


def world_weld(root):
    """The fixed joint under ``root`` that welds it to the world: body1 a rigid body, body0 empty or
    not a rigid body (the converter targets the asset's root Xform). Exactly one, or a ValueError."""
    stage = root.GetStage()

    def rigid(targets):
        if not targets:
            return False
        prim = stage.GetPrimAtPath(targets[0])  # a dangling target is no rigid body
        return prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)

    welds = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.FixedJoint)
             and rigid(UsdPhysics.Joint(p).GetBody1Rel().GetTargets())
             and not rigid(UsdPhysics.Joint(p).GetBody0Rel().GetTargets())]
    if len(welds) != 1:
        raise ValueError(f"{root.GetPath()}: {len(welds)} world welds {welds}; guide.fixture_pose places a "
                         f"fixture welded to the world by exactly one fixed joint")
    return stage.GetPrimAtPath(welds[0])


def release_body0(weld) -> None:
    """Empty the weld's body0: PhysX then reads localPos0/localRot0 as a world pose when it parses
    the stage (a stop and play) as well as at run time (spike K)."""
    UsdPhysics.Joint(weld).GetBody0Rel().SetTargets([])


def local1(weld) -> tuple[np.ndarray, np.ndarray]:
    """The weld's frame on body1: (localPos1, localRot1 as w, x, y, z); the identity where unset."""
    joint = UsdPhysics.Joint(weld)
    position, rotation = joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
    p = np.zeros(3) if position is None else np.array(position, float)
    q = np.array([1.0, 0.0, 0.0, 0.0]) if rotation is None else np.array([rotation.GetReal(), *rotation.GetImaginary()], float)
    return p, q


def scene_matrix(stage, scene_prefix: str) -> np.ndarray:
    """The scene root's world transform, 4 x 4 in USD's row-vector layout (translation in row 3)."""
    prim = stage.GetPrimAtPath(scene_prefix)
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def weld_world_pose(scene, position, rotation_deg, local1_position, local1_wxyz):
    """The world anchor (position, quat w, x, y, z) that puts body1 at ``position`` / ``rotation_deg``
    in the scene: scene root * body1 pose * the weld's body1 frame. ``rotation_deg`` is Euler XYZ in
    degrees, the modify.pose convention (USD rotateXYZ, scipy "xyz")."""
    def homogeneous(p, r):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = r.as_matrix(), np.asarray(p, float)
        return m

    m = (np.asarray(scene, float).T
         @ homogeneous(position, Rotation.from_euler("xyz", np.asarray(rotation_deg, float), degrees=True))
         @ homogeneous(local1_position, Rotation.from_quat(np.asarray(local1_wxyz, float), scalar_first=True)))
    return m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat(scalar_first=True)


def place_root(root, position, rotation_deg) -> None:
    """Give the fixture's root prim the build pose (scene frame, Euler XYZ degrees), so the weld's
    anchor and body1 start together and PhysX does not snap the body onto the anchor at parse time."""
    api = UsdGeom.XformCommonAPI(root)
    ok = api.SetTranslate(Gf.Vec3d(*(float(v) for v in position)))
    ok = api.SetRotate(Gf.Vec3f(*(float(v) for v in rotation_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ) and ok
    if not ok:  # the common API refuses a root whose xform ops are not translate/rotate/scale
        raise ValueError(f"{root.GetPath()} has xform ops XformCommonAPI cannot pose "
                         f"({[op.GetOpName() for op in UsdGeom.Xformable(root).GetOrderedXformOps()]})")


def write_weld(weld, position, quat_wxyz) -> None:
    """localPos0 / localRot0: the world anchor of a weld whose body0 was released."""
    joint = UsdPhysics.Joint(weld)
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(float(v) for v in position)))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(float(quat_wxyz[0]), *(float(v) for v in quat_wxyz[1:])))


def prepare_joint(joint) -> str:
    """Give a prismatic or revolute joint PhysX's JointStateAPI (spike J) and zero any holding
    drive's stiffness (damping kept): LIBERO's drawers and knobs are damped only, and a robot must
    be able to move them. Returns "linear" or "angular"."""
    kind = KINDS.get(joint.GetTypeName())
    if kind is None:
        raise ValueError(f"{joint.GetPath()} is a {joint.GetTypeName()}; guide.joint_state takes a "
                         f"prismatic or revolute joint")
    if f"PhysicsJointStateAPI:{kind}" not in joint.GetAppliedSchemas():
        joint.AddAppliedSchema(f"PhysicsJointStateAPI:{kind}")
    for name in ("position", "velocity"):
        attribute = joint.GetAttribute(f"state:{kind}:physics:{name}")
        if not attribute or attribute.Get() is None:
            joint.CreateAttribute(f"state:{kind}:physics:{name}", Sdf.ValueTypeNames.Float).Set(0.0)
    stiffness = joint.GetAttribute(f"drive:{kind}:physics:stiffness")
    if stiffness and stiffness.Get():
        stiffness.Set(0.0)
    return kind


def write_joint_state(joint, kind: str, value: float) -> None:
    """The joint's state from an SI ``value`` (m, or rad: USD keeps angles in degrees), velocity zeroed."""
    joint.GetAttribute(f"state:{kind}:physics:position").Set(math.degrees(value) if kind == "angular" else float(value))
    joint.GetAttribute(f"state:{kind}:physics:velocity").Set(0.0)


def articulation_root(prim):
    """``prim`` or its nearest ancestor with ArticulationRootAPI, or None."""
    while prim and prim.IsValid():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim
        prim = prim.GetParent()
    return None
