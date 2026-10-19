"""Spike S1: load the converted LIBERO-Spatial USDs in Isaac 6.0.1 and check them.

  OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python \
      guide_core/scripts/spike_libero_isaac.py

Checks load, a 5 cm drop onto a static table, the fixtures' joints and two RGB renders;
writes docs/design/spike_libero_isaac.json. Every omni/pxr/isaacsim import lives inside a
function, so another script can import build_scene() while its own SimulationApp runs.
"""
import json
from pathlib import Path

import numpy as np

USD = Path.home() / "libero_assets/usd"
RENDERS = Path.home() / "libero_assets/spike/renders"
DESIGN = Path(__file__).resolve().parents[2] / "docs/design"
SCENE = "/Scene_0"
TABLE = (1.0, 1.2, 0.05)  # LIBERO table_full_size; top at scene z = 0, centre at the origin
DT = 1.0 / 60.0
FIXTURES = ("wooden_cabinet", "flat_stove")
# Which limit is "open" (inferred from the ranges; LIBERO's env code is not checked out):
# the drawers slide out along -Y (range -0.16..0.01 m), the knob turns on toward +rad (-0.005..2.1).
OPEN_AT = {"wooden_cabinet": "lower", "flat_stove": "upper"}
MOVABLE_SPOTS = [(-0.25, -0.2), (0.0, -0.2), (0.25, -0.2), (-0.125, -0.45), (0.125, -0.45)]  # >= 25 cm apart
FIXTURE_SPOTS = {"wooden_cabinet_1": (-0.25, 0.3), "flat_stove_1": (0.25, 0.3)}


def world_joints(stage, root):
    """Joints with an end that is not a rigid body. The converter targets the asset's root
    Xform (no RigidBodyAPI) with body0, so PhysX anchors that end to the world."""
    from pxr import Usd, UsdPhysics

    out = []
    for p in Usd.PrimRange(root):
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            ends = [j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()]
            if any(not t or not stage.GetPrimAtPath(t[0]).HasAPI(UsdPhysics.RigidBodyAPI) for t in ends):
                out.append((p, ends))
    return out


def massless_bodies(root):
    """Rigid bodies with no MassAPI and no collider of their own (a collider belongs to its
    nearest rigid-body ancestor): the converter's bodies for MJCF's geom-less outer <body>.
    PhysX warns and falls back to a default mass for them."""
    from pxr import Usd, UsdPhysics

    prims = list(Usd.PrimRange(root))
    owners = set()
    for p in prims:
        if p.HasAPI(UsdPhysics.CollisionAPI):
            while p and not p.HasAPI(UsdPhysics.RigidBodyAPI):
                p = p.GetParent()
            owners.add(p.GetPath() if p else None)
    return [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and not p.HasAPI(UsdPhysics.MassAPI) and p.GetPath() not in owners]


def build_scene(stage, assets: dict[str, list[str]]) -> dict[str, str]:
    """/Scene_0 with a physics scene, a static table (top at z = 0), lights and one reference
    per instance under /Scene_0/objects (reference root at identity, with a translate op).

    Collision-only geoms (MJCF group 0) are hidden from rendering. Movable instances:
    world-anchoring joints disabled, and each massless collider-less rigid body gets a 0.1 g
    mass -- as converted, PhysX's default mass on the bowl's outer body (10 cm off the bowl)
    tips it over. Fixtures keep their world joint, which anchors to the reference root: place
    a fixture by moving the root, not its body (PhysX snaps the body back to the anchor).
    Fixture prismatic/revolute joints without a drive get one (stiffness 1e4, damping 1e3).
    What was changed is kept in the root's customData (spike:*).
    Returns {instance: first prim carrying RigidBodyAPI or ArticulationRootAPI}.

    A World/SimulationContext created afterwards reuses /Scene_0/PhysicsScene."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

    UsdGeom.Xform.Define(stage, SCENE)
    phys = UsdPhysics.Scene.Define(stage, f"{SCENE}/PhysicsScene")
    phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    phys.CreateGravityMagnitudeAttr(9.81)

    table = UsdGeom.Cube.Define(stage, f"{SCENE}/table")
    table.CreateSizeAttr(1.0)
    table.CreateDisplayColorAttr([(0.55, 0.45, 0.35)])
    table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -TABLE[2] / 2))
    table.AddScaleOp().Set(Gf.Vec3f(*TABLE))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())

    UsdLux.DistantLight.Define(stage, f"{SCENE}/distant_light").CreateIntensityAttr(1000)
    UsdLux.DomeLight.Define(stage, f"{SCENE}/dome_light").CreateIntensityAttr(300)
    UsdGeom.Xform.Define(stage, f"{SCENE}/objects")

    bodies = {}
    for name, instances in assets.items():
        for inst in instances:
            root = stage.DefinePrim(f"{SCENE}/objects/{inst}", "Xform")
            root.GetReferences().AddReference(str(USD / name / f"{name}.usda"))
            UsdGeom.Xform(root).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
            movable = name not in FIXTURES
            disabled = [j for j, _ in world_joints(stage, root)] if movable else []
            for j in disabled:
                UsdPhysics.Joint(j).CreateJointEnabledAttr(False)
            light = massless_bodies(root) if movable else []
            for b in light:
                mass = UsdPhysics.MassAPI.Apply(b)
                mass.CreateMassAttr(1e-4)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9, 1e-9, 1e-9))
            # MJCF group-0 geoms are collision-only (robosuite never renders them); the converter
            # keeps them as visible grey Cubes over the textured meshes. PhysX ignores purpose.
            hidden = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Gprim) and p.HasAPI(UsdPhysics.CollisionAPI)
                      and not (p.GetAttribute("mjc:group") and p.GetAttribute("mjc:group").Get())]
            for p in hidden:
                UsdGeom.Imageable(p).CreatePurposeAttr(UsdGeom.Tokens.guide)
            n = len(str(root.GetPath()))
            root.SetCustomDataByKey("spike:disabled_world_joints", Vt.StringArray([str(j.GetPath())[n:] for j in disabled]))
            root.SetCustomDataByKey("spike:massless_bodies_given_0.1g", Vt.StringArray([str(b.GetPath())[n:] for b in light]))
            root.SetCustomDataByKey("spike:hidden_collision_geoms", len(hidden))
            for p in Usd.PrimRange(root):
                kind = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(p.GetTypeName())
                if name in FIXTURES and kind and f"PhysicsDriveAPI:{kind}" not in p.GetAppliedSchemas():
                    drive = UsdPhysics.DriveAPI.Apply(p, kind)
                    drive.CreateStiffnessAttr(1e4)
                    drive.CreateDampingAttr(1e3)
            bodies[inst] = next(
                (str(p.GetPath()) for p in Usd.PrimRange(root)
                 if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI)),
                None,
            )
    return bodies


def bbox(stage, path):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def place(stage, inst_path, prim_path, xy, lift):
    """Translate prim_path (the body, or the reference root of a fixture) so the instance bbox
    is centred on xy with its bottom `lift` above the table top. Both start at identity under
    an identity parent chain, so their local frame is the scene's."""
    from pxr import Gf

    lo, hi = bbox(stage, inst_path)
    attr = stage.GetPrimAtPath(prim_path).GetAttribute("xformOp:translate")
    t = np.array(attr.Get()) + [xy[0] - (lo[0] + hi[0]) / 2, xy[1] - (lo[1] + hi[1]) / 2, lift - lo[2]]
    attr.Set(Gf.Vec3d(*t))
    return float(t[2])


def physx_state(path):
    """(position, quat wxyz, linear speed) of a body as PhysX sees it."""
    from isaacsim.core.prims import RigidPrim

    view = RigidPrim(prim_paths_expr=path)
    pos, quat = view.get_world_poses()
    vel = view.get_linear_velocities()
    return np.asarray(pos)[0], np.asarray(quat)[0], float(np.linalg.norm(np.asarray(vel)[0]))


def step(world, seconds, render=False):
    for _ in range(round(seconds / DT)):
        world.step(render=render)


def check_joints(world, stage, body_path, open_at, seconds=2.0):
    """Drive every DOF to its upper then its lower limit; record where it got after `seconds`."""
    from isaacsim.core.prims import Articulation
    from pxr import Usd

    art = Articulation(prim_paths_expr=body_path)
    types = {p.GetName(): p.GetTypeName() for p in Usd.PrimRange(stage.GetPrimAtPath(body_path))}
    limits = np.asarray(art.get_dof_limits())[0]
    out = {}
    for i, name in enumerate(art.dof_names):
        lo, hi = (float(v) for v in limits[i])
        reached = {}
        for end, target in (("upper", hi), ("lower", lo)):
            art.set_joint_position_targets(np.array([[target]]), joint_indices=np.array([i]))
            step(world, seconds)
            reached[end] = round(float(np.asarray(art.get_joint_positions())[0, i]), 4)
        closed_at = "upper" if open_at == "lower" else "lower"
        out[name] = {"type": types.get(name), "lower": round(lo, 4), "upper": round(hi, 4), "open_at": open_at,
                     "reached_open": reached[open_at], "reached_closed": reached[closed_at]}
    return art, out


def render(world, stage, eye, target, path, frames=60, res=(1280, 720)):
    """One RGB frame from a look-at camera; saved only when not uniform. Returns np.ptp."""
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, f"{SCENE}/cam_{Path(path).stem}")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))  # USD default near plane is 1 unit = 1 m here
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    rp = rep.create.render_product(str(cam.GetPath()), res)
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    for _ in range(frames):  # warm-up: shaders, textures
        world.step(render=True)
    img = np.asarray(ann.get_data())
    ann.detach()
    rp.destroy()
    ptp = int(np.ptp(img[..., :3])) if img.ndim == 3 and img.size else 0
    if ptp > 0:
        Image.fromarray(img[..., :3].astype(np.uint8)).save(path)
    return ptp


def err(e):
    return f"{type(e).__name__}: {e}"


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    import importlib.metadata

    import omni.usd
    from isaacsim.core.api import World

    convert = json.loads((DESIGN / "spike_libero_convert.json").read_text())
    assets = {n: [f"{n}_{i + 1}" for i in range(2 if n == "akita_black_bowl" else 1)]
              for n, r in convert.items() if r["ok"]}
    asset_of = {inst: n for n, insts in assets.items() for inst in insts}
    out = {"_isaac": importlib.metadata.version("isaacsim"), "_articulation": {}, "_renders": [], "_render_ptp": {}}
    out_path = DESIGN / "spike_libero_isaac.json"
    RENDERS.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    try:
        bodies = build_scene(stage, assets)

        # 1. load
        spots = iter(MOVABLE_SPOTS)
        for inst, body in bodies.items():
            root = stage.GetPrimAtPath(f"{SCENE}/objects/{inst}")
            rec = out[inst] = {"loaded": bool(body) and bool(root.GetChildren()), "body_prim": body}
            fixture = asset_of[inst] in FIXTURES
            rec["world_joints"] = [
                {"joint": str(j.GetPath()).removeprefix(str(root.GetPath())),
                 "body0": [str(t) for t in ends[0]], "body1": [str(t) for t in ends[1]]}
                for j, ends in world_joints(stage, root)]
            rec.update({k: v if isinstance(v, int) else list(v) for k, v in root.GetCustomDataByKey("spike").items()})
            if not rec["loaded"]:
                continue
            lo, hi = bbox(stage, str(root.GetPath()))
            rec["bbox_size"] = np.round(hi - lo, 4).tolist()
            # 2. placement: movables 5 cm above resting height, fixtures on the table
            xy = FIXTURE_SPOTS[inst] if fixture else next(spots)
            rec["place_prim"] = str(root.GetPath()) if fixture else body
            z = place(stage, str(root.GetPath()), rec["place_prim"], xy, 0.0 if fixture else 0.05)
            rec["placed_xy"] = list(xy)
            rec["dropped_from_z"] = None if fixture else round(z, 4)  # body origin, scene frame

        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        world.reset()

        # 2. drop test: 3 s at 60 Hz, then the PhysX pose and speed
        step(world, 3.0)
        for inst, rec in out.items():
            if inst.startswith("_") or not rec["loaded"]:
                continue
            try:
                pos, quat, speed = physx_state(rec["body_prim"])
                rec["rest_pose"] = np.round(np.concatenate([pos, quat]).astype(float), 4).tolist()
                rec["rest_z_above_table"] = round(float(pos[2]), 4)  # table top is z = 0
                rec["final_speed"] = round(speed, 4)
                rec["fell_through"] = bool(pos[2] < -0.05)
            except Exception as e:  # noqa: BLE001
                rec["error"] = err(e)

        # 3. articulation
        arts = {}
        for inst in FIXTURE_SPOTS:
            if not out.get(inst, {}).get("loaded"):
                continue
            try:
                arts[inst], joints = check_joints(world, stage, out[inst]["body_prim"], OPEN_AT[asset_of[inst]])
                out["_articulation"][inst] = {"joints": joints}
            except Exception as e:  # noqa: BLE001
                out["_articulation"][inst] = {"error": err(e)}

        # 4. renders: overview from 1.2 m above the table, pitched 60 deg down (aimed 15 cm in
        # front of the centre so the front row is in frame); then the cabinet with only its
        # top drawer open
        shots = [("isaac_overview.png", (0.0, -0.15 - 1.2 / np.tan(np.radians(60)), 1.2), (0.0, -0.15, 0.0))]
        if "wooden_cabinet_1" in arts:
            art = arts["wooden_cabinet_1"]
            names = list(art.dof_names)
            lim = np.asarray(art.get_dof_limits())[0]
            art.set_joint_position_targets(np.array([[lim[i][0] if n == "top_level" else 0.0 for i, n in enumerate(names)]]))
            step(world, 2.0)
            out["_articulation"]["wooden_cabinet_1"]["render_positions"] = dict(
                zip(names, np.round(np.asarray(art.get_joint_positions(), dtype=float)[0], 4).tolist()))
            # aimed from the placement: the USD bbox does not follow PhysX
            c = np.array([*FIXTURE_SPOTS["wooden_cabinet_1"], out["wooden_cabinet_1"]["bbox_size"][2] / 2])
            shots.append(("isaac_cabinet_open.png", tuple(c + [0.2, -0.55, 0.3]), tuple(c)))
        for fname, eye, target in shots:
            path = str(RENDERS / fname)
            try:
                ptp = out["_render_ptp"][fname] = render(world, stage, eye, target, path)
                if ptp > 0:
                    out["_renders"].append(path)
            except Exception as e:  # noqa: BLE001
                out["_render_ptp"][fname] = err(e)
    except Exception as e:  # noqa: BLE001
        out["_error"] = err(e)
        raise
    finally:
        out_path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1), flush=True)
    app.close()


if __name__ == "__main__":
    main()
