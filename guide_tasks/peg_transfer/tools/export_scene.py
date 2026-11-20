"""Export davinci_sim's peg-transfer scene as one self-contained USD for GUIDE.

Run (plain usd-core pxr, no Isaac needed):
    ~/ros2_ws/.venv/bin/python -I guide_tasks/peg_transfer/tools/export_scene.py

Input: ~/ros2_ws/src/davinci_sim/scene/davinci_scene.usda + scene_info.json
(built by `~/isaacsim/python.sh build_scene.py` in davinci_sim). Output:
guide_tasks/peg_transfer/assets/peg_transfer.usd (flattened usdc, defaultPrim /World).
The PhysX schemas are authored by name: usd-core has no PhysxSchema module.
"""
import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# Tuning knobs (T10).
BLOCK_MASS = 0.002  # kg
JAW_STIFFNESS = 0.5  # N m/rad
JAW_DAMPING = 0.01  # N m s/rad
JAW_MAX_FORCE = 0.05  # N m
GRAFT_MASS = 1e-4  # kg, visual-only dvrk_model parallelogram links
SOLVER_POS_ITERS = 32
CONTACT_OFFSET = 0.001  # m
HANDOVER_HEIGHT = 0.05  # m above the board centre

ARMS = {'ECM': '/World/ECM/ecm', 'PSM1': '/World/PSM1/psm', 'PSM2': '/World/PSM2/psm'}
BLOCKS = '/World/PegTransfer/Blocks'
BOARD = '/World/PegTransfer/Board'
SRC = Path.home() / 'ros2_ws/src/davinci_sim/scene'
OUT = Path(__file__).resolve().parents[1] / 'assets/peg_transfer.usd'


def _set(prim, name, type_, value, schema=None):
    if schema and schema not in (prim.GetMetadata('apiSchemas') or Sdf.TokenListOp()).GetAddedOrExplicitItems():
        prim.AddAppliedSchema(schema)
    prim.CreateAttribute(name, type_).Set(value)


def _check_anchored(stage, info):
    """Each arm root_joint hangs off the world at the SUJ-solved arm pose (4 articulations)."""
    for arm, path in ARMS.items():
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f'{path}/root_joint'))
        assert not joint.GetBody0Rel().GetTargets(), f'{arm} root_joint body0 is not empty'
        want = Gf.Matrix4d(info['arms'][arm]['arm_world']).GetTranspose().ExtractTranslation()
        assert (Gf.Vec3d(joint.GetLocalPos0Attr().Get()) - want).GetLength() < 1e-5, f'{arm} not at arm_world'


def orient_board(stage):
    """Board as translate + orient: Replicator's modify.pose rotation silently skips a rotateZ op."""
    board = UsdGeom.Xformable(stage.GetPrimAtPath(BOARD))
    m = board.GetLocalTransformation()
    board.ClearXformOpOrder()
    for op in ('xformOp:translate', 'xformOp:rotateZ'):
        board.GetPrim().RemoveProperty(op)
    board.AddTranslateOp().Set(m.ExtractTranslation())
    board.AddOrientOp().Set(Gf.Quatf(m.ExtractRotationQuat()))


def export(src=SRC, out=OUT):
    stage = Usd.Stage.Open(str(src / 'davinci_scene.usda'))
    _check_anchored(stage, json.loads((src / 'scene_info.json').read_text()))
    stage.RemovePrim('/World/PhysicsScene')  # GUIDE owns physics

    mat = UsdShade.Material.Define(stage, '/World/PegTransfer/PhysicsMaterial')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(1.0)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)

    def contact(prim):
        _set(prim, 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, CONTACT_OFFSET, 'PhysxCollisionAPI')
        _set(prim, 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, 0.0)

    def bind(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')

    for block in stage.GetPrimAtPath(BLOCKS).GetChildren():
        UsdPhysics.MassAPI(block).GetMassAttr().Set(BLOCK_MASS)
        _set(block, 'physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool, True, 'PhysxRigidBodyAPI')
        _set(block, 'physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, SOLVER_POS_ITERS)
        _set(block, 'physxRigidBody:maxDepenetrationVelocity', Sdf.ValueTypeNames.Float, 0.05)
        bind(block)
        for c in block.GetChildren():
            if c.HasAPI(UsdPhysics.CollisionAPI):
                contact(c)
    for peg in stage.GetPrimAtPath(BOARD).GetChildren():
        if peg.GetName().startswith('peg_'):
            contact(peg)
            bind(peg)

    deg = math.pi / 180  # USD angular drive gains are per degree
    for arm, path in ARMS.items():
        _set(stage.GetPrimAtPath(path), 'physxArticulation:solverPositionIterationCount',
             Sdf.ValueTypeNames.Int, SOLVER_POS_ITERS, 'PhysxArticulationAPI')
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            name = prim.GetName()
            if name.startswith('graft_') and prim.HasAPI(UsdPhysics.MassAPI):
                UsdPhysics.MassAPI(prim).GetMassAttr().Set(GRAFT_MASS)
            elif name in ('psm_tool_gripper1_joint', 'psm_tool_gripper2_joint'):
                drive = UsdPhysics.DriveAPI(prim, 'angular')
                drive.GetStiffnessAttr().Set(JAW_STIFFNESS * deg)
                drive.GetDampingAttr().Set(JAW_DAMPING * deg)
                drive.GetMaxForceAttr().Set(JAW_MAX_FORCE)
            elif name in ('psm_tool_gripper1_link', 'psm_tool_gripper2_link'):
                bind(stage.GetPrimAtPath(prim.GetPath().AppendPath('collisions_xform/collisions')))
            # Isaac's save left joint velocities in the file (one jaw at -8292 deg/s): start at rest.
            for attr in prim.GetAuthoredAttributes():
                if attr.GetName().startswith('state:') and attr.GetName().endswith(':physics:velocity'):
                    attr.Set(0.0)

    orient_board(stage)
    centre = UsdGeom.Xformable(stage.GetPrimAtPath(BOARD)).ComputeLocalToWorldTransform(0).ExtractTranslation()
    UsdGeom.Xform.Define(stage, '/World/PegTransfer/handover').AddTranslateOp().Set(
        centre + Gf.Vec3d(0, 0, HANDOVER_HEIGHT))

    layer = stage.Flatten()
    layer.defaultPrim = 'World'
    assert not [r for r in layer.GetExternalReferences() if r], 'external references left'  # '' = internal
    assets = set()

    def visit(path):
        v = layer.GetAttributeAtPath(path).default if path.IsPropertyPath() and layer.GetAttributeAtPath(path) else None
        if isinstance(v, Sdf.AssetPath):
            assets.add(v.path)
        elif isinstance(v, Sdf.AssetPathArray):
            assets.update(x.path for x in v)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    # Only MDL by name (OmniPBR.mdl ships with Kit) or by NVIDIA Materials URL may remain, no local files.
    local = sorted(a for a in assets if a and not a.startswith('http') and '/' in a)
    assert not local, f'local asset paths left: {local}'
    print('asset paths:', sorted(assets))
    out.parent.mkdir(parents=True, exist_ok=True)
    layer.Export(str(out))
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--src', type=Path, default=SRC, help='davinci_sim scene/ directory')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    export(a.src, a.out)
