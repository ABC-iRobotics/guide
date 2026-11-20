"""Checks on the exported peg_transfer.usd (tools/export_scene.py)."""
from pathlib import Path

import pytest

USD = Path(__file__).resolve().parents[1] / 'assets/peg_transfer.usd'
pytest.importorskip('pxr')
if not USD.exists():
    pytest.skip(f'{USD} absent', allow_module_level=True)

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# Spec: i4h PSM chain + jaws (robots/psm arm_joints).
PSM_JOINTS = ['psm_yaw_joint', 'psm_pitch_end_joint', 'psm_main_insertion_joint', 'psm_tool_roll_joint',
              'psm_tool_pitch_joint', 'psm_tool_yaw_joint', 'psm_tool_gripper1_joint', 'psm_tool_gripper2_joint']
ARMS = ['/World/ECM/ecm', '/World/PSM1/psm', '/World/PSM2/psm']


@pytest.fixture(scope='module')
def stage():
    return Usd.Stage.Open(str(USD))


def test_four_articulations_anchored_to_world(stage):
    roots = sorted(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    assert roots == sorted(ARMS + ['/World/Cart/Geometry/world'])
    for arm in ARMS:
        assert not UsdPhysics.Joint(stage.GetPrimAtPath(f'{arm}/root_joint')).GetBody0Rel().GetTargets()


def test_no_physics_scene_and_self_contained(stage):
    assert not [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    layer = Sdf.Layer.FindOrOpen(str(USD))
    assert not [r for r in layer.GetExternalReferences() if r]
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path('/World')


def test_blocks_and_pegs(stage):
    blocks = stage.GetPrimAtPath('/World/PegTransfer/Blocks').GetChildren()
    assert [b.GetName() for b in blocks] == ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    for b in blocks:
        assert UsdPhysics.MassAPI(b).GetMassAttr().Get() == pytest.approx(0.002)
    pegs = [p for p in stage.GetPrimAtPath('/World/PegTransfer/Board').GetChildren() if p.GetName().startswith('peg_')]
    assert len(pegs) == 12
    for p in pegs:
        c = UsdGeom.Cylinder(p)
        assert (c.GetRadiusAttr().Get(), c.GetHeightAttr().Get()) == pytest.approx((0.0015, 0.018))


def test_psm_joint_names(stage):
    for arm in ARMS[1:]:
        names = [p.GetName() for p in Usd.PrimRange(stage.GetPrimAtPath(arm)) if p.IsA(UsdPhysics.Joint)
                 and not p.IsA(UsdPhysics.FixedJoint) and not p.GetName().startswith('graft_')]
        assert sorted(names) == sorted(PSM_JOINTS)
