"""Fixture welds and joint states as PhysX reads them (spikes K, J): plain pxr, no Isaac."""
import math

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from guide_core.types.randomization import fixtures


def cabinet(stage, root="/Scene_0/objects/cab"):
    """The converted wooden_cabinet's physics skeleton: a non-rigid root, an articulation root link
    welded to it, an inner rigid body tied to the link, and a drawer joint."""
    UsdGeom.Xform.Define(stage, root)
    body = UsdGeom.Xform.Define(stage, f"{root}/Geometry/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    weld = UsdPhysics.FixedJoint.Define(stage, f"{root}/Geometry/Body/PhysicsFixedJoint")
    weld.CreateBody0Rel().SetTargets([root])
    weld.CreateBody1Rel().SetTargets([body.GetPath()])
    inner = UsdGeom.Xform.Define(stage, f"{root}/Geometry/Body/object").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(inner)
    tie = UsdPhysics.FixedJoint.Define(stage, f"{root}/Geometry/Body/object/PhysicsFixedJoint")
    tie.CreateBody0Rel().SetTargets([body.GetPath()])
    tie.CreateBody1Rel().SetTargets([inner.GetPath()])
    drawer = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/Geometry/Body/object/top_level").GetPrim()
    return weld.GetPrim(), drawer


def test_only_matches_as_replicator_does_and_wants_exactly_one():
    stage = Usd.Stage.CreateInMemory()
    cabinet(stage)
    assert str(fixtures.only(stage, "/Scene_0/objects/cab$", "t").GetPath()) == "/Scene_0/objects/cab"
    with pytest.raises(ValueError, match="exactly one"):
        fixtures.only(stage, "/Scene_0/objects/cab/", "t")


def test_the_world_weld_is_the_fixed_joint_to_a_non_rigid_end():
    stage = Usd.Stage.CreateInMemory()
    weld, _ = cabinet(stage)
    assert fixtures.world_weld(stage.GetPrimAtPath("/Scene_0/objects/cab")).GetPath() == weld.GetPath()


def test_a_root_with_no_weld_or_two_is_refused():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/empty")
    with pytest.raises(ValueError, match="0 world welds"):
        fixtures.world_weld(stage.GetPrimAtPath("/empty"))
    cabinet(stage, "/two")
    extra = UsdPhysics.FixedJoint.Define(stage, "/two/Geometry/Body/second")
    extra.CreateBody0Rel().SetTargets(["/two"])
    extra.CreateBody1Rel().SetTargets(["/two/Geometry/Body"])
    with pytest.raises(ValueError, match="2 world welds"):
        fixtures.world_weld(stage.GetPrimAtPath("/two"))


def test_a_released_weld_is_still_the_weld_and_has_no_body0():
    """Review focus 1: with body0 empty PhysX reads localPos0 as a world pose at parse time too, so
    a stop and play (a second RegisterScene) no longer adds the root's transform to the anchor."""
    stage = Usd.Stage.CreateInMemory()
    weld, _ = cabinet(stage)
    fixtures.release_body0(weld)
    assert UsdPhysics.Joint(weld).GetBody0Rel().GetTargets() == []
    assert fixtures.world_weld(stage.GetPrimAtPath("/Scene_0/objects/cab")).GetPath() == weld.GetPath()


def test_local1_defaults_to_the_identity():
    stage = Usd.Stage.CreateInMemory()
    weld, _ = cabinet(stage)
    position, quat = fixtures.local1(weld)
    assert position.tolist() == [0.0, 0.0, 0.0] and quat.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_scene_matrix_is_the_scene_roots_world_transform():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Scene_1").AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))
    assert np.allclose(fixtures.scene_matrix(stage, "/Scene_1")[3, :3], [0.0, 2.0, 0.0])


def test_the_anchor_is_scene_origin_then_body_pose_then_weld_frame():
    scene = np.array(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 2.0, 0.0)))
    position, quat = fixtures.weld_world_pose(scene, [0.03, -0.27, 0.005], [0, 0, 154], [0, 0, 0], [1, 0, 0, 0])
    assert np.allclose(position, [0.03, 1.73, 0.005])
    assert np.isclose(R.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=True)[2], 154.0)
    # the weld frame 0.1 m along body1's x, which a 90 deg yaw turns to the scene's +y
    position, _ = fixtures.weld_world_pose(scene, [0.03, -0.27, 0.005], [0, 0, 90], [0.1, 0, 0], [1, 0, 0, 0])
    assert np.allclose(position, [0.03, 1.83, 0.005])


def test_write_weld_sets_the_world_anchor():
    stage = Usd.Stage.CreateInMemory()
    weld, _ = cabinet(stage)
    quat = R.from_euler("z", 30, degrees=True).as_quat(scalar_first=True)
    fixtures.write_weld(weld, [0.1, 2.0, 0.005], quat)
    joint = UsdPhysics.Joint(weld)
    assert np.allclose(joint.GetLocalPos0Attr().Get(), [0.1, 2.0, 0.005])
    q = joint.GetLocalRot0Attr().Get()
    assert np.allclose([q.GetReal(), *q.GetImaginary()], quat, atol=1e-6)


def test_a_joint_gets_the_joint_state_api_and_loses_its_holding_drive():
    stage = Usd.Stage.CreateInMemory()
    weld, drawer = cabinet(stage)
    drive = UsdPhysics.DriveAPI.Apply(drawer, "linear")
    drive.CreateStiffnessAttr(1e4)
    drive.CreateDampingAttr(50.0)
    assert fixtures.prepare_joint(drawer) == "linear"
    # read the authored list: usd-core does not know PhysX's schema, so GetAppliedSchemas() drops it
    assert "PhysicsJointStateAPI:linear" in drawer.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert drawer.GetAttribute("state:linear:physics:position").Get() == 0.0
    assert drive.GetStiffnessAttr().Get() == 0.0 and drive.GetDampingAttr().Get() == 50.0
    assert fixtures.prepare_joint(UsdPhysics.RevoluteJoint.Define(stage, "/knob").GetPrim()) == "angular"
    with pytest.raises(ValueError, match="prismatic or revolute"):
        fixtures.prepare_joint(weld)


def test_joint_states_are_written_in_usd_units_from_si():
    """Review focus 3: GUIDE speaks radians; USD keeps a revolute joint's state in degrees."""
    stage = Usd.Stage.CreateInMemory()
    knob = UsdPhysics.RevoluteJoint.Define(stage, "/knob").GetPrim()
    fixtures.prepare_joint(knob)
    knob.GetAttribute("state:angular:physics:velocity").Set(3.0)
    fixtures.write_joint_state(knob, "angular", 1.0)
    assert math.isclose(knob.GetAttribute("state:angular:physics:position").Get(), math.degrees(1.0), rel_tol=1e-6)
    assert knob.GetAttribute("state:angular:physics:velocity").Get() == 0.0
    _, drawer = cabinet(stage)
    fixtures.prepare_joint(drawer)
    fixtures.write_joint_state(drawer, "linear", -0.12)
    assert math.isclose(drawer.GetAttribute("state:linear:physics:position").Get(), -0.12, rel_tol=1e-6)


def test_a_joint_finds_its_articulation_root():
    stage = Usd.Stage.CreateInMemory()
    _, drawer = cabinet(stage)
    assert str(fixtures.articulation_root(drawer).GetPath()) == "/Scene_0/objects/cab/Geometry/Body"
    assert fixtures.articulation_root(UsdPhysics.RevoluteJoint.Define(stage, "/loose").GetPrim()) is None


def test_a_weld_to_a_dangling_body0_target_counts_as_a_world_weld():
    stage = Usd.Stage.CreateInMemory()
    weld, _ = cabinet(stage)
    UsdPhysics.Joint(weld).GetBody0Rel().SetTargets(["/gone"])
    assert fixtures.world_weld(stage.GetPrimAtPath("/Scene_0/objects/cab")).GetPath() == weld.GetPath()
    UsdPhysics.Joint(weld).GetBody1Rel().SetTargets(["/gone"])  # body1 must be a real rigid body
    with pytest.raises(ValueError, match="0 world welds"):
        fixtures.world_weld(stage.GetPrimAtPath("/Scene_0/objects/cab"))
