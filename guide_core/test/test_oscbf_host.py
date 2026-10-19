"""OscbfHost's per-substep logic against a fake articulation, with Isaac stubbed out."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from guide_motion import description
from guide_motion.model import Model
from guide_motion.targets import POSE, Target

ROBOT = "fr3"
STUBS = ("isaacsim", "isaacsim.core", "isaacsim.core.experimental", "isaacsim.core.experimental.prims",
         "isaacsim.core.experimental.utils", "isaacsim.core.experimental.utils.stage", "isaacsim.core.simulation_manager")
DT = 1.0 / 120.0


class Arr:
    """What the experimental Articulation's getters return: .numpy() of shape (1, ...)."""

    def __init__(self, a):
        self.a = np.asarray(a, np.float32)[None]
        self.shape, self.device = self.a.shape, "cpu"

    def numpy(self):
        return self.a


def usd_frame_mass_properties(model):
    """The description's URDF inertials as PhysX reports them: in USD body frames."""
    import xml.etree.ElementTree as ET

    raw = {}
    for link in ET.fromstring(model.urdf_text).iter("link"):
        inertial = link.find("inertial")
        if inertial is None or link.get("name") not in model.usd_to_urdf:
            continue
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        raw[link.get("name")] = (float(inertial.find("mass").get("value")), np.array(inertial.find("origin").get("xyz").split(), float),
                                 np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]))
    moved = description.to_urdf_frames({n: v[0] for n, v in raw.items()}, {n: v[1] for n, v in raw.items()},
                                       {n: v[2] for n, v in raw.items()}, {n: np.linalg.inv(model.usd_to_urdf[n]) for n in raw})
    return moved


class FakeArm:
    def __init__(self, model, q, scale_M=1.0):
        self.model = model
        self.dof_names = model.arm_joints + [model.gripper_joint, model.gripper_joint.replace("1", "2")]
        self.props = usd_frame_mass_properties(model)
        self.link_names = list(self.props)
        self.q = np.r_[q, 0.0, 0.0]
        self.scale_M = scale_M
        self.efforts, self.velocity_targets, self.finger_targets, self.modes = [], [], [], []

    def get_link_masses(self):
        return Arr([self.props[n][0] for n in self.link_names])

    def get_link_coms(self):
        return Arr([self.props[n][1] for n in self.link_names]), None

    def get_link_inertias(self):
        return Arr([self.props[n][2].reshape(9) for n in self.link_names])

    def get_mass_matrices(self):
        M = np.zeros((9, 9))
        M[:7, :7] = self.scale_M * np.asarray(self.model.manipulator.mass_matrix(self.q[:7]))
        return Arr(M)

    def get_dof_gravity_compensation_forces(self):
        return Arr(np.r_[np.asarray(self.model.manipulator.gravity_vector(self.q[:7])), 0, 0])

    def get_dof_positions(self):
        return Arr(self.q)

    def get_dof_limits(self):
        lower, upper = np.r_[self.model.lower, 0.0, 0.0], np.r_[self.model.upper, 0.04, 0.04]
        return Arr(lower), Arr(upper)

    def get_dof_velocities(self):
        return Arr(np.zeros(9))

    def switch_dof_control_mode(self, mode, dof_indices):
        self.modes.append((mode, list(dof_indices)))

    def set_dof_efforts(self, u, dof_indices):
        self.efforts.append(np.array(u[0]))

    def set_dof_velocity_targets(self, u, dof_indices):
        self.velocity_targets.append(np.array(u[0]))

    def set_dof_position_targets(self, x, dof_indices):
        self.finger_targets.append((float(x[0][0]), list(dof_indices)))


class FakeController:
    made = 0

    def __init__(self, model, mode, *args):
        self.model, self.mode, self.events, self.targets = model, mode, {"infeasible": 0}, []
        FakeController.made += 1

    def warm_up(self):
        return 0.0

    def step(self, target, q, qd):
        self.targets.append(target)
        return np.full(7, 0.5)


@pytest.fixture
def host_module(monkeypatch):
    before = dict(sys.modules)
    for name in STUBS:
        sys.modules[name] = MagicMock()
    sys.modules.pop("guide_core.control.oscbf_host", None)
    module = importlib.import_module("guide_core.control.oscbf_host")
    monkeypatch.setattr(module, "Controller", FakeController)
    monkeypatch.setattr(module, "enable_compilation_cache", lambda: None)
    yield module
    for name in set(sys.modules) - set(before):
        if name.startswith(("isaacsim", "guide_core.control")):
            del sys.modules[name]
    sys.modules.update(before)


def robot_stage(drop=()):
    """The scene's robot as block_bin.usd payloads the bare FR3: its link prims under /Scene_0/fr3."""
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    for name in Model.load(ROBOT).usd_to_urdf:
        if name not in drop:
            stage.DefinePrim(f"/Scene_0/fr3/{name}", "Xform")
    return stage


def make_host(module, monkeypatch, mode="torque", rate=None, stage=None, **arm):
    model = Model.load(ROBOT)
    fake = FakeArm(Model.load(ROBOT), model.home, **arm)
    stage = robot_stage() if stage is None else stage
    monkeypatch.setattr(module, "Articulation", lambda path: fake)
    monkeypatch.setattr(module, "get_current_stage", lambda: stage)
    host = module.OscbfHost("/Scene_0/fr3", robot=ROBOT, mode=mode, rate=rate)
    return host, fake


def run(host, n):
    for _ in range(n):
        host._on_step(DT, None)


def test_no_target_holds_the_pose_with_the_arm_in_effort_mode(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    run(host, 3)
    assert arm.modes == [("effort", list(range(7)))]
    assert host.diagnostics["stale"] and len(arm.efforts) == 3
    held = host._controller.targets
    assert held[0].mode == POSE and all(t is held[0] for t in held)
    np.testing.assert_allclose(held[0].position, host.model.ee_pose(host.model.home)[0], atol=1e-6)


def test_a_fresh_target_is_used_until_it_goes_stale(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    target = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(target)
    run(host, 30)  # 0.25 s
    assert host._controller.targets[-1] is target and not host.diagnostics["stale"]
    run(host, 2)
    assert host.diagnostics["stale"] and host._controller.targets[-1] is not target


def test_velocity_mode_runs_once_per_frame_and_stops_when_stale(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch, mode="velocity")  # no rate: velocity defaults to 60 Hz
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4])))
    run(host, 20)
    assert arm.modes == [("velocity", list(range(7)))] and host.every == 2
    assert len(host._controller.targets) == 10 and len(arm.velocity_targets) == 10
    run(host, 40)
    assert host.diagnostics["stale"] and np.all(arm.velocity_targets[-1] == 0)


def test_gripper_width_is_one_finger_target_written_once(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4]), gripper_width=0.08))
    run(host, 5)
    assert arm.finger_targets == [(pytest.approx(0.04), [7])]


def test_a_dynamics_mismatch_refuses_the_robot(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch, scale_M=1.05)
    run(host, 5)
    assert "dynamics check failed" in host.diagnostics["refused"]
    assert arm.modes == [] and arm.efforts == []


def test_a_stop_attaches_again_on_the_next_play_without_recompiling(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    run(host, 2)
    made = FakeController.made
    host._on_stop(None)
    run(host, 2)
    assert len(arm.modes) == 2 and FakeController.made == made  # same masses: same compiled step


def test_a_target_sent_before_the_first_substep_is_used(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    target = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(target)
    run(host, 1)
    assert host._controller.targets == [target]


def test_a_gripper_width_outside_the_hand_is_clamped(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4]), gripper_width=0.2))
    run(host, 2)
    assert arm.finger_targets == [(pytest.approx(0.04), [7])]


def test_tensor_view_io_uses_the_views_device(host_module, monkeypatch):
    writes = []
    fake_wp = SimpleNamespace(int32="int32", float32="float32",
                              array=lambda data, dtype, device: ("index", device),
                              from_numpy=lambda data, dtype, device: writes.append(device) or ("data", device))
    monkeypatch.setattr(host_module, "wp", fake_wp)
    host, arm = make_host(host_module, monkeypatch)
    on_gpu = SimpleNamespace(shape=(1, 9), device="cuda:0", numpy=lambda: arm.get_dof_positions().numpy())
    arm._physics_articulation_view = SimpleNamespace(
        get_dof_positions=lambda: on_gpu,
        get_dof_velocities=arm.get_dof_velocities,
        set_dof_actuation_forces=lambda data, index: writes.append(("forces", data, index)),
    )
    run(host, 3)
    assert writes[0] == "cuda:0" and writes[1] == ("forces", ("data", "cuda:0"), ("index", "cuda:0"))
    assert arm.efforts == []  # the API setter was not used


def test_an_extra_physx_link_folds_into_its_parent(host_module):
    host = host_module.OscbfHost.__new__(host_module.OscbfHost)
    host.model = Model.load(ROBOT)
    names = ["fr3_hand", "camera"]
    T_camera = np.eye(4)
    T_camera[:3, 3] = [0.0, 0.0, 0.05]
    host._view = SimpleNamespace(
        shared_metatype=SimpleNamespace(link_names=names, link_parents=["fr3_link7", "fr3_hand"]),
        get_link_transforms=lambda: Arr([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0.05, 0, 0, 0, 1]]),
    )
    (parent, T), = host._extra_links(names).values()
    assert parent == "fr3_hand"
    np.testing.assert_allclose(T, host.model.usd_to_urdf["fr3_hand"] @ T_camera, atol=1e-7)


def test_an_extra_physx_link_without_the_tensor_view_refuses_the_robot(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch)
    arm.link_names = arm.link_names + ["camera"]  # FakeArm has no _physics_articulation_view: API I/O
    run(host, 3)
    assert "camera" in host.diagnostics["refused"] and arm.modes == [] and arm.efforts == []


def test_the_committed_mass_overrides_reach_the_scenes_robot_before_play(host_module, monkeypatch):
    """Controller ruling 2026-10-20: at registration the host copies robots/fr3/fr3.usda's mass, COM and
    inertia onto the scene's FR3 (Ruling 10: hand and fingers from franka_description, link8 and
    hand_tcp 1 g; link0..7 keep the asset's). A scene without those links refuses the robot."""
    from pxr import Sdf, UsdPhysics

    stage = robot_stage()
    host, arm = make_host(host_module, monkeypatch, stage=stage)
    layer = Sdf.Layer.FindOrOpen(str(description.ROBOTS_DIR / ROBOT / f"{ROBOT}.usda"))
    overs = layer.GetPrimAtPath(f"/{layer.defaultPrim}").nameChildren
    assert {o.name for o in overs} == {"fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link8", "fr3_hand_tcp"}
    for over in overs:
        prim = stage.GetPrimAtPath(f"/Scene_0/fr3/{over.name}")
        assert prim.HasAPI(UsdPhysics.MassAPI)
        assert {a.name: prim.GetAttribute(a.name).Get() for a in over.attributes} == {a.name: a.default for a in over.attributes}
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath("/Scene_0/fr3/fr3_link8")).GetMassAttr().Get() == pytest.approx(0.001)
    assert not stage.GetPrimAtPath("/Scene_0/fr3/fr3_link7").HasAPI(UsdPhysics.MassAPI)
    assert host.diagnostics["refused"] is None

    host, arm = make_host(host_module, monkeypatch, stage=robot_stage(drop={"fr3_link8"}))
    run(host, 2)
    assert "fr3_link8" in host.diagnostics["refused"] and arm.modes == [] and arm.efforts == []


def test_home_holds_the_home_configuration_until_the_next_fresh_target(host_module, monkeypatch):
    """Ruling 2: GUIDE's Reset. Drive targets do not move an effort-driven arm, so the host holds a
    JOINTS target at home, never stale, until the solver's next target replaces it."""
    from guide_motion.targets import JOINTS

    host, arm = make_host(host_module, monkeypatch)
    host.home()
    run(host, 60)  # 0.5 s: twice the staleness timeout
    home = host._controller.targets[-1]
    assert home.mode == JOINTS and np.array_equal(home.joints, host.model.home)
    assert not host.diagnostics["stale"] and all(t is home for t in host._controller.targets)
    target = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(target)
    run(host, 1)
    assert host._controller.targets[-1] is target
    run(host, 31)
    assert host.diagnostics["stale"] and host._controller.targets[-1].mode == POSE  # the ordinary stale hold


def test_the_first_substep_after_attach_runs_the_controller(host_module, monkeypatch):
    """TESTER B2-1: below the physics rate, the attach substep rewrote the last efforts, which did
    not exist yet: Isaac got None. Every play now starts with a controller step."""
    host, arm = make_host(host_module, monkeypatch, rate=60.0)
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4])))
    run(host, 3)
    assert host.every == 2 and len(host._controller.targets) == 2
    assert len(arm.efforts) == 3 and all(np.isfinite(np.asarray(u, float)).all() for u in arm.efforts)


def test_a_stop_does_not_revive_the_last_target_on_the_next_play(host_module, monkeypatch):
    """TESTER B2-2: the host's physics clock restarts on play. A target put before the stop must stay
    old once the new clock passes the time it was put at."""
    host, arm = make_host(host_module, monkeypatch)
    run(host, 40)
    old = Target(position=np.array([0.4, 0.0, 0.4]))
    host.holder.put(old)
    run(host, 20)
    host._on_stop(None)
    run(host, 80)  # the new clock passes 0.34 s, where the old target was put
    assert all(t is not old for t in host._controller.targets[60:]) and host.diagnostics["stale"]
