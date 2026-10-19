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
        self.damping = np.full(9, 100.0)  # the drives' damping (velocity mode keeps it, stiffness 0)
        self.drive_types = ["force"] * 9
        self.efforts, self.velocity_targets, self.finger_targets, self.modes = [], [], [], []
        self.teleports = []

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

    def set_dof_positions(self, x, dof_indices):
        self.q[list(dof_indices)] = np.asarray(x)[0]
        self.teleports.append(np.array(x[0]))

    def set_dof_velocities(self, x, dof_indices):
        pass

    def get_dof_drive_types(self):
        return [list(self.drive_types)]

    def get_dof_gains(self):
        return Arr(np.zeros(9)), Arr(self.damping)

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
        self.q_rest = np.asarray(args[2] if len(args) > 2 and args[2] is not None else model.home, float)
        self.barrier_values = lambda q: np.r_[q - model.lower, model.upper - q]  # the joint-limit barriers
        self.gravity = lambda q: model.manipulator.gravity_vector(q)  # Controller's compiled gravity
        self.limit = model.max_velocity if mode == "velocity" else model.max_effort
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


def make_host(module, monkeypatch, mode="torque", rate=None, stage=None, q_rest=None, **arm):
    model = Model.load(ROBOT)
    fake = FakeArm(Model.load(ROBOT), model.home, **arm)
    stage = robot_stage() if stage is None else stage
    monkeypatch.setattr(module, "Articulation", lambda path: fake)
    monkeypatch.setattr(module, "get_current_stage", lambda: stage)
    host = module.OscbfHost("/Scene_0/fr3", robot=ROBOT, mode=mode, rate=rate, q_rest=q_rest)
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


def test_velocity_mode_runs_once_per_frame(host_module, monkeypatch):
    host, arm = make_host(host_module, monkeypatch, mode="velocity")  # no rate: velocity defaults to 60 Hz
    host.holder.put(Target(position=np.array([0.4, 0.0, 0.4])))
    run(host, 20)
    assert arm.modes == [("velocity", list(range(7)))] and host.every == 2
    assert len(host._controller.targets) == 10 and len(arm.velocity_targets) == 10


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
    host._on_stop(None)
    run(host, 1)  # after a stop too: the first write is a fresh controller output, not the last play's efforts
    assert len(host._controller.targets) == 3 and len(arm.efforts) == 4


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


def test_the_home_hold_survives_a_stop(host_module, monkeypatch):
    """Ruling 2 pin (controller, B3b): a stop forgets old targets, but Reset's home hold is exempt
    from staleness and stays held on the next play, until a fresh target replaces it."""
    host, arm = make_host(host_module, monkeypatch)
    host.home()
    run(host, 5)
    home = host._controller.targets[-1]
    host._on_stop(None)
    run(host, 60)  # re-planned from the arm's q at the new play: equal targets, new objects
    assert all(t.mode == home.mode and np.array_equal(t.joints, home.joints) for t in host._controller.targets)
    assert not host.diagnostics["stale"]


def test_an_error_in_the_physics_step_refuses_the_robot_and_restores_its_position_drives(host_module, monkeypatch):
    """REVIEWER C B2-I2: an exception in the callback left an effort-driven arm limp (stiffness and
    damping 0, no effort) and was retried every substep. Now it refuses the robot, named, and puts
    the arm back on its position drives at its current pose."""
    host, arm = make_host(host_module, monkeypatch)
    run(host, 2)

    def broken(*args):
        raise RuntimeError("XLA runtime error")

    host._controller.step = broken
    run(host, 3)
    assert host.diagnostics["refused"] == "RuntimeError: XLA runtime error"
    assert arm.modes[-1] == ("position", list(range(7))) and arm.finger_targets[-1][1] == list(range(7))
    assert len(arm.efforts) == 2


def test_an_error_before_the_drives_switch_refuses_the_robot_once(host_module, monkeypatch):
    """REVIEWER C B2-I2: an error at attach (Isaac left the physics view None, say) was retried every
    substep and never became a refusal. Now it refuses once; nothing was switched, nothing is reverted."""
    host, arm = make_host(host_module, monkeypatch)
    attempts = []
    arm.get_link_masses = lambda: attempts.append(1) or (_ for _ in ()).throw(AssertionError("no physics view"))
    run(host, 3)
    assert host.diagnostics["refused"] == "AssertionError: no physics view" and attempts == [1]
    assert arm.modes == [] and arm.finger_targets == [] and arm.efforts == []


def test_a_missing_override_layer_refuses_the_robot_unless_it_has_no_override(host_module, monkeypatch, tmp_path):
    """Controller (B2 minor): a robot whose ROBOTS entry has a mass_override needs its <robot>.usda;
    without the file the robot is refused, with the path named. Without a mass_override: nothing to copy."""
    import shutil

    shutil.copytree(description.ROBOTS_DIR / ROBOT, tmp_path / ROBOT, ignore=shutil.ignore_patterns("*.usda"))
    monkeypatch.setattr(description, "ROBOTS_DIR", tmp_path)
    host, arm = make_host(host_module, monkeypatch)
    run(host, 2)
    assert str(tmp_path / ROBOT / f"{ROBOT}.usda") in host.diagnostics["refused"] and arm.modes == []

    monkeypatch.setitem(description.ROBOTS, ROBOT, {k: v for k, v in description.ROBOTS[ROBOT].items() if k != "mass_override"})
    host, arm = make_host(host_module, monkeypatch)
    run(host, 2)
    assert host.diagnostics["refused"] is None and arm.modes == [("effort", list(range(7)))]


def test_velocity_mode_holds_the_latched_pose_when_stale(host_module, monkeypatch):
    """FIX B4 (controller ruling): on PhysX velocity drives (stiffness 0) a zero velocity target is only
    a damper, and the FR3 sagged 9.5 mm in 5 s. Stale, velocity mode steps the controller on the latched
    pose, as torque mode does."""
    host, arm = make_host(host_module, monkeypatch, mode="velocity")
    run(host, 6)
    held = host._controller.targets
    assert host.diagnostics["stale"] and len(held) == 3 and held[0].mode == POSE and all(t is held[0] for t in held)
    np.testing.assert_allclose(held[0].position, host.model.ee_pose(host.model.home)[0], atol=1e-6)
    assert len(arm.velocity_targets) == 3


def test_home_streams_planned_joint_targets_on_the_physics_clock(host_module, monkeypatch):
    """Controller (B3b follow-up): a step target at home ran the FR3's joint 1 to 1.04x its URDF
    velocity limit (velocity mode, 1 rad off home on joints 1, 2, 4). home() now streams Plan A's
    plan(model, JOINTS home, q, speed=1.0), one target per 50 ms of the host's physics clock."""
    from guide_motion.planner import plan
    from guide_motion.targets import JOINTS

    model = Model.load(ROBOT)
    far = model.home + np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    expected = plan(model, Target(mode=JOINTS, joints=model.home), far, speed=1.0)
    host, arm = make_host(host_module, monkeypatch)
    arm.q[:7] = far
    host.home()
    run(host, 6 * len(expected) + 30)  # the plan, then 0.25 s more at the goal
    steps = host._controller.targets
    assert len(expected) > 10 and not host.diagnostics["stale"]
    for i, step in enumerate(steps):
        np.testing.assert_allclose(step.joints, expected[min(i // 6, len(expected) - 1)].joints, atol=1e-6, err_msg=f"substep {i}")  # q is float32
    np.testing.assert_array_equal(steps[-1].joints, model.home)


def test_velocity_mode_adds_gravity_over_the_drive_damping(host_module, monkeypatch):
    """FIX B4-2 (controller ruling): a velocity drive pushes D (v_target - v), so with stiffness 0 it
    holds gravity only with a standing pose error (Panda 0.113 mm). The host adds tau_g(q) / D to every
    velocity target, D read from the drives at attach; a joint without damping refuses the robot."""
    host, arm = make_host(host_module, monkeypatch, mode="velocity")
    arm.damping[:7] = np.linspace(50.0, 200.0, 7)
    run(host, 2)
    q = arm.q[:7].astype(np.float32).astype(float)  # what the host reads
    g = np.asarray(host.model.manipulator.gravity_vector(q))
    np.testing.assert_allclose(arm.velocity_targets[-1], 0.5 + g / arm.damping[:7], rtol=1e-6)

    host, arm = make_host(host_module, monkeypatch, mode="velocity")
    arm.damping[3] = 0.0
    run(host, 2)
    assert "fr3_joint4" in host.diagnostics["refused"] and arm.modes == [] and arm.velocity_targets == []


def test_the_velocity_feedforward_keeps_the_nan_guard_and_the_limit(host_module, monkeypatch):
    """REVIEWER C B4-I3: tau_g / D was added after Controller.step's finite guard and clip. A NaN q
    wrote NaN velocity targets, and a saturated command plus the feedforward crossed the URDF limit."""
    host, arm = make_host(host_module, monkeypatch, mode="velocity")
    run(host, 2)
    arm.q[1] = np.nan
    run(host, 2)
    assert np.all(np.isfinite(arm.velocity_targets[-1]))
    arm.q[1] = host.model.home[1]
    host._controller.step = lambda *args: host._controller.limit.copy()  # saturated
    run(host, 2)
    assert np.all(np.abs(arm.velocity_targets[-1]) <= host._controller.limit)


def test_velocity_drives_that_are_not_force_drives_refuse_the_robot(host_module, monkeypatch):
    """REVIEWER C B4-M2: an acceleration drive scales its force by the joint inertia, so tau_g / D would
    be silently wrong. Velocity mode refuses such a robot, the joint named, before switching its drives."""
    host, arm = make_host(host_module, monkeypatch, mode="velocity")
    arm.drive_types[1] = "acceleration"
    run(host, 2)
    assert "fr3_joint2" in host.diagnostics["refused"] and arm.modes == [] and arm.velocity_targets == []


def test_an_arm_restarted_outside_its_safe_set_is_put_at_q_rest(host_module, monkeypatch):
    """Smoke test, second RegisterScene: GUIDE's stop/play reverts the joints to the asset's authored
    zeros, outside the FR3's j4 and j6 limits. Started outside its barrier set the controller
    deadlocked against PhysX's joint limit: a limp-looking arm that no target moved. At attach the
    host now puts such an arm at q_rest (GUIDE's default_joint_states), as position drives would."""
    host, arm = make_host(host_module, monkeypatch)
    arm.q[:7] = 0.0  # j4 0 > -0.1518, j6 0 < 0.5445
    run(host, 2)
    assert host.diagnostics["refused"] is None and arm.modes == [("effort", list(range(7)))]
    np.testing.assert_allclose(arm.q[:7], host._controller.q_rest)
    held = host._controller.targets[-1]
    np.testing.assert_allclose(held.position, host.model.ee_pose(host._controller.q_rest)[0], atol=1e-6)


def test_an_unsafe_q_rest_refuses_the_robot_at_registration(host_module, monkeypatch):
    """Controller (B5-2): q_rest outside the barriers refuses the robot when it registers, not later,
    when another scene's RegisterScene restarts the sim and the arm would be sent there."""
    host, arm = make_host(host_module, monkeypatch, q_rest=[0.0] * 7)
    assert "fr3_joint4" in host.diagnostics["refused"] and "q_rest" in host.diagnostics["refused"]
    arm.q[:7] = 0.0
    run(host, 2)
    assert arm.modes == [] and arm.efforts == [] and arm.teleports == []
