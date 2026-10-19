import math
import time

import numpy as np
import pytest

from fr3_fixture import HOME
from guide_motion.control import GAINS, Controller
from guide_motion.targets import JOINTS, TWIST, Target


@pytest.fixture(scope="module", params=["torque", "velocity"])
def controller(request, fr3_model):
    c = Controller(fr3_model, request.param)
    c.warm_up()
    return c


def hold(model, q=HOME):
    p, R = model.ee_pose(q)
    return Target(position=p, rotation=R)


def test_every_barrier_is_satisfied_at_home(controller):
    assert controller.barrier_values(HOME).min() > 0


def test_hold_is_gravity_compensation_or_rest(controller):
    u = controller.step(hold(controller.model), HOME, np.zeros(7))
    if controller.mode == "torque":
        np.testing.assert_allclose(u, np.asarray(controller.model.manipulator.gravity_vector(HOME)), atol=0.02)
    else:
        assert np.abs(u).max() < 0.05


def test_recorded_inputs_give_the_same_command_twice(controller):
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, qd = HOME + 0.1 * rng.standard_normal(7), 0.1 * rng.standard_normal(7)
        target = hold(controller.model, HOME + 0.05 * rng.standard_normal(7))
        assert np.array_equal(controller.step(target, q, qd), controller.step(target, q, qd))


def test_a_target_ahead_pulls_the_tool_forward(controller):
    model = controller.model
    p, R = model.ee_pose(HOME)
    u = controller.step(Target(position=p + [0.01, 0, 0], rotation=R), HOME, np.zeros(7))
    J = np.asarray(model.manipulator.ee_jacobian(HOME))[:3]
    if controller.mode == "torque":
        man = model.manipulator
        u = np.linalg.solve(np.asarray(man.mass_matrix(HOME)), u - np.asarray(man.gravity_vector(HOME)))
    assert (J @ u)[0] > 0 and abs((J @ u)[0]) > 5 * np.abs((J @ u)[1:]).max()


def test_twist_and_joint_targets(controller):
    u = controller.step(Target(mode=TWIST, twist=np.r_[0.05, 0, 0, 0, 0, 0]), HOME, np.zeros(7))
    assert np.all(np.isfinite(u))
    u = controller.step(Target(mode=JOINTS, joints=HOME), HOME, np.zeros(7))
    assert np.all(np.isfinite(u))


def test_infeasible_step_falls_back_to_damping_and_gravity(controller, monkeypatch):
    """Ruling 1 (2026-10-19): torque mode applies joint damping plus gravity compensation, never damping alone."""
    before = controller.events["infeasible"]
    monkeypatch.setattr(controller, "_fn", lambda *a: (np.full(7, np.nan), 0.0))
    qd = np.full(7, 0.1)
    u = controller.step(hold(controller.model), HOME, qd)
    assert controller.events["infeasible"] == before + 1
    if controller.mode == "torque":
        g = np.asarray(controller.model.manipulator.gravity_vector(HOME))
        np.testing.assert_allclose(u, g - controller.gains["kd_null"] * qd, atol=1e-12)
    else:
        np.testing.assert_allclose(u, 0.0)


def test_command_is_clamped_to_the_urdf_limits(controller, monkeypatch):
    monkeypatch.setattr(controller, "_fn", lambda *a: (np.full(7, 1e3), 0.0))
    np.testing.assert_allclose(controller.step(hold(controller.model), HOME, np.zeros(7)), controller.limit)


def test_step_cost_regression(controller):
    """Standalone guard only; the RTF gate is the Isaac check (Plan B)."""
    target, rng = hold(controller.model), np.random.default_rng(2)
    times = []
    for _ in range(200):
        q = HOME + 0.01 * rng.standard_normal(7)
        start = time.perf_counter()
        controller.step(target, q, np.zeros(7))
        times.append(time.perf_counter() - start)
    assert np.mean(times[20:]) < (0.5e-3 if controller.mode == "torque" else 0.3e-3), np.mean(times[20:])


def test_a_state_past_a_joint_limit_gives_a_finite_inward_command(controller):
    q = HOME.copy()
    q[3] = controller.model.upper[3] + 0.01  # joint 4 a hair past its upper limit (a measurement can be)
    u = controller.step(hold(controller.model, HOME), q, np.zeros(7))
    assert np.all(np.isfinite(u))
    if controller.mode == "velocity":
        assert u[3] < 0


def test_float32_states_give_the_float64_command(controller):
    q32 = (HOME + 0.01).astype(np.float32)
    target = hold(controller.model)
    np.testing.assert_array_equal(controller.step(target, q32, np.zeros(7, np.float32)),
                                  controller.step(target, q32.astype(np.float64), np.zeros(7)))


def test_the_gains_are_the_specs(controller):
    """S2b spec 5: LIBERO-like torque gains (zeta = 1), OSCBF's velocity gains."""
    assert GAINS["torque"] == {"kp": 150.0, "kd": 2 * math.sqrt(150.0), "kp_null": 10.0, "kd_null": 2 * math.sqrt(10.0)}
    assert GAINS["velocity"] == {"kp_pos": 50.0, "kp_rot": 20.0, "kp_null": 10.0}
    assert controller.gains == GAINS[controller.mode]


def test_the_barrier_rows_catch_self_collision_and_the_workspace(controller):
    """FR3: 21 self-collision group pairs, 14 joint limits, 6 workspace faces, 1 singularity."""
    assert controller.barrier_values(HOME).shape == (42,)
    folded = controller.barrier_values(np.array([0, 0.5, 0, -2.9, 0, 0.8, 0.785]))  # hand folded onto the upper arm
    assert folded[:21].min() < 0 and folded[21:].min() > 0
    low = HOME.copy()
    low[1], low[3] = 1.7, -0.3  # the tool 10 cm below the base plane
    h = controller.barrier_values(low)
    assert h[37] < 0 and np.delete(h, 37).min() > 0  # row 37: tool z above the box's floor


def test_a_non_finite_state_gives_a_zero_command(controller):
    """Every host routes through step: a NaN measurement must not become a NaN command."""
    before = controller.events["infeasible"]
    q = HOME.copy()
    q[2] = np.nan
    np.testing.assert_array_equal(controller.step(hold(controller.model), q, np.zeros(7)), np.zeros(7))
    assert controller.events["infeasible"] == before + 1


def test_a_float32_pose_target_does_not_recompile(controller):
    """A recompile is a ~4 s stall inside the physics callback."""
    p, R = controller.model.ee_pose(HOME)
    compiled = controller._fn._cache_size()
    controller.step(Target(position=p.astype(np.float32), rotation=R.astype(np.float32)), HOME, np.zeros(7))
    assert controller._fn._cache_size() == compiled
