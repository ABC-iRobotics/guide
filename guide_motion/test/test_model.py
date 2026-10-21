import numpy as np
import pinocchio as pin

from fr3_fixture import ARM, HOME, fd_dynamics, fd_with_camera, simulator_mass_properties

RNG = np.random.default_rng(0)


def random_q(model, n=10):
    return RNG.uniform(model.lower, model.upper, size=(n, len(model.lower)))


def test_description_basics(fr3_model):
    assert fr3_model.arm_joints == ARM and fr3_model.gripper_joint == "fr3_finger_joint1"
    assert fr3_model.tool_frame == "fr3_hand_tcp" and fr3_model.base_link == "fr3_link0"
    np.testing.assert_allclose(fr3_model.home, HOME, atol=1e-4)  # the XRDF keeps 4 decimals
    np.testing.assert_allclose(fr3_model.max_effort, [87, 87, 87, 87, 12, 12, 12])
    assert set(np.unique(fr3_model.sphere_groups)) == set(range(8))  # base + 7 joints, hand on joint 7


def test_oscbf_matches_pinocchio_on_the_same_urdf(fr3_model):
    full = pin.buildModelFromXML(fr3_model.urdf_text)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    man = fr3_model.manipulator
    tool = model.getFrameId("fr3_hand_tcp")
    for q in random_q(fr3_model):
        M = pin.crba(model, data, q)
        np.testing.assert_allclose(np.asarray(man.mass_matrix(q)), np.triu(M) + np.triu(M, 1).T, atol=1e-9)
        np.testing.assert_allclose(np.asarray(man.gravity_vector(q)), pin.computeGeneralizedGravity(model, data, q), atol=1e-9)
        pin.framesForwardKinematics(model, data, q)
        np.testing.assert_allclose(np.asarray(man.ee_transform(q)), data.oMf[tool].homogeneous, atol=1e-9)
        p, R = fr3_model.ee_pose(q)
        np.testing.assert_allclose(np.c_[R, p], data.oMf[tool].homogeneous[:3], atol=1e-12)


def test_simulator_mass_properties_give_the_physical_dynamics(fr3_model):
    """Inertials handed over in USD frames reproduce franka_description's M and g."""
    for q in random_q(fr3_model):
        M_ref, g_ref = fd_dynamics(q)
        M, g = np.asarray(fr3_model.manipulator.mass_matrix(q)), np.asarray(fr3_model.manipulator.gravity_vector(q))
        assert np.linalg.norm(M - M_ref) / np.linalg.norm(M_ref) < 1e-6
        assert np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref) < 1e-6


def test_same_mass_properties_keep_the_compiled_model(fr3_model):
    man = fr3_model.manipulator
    assert fr3_model.set_mass_properties(*simulator_mass_properties()) is False
    assert fr3_model.manipulator is man


def with_camera(mass=0.2):
    masses, coms, inertias = simulator_mass_properties()
    T = np.eye(4)
    T[:3, 3] = [0.05, 0.0, 0.0]
    return ({**masses, "camera": mass}, {**coms, "camera": np.zeros(3)}, {**inertias, "camera": 1e-4 * np.eye(3)},
            {"camera": ("fr3_hand", T)}), T


def test_an_extra_simulator_link_is_folded_into_its_parent(fr3_dir):
    from guide_motion.model import Model

    model = Model.load(fr3_dir)
    (masses, coms, inertias, extra), T = with_camera()
    assert model.set_mass_properties(masses, coms, inertias, extra_links=extra)
    g, g_ref = np.asarray(model.manipulator.gravity_vector(HOME)), fd_dynamics(HOME, fd_with_camera(T, 0.2))[1]
    assert np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref) < 1e-6
    assert np.linalg.norm(g - fd_dynamics(HOME)[1]) / np.linalg.norm(g_ref) > 1e-3  # 0.2 kg on the hand shows in g


def test_the_same_extra_link_twice_keeps_the_compiled_model(fr3_dir):
    """A re-registration (next episode, Reset) hands the same camera over again."""
    from guide_motion.model import Model

    model = Model.load(fr3_dir)
    (masses, coms, inertias, extra), _ = with_camera()
    assert model.set_mass_properties(masses, coms, inertias, extra_links=extra)
    man = model.manipulator
    assert model.set_mass_properties(masses, coms, inertias, extra_links=extra) is False
    assert model.manipulator is man and model.urdf_text.count('<link name="camera"') == 1
    (masses, coms, inertias, extra), _ = with_camera(mass=0.3)  # a heavier camera replaces it, not adds to it
    assert model.set_mass_properties(masses, coms, inertias, extra_links=extra)
    assert model.urdf_text.count('<link name="camera"') == 1 and model.manipulator is not man


def test_tcp_twist_is_the_tool_frames_finite_difference_velocity(fr3_model):
    """J(q) qd against the tool pose moved by qd dt: linear velocity of the TCP point, angular velocity, in the TCP axes."""
    dt = 1e-6
    for q in random_q(fr3_model):
        qd = RNG.uniform(-1, 1, len(q))
        p0, R0 = fr3_model.ee_pose(q)
        p1, R1 = fr3_model.ee_pose(q + qd * dt)
        twist = fr3_model.tcp_twist(q, qd)
        assert twist.shape == (6,)
        np.testing.assert_allclose(twist[:3], R0.T @ (p1 - p0) / dt, atol=1e-5)
        np.testing.assert_allclose(twist[3:], R0.T @ pin.log3(R1 @ R0.T) / dt, atol=1e-5)
        assert np.isclose(fr3_model.ee_speed(q, qd), np.linalg.norm(twist[:3]))
