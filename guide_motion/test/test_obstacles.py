"""Scene objects as OSCBF barriers: the box SDF, the barrier rows, no recompile on a pose update."""

import jax
import numpy as np
import pinocchio as pin
import pytest

from fr3_fixture import HOME
from guide_motion.control import Controller, box_sdf
from guide_motion.model import Model
from guide_motion.targets import Target


def boxes(centres, halves, margin=0.0, R=np.eye(3)):
    """Controller.obstacles rows: rotation, centre, half extents, margin."""
    centres, halves = np.atleast_2d(centres), np.atleast_2d(halves)
    return np.c_[np.tile(np.ravel(R), (len(centres), 1)), centres, halves, np.full(len(centres), margin)]


BOX = boxes([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
CASES = {  # point -> (distance, gradient)
    "inside": ([1.0, 2.15, 3.0], -0.05, [0, 1, 0]),
    "centre": ([1.0, 2.0, 3.0], -0.1, None),  # equidistant faces: any unit normal
    "face": ([1.0, 2.0, 3.5], 0.2, [0, 0, 1]),
    "edge": ([1.4, 2.6, 3.0], 0.5, [0.6, 0.8, 0]),
    "corner": ([0.8, 1.6, 2.5], np.sqrt(0.1**2 + 0.2**2 + 0.2**2), -np.array([0.1, 0.2, 0.2]) / np.sqrt(0.09)),
    "on the edge": ([1.1, 2.2, 3.0], 0.0, None),  # the max(q, 0) norm at 0: the gradient sqrt(0) would NaN
}


@pytest.mark.parametrize("case", CASES)
def test_the_box_sdf_and_its_gradient(case):
    point, distance, normal = CASES[case]
    value, grad = jax.value_and_grad(lambda p: box_sdf(p, BOX[0]))(np.array(point, float))
    assert abs(float(value) - distance) < 2e-6 and np.all(np.isfinite(grad))
    if normal is not None:
        np.testing.assert_allclose(grad, normal, atol=1e-6)


def test_a_rotated_box():
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # box x axis along base (1, 1, 0)
    box = boxes([0, 0, 0], [0.5, 0.1, 0.1], R=R)[0]
    assert abs(float(box_sdf(np.array([0.4, 0.4, 0.0]), box)) - (0.4 * np.sqrt(2) - 0.5)) < 2e-6


WALL, FLOOR = [0.005, 0.2, 0.3], [0.2, 0.2, 0.005]  # half extents: a wall 10 cm ahead of the tool, a floor 10 cm under it


@pytest.fixture(scope="module", params=["torque", "velocity"])
def walled(request):
    """A controller for two boxes: a wall and a floor, placed by drive (or 1 km away)."""
    c = Controller(Model.load("fr3"), request.param, obstacle_boxes=2)
    c.warm_up()
    return c


def drive(c, wall=False, floor=False, goal=(0.2, 0.0, 0.0), seconds=2.0):
    """The tool from HOME towards its pose there + goal (a 0.2 m/s ramp, then held) -> (tool offset, lowest obstacle row).
    velocity: Euler at 240 Hz (at 60 Hz kinematic Euler chatters against an active barrier; PhysX's drives do not).
    torque: the controller at 60 Hz on Pinocchio's dynamics at 120 Hz, as the host runs it."""
    model = c.model
    p, R = model.ee_pose(HOME)
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[0, :3, 3] = p + [0.105, 0.0, 0.0] if wall else [1e3, 0.0, 0.0]
    poses[1, :3, 3] = p + [0.0, 0.0, -0.105] if floor else [-1e3, 0.0, 0.0]
    c.set_obstacles(poses, [WALL, FLOOR], [0.01, 0.01])
    c.events["infeasible"] = 0
    goal, hz = np.asarray(goal, float), 240 if c.mode == "velocity" else 120
    q, qd, low = HOME.copy(), np.zeros(7), np.inf
    for k in range(int(seconds * hz)):
        if c.mode == "velocity" or k % 2 == 0:
            ramp = min(1.0, k / hz * 0.2 / np.linalg.norm(goal))
            u = c.step(Target(position=p + ramp * goal, rotation=R), q, qd)
            low = min(low, c.barrier_values(q)[42:].min())
        if c.mode == "velocity":
            q = q + u / hz
        else:
            qd = qd + pin.aba(model.pin_model, model.pin_data, q, qd, u) / hz
            q = q + qd / hz
    return model.ee_pose(q)[0] - p, low


def test_a_wall_in_front_stops_the_tool_short(walled):
    """Without it the tool reaches the target 20 cm ahead; with a wall 10 cm ahead the hand stops in front of it,
    the barrier rows never below zero."""
    free, _ = drive(walled)
    assert free[0] > 0.19
    stopped, low = drive(walled, wall=True)
    assert 0.05 < stopped[0] < 0.1 - 0.01 and low > -1e-3 and walled.events["infeasible"] == 0


def test_an_inner_edge_stops_the_tool_on_both_faces(walled):
    """REVIEW (HIGH): a wall and a floor of one object, the target beyond their edge. With both boxes in one
    barrier row the gradient jumps at the bisector, only one face is constrained, and the QP slid the hand
    along one into the other (velocity, 240 Hz: 441 infeasible steps, the tool through both). One row per box."""
    moved, low = drive(walled, wall=True, floor=True, goal=(0.2, 0.0, -0.2))
    assert walled.events["infeasible"] == 0 and low > -1e-3
    assert moved[0] < 0.1 - 0.01 and moved[2] > -0.1 + 0.01


def test_an_obstacle_pose_update_never_recompiles(walled):
    """One barrier row per box, after today's 42; a new pose is an input of the compiled step, never a ~4 s
    recompile in the physics callback."""
    p, R = walled.model.ee_pose(HOME)
    drive(walled, seconds=0.1)
    compiled = walled._fn._cache_size(), walled._h._cache_size()
    far = walled.barrier_values(HOME)
    drive(walled, floor=True, seconds=0.1)
    near = walled.barrier_values(HOME)
    assert far.shape == near.shape == (42 + 2,) and far[-2] == near[-2] > 100 and 0.0 < near[-1] < 0.12 < 100 < far[-1]
    walled.step(Target(position=p, rotation=R), HOME, np.zeros(7))
    assert (walled._fn._cache_size(), walled._h._cache_size()) == compiled


def test_spheres_added_to_a_link_ride_on_its_joint():
    """An attached body's spheres (the wrist camera's) join the link's group: they move with it."""
    model = Model.load("fr3")
    n = len(model.sphere_radii)
    model.add_spheres("fr3_hand", [[0.0, 0.0, 0.1], [0.05, 0.0, 0.1]], [0.02, 0.03])
    assert len(model.sphere_radii) == n + 2 and model.sphere_links[-2:] == ["fr3_hand"] * 2
    assert set(model.sphere_groups[-2:]) == {len(model.arm_joints)}
    pin.framesForwardKinematics(model.pin_model, model.pin_data, HOME)
    hand = model.pin_data.oMf[model.pin_model.getFrameId("fr3_hand")].homogeneous
    joint = model.pin_data.oMi[len(model.arm_joints)].homogeneous
    np.testing.assert_allclose(joint @ np.r_[model.sphere_centers[-1], 1.0], hand @ [0.05, 0.0, 0.1, 1.0], atol=1e-9)
