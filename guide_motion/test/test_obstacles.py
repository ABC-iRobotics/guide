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


def tool_gap_run(model, obstacle, seconds=1.0, hz=240):
    """Velocity mode integrated at `hz` toward a target 20 cm ahead (+x) of the tool at HOME. Kinematic
    Euler at 60 Hz chatters against an active barrier (|u| ~5 rad/s per frame); PhysX's drives do not."""
    c = Controller(model, "velocity", obstacle_boxes=(1,) if obstacle is not None else ())
    if obstacle is not None:
        pose = np.eye(4)
        pose[:3, 3] = obstacle[0]
        c.set_obstacles(pose[None], [obstacle[1]], [0.01])
    p, R = model.ee_pose(HOME)
    target = Target(position=p + [0.2, 0, 0], rotation=R)
    q, rows = HOME.copy(), []
    for k in range(int(seconds * hz)):
        q = q + c.step(target, q, np.zeros(7)) / hz
        if k % 4 == 0:
            rows.append(c.barrier_values(q)[-1])
    return model.ee_pose(q)[0][0] - p[0], np.array(rows)


def test_a_box_in_front_stops_the_tool_short(fr3_model):
    """A wall 10 cm ahead of the tool: without it the tool reaches the target 20 cm ahead; with it,
    the hand stops in front of it and the obstacle's barrier row never goes below zero."""
    free, _ = tool_gap_run(fr3_model, None)
    p = fr3_model.ee_pose(HOME)[0]
    wall = (p + [0.105, 0, 0], [0.005, 0.2, 0.3])  # a 1 cm wall, its near face 10 cm ahead
    stopped, rows = tool_gap_run(fr3_model, wall)
    assert free > 0.19
    assert 0.05 < stopped < 0.1 - 0.01  # short of the face by more than the margin: the hand's spheres
    assert rows.min() > -1e-4


def test_an_obstacle_pose_update_never_recompiles(fr3_model):
    """One barrier row per obstacle (after today's 42), the softmin over its own boxes; a new pose is an
    input of the compiled step, never a ~4 s recompile in the physics callback."""
    c = Controller(fr3_model, "velocity", obstacle_boxes=(1, 2))  # the same input path as torque, compiled in a third of the time
    c.warm_up()
    compiled = c._fn._cache_size(), c._h._cache_size()
    h = c.barrier_values(HOME)
    assert h.shape == (42 + 2,) and h[-2:].min() > 100  # unset obstacles are far away
    p = fr3_model.ee_pose(HOME)[0]
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, 3] = [p + [0, 0, 0.5], p + [0, 0, 0.6], p + [0, 0, 0.15]]  # obstacle 2's second box near the hand
    c.set_obstacles(poses, np.full((3, 3), 0.02), np.zeros(3))
    near = c.barrier_values(HOME)
    poses[2, :3, 3] = p + [0, 0, 0.3]
    c.set_obstacles(poses, np.full((3, 3), 0.02), np.zeros(3))
    farther = c.barrier_values(HOME)
    assert near[-1] < farther[-1] < near[-2] == farther[-2]  # obstacle 2's row sees its boxes, obstacle 1's its own
    p, R = fr3_model.ee_pose(HOME)
    c.step(Target(position=p, rotation=R), HOME, np.zeros(7))
    assert (c._fn._cache_size(), c._h._cache_size()) == compiled


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
