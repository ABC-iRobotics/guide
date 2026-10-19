import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fr3_fixture import HOME
from guide_motion.planner import DT, JOINT_SPEED, MAX_SPEED, OMEGA, SPEED, V_FLOOR, PlanningError, plan, profile
from guide_motion.targets import JOINTS, POSE, Target


def test_one_step_goal_is_sent_as_is(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p + [0.002, 0, 0], rotation=R, gripper_width=0.08)
    assert plan(fr3_model, goal, HOME) == [goal]


def test_pose_goal_is_a_reachable_line_ending_at_the_goal(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p + [0.1, 0.15, -0.2], rotation=Rotation.from_euler("z", 0.6).as_matrix() @ R, gripper_width=0.03)
    targets = plan(fr3_model, goal, HOME)
    assert targets[-1] is goal and all(t.mode == POSE and t.gripper_width == 0.03 for t in targets)
    steps = np.linalg.norm(np.diff([t.position for t in targets], axis=0), axis=1)
    assert steps.max() < 1.2 * SPEED * DT
    assert len(targets) >= np.linalg.norm(goal.position - p) / (SPEED * DT)


def test_unreachable_goal_raises(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    with pytest.raises(PlanningError):
        plan(fr3_model, Target(position=p + [1.5, 0, 0], rotation=R), HOME)


def test_joint_goal_follows_the_profile_to_the_goal(fr3_model):
    goal = HOME + np.r_[0.5, 0, 0, 0, 0, 0, -0.2]
    targets = plan(fr3_model, Target(mode=JOINTS, joints=goal), HOME)
    assert len(targets) >= 10 and all(t.mode == JOINTS for t in targets)  # 0.5 rad at 1 rad/s, plus ramp and approach
    np.testing.assert_allclose(targets[-1].joints, goal)
    steps = np.abs(np.diff([HOME] + [t.joints for t in targets], axis=0)).max(axis=1)
    assert steps.max() <= JOINT_SPEED * DT + 1e-12


def test_joint_goal_outside_the_limits_raises(fr3_model):
    with pytest.raises(PlanningError):
        plan(fr3_model, Target(mode=JOINTS, joints=fr3_model.upper + 0.1), HOME)


def test_a_half_turn_of_the_hand_is_planned(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p, rotation=Rotation.from_euler("z", np.radians(179)).as_matrix() @ R)
    targets = plan(fr3_model, goal, HOME)
    assert targets[-1] is goal and all(np.all(np.isfinite(t.rotation)) for t in targets)


def test_guide_ex_speed_sets_the_cruise_speed(fr3_model):
    """Ruling 3: v_max = speed x 0.5 m/s."""
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p + [0.0, 0.25, 0.0], rotation=R)
    for speed in (0.2, 0.6):
        steps = np.linalg.norm(np.diff([t.position for t in plan(fr3_model, goal, HOME, speed=speed)], axis=0), axis=1)
        assert 0.9 * speed * MAX_SPEED * DT < steps.max() < 1.1 * speed * MAX_SPEED * DT, (speed, steps.max())
    with pytest.raises(ValueError):
        plan(fr3_model, goal, HOME, speed=0.0)


# The profile itself (ruling 3), exactly: distances along the path at each 50 ms target.

def v_start(t, v_max, a_max):
    return min(v_max * (1 - math.exp(-OMEGA * t)), a_max * t)


def v_approach(d, v_max):
    return v_max * (1 - math.exp(-d / (v_max / OMEGA)))


def speeds(s):
    return np.diff(np.r_[0.0, s]) / DT


@pytest.mark.parametrize("v_max", [0.02, 0.1, 0.5])
def test_profile_cruises_at_v_max_and_ends_exactly_at_the_length(v_max):
    s = profile(1.0, v_max)
    assert s[-1] == 1.0 and np.all(np.diff(s) > 0)
    assert v_max * (1 - 1e-4) < speeds(s).max() <= v_max + 1e-12  # the ramp's 1 - e^(-omega t) only tends to v_max


@pytest.mark.parametrize("v_max, a_max", [(0.1, math.inf), (0.5, math.inf), (0.5, 2.0)])
def test_profile_starts_from_rest_on_the_mirrored_ramp_within_the_acceleration_limit(v_max, a_max):
    v = speeds(profile(0.4, v_max, a_max))
    t = DT * np.arange(1, len(v) + 1)  # each step's end
    assert all(v[k] <= v_start(t[k], v_max, a_max) + 1e-12 for k in range(len(v) - 1))
    assert np.diff(v[: np.argmax(v)]).max() <= a_max * DT + 1e-12
    assert v[0] < 0.6 * v_max  # not a jump to cruise


@pytest.mark.parametrize("v_max", [0.05, 0.1, 0.5])
def test_profile_approach_follows_the_exponential(v_max):
    """v(d) = v_max (1 - e^(-d / lam)), lam = v_max / omega: each step's mean speed lies between
    v at the step's start and at its end, wherever the ramp and the floor do not bind."""
    s = profile(0.4, v_max)
    v, d = speeds(s), 0.4 - np.r_[0.0, s]  # d[k]: remaining at step k's start
    checked = 0
    for k in range(len(v) - 1):
        if v_start(DT * k, v_max, math.inf) >= v_approach(d[k], v_max) and v_approach(d[k + 1], v_max) > V_FLOOR:
            assert v_approach(d[k + 1], v_max) - 1e-12 <= v[k] <= v_approach(d[k], v_max) + 1e-12, k
            checked += 1
    assert checked >= 3


def test_profile_arrives_on_the_floor_and_sends_the_goal_within_one_step():
    s = profile(0.3, 0.1)
    steps = np.diff(np.r_[0.0, s])
    assert steps[:-1].min() >= V_FLOOR * DT - 1e-12  # never slower than 5 mm/s
    assert steps[-1] <= V_FLOOR * DT + 1e-12  # the goal follows once d is below one (floor) step
    assert steps[-2] < 2 * V_FLOOR * DT  # the stream slowed to the floor first, it was not cut short
