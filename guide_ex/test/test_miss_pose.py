import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Pose
from guide_ex.utility.pose import MissPose, miss_offset

GOAL = Pose.from_numpy(np.array([0.3, 0.1, 0.05]), R.from_euler("z", 0.4) * R.from_euler("x", np.pi))


def miss(node, pose=GOAL, m=None, seed=5):
    return node.run(pose=pose, miss=m or {"axes": "x", "r": [0.045, 0.06]}, seed=seed).outputs["pose"]


@pytest.mark.parametrize("axes", ["x", "y", "z", "xy", "yx", "xz", "zy", "xyz", "zxy"])
def test_an_offset_lies_in_the_span_of_its_axes_at_a_length_in_r(axes):
    rng = np.random.default_rng(1)
    for _ in range(50):
        d = miss_offset(rng, axes, [0.012, 0.018])
        assert 0.012 <= np.linalg.norm(d) <= 0.018
        assert all(d["xyz".index(a)] == 0 for a in set("xyz") - set(axes))


@pytest.mark.parametrize("axes", ["", "w", "xx", "xyzx"])
def test_bad_axes(axes):
    with pytest.raises(ValueError):
        miss_offset(np.random.default_rng(), axes, [0.01, 0.02])


def test_a_miss_lies_along_the_poses_own_axis_once_per_goal():
    node = MissPose()
    off = miss(node).position.to_numpy() - GOAL.position.to_numpy()
    assert 0.045 <= np.linalg.norm(off) <= 0.06
    assert np.allclose(np.cross(off, GOAL.orientation.to_numpy()[:, 0]), 0)  # along the pose's x
    assert np.allclose(miss(node).position.to_numpy(), GOAL.position.to_numpy())  # the retry reaches the goal
    other = Pose.from_numpy(GOAL.position.to_numpy() + [0.1, 0, 0], GOAL.orientation.to_scipy())
    assert not np.allclose(miss(node, other).position.to_numpy(), other.position.to_numpy())  # a new goal may miss


def test_off_and_seeded():
    assert miss(MissPose(), m={"p": 0.0, "axes": "xy", "r": [0.03, 0.04]}) is GOAL
    assert MissPose().run(pose=GOAL, miss=None).outputs["pose"] is GOAL
    a, b = miss(MissPose(), seed=9), miss(MissPose(), seed=9)
    assert np.allclose(a.position.to_numpy(), b.position.to_numpy())


def test_planned_misses_fire_on_the_listed_keys_once_even_if_the_goal_moved():
    node, m = MissPose(), {"at": {"/blocks/red_block": {"axes": "x", "r": [0.045, 0.06]}}}
    run = lambda key, pose=GOAL: node.run(pose=pose, miss=m, seed=3, key=key).outputs["pose"]
    assert run("/blocks/blue_block") is GOAL  # not listed
    assert run("/blocks/red_block") is not GOAL  # listed: missed
    moved = Pose.from_numpy(GOAL.position.to_numpy() + [0.08, 0.02, 0], GOAL.orientation.to_scipy())
    assert run("/blocks/red_block", moved) is moved  # the fallen cube's re-pick is not missed again
    assert run(None) is GOAL  # a recipe without a key never misses


def test_face_narrow_side_turns_the_hands_x_at_a_near_obstacle_only():
    from guide_ex.utility.pose import FaceNarrowSide

    grasp = Pose.from_numpy(np.array([0.4, 0.0, 0.03]), R.from_euler("z", 0.1) * R.from_euler("x", np.pi))
    tower = Pose.from_numpy(np.array([0.4, 0.07, 0.03]), R.identity())  # 7 cm along +y
    out = FaceNarrowSide().run(pose=grasp, obstacle=tower).outputs["pose"]
    x = out.orientation.to_numpy()[:, 0]
    assert abs(x[1]) > 0.99  # the hand's x points at the tower, its 20 cm y body runs along it
    assert np.allclose(out.position.to_numpy(), grasp.position.to_numpy())
    assert np.allclose(out.orientation.to_numpy()[:, 2], grasp.orientation.to_numpy()[:, 2])  # still top-down
    far = Pose.from_numpy(np.array([0.4, 0.3, 0.03]), R.identity())
    assert FaceNarrowSide().run(pose=grasp, obstacle=far).outputs["pose"] is grasp


def test_no_drop_fails_only_when_the_count_falls_below_its_peak():
    from guide_ex.core.states import DemoStatus
    from guide_ex.utility.collection import NoDrop

    node = NoDrop()
    assert [node.run(value=v).status for v in (1, 1, 2, 3)] == [DemoStatus.PERFECT] * 4
    assert node.run(value=2).status == DemoStatus.FAILURE


def test_face_narrow_side_takes_the_turn_joint_7_can_reach():
    from guide_ex.utility.pose import FaceNarrowSide

    # q1 = 0, hand x at -30 deg, tower at +60 deg: x at +60 (+90 turn) gives q7 = 45 - 60 = -15 deg,
    # x at -120 (-90 turn) q7 = 45 + 120 = 165 deg, past the limit (the pick that pinned joint 7).
    grasp = Pose.from_numpy(np.array([0.4, 0.0, 0.03]), R.from_euler("z", -np.pi / 6) * R.from_euler("x", np.pi))
    tower = Pose.from_numpy(np.array([0.4 + 0.07 * np.cos(np.pi / 3), 0.07 * np.sin(np.pi / 3), 0.03]), R.identity())
    x = FaceNarrowSide().run(pose=grasp, obstacle=tower).outputs["pose"].orientation.to_numpy()[:, 0]
    assert np.allclose(x[:2], [0.5, np.sqrt(0.75)], atol=1e-6)
