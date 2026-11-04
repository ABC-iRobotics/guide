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
