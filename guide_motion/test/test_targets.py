import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from guide_motion.targets import (
    OPEN_WIDTH,
    POSE,
    Target,
    TargetHolder,
    delta_action_to_target,
    from_libero_normalized,
    to_libero_normalized,
)

LIBERO = sorted((Path.home() / "Downloads" / "libero_spatial").glob("*.hdf5"))
needs_libero = pytest.mark.skipif(not LIBERO, reason="no LIBERO-Spatial demos in ~/Downloads/libero_spatial")


def libero_demo(file_index=0, demo="demo_0"):
    h5py = pytest.importorskip("h5py")
    with h5py.File(LIBERO[file_index], "r") as f:
        d = f[f"data/{demo}"]
        actions, ee_pos, ee_ori = d["actions"][()], d["obs/ee_pos"][()], d["obs/ee_ori"][()]
    assert actions.shape[1] == 7 and ee_pos.shape[1] == 3 and ee_ori.shape[1] == 3, (actions.shape, ee_ori.shape)
    return actions, ee_pos, ee_ori


def test_clamp_holds_at_libero_bounds():
    p, R = np.array([0.4, 0.0, 0.3]), np.eye(3)
    t = delta_action_to_target([0.2, -0.2, 0.01, 1.0, -1.0, 0.1, 0.04], (p, R))
    np.testing.assert_allclose(t.position - p, [0.05, -0.05, 0.01])
    np.testing.assert_allclose(Rotation.from_matrix(t.rotation).as_rotvec(), [0.5, -0.5, 0.1], atol=1e-12)
    assert t.mode == POSE and t.gripper_width == 0.04


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis", [0, 4, 6])
def test_a_non_finite_action_is_refused(bad, axis):
    """A policy's output is a trust boundary: a NaN there would become NaN torques."""
    action = np.zeros(7)
    action[axis] = bad
    with pytest.raises(ValueError):
        delta_action_to_target(action, (np.zeros(3), np.eye(3)))


def test_libero_gripper_is_open_at_minus_one_and_closed_at_plus_one():
    assert from_libero_normalized([0, 0, 0, 0, 0, 0, -1])[6] == pytest.approx(OPEN_WIDTH)
    assert from_libero_normalized([0, 0, 0, 0, 0, 0, 1])[6] == pytest.approx(0.0)


@needs_libero
def test_libero_actions_round_trip_through_metric():
    for i in range(len(LIBERO)):
        actions, _, _ = libero_demo(i)
        back = np.array([to_libero_normalized(from_libero_normalized(a)) for a in actions])
        np.testing.assert_allclose(back, actions, atol=1e-12)


@needs_libero
def test_delta_is_applied_to_the_measured_pose():
    actions, ee_pos, ee_ori = libero_demo()
    for t in range(0, len(actions), 7):
        metric = from_libero_normalized(actions[t])
        R = Rotation.from_rotvec(ee_ori[t]).as_matrix()
        target = delta_action_to_target(metric, (ee_pos[t], R))
        np.testing.assert_allclose(target.position - ee_pos[t], np.clip(metric[:3], -0.05, 0.05), atol=1e-12)
        np.testing.assert_allclose(target.rotation, Rotation.from_rotvec(np.clip(metric[3:6], -0.5, 0.5)).as_matrix() @ R, atol=1e-12)


@needs_libero
def test_libero_arm_moves_the_way_the_delta_points():
    """Real demos: OSC_POSE's delta lives in the frame ee_pos is reported in, same sign."""
    actions, ee_pos, _ = libero_demo()
    commanded = actions[:-1, :3] * 0.05
    moved = np.diff(ee_pos, axis=0)
    big = np.linalg.norm(commanded, axis=1) > 0.01
    cosine = np.sum(commanded[big] * moved[big], axis=1) / (
        np.linalg.norm(commanded[big], axis=1) * np.linalg.norm(moved[big], axis=1)
    )
    assert big.sum() > 20 and cosine.mean() > 0.7, (big.sum(), cosine.mean())


def test_holder_latest_wins_and_goes_stale():
    holder = TargetHolder(stale_after=0.25)
    assert holder.get(0.0) == (None, True)
    a, b = Target(position=np.ones(3)), Target(position=2 * np.ones(3))
    holder.put(a)
    holder.put(b)
    target, stale = holder.get(1.0)
    assert target is b and not stale
    assert not holder.get(1.25)[1]
    assert holder.get(1.26)[1]
    holder.put(b)  # re-sending the same target refreshes it
    assert not holder.get(1.3)[1]


def test_holder_put_from_another_thread():
    holder = TargetHolder()
    targets = [Target(position=np.full(3, i)) for i in range(1000)]
    worker = threading.Thread(target=lambda: [holder.put(t) for t in targets])
    worker.start()
    worker.join()
    assert holder.get(0.0)[0] is targets[-1]
