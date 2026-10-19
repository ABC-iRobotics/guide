"""What one 20 Hz step hands the controller (S2b spec 4.1, 4.4).

Actions are metric everywhere in GUIDE: dp in metres, dw a rotation vector in radians,
gripper width in metres. LIBERO's [-1, 1] convention exists only at the evaluation
boundary (from_libero_normalized / to_libero_normalized).
"""

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

POSE, TWIST, JOINTS = 0, 1, 2  # == guide_msgs/MotionTarget MODE_*
MAX_DPOS, MAX_DROT = 0.05, 0.5  # LIBERO's per-step bound (m, rad per axis), kept as a safety limit
OPEN_WIDTH = 0.08  # Franka hand fully open, m


@dataclass(frozen=True)
class Target:
    """One motion target in the robot base frame."""

    mode: int = POSE
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [v m/s, w rad/s]
    joints: np.ndarray = field(default_factory=lambda: np.zeros(0))  # rad, arm joints
    gripper_width: float = float("nan")  # m, absolute; NaN leaves the gripper as it is


def delta_action_to_target(action, measured_pose) -> Target:
    """robosuite OSC_POSE's goal, in metres: clamp per axis, then apply to the MEASURED pose.

    action = [dx, dy, dz (m), rx, ry, rz (rotation vector, rad), gripper width (m)];
    measured_pose = (position (3,), rotation (3, 3)) of the end effector at this policy step.
    goal_pos = p + dp, goal_ori = R(dw) @ R. The caller holds the result for one 50 ms step.
    """
    a = np.asarray(action, dtype=float)
    position, rotation = measured_pose
    dp = np.clip(a[:3], -MAX_DPOS, MAX_DPOS)
    dw = np.clip(a[3:6], -MAX_DROT, MAX_DROT)
    return Target(
        POSE,
        np.asarray(position, dtype=float) + dp,
        Rotation.from_rotvec(dw).as_matrix() @ np.asarray(rotation, dtype=float),
        gripper_width=float(a[6]),
    )


def from_libero_normalized(action) -> np.ndarray:
    """LIBERO action -> metric: x0.05 m, x0.5 rad, gripper -1 (open) .. +1 (closed) -> width."""
    a = np.asarray(action, dtype=float)
    return np.r_[a[:3] * MAX_DPOS, a[3:6] * MAX_DROT, (1.0 - a[6]) / 2.0 * OPEN_WIDTH]


def to_libero_normalized(action) -> np.ndarray:
    """Metric action -> LIBERO's normalized one (the inverse of from_libero_normalized)."""
    a = np.asarray(action, dtype=float)
    return np.r_[a[:3] / MAX_DPOS, a[3:6] / MAX_DROT, 1.0 - 2.0 * a[6] / OPEN_WIDTH]


class TargetHolder:
    """Single-slot latest target: the ROS thread puts, the physics callback gets."""

    def __init__(self, stale_after: float = 0.25):
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._slot = (0, None)  # (sequence number, target)
        self._seen = 0
        self._fresh_at = -np.inf

    def put(self, target: Target) -> None:
        with self._lock:
            self._slot = (self._slot[0] + 1, target)

    def get(self, now: float):
        """(latest target or None, stale): stale = nothing new for stale_after s of `now`'s clock."""
        seq, target = self._slot
        if seq != self._seen:
            self._seen, self._fresh_at = seq, now
        return target, target is None or now - self._fresh_at > self.stale_after
