"""Opt-in recorded modalities beyond observation.state (pure numpy: no pxr, no Isaac).

``dataset.velocity: true`` in a task's init.yaml adds
  observation.velocity   joint velocities, the joints and order of observation.state (".vel" for ".pos")
  observation.tcp_twist  J(q) qd at the TCP: linear (m/s) and angular (rad/s) velocity, robot base frame
``dataset.tcp_wrench: true`` adds
  observation.tcp_wrench the 6-D wrench the ENVIRONMENT applies to the TCP (N, N m), robot base frame,
                         moments about the TCP point. Computed, not a sensor: see tcp_wrench().
Both default to off; a dataset without them keeps exactly its old features.
"""
from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])  # world frame, m/s^2
TWIST_NAMES = ["vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"]
WRENCH_NAMES = ["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"]
BASE = "robot base frame (the frame of observation.state x..wz)"


def velocity_names(dataset_cfg: dict) -> list[str]:
    """observation.velocity's names: observation.state's joints, ``<joint>.pos`` -> ``<joint>.vel``."""
    return [k[:-4] + ".vel" for item in dataset_cfg.get("observations", []) for k in item if k.endswith(".pos")]


def _vector(names: list[str], **info) -> dict:
    return {"dtype": "float32", "shape": (len(names),), "names": names, "info": info}


def extra_features(dataset_cfg: dict) -> dict[str, dict]:
    """LeRobot features of the modalities ``dataset_cfg`` turns on; {} when none is."""
    out = {}
    if dataset_cfg.get("velocity"):
        names = velocity_names(dataset_cfg)
        if not names:
            raise ValueError("dataset.velocity needs <joint>.pos entries under dataset.observations")
        out["observation.velocity"] = _vector(names, unit="rad/s (revolute joints), m/s (prismatic gripper)")
        out["observation.tcp_twist"] = _vector(
            TWIST_NAMES, frame=BASE, point="TCP", unit="m/s (v), rad/s (w)",
            source="J(q) qd, guide_motion model, q and qd read from the articulation")
    if dataset_cfg.get("tcp_wrench"):
        out["observation.tcp_wrench"] = _vector(
            WRENCH_NAMES, frame=BASE, point="TCP", unit="N (f), N m (t)",
            sign="wrench the environment applies to the TCP; gripper weight and static hand load removed",
            source="PhysX incoming joint force of the hand link, gravity-tared, moved to the TCP")
    return out


def tcp_wrench(f, tau, R_link, p_link, p_tcp, R_base, tare_mass, tare_com, gravity=GRAVITY) -> np.ndarray:
    """The wrench the environment applies to the TCP, (6,): force then moment about the TCP point, in the base frame.

    f, tau: PhysX's incoming joint force and torque of the hand link, in the link's frame, at its origin
    (the force the parent link applies to the hand and everything below it). R_link, p_link: the link's world
    pose; p_tcp: the TCP's world position; R_base: the base frame's world rotation. tare_mass, tare_com: mass
    (kg) and world COM of the hand and everything below it (fingers, camera mount): the joint also carries
    that body's weight, which is not the environment's doing.

    Newton for that body: F_joint + F_env + m g = m a, so F_env = -F_joint - m g + m a. The inertial term
    m a is dropped (ponytail: ~0.7 kg x a few m/s^2, under 2 N on a 7 N weight; add a from the link's
    acceleration when a task needs it). Moments are taken about the TCP, not the joint.
    """
    F_joint = np.asarray(R_link) @ np.asarray(f, float)
    T_joint = np.asarray(R_link) @ np.asarray(tau, float) + np.cross(np.asarray(p_link) - p_tcp, F_joint)
    weight = tare_mass * np.asarray(gravity, float)
    F = -F_joint - weight
    T = -T_joint - np.cross(np.asarray(tare_com) - p_tcp, weight)
    R_inv = np.asarray(R_base).T
    return np.r_[R_inv @ F, R_inv @ T]
