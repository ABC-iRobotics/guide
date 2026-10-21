"""Named keys of observation.state and action beyond the joint positions (pure numpy: no pxr, no Isaac).

A task's init.yaml lists them under dataset.observations / dataset.action like ``joint1.pos``:
  <joint>.vel              joint velocity (rad/s; m/s for the prismatic gripper), the ``.vel`` attr
  <joint>.eff              joint effort PhysX measures (N m; N for the gripper)
  vx vy vz wx wy wz        J(q) qd at the TCP: linear (m/s) and angular (rad/s) velocity, in the TCP's own axes
  fx fy fz tx ty tz        the 6-D wrench the ENVIRONMENT applies to the TCP (N, N m), in the TCP's own axes,
                           moments about the TCP point. Computed, not a sensor: see tcp_wrench().
Twist is computed only when a vx..wz key is listed, the wrench (and its Articulation) only when an fx..tz key is.
In the action each of these keys is a copy of the measured value, as x y z / rx ry rz copy the measured pose delta.
The pose keys are always recorded: state x y z (m) and rx ry rz (rad, rotation vector), both in the base frame; the
action's x y z rx ry rz are the per-frame delta (rotation vector).
"""
from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])  # world frame, m/s^2
POSE = ["x", "y", "z", "rx", "ry", "rz"]  # rx ry rz a rotation vector; the state's is absolute, the action's a per-frame delta
TWIST = ["vx", "vy", "vz", "wx", "wy", "wz"]
WRENCH = ["fx", "fy", "fz", "tx", "ty", "tz"]
CARTESIAN = {*POSE, *TWIST, *WRENCH}  # keys that are not a joint attr: parse_mapping skips them
TCP_FRAME = "TCP axes (the tool frame's own; the point is the TCP origin)"


def listed(dataset_cfg: dict, names: list[str]) -> list[str]:
    """The ``names`` that dataset.observations or dataset.action list, in ``names`` order."""
    have = {k for sec in ("observations", "action") for item in dataset_cfg.get(sec, []) for k in item}
    return [n for n in names if n in have]


def check_layout(dataset_cfg: dict) -> None:
    """Refuse an old-layout config: there wx wy wz was the state's rotation vector, now it is rx ry rz."""
    if listed(dataset_cfg, TWIST[3:]) and not listed(dataset_cfg, TWIST[:3]):
        raise ValueError("dataset lists wx/wy/wz without vx/vy/vz: the rotation vector is rx ry rz now; "
                         "wx wy wz are the TCP's angular velocity (list vx vy vz wx wy wz together)")


def key_info(name: str, action: bool = False) -> tuple[str, str]:
    """(unit, frame) of one observation.state / action key; frame is '' for a joint key."""
    if name in POSE:
        return ("m" if name in ("x", "y", "z") else "rad"), ("per-frame delta in the base frame" if action else "base frame") + ("; rotation vector" if name[0] == "r" else "")
    if name in TWIST:
        return ("m/s" if name[0] == "v" else "rad/s"), TCP_FRAME
    if name in WRENCH:
        return ("N" if name[0] == "f" else "N m"), TCP_FRAME + "; wrench the environment applies, gripper weight and static hand load removed"
    gripper = name.startswith("gripper")  # prismatic; the arm's joints are revolute
    unit = {"pos": ("rad", "m"), "vel": ("rad/s", "m/s"), "eff": ("N m", "N")}.get(name.rsplit(".", 1)[-1], ("", ""))[gripper]
    return unit, ""


def with_info(feature: dict, action: bool = False) -> dict:
    """``feature`` (observation.state or action) plus an info entry with each key's unit and frame."""
    pairs = {n: key_info(n, action) for n in feature["names"]}
    return {**feature, "info": {"unit": {n: u for n, (u, _) in pairs.items()},
                                "frame": {n: f for n, (_, f) in pairs.items() if f}}}


def tcp_wrench(f, tau, R_link, p_link, p_tcp, R_tcp, tare_mass, tare_com, gravity=GRAVITY) -> np.ndarray:
    """The wrench the environment applies to the TCP, (6,): force then moment about the TCP point, in the TCP frame's axes.

    f, tau: PhysX's incoming joint force and torque of the hand link, in the link's frame, at its origin
    (the force the parent link applies to the hand and everything below it). R_link, p_link: the link's world
    pose; p_tcp, R_tcp: the TCP's world position and rotation. tare_mass, tare_com: mass
    (kg) and world COM of the hand and everything below it (fingers, camera mount): the joint also carries
    that body's weight, which is not the environment's doing.

    Newton for that body: F_joint + F_env + m g = m a, so F_env = -F_joint - m g + m a. The inertial term
    m a is dropped (ponytail: ~0.7 kg x a few m/s^2, under 2 N on a 7 N weight; add a from the link's
    acceleration when a task needs it). Moments are taken about the TCP, not the joint.

    f = tau = 0 exactly means PhysX has no joint force yet (just after a Reset or teleport): record a zero
    wrench, not the tare (that would be a spurious +m g).
    """
    if not (np.any(f) or np.any(tau)):
        return np.zeros(6)
    F_joint = np.asarray(R_link) @ np.asarray(f, float)
    T_joint = np.asarray(R_link) @ np.asarray(tau, float) + np.cross(np.asarray(p_link) - p_tcp, F_joint)
    weight = tare_mass * np.asarray(gravity, float)
    F = -F_joint - weight
    T = -T_joint - np.cross(np.asarray(tare_com) - p_tcp, weight)
    R_inv = np.asarray(R_tcp).T  # world -> TCP axes
    return np.r_[R_inv @ F, R_inv @ T]
