"""Pink IK planner (S2b spec 4.1): a goal more than one 20 Hz step away -> a list of Targets.

Speed profile (user ruling 3, 2026-10-19): GUIDE-EX's `speed` sets the cruise speed
v_max = speed x 0.5 m/s. Approaching the goal, v(d) = v_max (1 - e^(-d / lam)) with d the
remaining distance and lam = v_max / omega, omega = sqrt(kp_task): the commanded deceleration
matches the controller's critically damped decay. From rest, the mirrored ramp
v_max (1 - e^(-omega t)), capped by the robot's acceleration limit. A 5 mm/s floor, and the
goal itself once d is below one 50 ms step.
"""

import math

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation, Slerp

from guide_motion.control import GAINS
from guide_motion.targets import JOINTS, POSE, Target

DT = 0.05  # one target step, s (20 Hz)
MAX_SPEED = 0.5  # m/s: the cruise speed at GUIDE-EX speed 1.0
SPEED = 0.1  # m/s: the cruise speed at the default speed 0.2
ANGULAR_SPEED = 1.0  # rad/s of hand turn at SPEED; scales with speed like the line
JOINT_SPEED = 1.0  # rad/s of the fastest joint at SPEED, joint goals; scales likewise
OMEGA = math.sqrt(GAINS["torque"]["kp"])  # rad/s, the task-space PD's natural frequency
V_FLOOR = 0.005  # m/s, arrival floor
SUBSTEPS = 10  # integration steps of the profile per target
POS_TOL, ROT_TOL = 1e-3, math.radians(0.5)  # IK has reached the goal
EXTRA_STEPS = 20  # IK steps allowed at the goal before giving up


class PlanningError(RuntimeError):
    pass


def profile(length: float, v_max: float, a_max: float = math.inf) -> np.ndarray:
    """Distances along a path of `length` m at each 50 ms target; the last one is `length`."""
    lam, floor, dt = v_max / OMEGA, min(V_FLOOR, v_max), DT / SUBSTEPS
    s, t, out = 0.0, 0.0, []
    while True:
        step = 0.0
        for _ in range(SUBSTEPS):
            t += dt
            d = length - s - step
            v = min(-v_max * math.expm1(-OMEGA * t), a_max * t, -v_max * math.expm1(-d / lam))
            step += max(v, floor) * dt
        if length - s <= step:
            return np.array(out + [length])
        s += step
        out.append(s)


def _accel_limit(model, rate) -> float:
    """Path acceleration (m/s^2) at which the first joint reaches its XRDF acceleration limit.

    rate: joint motion per metre of path at the start (rad/m).
    """
    limits = np.asarray(model.xrdf["cspace"]["acceleration_limits"], float)
    return float(np.min(limits / np.maximum(np.abs(rate), 1e-9)))


def plan(model, goal: Target, q, speed: float = SPEED / MAX_SPEED) -> list:
    """Targets 50 ms apart from the configuration q to `goal`, the last one the goal itself.

    speed: GUIDE-EX's speed (0..1]. POSE goals follow a straight line (slerp for the
    orientation), each waypoint tracked by Pink IK under the URDF joint limits, so every target
    is reachable. JOINTS goals are interpolated in joint space. Both follow the module's speed
    profile; a goal within one cruise step is sent as is. TWIST goals pass through unchanged.
    """
    if not speed > 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    q = np.asarray(q, float)
    v_max, width = speed * MAX_SPEED, goal.gripper_width
    if goal.mode == JOINTS:
        goal_q = np.asarray(goal.joints, float)
        if np.any(goal_q < model.lower) or np.any(goal_q > model.upper):
            raise PlanningError(f"joint goal outside the URDF limits: {np.round(goal_q, 4).tolist()}")
        dq = goal_q - q
        length = np.abs(dq).max() * SPEED / JOINT_SPEED  # metres of "path" per the profile
        if length <= v_max * DT:
            return [Target(JOINTS, joints=goal_q, gripper_width=width)]
        sigma = profile(length, v_max, _accel_limit(model, dq / length)) / length
        return [Target(JOINTS, joints=q + dq * s, gripper_width=width) for s in sigma]
    if goal.mode != POSE:
        return [goal]
    p0, R0 = model.ee_pose(q)
    turn = Rotation.from_matrix(goal.rotation @ R0.T).as_rotvec()
    # ponytail: one turn-to-metres ratio for every speed (5 rad/s at speed 1.0, above Franka's
    # 2.5 rad/s Cartesian limit); cap ANGULAR_SPEED separately if fast turns ever track badly.
    length = max(np.linalg.norm(goal.position - p0), np.linalg.norm(turn) * SPEED / ANGULAR_SPEED)
    if length <= v_max * DT:
        return [goal]
    tool = model.pin_model.getFrameId(model.tool_frame)
    J = pin.computeFrameJacobian(model.pin_model, model.pin_data, q, tool, pin.LOCAL_WORLD_ALIGNED)
    sigma = profile(length, v_max, _accel_limit(model, np.linalg.pinv(J) @ np.r_[goal.position - p0, turn] / length)) / length
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R0, goal.rotation]))
    configuration = pink.Configuration(model.pin_model, model.pin_model.createData(), q.copy())
    task = FrameTask(model.tool_frame, position_cost=1.0, orientation_cost=1.0)
    posture = PostureTask(cost=1e-3)
    posture.set_target(q)
    targets = []
    for s in np.r_[sigma, np.ones(EXTRA_STEPS)]:
        task.set_target(pin.SE3(slerp(s).as_matrix(), (1.0 - s) * p0 + s * goal.position))
        for _ in range(3):
            velocity = pink.solve_ik(configuration, [task, posture], DT, solver="quadprog", safety_break=False)
            configuration.integrate_inplace(velocity, DT)
        T = configuration.get_transform_frame_to_world(model.tool_frame)
        if s < 1.0:
            targets.append(Target(POSE, T.translation.copy(), T.rotation.copy(), gripper_width=width))
        elif (np.linalg.norm(T.translation - goal.position) < POS_TOL
              and Rotation.from_matrix(T.rotation @ goal.rotation.T).magnitude() < ROT_TOL):
            return targets + [goal]
    raise PlanningError(f"IK did not reach the goal within {EXTRA_STEPS} steps (reach or joint limits)")
