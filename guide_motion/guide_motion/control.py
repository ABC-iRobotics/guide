"""OSCBF control (S2b spec 4.1, 5): one small QP per step, torque or joint velocity."""

import math
import time
from pathlib import Path

import numpy as np

from guide_motion.targets import JOINTS, TWIST, Target

GAINS = {
    # LIBERO-like: robosuite OSC_POSE as LIBERO configures it (kp 150, zeta 1, all six axes);
    # nullspace PD towards the posture at the episode start.
    "torque": {"kp": 150.0, "kd": 2 * math.sqrt(150.0), "kp_null": 10.0, "kd_null": 2 * math.sqrt(10.0)},
    # OSCBF's own velocity gains (its dynamic_motion.py).
    "velocity": {"kp_pos": 50.0, "kp_rot": 20.0, "kp_null": 10.0},
}
BARRIERS = {
    "workspace": [[-1.0, 1.0], [-1.0, 1.0], [-0.05, 1.3]],  # tool box in the base frame, m
    "singularity": 1e-3,  # prod of the Jacobian's singular values
    "min_group_gap": 2,  # self-collision between link groups at least this many joints apart
    "softmin_beta": 1000.0,  # 1/m; the soft minimum undershoots the true gap by <= ln(pairs)/beta
}
INFEASIBLE_TOL = 1e-2  # a QP row the relaxed solution breaks by more than this = infeasible
CACHE_DIR = Path.home() / ".cache" / "guide_motion" / "jax"


def enable_compilation_cache(path=CACHE_DIR) -> None:
    """JAX's persistent cache: a restart loads the compiled step instead of recompiling.

    Call it before any JAX computation in the process: after one, the cache stays silently off.
    """
    import jax

    Path(path).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


class Controller:
    """step(target, q, qd) -> joint torques (torque mode) or joint velocities (velocity mode)."""

    def __init__(self, model, mode: str = "torque", gains=None, barriers=None, q_rest=None):
        assert mode in GAINS, mode
        self.model, self.mode = model, mode
        self.gains = {**GAINS[mode], **(gains or {})}
        self.barriers = {**BARRIERS, **(barriers or {})}
        self.q_rest = np.asarray(model.home if q_rest is None else q_rest, float)
        self.limit = model.max_effort if mode == "torque" else model.max_velocity
        self.events = {"infeasible": 0}
        self._fn, self._h, self.gravity = _build(model, mode, self.gains, self.barriers)

    def warm_up(self) -> float:
        """Compile the step (seconds taken). Without it the first physics step stalls."""
        start = time.perf_counter()
        position, rotation = self.model.ee_pose(self.q_rest)
        self.step(Target(position=position, rotation=rotation), self.q_rest, np.zeros_like(self.q_rest))
        self.gravity(self.q_rest)  # the fallback's, so an infeasible step never compiles mid-episode
        self.barrier_values(self.q_rest)  # the host checks the barriers at registration and attach
        self.events["infeasible"] = 0
        return time.perf_counter() - start

    def step(self, target: Target, q, qd) -> np.ndarray:
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        des_q, twist = self.q_rest, np.zeros(6)
        if target.mode == JOINTS:
            des_q = np.asarray(target.joints, float)
            position, rotation = self.model.ee_pose(des_q)
        elif target.mode == TWIST:
            position, rotation = self.model.ee_pose(q)
            twist = np.asarray(target.twist, float)
        else:
            # float64 like every other input: a float32 pose would recompile the step (~4 s)
            position, rotation = np.asarray(target.position, float), np.asarray(target.rotation, float)
            # The planner's IK configuration, when given, is the posture: with q_rest instead, the
            # null space drifts off the planned branch where joints 1 and 3 line up (q2 ~ 0) and
            # carried q1 into its limit on block_bin's place moves.
            if len(target.joints) == len(des_q):
                des_q = np.asarray(target.joints, float)
        u, violation = self._fn(q, qd, position, rotation, twist[:3], twist[3:], des_q)
        u = np.asarray(u)
        if not np.all(np.isfinite(u)) or float(violation) > INFEASIBLE_TOL:
            self.events["infeasible"] += 1
            # Ruling 1 (2026-10-19): torque mode holds the arm up (gravity) and damps it, never damping alone.
            if self.mode == "torque":
                u = np.asarray(self.gravity(q)) - self.gains["kd_null"] * qd
            else:
                u = np.zeros_like(q)
        # a non-finite state (a NaN measurement) leaves even the fallback non-finite: command nothing there
        return np.clip(np.where(np.isfinite(u), u, 0.0), -self.limit, self.limit)

    def barrier_values(self, q) -> np.ndarray:
        """Every barrier h(q); all > 0 means q is inside the safe set."""
        return np.asarray(self._h(np.asarray(q, float)))


def _barrier_fn(model, man, b):
    import jax
    import jax.numpy as jnp

    C = np.c_[model.sphere_centers, np.ones(len(model.sphere_radii))]
    r, group, links = model.sphere_radii, model.sphere_groups, model.sphere_links
    pairs = {}
    for a in range(len(r)):
        for c in range(a + 1, len(r)):
            if abs(group[a] - group[c]) >= b["min_group_gap"] and frozenset((links[a], links[c])) not in model.ignore:
                pairs.setdefault((min(group[a], group[c]), max(group[a], group[c])), []).append((a, c))
    index = [(np.array([p[0] for p in v]), np.array([p[1] for p in v])) for v in pairs.values()]
    lo, hi = np.array(man.joint_lower_limits), np.array(man.joint_upper_limits)
    box, beta = np.asarray(b["workspace"], float), float(b["softmin_beta"])

    def h(q):
        T = jnp.concatenate([jnp.eye(4)[None], man.joint_to_world_transforms(q)])  # [base, joint 1..7]
        p = jnp.einsum("sij,sj->si", T[group], C)[:, :3]
        # One barrier per pair of link groups: a smooth minimum of their sphere gaps (never above the true one).
        selfcol = [-jax.nn.logsumexp(-beta * (jnp.linalg.norm(p[i] - p[j], axis=1) - r[i] - r[j])) / beta for i, j in index]
        tool = man.ee_position(q)
        sigma = jax.lax.linalg.svd(man.ee_jacobian(q), compute_uv=False)
        return jnp.concatenate([
            jnp.stack(selfcol) if selfcol else jnp.zeros(0),
            hi - q, q - lo,
            tool - box[:, 0], box[:, 1] - tool,
            jnp.array([jnp.prod(sigma) - b["singularity"]]),
        ])

    return h


def _build(model, mode, gains, barriers):
    import jax
    import jax.numpy as jnp
    import qpax
    from cbfpy import CBF
    from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController
    from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig

    man = model.manipulator
    n = man.num_joints
    h = _barrier_fn(model, man, barriers)
    z3 = jnp.zeros(3)
    if mode == "torque":

        @jax.tree_util.register_static
        class Config(OSCBFTorqueConfig):
            def __init__(self):
                super().__init__(man, rot_obj_weight=0.1)

            def h_2(self, z, **kwargs):
                return h(z[:n])

            def alpha(self, x):
                return 10.0 * x

            def alpha_2(self, x):
                return 10.0 * x

        task = PoseTaskTorqueController(n, gains["kp"], gains["kd"], gains["kp_null"], gains["kd_null"], None, None)

        def nominal(q, qd, p, R, v, w, des_q):
            M, M_inv, g, c, J, T = man.torque_control_matrices(q, qd)
            u = task(q, qd, T[:3, 3], T[:3, :3], p, R, v, w, z3, z3, des_q, jnp.zeros(n), J, M, M_inv, g, c)
            return jnp.concatenate([q, qd]), u

    else:

        @jax.tree_util.register_static
        class Config(OSCBFVelocityConfig):
            def __init__(self):
                super().__init__(man, rot_obj_weight=0.1)

            def h_1(self, z, **kwargs):
                return h(z)

            def alpha(self, x):
                return 10.0 * x

        kp = np.r_[np.full(3, gains["kp_pos"]), np.full(3, gains["kp_rot"])]
        task = PoseTaskVelocityController(n, kp, gains["kp_null"], None, None)

        def nominal(q, qd, p, R, v, w, des_q):
            M_inv, J, T = man.dynamically_consistent_velocity_control_matrices(q)
            return q, task(q, T[:3, 3], T[:3, :3], p, R, v, w, des_q, J, M_inv)

    cbf = CBF.from_config(Config())

    @jax.jit
    def fn(q, qd, p, R, v, w, des_q):
        z, u_nom = nominal(q, qd, p, R, v, w, des_q)
        # CBF.safety_filter's relaxed QP, spelled out so the step also reports how far the
        # solution breaks a row (the relaxed QP never says "infeasible" by itself).
        P, c, _, _, G, hq = cbf.qp_data(z, u_nom)
        penalty = jnp.asarray(cbf.constraint_relaxation_penalties)
        u = qpax.solve_qp_elastic_primal(P, c, G, hq, penalty=penalty, solver_tol=cbf.solver_tol)
        return u, jnp.max(G @ u - hq)

    return fn, jax.jit(h), jax.jit(man.gravity_vector)
