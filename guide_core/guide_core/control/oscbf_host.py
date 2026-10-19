"""guide_motion's OSCBF controller inside Isaac's physics step, for one robot (S2b spec 4.3).

Before each physics substep (SimulationManager PHYSICS_PRE_STEP, main thread): read q and
qd, take the latest target from a single-slot holder that GUIDE's ROS thread writes, step
the controller, write joint efforts (torque) or velocity targets (velocity). Nothing goes
over ROS per substep; states return over the existing per-frame /joint_states.
"""

import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import get_current_stage
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from pxr import Sdf, UsdPhysics
from scipy.spatial.transform import Rotation

from guide_motion import description
from guide_motion.control import Controller, enable_compilation_cache
from guide_motion.model import Model
from guide_motion.planner import DT as TARGET_DT
from guide_motion.planner import plan
from guide_motion.targets import JOINTS, Target, TargetHolder

DYNAMICS_TOL = 1e-3  # controller M and g vs PhysX, relative (S2b spec 4.2.5)
HOME_SPEED = 1.0  # Reset's homing at GUIDE-EX speed 1.0: the planner caps each joint below its URDF velocity limit
RATE = {"torque": 120.0, "velocity": 60.0}  # Hz without an oscbf.rate: every substep / once per frame (spec Q4)


def apply_mass_overrides(stage, articulation_root: str, robot: str) -> list:
    """Copy robots/<robot>/<robot>.usda's mass, COM and inertia onto the scene's robot (before play).

    Controller ruling 2026-10-20: a scene that embeds the bare asset (block_bin.usd: the FR3) then
    simulates the masses the description was built on. Only the layer is read; its reference to
    Isaac's asset is never resolved. Returns the override links the scene lacks (nothing written then).
    """
    layer = Sdf.Layer.FindOrOpen(str(description.ROBOTS_DIR / robot / f"{robot}.usda"))
    overs = list(layer.GetPrimAtPath(f"/{layer.defaultPrim}").nameChildren)  # ponytail: direct children, as both layers are
    prims = {over.name: stage.GetPrimAtPath(f"{articulation_root}/{over.name}") for over in overs}
    missing = [name for name, prim in prims.items() if not prim.IsValid()]
    if missing:
        return missing
    for over in overs:
        prim = prims[over.name]
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        for attribute in over.attributes:  # a value already there stays unwritten: nothing for PhysX to re-parse
            if prim.GetAttribute(attribute.name).Get() != attribute.default:
                prim.CreateAttribute(attribute.name, attribute.typeName, custom=attribute.custom).Set(attribute.default)
    return []


class OscbfHost:
    def __init__(self, articulation_root: str, robot: str = "fr3", mode: str = "torque", gains=None,
                 rate=None, barriers=None, q_rest=None, logger=None):
        self.holder = TargetHolder(stale_after=0.25)
        self.mode, self.rate, self._logger = mode, rate or RATE[mode], logger
        self.diagnostics = {"stale": True, "infeasible": 0, "refused": None}
        self.model = Model.load(robot)
        layer = description.ROBOTS_DIR / robot / f"{robot}.usda"
        if "mass_override" not in description.ROBOTS.get(robot, {}):
            pass  # no override recipe: the description was built on the asset's own masses
        elif not layer.is_file():
            self._refuse(f"mass overrides: {layer} does not exist")
        elif missing := apply_mass_overrides(get_current_stage(), articulation_root, robot):
            # else the step would recompile to the asset's masses at attach and pass its M/g check
            self._refuse(f"mass overrides: {articulation_root} has no {missing}")
        self._settings = (mode, gains, barriers, q_rest)
        enable_compilation_cache()
        self._controller = Controller(self.model, *self._settings)
        self._log("info", f"{robot} {mode}: OSCBF step compiled in {self._controller.warm_up():.1f} s")
        self._arm = Articulation(articulation_root)
        self._attached, self._switched, self._t, self._k = False, False, 0.0, 0
        self._u, self._hold, self._width, self._home, self._homing = None, None, None, None, None
        self._ids = [
            SimulationManager.register_callback(self._on_step, SimulationEvent.PHYSICS_PRE_STEP),
            SimulationManager.register_callback(self._on_stop, SimulationEvent.SIMULATION_STOPPED),
        ]

    def home(self) -> None:
        """Ruling 2: GUIDE's Reset under controller: oscbf. An effort- or velocity-driven arm ignores
        drive targets, so drive it home: Plan A's joint plan from the q at the next controller step,
        one target per 50 ms of the physics clock, then the home configuration held. Exempt from
        staleness until the next fresh target arrives (it replaces this one in the holder). A step
        target instead ran joints past their URDF velocity limits (controller, B3b)."""
        self._home = Target(mode=JOINTS, joints=self.model.home.copy())
        self.holder.put(self._home)

    def _log(self, level, message):
        if self._logger is not None:
            getattr(self._logger, level)(f"[oscbf] {message}")

    def _refuse(self, reason: str) -> bool:
        """Spec 8: not registered for OSCBF control, with the failing check named; the robot stays position-driven."""
        self.diagnostics["refused"] = reason
        self._log("error", f"robot not registered for OSCBF control, it stays position-driven: {reason}")
        return False

    def _on_stop(self, event) -> None:
        self._attached = self._switched = False  # PhysX rebuilds the articulation (gains, modes) on the next play
        # The physics clock restarts on play: a target from before the stop would turn fresh again
        # once the new clock passed the time it was put at. Targets put from now on go to the new slot.
        # ponytail: a put racing this swap lands in the old slot and is lost; the next 20 Hz target replaces it
        latest, _ = self.holder.get(self._t)
        self.holder = TargetHolder(stale_after=0.25)
        if latest is not None and latest is self._home:
            self.holder.put(latest)  # Ruling 2: Reset's home hold never goes stale, a stop included

    def _on_step(self, dt, context) -> None:
        if self.diagnostics["refused"]:
            return
        try:
            if self._attached or self._attach(dt):
                self._control(dt)
        except Exception as error:  # in the physics callback: never leave an effort-driven arm limp, nor retry
            self._refuse(f"{type(error).__name__}: {error}")
            self._release()

    def _release(self) -> None:
        """After an error: the arm back on its position drives at its current pose, if they were switched."""
        if not self._switched:
            return
        try:
            q = self._arm.get_dof_positions().numpy()[0][self._arm_dofs]
            self._arm.switch_dof_control_mode("position", dof_indices=self._arm_dofs)
            self._arm.set_dof_position_targets(np.asarray(q)[None], dof_indices=self._arm_dofs)
        except Exception as error:
            self._log("error", f"could not put the arm back on its position drives: {type(error).__name__}: {error}")

    def _control(self, dt) -> None:
        self._t += dt
        self._k += 1
        # the first substep after attach steps the controller: there are no efforts to repeat yet
        if (self._k - 1) % self.every:  # velocity targets persist between controller steps; efforts do not
            if self.mode == "torque":
                self._write(self._u)
            return
        q, qd = self._read()
        target, stale = self.holder.get(self._t)
        if target is not None and target is self._home:
            stale = False  # Ruling 2: Reset's home target never goes stale
            if self._homing is None or self._homing[0] is not target:
                self._homing = (target, self._t, plan(self.model, target, q, speed=HOME_SPEED))
            _, start, path = self._homing
            target = path[min(int((self._t - start) / TARGET_DT + 1e-6), len(path) - 1)]
        if stale != self.diagnostics["stale"]:
            self._log("warning" if stale else "info", "no target for 0.25 s: holding the pose" if stale else "targets resumed")
        self.diagnostics["stale"] = stale
        if not stale:
            self._hold = None
        elif self._hold is None:
            position, rotation = self.model.ee_pose(q)
            self._hold = Target(position=position, rotation=rotation)
        # stale: the controller holds the latched pose in both modes. A zero velocity target cannot:
        # on velocity drives (stiffness 0) it is only a damper, and the arm sags (FIX B4)
        u = self._controller.step(self._hold or target, q, qd)
        if self.mode == "velocity":  # the drive pushes D (v_target - v): gravity needs v_target += tau_g / D (FIX B4-2)
            limit = self._controller.limit  # step's NaN guard and URDF clip, kept for the sum
            u = np.clip(np.nan_to_num(u + np.asarray(self._controller.gravity(q)) / self._damping), -limit, limit)
        self.diagnostics["infeasible"] = self._controller.events["infeasible"]
        self._u = u
        self._write(u)
        width = (self._hold or target).gripper_width
        if np.isfinite(width) and width != self._width:
            self._width = width  # one finger travels half the opening; clamped to its joint range
            finger = float(np.clip(width / 2.0, *self._finger_range))
            self._arm.set_dof_position_targets(np.array([[finger]]), dof_indices=[self._finger])

    def _attach(self, dt) -> bool:
        """First substep after play: mass properties from PhysX, the dynamics check, control modes."""
        arm = self._arm
        dofs = list(arm.dof_names)
        self._arm_dofs = [dofs.index(j) for j in self.model.arm_joints]
        self._finger = dofs.index(self.model.gripper_joint)
        lower, upper = arm.get_dof_limits()
        self._finger_range = (float(lower.numpy()[0][self._finger]), float(upper.numpy()[0][self._finger]))
        self.every = max(1, round(1.0 / (dt * self.rate)))
        view = getattr(arm, "_physics_articulation_view", None)  # private: 0.2 ms cheaper per call
        self._view = view if view is not None and hasattr(view, "set_dof_actuation_forces") else None
        names = list(arm.link_names)
        extra = [n for n in names if n not in self.model.usd_to_urdf]
        if extra and self._view is None:  # their parents and poses come from the tensor view only
            return self._refuse(f"PhysX links {extra} are not in the description, and the API I/O fallback cannot fold them")
        masses = arm.get_link_masses().numpy()[0]
        coms = arm.get_link_coms()[0].numpy()[0]
        inertias = arm.get_link_inertias().numpy()[0].reshape(-1, 3, 3)
        properties = [dict(zip(names, values)) for values in (masses, coms, inertias)]
        if self.model.set_mass_properties(*properties, extra_links=self._extra_links(names)):
            self._controller = Controller(self.model, *self._settings)
            self._log("warning", f"PhysX mass properties differ from the description: recompiled in {self._controller.warm_up():.1f} s")
        q, _ = self._read_api()
        M = arm.get_mass_matrices().numpy()[0][np.ix_(self._arm_dofs, self._arm_dofs)]
        g = arm.get_dof_gravity_compensation_forces().numpy()[0][self._arm_dofs]
        man = self.model.manipulator
        error_M = np.linalg.norm(np.asarray(man.mass_matrix(q)) - M) / np.linalg.norm(M)
        error_g = np.linalg.norm(np.asarray(man.gravity_vector(q)) - g) / np.linalg.norm(g)
        if max(error_M, error_g) > DYNAMICS_TOL:
            return self._refuse(f"dynamics check failed: M {error_M:.1e}, g {error_g:.1e} relative > {DYNAMICS_TOL}; PhysX links {names}")
        if np.min(self._controller.barrier_values(q)) <= 0.0:
            # A stop reverts the joints to the asset's authored state (the FR3's zeros break its j4 and j6
            # limits). Started outside its barriers the controller deadlocks against PhysX's joint limits,
            # so the arm goes where position drives would take it: q_rest (GUIDE's default_joint_states).
            rest = np.asarray(self._controller.q_rest, float)
            outside = [j for j, x, lo, hi in zip(self.model.arm_joints, q, self.model.lower, self.model.upper) if not lo <= x <= hi]
            where = f"q {np.round(q, 4).tolist()} is outside the controller's barriers (joints outside their limits: {outside})"
            if np.min(self._controller.barrier_values(rest)) <= 0.0:
                return self._refuse(f"{where}, and so is q_rest {rest.tolist()}")
            arm.set_dof_positions(rest[None], dof_indices=self._arm_dofs)
            arm.set_dof_velocities(np.zeros((1, len(rest))), dof_indices=self._arm_dofs)
            self._log("warning", f"{where}: set to q_rest")
        if self.mode == "velocity":  # read before the switch: velocity mode keeps the drives' damping
            # tau_g / D needs force drives: an acceleration drive scales its force by the joint inertia
            types = [arm.get_dof_drive_types()[0][i] for i in self._arm_dofs]
            self._damping = arm.get_dof_gains()[1].numpy()[0][self._arm_dofs].astype(np.float64)
            bad = [j for j, t, d in zip(self.model.arm_joints, types, self._damping) if t != "force" or not (np.isfinite(d) and d > 0)]
            if bad:
                return self._refuse(f"velocity drives must be force drives with damping, not on {bad}: types {types}, damping {self._damping.tolist()}")
        arm.switch_dof_control_mode("effort" if self.mode == "torque" else "velocity", dof_indices=self._arm_dofs)
        self._switched = True
        if self._view is not None:
            sample = self._view.get_dof_positions()  # its device: "cpu", or the GPU with GPU physics
            self._buffer, self._device = np.zeros(sample.shape, np.float32), sample.device
            self._index = wp.array([0], dtype=wp.int32, device=self._device)
        self._t, self._k, self._hold, self._width, self._homing = 0.0, 0, None, None, None  # the clock restarts
        self._attached = True
        damping = f", drive damping {np.round(self._damping, 3).tolist()}" if self.mode == "velocity" else ""
        self._log("info", f"attached: {self.mode}, controller every {self.every} substep(s), {'tensor view' if self._view else 'API'} I/O{damping}")
        return True

    def _extra_links(self, names):
        """PhysX links the description lacks (a camera mount, say) as {name: (parent, T_parent_urdf_child)}.

        Needs the tensor view: _attach refuses the robot when such links come with the API fallback.
        """
        extra = [n for n in names if n not in self.model.usd_to_urdf]
        if not extra:
            return None
        parents = dict(zip(self._view.shared_metatype.link_names, self._view.shared_metatype.link_parents))
        poses = self._view.get_link_transforms().numpy().reshape(-1, len(names), 7)[0]
        T = {}
        for name, (x, y, z, qx, qy, qz, qw) in zip(names, poses):
            T[name] = np.eye(4)
            T[name][:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[name][:3, 3] = [x, y, z]
        to_urdf = lambda name: self.model.usd_to_urdf.get(name, np.eye(4))  # noqa: E731
        return {n: (parents[n], to_urdf(parents[n]) @ np.linalg.inv(T[parents[n]]) @ T[n]) for n in extra}

    def _read(self):
        if self._view is None:
            return self._read_api()
        q = self._view.get_dof_positions().numpy()[0, self._arm_dofs]
        qd = self._view.get_dof_velocities().numpy()[0, self._arm_dofs]
        return q.astype(np.float64), qd.astype(np.float64)

    def _read_api(self):
        q = self._arm.get_dof_positions().numpy()[0][self._arm_dofs]
        qd = self._arm.get_dof_velocities().numpy()[0][self._arm_dofs]
        return q.astype(np.float64), qd.astype(np.float64)

    def _write(self, u) -> None:
        if self._view is None:
            setter = self._arm.set_dof_efforts if self.mode == "torque" else self._arm.set_dof_velocity_targets
            setter(np.asarray(u)[None], dof_indices=self._arm_dofs)
            return
        self._buffer[0, self._arm_dofs] = u
        data = wp.from_numpy(self._buffer, dtype=wp.float32, device=self._device)
        setter = self._view.set_dof_actuation_forces if self.mode == "torque" else self._view.set_dof_velocity_targets
        setter(data, self._index)

    def close(self) -> None:
        for uid in self._ids:
            SimulationManager.deregister_callback(uid)
        self._ids = []
