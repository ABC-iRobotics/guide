"""A robot for guide_motion: URDF + XRDF -> kinematics, dynamics and collision spheres."""

import json
import tempfile
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from guide_motion import description


class Model:
    """Arm kinematics (Pinocchio, fingers locked) and dynamics (OSCBF, built lazily)."""

    def __init__(self, urdf_text: str, xrdf: dict, usd_to_urdf: dict | None = None):
        self.urdf_text, self.xrdf = urdf_text, xrdf
        self._description_urdf = urdf_text  # set_mass_properties always starts from it, never accumulates
        self.usd_to_urdf = {name: np.asarray(X, float) for name, X in (usd_to_urdf or {}).items()}
        self.arm_joints = list(xrdf["cspace"]["joint_names"])
        self.gripper_joint = self.arm_joints[0].rsplit("_joint", 1)[0] + "_finger_joint1"
        self.tool_frame = xrdf["tool_frames"][0]
        root = ET.fromstring(urdf_text)
        children = {j.find("child").get("link") for j in root.iter("joint")}
        self.base_link = next(link.get("name") for link in root.iter("link") if link.get("name") not in children)

        full = pin.buildModelFromXML(urdf_text)
        locked = [full.getJointId(n) for n in full.names[1:] if n not in self.arm_joints]
        self.pin_model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.pin_data = self.pin_model.createData()
        assert list(self.pin_model.names[1:]) == self.arm_joints, list(self.pin_model.names[1:])
        self.lower = self.pin_model.lowerPositionLimit.copy()
        self.upper = self.pin_model.upperPositionLimit.copy()
        self.max_effort = self.pin_model.effortLimit.copy()
        self.max_velocity = self.pin_model.velocityLimit.copy()
        self.home = np.array([xrdf["default_joint_positions"][j] for j in self.arm_joints], float)
        assert self.pin_model.existFrame(self.tool_frame), self.tool_frame
        self._tool = self.pin_model.getFrameId(self.tool_frame)
        assert self.pin_model.frames[self._tool].parentJoint == len(self.arm_joints), "TCP must be rigid on the last joint"
        self.ee_offset = self.pin_model.frames[self._tool].placement.homogeneous

        # Self-collision spheres, each in its parent joint's frame; group 0 = the fixed base.
        self.sphere_centers, self.sphere_radii, self.sphere_groups, self.sphere_links = np.zeros((0, 3)), np.zeros(0), np.zeros(0, int), []
        for link, sphere in self._spheres(xrdf["self_collision"]["geometry"]):
            self.add_spheres(link, [sphere["center"]], [sphere["radius"]])
        ignore = xrdf["self_collision"].get("ignore", {})
        self.ignore = {frozenset((a, b)) for a, others in ignore.items() for b in others}

    def add_spheres(self, link: str, centers, radii) -> None:
        """Collision spheres on `link` (centres in its URDF frame); a controller built after sees them."""
        assert self.pin_model.existFrame(link), link
        frame = self.pin_model.frames[self.pin_model.getFrameId(link)]
        centers = np.c_[np.reshape(centers, (-1, 3)), np.ones(len(radii))]
        self.sphere_centers = np.r_[self.sphere_centers, (centers @ frame.placement.homogeneous.T)[:, :3]]
        self.sphere_radii = np.r_[self.sphere_radii, np.asarray(radii, float)]
        self.sphere_groups = np.r_[self.sphere_groups, np.full(len(radii), int(frame.parentJoint))]
        self.sphere_links = self.sphere_links + [link] * len(radii)

    @classmethod
    def load(cls, robot) -> "Model":
        """`robot`: a name under guide_motion/robots ('fr3', 'panda') or a directory."""
        d = Path(robot)
        if str(robot) in description.ROBOTS or not d.is_dir():  # a known name never means the cwd's dir
            d = description.ROBOTS_DIR / str(robot)
        meta = json.loads((d / "description.json").read_text()) if (d / "description.json").exists() else {}
        model = cls((d / f"{d.name}.urdf").read_text(), yaml.safe_load((d / f"{d.name}.xrdf").read_text()), meta.get("usd_to_urdf"))
        model.robot_dir = d
        return model

    def _spheres(self, name: str) -> list:
        group = self.xrdf["geometry"][name]
        cloned = [s for other in group.get("clone", []) for s in self._spheres(other)]
        return cloned + [(link, s) for link, spheres in group.get("spheres", {}).items() for s in spheres]

    @cached_property
    def manipulator(self):
        """OSCBF's JAX model (fingers fixed at 0). Lazy: only the control process imports JAX."""
        from oscbf.core.manipulator import Manipulator

        root = ET.fromstring(self.urdf_text)
        for joint in root.iter("joint"):
            if joint.get("type") != "fixed" and joint.get("name") not in self.arm_joints:
                joint.set("type", "fixed")
                for mimic in joint.findall("mimic"):
                    joint.remove(mimic)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
            f.write(ET.tostring(root, encoding="unicode"))
            f.flush()
            return Manipulator.from_urdf(f.name, ee_offset=self.ee_offset)

    def ee_pose(self, q):
        """(position (3,), rotation (3, 3)) of the tool frame in the base frame."""
        pin.framesForwardKinematics(self.pin_model, self.pin_data, np.asarray(q, float))
        T = self.pin_data.oMf[self._tool]
        return T.translation.copy(), T.rotation.copy()

    def tcp_twist(self, q, qd) -> np.ndarray:
        """J(q) qd at the tool frame, (6,): linear velocity of the TCP point (m/s), then the angular
        velocity (rad/s), both in the base frame (world-aligned axes)."""
        J = pin.computeFrameJacobian(self.pin_model, self.pin_data, np.asarray(q, float), self._tool, pin.LOCAL_WORLD_ALIGNED)
        return J @ np.asarray(qd, float)

    def ee_speed(self, q, qd) -> float:
        """Linear tool speed, m/s."""
        return float(np.linalg.norm(self.tcp_twist(q, qd)[:3]))

    def set_mass_properties(self, masses: dict, coms: dict, inertias: dict, extra_links: dict | None = None) -> bool:
        """Use the simulator's mass properties (COM and inertia in USD body frames, keyed by link).

        extra_links {name: (parent, T_parent_child)}: simulator links the URDF lacks (a camera
        mount, say), added as fixed links so OSCBF folds them into their parent.
        Returns False, and keeps the compiled dynamics, when nothing changed.
        """
        inertials = description.to_urdf_frames(masses, coms, inertias, self.usd_to_urdf)
        text = description.set_inertials(self._description_urdf, inertials, extra_links)
        if text == self.urdf_text:
            return False
        self.urdf_text = text
        self.__dict__.pop("manipulator", None)
        return True
