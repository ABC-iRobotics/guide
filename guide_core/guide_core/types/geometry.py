from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import Pose as RosPose
from geometry_msgs.msg import Quaternion as RosQuaternion
from scipy.spatial.transform import Rotation as R

from guide_core.types import conversion

# * Geometry classes wrapping common representations used in Isaac Sim and ROS


def _vec3(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError("Vector has non-finite values.")
    return a


# -----------------------------
# Point / Vector3
# -----------------------------


@dataclass(slots=True)
class Point:
    coordinates: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    random_low: np.ndarray | None = None
    random_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", _vec3(self.coordinates))

        if self.random_low is not None:
            object.__setattr__(self, "random_low", _vec3(self.random_low))

        if self.random_high is not None:
            object.__setattr__(self, "random_high", _vec3(self.random_high))

        if (self.random_low is None) != (self.random_high is None):
            raise ValueError("random_low and random_high must be both set or both None")

        if self.random_low is not None:
            if np.any(self.random_low > self.random_high):
                raise ValueError("random_low must be <= random_high")

    @property
    def is_randomized(self) -> bool:
        return self.random_low is not None

    def _sample_delta(self) -> np.ndarray:
        if not self.is_randomized:
            return np.zeros(3, dtype=float)

        return np.random.uniform(self.random_low, self.random_high)

    @staticmethod
    def zero() -> Point:
        return Point()

    @staticmethod
    def from_numpy(
        arr: np.ndarray, random_low: np.ndarray | None = None, random_high: np.ndarray | None = None
    ) -> Point:
        return Point(coordinates=_vec3(arr), random_low=random_low, random_high=random_high)

    @staticmethod
    def from_ros(p: conversion.RosPoint) -> Point:
        return Point.from_numpy(conversion.ros_to_vec3(p))

    def to_numpy(self) -> np.ndarray:
        return self.coordinates + self._sample_delta()

    def to_ros(self) -> conversion.RosPoint:
        return conversion.vec3_to_ros_point(self.coordinates + self._sample_delta())

    def __add__(self, other: Vector3 | np.ndarray) -> Point:
        if isinstance(other, Vector3):
            return Point.from_numpy(
                self.coordinates + other.v,
                self.random_low + other.random_low,
                self.random_high + other.random_high,
            )
        return Point.from_numpy(
            self.coordinates + _vec3(other),
            self.random_low + other.random_low,
            self.random_high + other.random_high,
        )

    def __sub__(self, other: Point | Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Point):
            return Vector3.from_numpy(
                self.coordinates - other.coordinates,
                self.random_low - other.random_high,
                self.random_high - other.random_low,
            )
        if isinstance(other, Vector3):
            return Vector3.from_numpy(
                self.coordinates - other.v,
                self.random_low - other.random_high,
                self.random_high - other.random_low,
            )
        return Vector3.from_numpy(self.coordinates - _vec3(other))

    def __neg__(self) -> Point:
        return Point.from_numpy(
            -self.coordinates,
            -self.random_high if self.random_high is not None else None,
            -self.random_low if self.random_low is not None else None,
        )


@dataclass(slots=True)
class Vector3:
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    random_low: np.ndarray | None = None
    random_high: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "v", _vec3(self.v))

        if self.random_low is not None:
            object.__setattr__(self, "random_low", _vec3(self.random_low))

        if self.random_high is not None:
            object.__setattr__(self, "random_high", _vec3(self.random_high))

        if (self.random_low is None) != (self.random_high is None):
            raise ValueError("random_low and random_high must be both set or both None")

        if self.random_low is not None:
            if np.any(self.random_low > self.random_high):
                raise ValueError("random_low must be <= random_high")

    @property
    def is_randomized(self) -> bool:
        return self.random_low is not None

    def _sample_delta(self) -> np.ndarray:
        if not self.is_randomized:
            return np.zeros(3, dtype=float)

        return np.random.uniform(self.random_low, self.random_high)

    @staticmethod
    def zero() -> Vector3:
        return Vector3()

    @staticmethod
    def from_numpy(
        arr: np.ndarray, random_low: np.ndarray | None = None, random_high: np.ndarray | None = None
    ) -> Vector3:
        return Vector3(v=_vec3(arr), random_low=random_low, random_high=random_high)

    @staticmethod
    def from_ros(v: conversion.RosVector3) -> Vector3:
        return Vector3.from_numpy(conversion.ros_to_vec3(v))

    def to_numpy(self) -> np.ndarray:
        return self.v + self._sample_delta()

    def to_ros(self) -> conversion.RosVector3:
        return conversion.vec3_to_ros_vec(self.v + self._sample_delta())

    def __add__(self, other: Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3.from_numpy(
                self.v + other.v,
                self.random_low + other.random_low,
                self.random_high + other.random_high,
            )
        return Vector3.from_numpy(self.v + _vec3(other), self.random_low, self.random_high)

    def __sub__(self, other: Vector3 | np.ndarray) -> Vector3:
        if isinstance(other, Vector3):
            return Vector3.from_numpy(
                self.v - other.v,
                self.random_low - other.random_high,
                self.random_high - other.random_low,
            )
        return Vector3.from_numpy(self.v - _vec3(other), self.random_low, self.random_high)

    def __neg__(self) -> Vector3:
        return Vector3.from_numpy(
            -self.v,
            -self.random_high if self.random_high is not None else None,
            -self.random_low if self.random_low is not None else None,
        )


# -----------------------------
# Rotation
# -----------------------------
@dataclass(slots=True)
class Rotation:
    rot: R = field(default_factory=R.identity)
    scale_factor: float = 1.0
    random_axis: np.ndarray | None = None
    random_max_angle: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_factor", float(self.scale_factor))

        if self.random_axis is not None:
            axis = _vec3(self.random_axis)
            norm = np.linalg.norm(axis)
            if norm == 0.0:
                raise ValueError("random_axis must be non-zero")
            object.__setattr__(self, "random_axis", axis / norm)

        if (self.random_axis is None) != (self.random_max_angle is None):
            raise ValueError("random_axis and random_max_angle must be both set or both None")

        if self.random_max_angle is not None:
            object.__setattr__(self, "random_max_angle", float(self.random_max_angle))
            if self.random_max_angle < 0.0:
                raise ValueError("random_max_angle must be >= 0")

    @property
    def is_randomized(self) -> bool:
        return self.random_axis is not None

    def _sample_angle(self) -> float:
        if not self.is_randomized:
            return 0.0
        return float(np.random.uniform(-self.random_max_angle, self.random_max_angle))

    def _sample_rot(self) -> R:
        if not self.is_randomized:
            return self.rot

        angle = self._sample_angle()
        delta_rot = R.from_rotvec(self.random_axis * angle)
        return self.rot * delta_rot

    @staticmethod
    def identity() -> Rotation:
        return Rotation()

    @staticmethod
    def from_scipy(
        rot: R,
        scale_factor: float = 1.0,
        random_axis: np.ndarray | None = None,
        random_max_angle: float | None = None,
    ) -> Rotation:
        return Rotation(
            rot=rot,
            scale_factor=float(scale_factor),
            random_axis=random_axis,
            random_max_angle=random_max_angle,
        )

    @staticmethod
    def from_ros_quat(q: conversion.RosQuaternion) -> Rotation:
        return Rotation.from_scipy(conversion.ros_quat_to_scipy_rot(q))

    def to_scipy(self) -> R:
        return self._sample_rot()

    def to_numpy(self) -> np.ndarray:
        return self._sample_rot().as_matrix()

    def to_numpy_quat(self) -> np.ndarray:
        rot = self._sample_rot().as_quat()
        return np.array([rot[3], rot[0], rot[1], rot[2]])  # xyzw -> wxyz

    def to_ros_quat(self) -> RosQuaternion:
        return conversion.scipy_rot_to_ros_quat(self._sample_rot())

    def inv(self) -> Rotation:
        return Rotation(
            rot=self.rot.inv(),
            scale_factor=self.scale_factor,
            random_axis=self.random_axis,
            random_max_angle=self.random_max_angle,
        )

    def __mul__(self, other: Rotation) -> Rotation:
        return Rotation(
            rot=self.rot * other.rot,
            scale_factor=self.scale_factor * other.scale_factor,
            random_axis=self.random_axis,
            random_max_angle=self.random_max_angle,
        )

    def __invert__(self) -> Rotation:
        return self.inv()

    def apply(self, x: Point | Vector3 | np.ndarray) -> np.ndarray:
        if isinstance(x, Point):
            v = x.coordinates
        elif isinstance(x, Vector3):
            v = x.v
        else:
            v = _vec3(x)
        return self._sample_rot().apply(v)


# -----------------------------
# Pose
# -----------------------------
@dataclass(slots=True)
class Pose:
    position: Point = field(default_factory=Point.zero)
    orientation: Rotation = field(default_factory=Rotation.identity)

    @staticmethod
    def identity() -> Pose:
        return Pose()

    @staticmethod
    def from_numpy(position: np.ndarray, orientation: R) -> Pose:
        return Pose(
            position=Point.from_numpy(position),
            orientation=Rotation.from_scipy(orientation),
        )

    @staticmethod
    def from_ros_pose(p: conversion.RosPose) -> Pose:
        pos = conversion.ros_to_vec3(p.position)
        rot = conversion.ros_quat_to_scipy_rot(p.orientation)
        return Pose.from_numpy(pos, rot)

    @staticmethod
    def from_ros_transform(t: conversion.RosTransform) -> Pose:
        pos = conversion.ros_to_vec3(t.translation)
        rot = conversion.ros_quat_to_scipy_rot(t.rotation)
        return Pose.from_numpy(pos, rot)

    def to_ros(self) -> RosPose:
        return RosPose(
            position=self.position.to_ros(),
            orientation=self.orientation.to_ros_quat(),
        )

    def to_numpy(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.orientation.to_numpy()
        T[:3, 3] = self.position.to_numpy()
        return T

    def toDict(self) -> dict:
        pos = self.position.to_numpy()
        rot = self.orientation.to_scipy().as_euler("xyz")
        return {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "roll": rot[0],
            "pitch": rot[1],
            "yaw": rot[2],
        }

    def to_Transform(self) -> conversion.RosTransform:
        return conversion.RosTransform(
            translation=self.position.to_ros(),
            rotation=self.orientation.to_ros_quat(),
        )

    def inv(self) -> Pose:
        inv_rot = self.orientation.inv()
        inv_pos = -inv_rot.apply(self.position)
        return Pose(position=Point(inv_pos), orientation=inv_rot)

    def __invert__(self) -> Pose:
        return self.inv()

    def __mul__(self, other: Pose | Transform) -> Pose:
        # t = t1 + R1 * t2
        if isinstance(other, Pose):
            pos = self.position.to_numpy() + self.orientation.apply(other.position)
            rot = self.orientation * other.orientation
        else:
            pos = self.position.to_numpy() + self.orientation.apply(other.translation)
            rot = self.orientation * other.rotation
        return Pose(position=Point.from_numpy(pos), orientation=rot)

    def __rmul__(self, other: Pose | Transform) -> Pose:
        if isinstance(other, Pose):
            pos = other.position + other.orientation.apply(self.position)
            rot = other.orientation * self.orientation
        else:
            pos = other.translation + other.rotation.apply(self.position)
            rot = other.rotation * self.orientation
        return Pose(position=Point.from_numpy(pos), orientation=rot)


# -----------------------------
# Transform
# -----------------------------
@dataclass(slots=True)
class Transform:
    translation: Vector3 = field(default_factory=Vector3.zero)
    rotation: Rotation = field(default_factory=Rotation.identity)

    @staticmethod
    def identity() -> Transform:
        return Transform()

    @staticmethod
    def from_ros(t: conversion.RosTransform) -> Transform:
        tr = conversion.gf_to_vec3(conversion.ros_vec_to_gf(t.translation))
        rot = conversion.ros_quat_to_scipy_rot(t.rotation)
        return Transform(translation=Vector3.from_numpy(tr), rotation=Rotation.from_scipy(rot))

    @staticmethod
    def from_numpy(translation: np.ndarray, rotation: R) -> Transform:
        return Transform(
            translation=Vector3.from_numpy(translation),
            rotation=Rotation.from_scipy(rotation),
        )

    def to_ros(self) -> conversion.RosTransform:
        return conversion.RosTransform(
            translation=self.translation.to_ros(),
            rotation=self.rotation.to_ros_quat(),
        )

    def to_numpy(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.rotation.to_numpy()
        T[:3, 3] = self.translation.to_numpy()
        return T

    def to_pose(self) -> Pose:
        return Pose(
            position=Point.from_numpy(self.translation.to_numpy()),
            orientation=self.rotation,
        )

    def inv(self) -> Transform:
        inv_rot = self.rotation.inv()
        inv_trans = -inv_rot.apply(self.translation)
        return Transform(translation=Vector3.from_numpy(inv_trans), rotation=inv_rot)

    def __invert__(self) -> Transform:
        return self.inv()

    def __mul__(self, other: Transform | Pose) -> Transform:
        # t = t1 + R1 * t2
        if isinstance(other, Transform):
            pos = self.translation.to_numpy() + self.rotation.apply(other.translation)
            rot = self.rotation * other.rotation
        else:
            pos = self.translation.to_numpy() + self.rotation.apply(other.position)
            rot = self.rotation * other.orientation
        return Transform(translation=Vector3.from_numpy(pos), rotation=rot)

    def __rmul__(self, other: Transform | Pose) -> Transform:
        if isinstance(other, Transform):
            pos = other.translation.to_numpy() + other.rotation.apply(self.translation)
            rot = other.rotation * self.rotation
        else:
            pos = other.position.to_numpy() + other.orientation.apply(self.translation)
            rot = other.orientation * self.rotation
        return Transform(translation=Vector3.from_numpy(pos), rotation=rot)
