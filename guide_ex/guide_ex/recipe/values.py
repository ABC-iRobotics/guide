"""Typed values in static_args: a mapping of `position` and/or `orientation` becomes a Pose."""

from scipy.spatial.transform import Rotation as R

from guide_core.types.geometry import Point, Pose, Rotation

POSE_KEYS = {"position", "orientation"}


def _rotation(spec) -> R:
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"orientation is one of {{euler: [axes, angles...]}}, {{quat: [x, y, z, w]}}, "
                         f"{{rotvec: [x, y, z]}}; got {spec!r}")
    [(kind, value)] = spec.items()
    if kind == "euler":
        if not (isinstance(value, list) and value and isinstance(value[0], str) and len(value[0]) == len(value) - 1):
            raise ValueError(
                f"euler is [axes, one angle per axis] in radians, e.g. ['xyz', 3.14, 0, 0]; got {value!r}")
        axes, angles = value[0], value[1:]
        return R.from_euler(axes, angles if len(angles) > 1 else angles[0])
    if kind == "quat":
        return R.from_quat(value)
    if kind == "rotvec":
        return R.from_rotvec(value)
    raise ValueError(f"orientation is euler, quat or rotvec; got {kind!r}")


def to_pose(spec: dict) -> Pose:
    position = spec.get("position", [0.0, 0.0, 0.0])
    if "orientation" not in spec:
        return Pose(position=Point(position))
    return Pose(position=Point(position), orientation=Rotation(_rotation(spec["orientation"])))


def convert(value):
    """`value` with every pose mapping in it, at any depth, made a Pose."""
    if isinstance(value, dict):
        if value and set(value) <= POSE_KEYS:
            return to_pose(value)
        return {key: convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert(item) for item in value]
    return value
