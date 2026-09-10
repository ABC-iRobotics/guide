"""Declarative randomization specs — "what may vary".

Each ``Distribution`` is an immutable, global-state-free description of a random
quantity. It can ``sample`` itself from a passed generator, serialize itself to
a plain ``dict`` (``to_spec`` / ``from_spec``) for config round-tripping, and
``decode`` a previously realized value back to its native form (the injection
path). Distributions never touch a global RNG and never know about scenes or
tasks. Rotation algebra is delegated to SciPy (``scipy.spatial.transform``),
consistent with the rest of ``guide_core``.

Output conventions (kept NumPy-native; the record stores JSON lists):
- positions / vectors: ``np.ndarray`` shape ``(3,)``
- orientations: quaternion ``np.ndarray`` shape ``(4,)`` as ``[w, x, y, z]``
- poses: flat ``np.ndarray`` shape ``(7,)`` = ``[x, y, z, w, x, y, z]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import _quat

T = TypeVar("T")


@runtime_checkable
class Distribution(Protocol[T]):
    """Structural contract every distribution implements."""

    def sample(self, rng: np.random.Generator) -> T: ...

    def to_spec(self) -> dict[str, Any]: ...

    def decode(self, raw: Any) -> T: ...


# --------------------------------------------------------------------------- #
# Concrete distributions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Constant(Generic[T]):
    """A fixed value — draws nothing from the RNG."""

    value: T

    def sample(self, rng: np.random.Generator) -> T:
        return self.value

    def to_spec(self) -> dict[str, Any]:
        from .record import to_jsonable

        return {"type": "constant", "value": to_jsonable(self.value)}

    def decode(self, raw: Any) -> T:
        return raw


@dataclass(frozen=True, slots=True)
class UniformVec:
    """Component-wise uniform vector in ``[low, high]``."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = _quat.as_vec(self.low)
        high = _quat.as_vec(self.high)
        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")
        if np.any(low > high):
            raise ValueError("low must be <= high component-wise")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high)

    def to_spec(self) -> dict[str, Any]:
        return {"type": "uniform_vec", "low": self.low.tolist(), "high": self.high.tolist()}

    def decode(self, raw: Any) -> np.ndarray:
        return np.asarray(raw, dtype=float)


@dataclass(frozen=True, slots=True)
class AxisAngle:
    """Random rotation: ``base`` composed with a delta about ``axis``.

    The delta angle is uniform in ``[-max_angle, max_angle]`` (radians). Output
    is a ``[w, x, y, z]`` quaternion. Rotation math uses SciPy.
    """

    axis: np.ndarray
    max_angle: float
    base_quat: np.ndarray = None  # type: ignore[assignment]  # None -> identity

    def __post_init__(self) -> None:
        axis = _quat.as_vec(self.axis, 3)
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("axis must be non-zero")
        max_angle = float(self.max_angle)
        if max_angle < 0.0:
            raise ValueError("max_angle must be >= 0")
        if self.base_quat is None:
            base = _quat.IDENTITY_QUAT
        else:
            base = _quat.xyzw_to_wxyz(R.from_quat(_quat.wxyz_to_xyzw(self.base_quat)).as_quat())
        object.__setattr__(self, "axis", axis / norm)
        object.__setattr__(self, "max_angle", max_angle)
        object.__setattr__(self, "base_quat", base)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        angle = float(rng.uniform(-self.max_angle, self.max_angle))
        delta = R.from_rotvec(self.axis * angle)
        base = R.from_quat(_quat.wxyz_to_xyzw(self.base_quat))
        return _quat.xyzw_to_wxyz((base * delta).as_quat())

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "axis_angle",
            "axis": self.axis.tolist(),
            "max_angle": self.max_angle,
            "base_quat": self.base_quat.tolist(),
        }

    def decode(self, raw: Any) -> np.ndarray:
        return np.asarray(raw, dtype=float)


@dataclass(frozen=True, slots=True)
class Categorical(Generic[T]):
    """Uniform choice over a fixed tuple of options."""

    options: tuple

    def __post_init__(self) -> None:
        opts = tuple(self.options)
        if len(opts) == 0:
            raise ValueError("options must be non-empty")
        object.__setattr__(self, "options", opts)

    def sample(self, rng: np.random.Generator) -> T:
        return self.options[int(rng.integers(0, len(self.options)))]

    def to_spec(self) -> dict[str, Any]:
        from .record import to_jsonable

        return {"type": "categorical", "options": [to_jsonable(o) for o in self.options]}

    def decode(self, raw: Any) -> T:
        return raw


@dataclass(frozen=True, slots=True)
class PoseDist:
    """Composes a position distribution and an orientation distribution.

    Samples to a flat ``(7,)`` array ``[x, y, z, w, x, y, z]``. Position is
    drawn before orientation (fixed order -> deterministic).
    """

    position: Distribution
    orientation: Distribution

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        pos = _quat.as_vec(self.position.sample(rng), 3)
        quat = _quat.as_vec(self.orientation.sample(rng), 4)
        return np.concatenate([pos, quat])

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "pose",
            "position": self.position.to_spec(),
            "orientation": self.orientation.to_spec(),
        }

    def decode(self, raw: Any) -> np.ndarray:
        return np.asarray(raw, dtype=float)


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "constant": lambda s: Constant(s["value"]),
    "uniform_vec": lambda s: UniformVec(s["low"], s["high"]),
    "axis_angle": lambda s: AxisAngle(s["axis"], s["max_angle"], s.get("base_quat")),
    "categorical": lambda s: Categorical(tuple(s["options"])),
    "pose": lambda s: PoseDist(from_spec(s["position"]), from_spec(s["orientation"])),
}


def from_spec(spec: dict) -> Distribution:
    """Rebuild a distribution from its ``to_spec`` dict."""
    try:
        kind = spec["type"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"distribution spec missing 'type': {spec!r}") from exc
    try:
        factory = _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"unknown distribution type {kind!r}") from exc
    return factory(spec)


def pose_from_yaml(pose_spec: dict) -> PoseDist:
    """Build a ``PoseDist`` from the ``randomize.yaml`` pose schema.

    Mirrors the semantics previously inlined in
    ``SceneOrchestrator.parse_instruction``: a ``value`` base plus an optional
    ``random`` block. Position randomization is a uniform range given as a 3x2
    ``[[min, max], ...]`` matrix of offsets from ``value``; orientation is a base
    Euler (XYZ, degrees) composed
    with a uniform rotation about ``axis`` up to ``angle`` degrees.
    """
    pose_spec = pose_spec or {}

    pos_spec = pose_spec.get("position") or {}
    base_pos = _quat.as_vec(pos_spec.get("value", [0.0, 0.0, 0.0]), 3)
    pos_rand = pos_spec.get("random")
    if pos_rand is not None:
        # random: [[x_min, x_max], [y_min, y_max], [z_min, z_max]], offsets from value
        low, high = _quat.as_range(pos_rand)
        position: Distribution = UniformVec(base_pos + low, base_pos + high)
    else:
        position = Constant(base_pos)

    ori_spec = pose_spec.get("orientation") or {}
    base_euler_deg = _quat.as_vec(ori_spec.get("value", [0.0, 0.0, 0.0]), 3)
    base_quat = _quat.xyzw_to_wxyz(R.from_euler("xyz", base_euler_deg, degrees=True).as_quat())
    ori_rand = ori_spec.get("random")
    if ori_rand is not None:
        axis = ori_rand.get("axis", [0.0, 0.0, 1.0])
        max_angle = float(np.deg2rad(ori_rand.get("angle", 0.0)))
        orientation: Distribution = AxisAngle(axis, max_angle, base_quat)
    else:
        orientation = Constant(base_quat)

    return PoseDist(position, orientation)


# convention boundary kept in _quat.py; rotation math via SciPy.
