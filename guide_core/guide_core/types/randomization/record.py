"""The realized values of one randomization pass — a pure value object.

``RandomizationRecord`` is the unit of value-level reproducibility: it knows
the seed that drove the draws and the realized value of every named draw, and
nothing about scenes, episodes, or tasks (that identity lives in the
orchestration layer). It is JSON-serializable for the dataset sidecar and for
injection (``_cmd_randomize_scene``'s ``params``; the ``Randomize`` service does not
expose it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy types to plain JSON-serializable Python."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


@dataclass(slots=True)
class RandomizationRecord:
    """Seed + the realized value of every named draw (all JSON-ready)."""

    seed: int
    values: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "values": to_jsonable(self.values),
            "schema_version": int(self.schema_version),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RandomizationRecord":
        return cls(
            seed=int(data["seed"]),
            values=dict(data.get("values", {})),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, payload: str) -> "RandomizationRecord":
        return cls.from_dict(json.loads(payload))
