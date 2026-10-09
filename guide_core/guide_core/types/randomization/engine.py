"""The single sampling authority.

``Randomizer`` is the one chokepoint for randomness in the framework. Every
draw goes through :meth:`Randomizer.draw`, which does exactly one of two things
— sample from the passed generator, or (when an injection record supplies the
name) decode the previously realized value — and **always** records the result.
Geometry never samples; tasks never call ``random``/``np.random`` directly.
The exception is a Replicator-dialect file: Replicator samples it itself, seeded
from the same seed, and only its samples are recorded (see ``replicator_guide``).
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from .distributions import Distribution
from .record import RandomizationRecord, to_jsonable

T = TypeVar("T")


class Randomizer:
    """Sample-xor-inject, and capture, every named draw."""

    __slots__ = ("_rng", "_record", "_inject")

    def __init__(
        self,
        rng: np.random.Generator,
        record: RandomizationRecord,
        *,
        inject: RandomizationRecord | None = None,
    ) -> None:
        self._rng = rng
        self._record = record
        self._inject = inject

    @property
    def record(self) -> RandomizationRecord:
        """The record being filled (realized values for this pass)."""
        return self._record

    def draw(self, name: str, dist: Distribution[T]) -> T:
        """Resolve ``name`` from ``dist``: inject if available, else sample.

        The realized value is always stored under ``name`` in the record.
        Re-using a name within one pass is a bug and raises.
        """
        if name in self._record.values:
            raise KeyError(f"duplicate randomization draw name: {name!r}")

        if self._inject is not None and name in self._inject.values:
            value = dist.decode(self._inject.values[name])
        else:
            value = dist.sample(self._rng)

        self._record.values[name] = to_jsonable(value)
        return value
