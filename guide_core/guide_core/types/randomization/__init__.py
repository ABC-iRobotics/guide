"""Reproducible scene randomization.

Layered so each concern has a single owner:

- ``distributions`` — declarative specs ("what may vary")
- ``seeding``       — ``SeedTree``, the only RNG source
- ``engine``        — ``Randomizer``, the only sampler (sample xor inject + record)
- ``record``        — ``RandomizationRecord``, the realized values (pure, JSON)
- ``apply``         — ``draw_instructions``, glue that draws specs attached to
                      scene instructions (geometry injected, so still pure)
- ``grid``          — ``Grid``, a 2D zone partition over a position region, plus the
                      ``single_grid`` scene validation and ``zone_plan`` request expansion
- ``replicator_guide`` — the Replicator-YAML front-end (not re-exported here)

Apart from ``replicator_guide`` (PyYAML, and Isaac Replicator imported inside its
Isaac-side functions), the package depends only on NumPy + SciPy (rotation math);
it is free of Isaac/ROS so it can be unit-tested in isolation.
"""

from __future__ import annotations

from .apply import draw_instructions, draw_name
from .distributions import (
    AxisAngle,
    Categorical,
    Constant,
    Distribution,
    PoseDist,
    UniformVec,
    from_spec,
    pose_from_yaml,
)
from .engine import Randomizer
from .grid import Grid, grid_from_yaml, single_grid, zone_plan
from .record import RandomizationRecord, to_jsonable
from .seeding import SeedTree

__all__ = [
    "Distribution",
    "Constant",
    "UniformVec",
    "AxisAngle",
    "Categorical",
    "PoseDist",
    "from_spec",
    "pose_from_yaml",
    "SeedTree",
    "RandomizationRecord",
    "to_jsonable",
    "Randomizer",
    "draw_instructions",
    "draw_name",
    "Grid",
    "grid_from_yaml",
    "single_grid",
    "zone_plan",
]
