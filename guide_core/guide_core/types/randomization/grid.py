"""Grid/zone model over a position-randomization region.

Partitions a 2D (x, y) region into ``resolution``-sized square cells ("zones"),
numbered **row-major** and **0-indexed** from the ``(min-x, min-y)`` corner:

    col = zone % ncols   (advances with x)
    row = zone // ncols  (advances with y)

Pure NumPy -- no Isaac/ROS, so it is fully unit-testable. A zone selects a cell;
the within-cell pose is drawn on the seeded RNG, so ``(seed, zone) -> pose`` is
deterministic and injection replays the stored pose verbatim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _quat
from .distributions import PoseDist, UniformVec


@dataclass(frozen=True, slots=True)
class Grid:
    """A 2D grid of zones over a position region ``[low, high]``."""

    low: np.ndarray        # (3,) region min (x, y, z)
    high: np.ndarray       # (3,) region max
    resolution: float      # cell size in metres

    def __post_init__(self) -> None:
        low = _quat.as_vec(self.low, 3)
        high = _quat.as_vec(self.high, 3)
        if float(self.resolution) <= 0.0:
            raise ValueError("grid resolution must be > 0")
        if np.any(low > high):
            raise ValueError("grid low must be <= high component-wise")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "resolution", float(self.resolution))

    @property
    def ncols(self) -> int:
        return max(1, math.ceil((self.high[0] - self.low[0]) / self.resolution))

    @property
    def nrows(self) -> int:
        return max(1, math.ceil((self.high[1] - self.low[1]) / self.resolution))

    @property
    def num_zones(self) -> int:
        return self.ncols * self.nrows

    def _check(self, zone: int) -> int:
        z = int(zone)
        if z < 0 or z >= self.num_zones:
            raise ValueError(f"zone {z} out of range [0, {self.num_zones})")
        return z

    def cell_bounds(self, zone: int) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` of the cell for ``zone``, clamped to the region.

        x, y are restricted to the cell; z keeps the region's (usually fixed) range.
        """
        z = self._check(zone)
        r = self.resolution
        col = z % self.ncols
        row = z // self.ncols
        cl = self.low.copy()
        ch = self.high.copy()
        cl[0] = self.low[0] + col * r
        ch[0] = min(self.low[0] + (col + 1) * r, self.high[0])
        cl[1] = self.low[1] + row * r
        ch[1] = min(self.low[1] + (row + 1) * r, self.high[1])
        return cl, ch

    def zone_of(self, point) -> int:
        """Which zone an ``(x, y)`` point falls in (clamped to the grid)."""
        p = _quat.as_vec(point, 3)
        col = int(math.floor((p[0] - self.low[0]) / self.resolution))
        row = int(math.floor((p[1] - self.low[1]) / self.resolution))
        col = min(max(col, 0), self.ncols - 1)
        row = min(max(row, 0), self.nrows - 1)
        return row * self.ncols + col

    def restrict(self, pose_dist: PoseDist, zone: int) -> PoseDist:
        """A ``PoseDist`` whose position samples inside ``zone``'s cell.

        Keeps the caller (``apply.draw_instructions``) free of distribution internals.
        """
        cl, ch = self.cell_bounds(zone)
        return PoseDist(UniformVec(cl, ch), pose_dist.orientation)


def single_grid(instructions) -> Grid | None:
    """The scene's one grid, or ``None`` when no instruction carries one.

    At most **one** ``grid:``-enabled instruction per scene: with two, a scalar ``zone``
    is ambiguous (whose grid does ``zone=16`` address?) and the ``zones``/``counts``
    generation API has no way to say "cube in zone 2, bin in zone 5". See
    ``docs/design/zoned-randomization-todo.md`` for the deferred multi-grid design.
    """
    grids = [i["grid"] for i in instructions if i.get("grid") is not None]
    if len(grids) > 1:
        raise ValueError(
            f"At most one grid-enabled randomization instruction is supported "
            f"(found {len(grids)}); see docs/design/zoned-randomization-todo.md."
        )
    return grids[0] if grids else None


def zone_plan(zones, counts, num_zones: int) -> list:
    """Per-episode target zones for a demonstration request.

    Returns one entry per episode to record; ``None`` means a free (unstratified)
    draw. Three cases:

    * **Empty ``zones``** -> ``counts[0]`` free episodes, gridded scene or not.
    * **``-1`` in ``zones``** -> that entry's count in *every* zone, ascending. Note
      this multiplies: 5 counts over 20 zones is 100 episodes. On an ungridded scene
      (``num_zones <= 1``) it is the same as a free draw.
    * **Explicit ``zones``** -> ``counts[i]`` episodes in ``zones[i]``, falling back to
      ``counts[0]`` when ``counts`` is shorter than ``zones``.

    Zones are not range-checked here: an out-of-range zone fails at draw time when
    ``Grid.cell_bounds`` raises.
    """
    zones = [int(z) for z in zones]
    counts = [int(c) for c in counts]
    if not zones:
        return [None] * (counts[0] if counts else 0)
    plan = []
    for i, z in enumerate(zones):
        c = counts[i] if i < len(counts) else (counts[0] if counts else 0)
        if z == -1:
            cells = range(int(num_zones)) if int(num_zones) > 1 else [None]
            plan += [cell for cell in cells for _ in range(c)]
        else:
            plan += [z] * c
    return plan


def grid_from_yaml(position_spec: dict | None) -> Grid | None:
    """Build a ``Grid`` from a ``pose.position`` spec that carries ``random`` +
    ``grid.enabled``. Returns ``None`` when the position has no enabled grid.

    The region is computed the SAME way ``pose_from_yaml`` builds the position
    ``UniformVec`` (base ``value`` + the ``random`` matrix) so the grid tiles exactly
    the free range.
    """
    if not position_spec:
        return None
    grid_cfg = position_spec.get("grid")
    if not grid_cfg or not grid_cfg.get("enabled", False):
        return None
    rand = position_spec.get("random")
    if not rand:
        raise ValueError("a grid requires a position.random [[min, max], ...] range")
    base = _quat.as_vec(position_spec.get("value", [0.0, 0.0, 0.0]), 3)
    low, high = _quat.as_range(rand)
    return Grid(base + low, base + high, float(grid_cfg.get("resolution", 0.1)))
