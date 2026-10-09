"""Apply distributions attached to scene instructions through a Randomizer.

Pure glue between the scene layer and the randomization engine. It walks a list
of instruction dicts and, for each one carrying a ``pose_dist`` (a
``Distribution``), draws it through the single ``Randomizer`` and replaces the
instruction's concrete ``kwargs['pose']`` using an injected ``pose_builder``.

Geometry is deliberately kept out of this module (the caller passes a
``pose_builder`` that turns a realized 7-vector ``[x, y, z, w, x, y, z]`` into a
geometry ``Pose``), so the module stays Isaac/ROS/SciPy-free and unit-testable.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from .engine import Randomizer


def is_prim_pattern(path: Any) -> bool:
    """True if ``path`` is an Isaac prim-path *pattern* (matches many prims).

    Isaac matches ``prim_paths_expr`` as a regex, so any regex metacharacter
    (``*``, ``[``, ``.`` ...) means the expression can resolve to more than one
    prim. ``re.escape`` changes exactly those characters, so a path that differs
    from its escaped form contains a special char and is treated as a pattern.
    """
    return isinstance(path, str) and re.escape(path) != path


def draw_name(instruction: dict) -> str:
    """Stable, human-readable draw key for an instruction.

    Uses the (already scene-prefixed) ``prim_path`` when present so the recorded
    key matches the target it positions; falls back to object identity.
    """
    kwargs = instruction.get("kwargs", {})
    name = kwargs.get("prim_path")
    if isinstance(name, list):
        return ",".join(str(n) for n in name)
    if name:
        return str(name)
    return f"pose@{id(instruction):x}"


def draw_instructions(
    instructions: Iterable[dict],
    randomizer: Randomizer,
    pose_builder: Callable[[Any], Any],
    prim_resolver: Callable[[str], list] | None = None,
    zone: int | None = None,
    zone_target: str | None = None,
) -> None:
    """Draw every instruction's ``pose_dist`` and write back a concrete pose.

    Mutates each instruction in place. Instructions without a ``pose_dist`` are
    left alone.

    Two cases:

    * **Single prim** (literal ``prim_path``): one draw for the instruction,
      recorded under :func:`draw_name`, ``kwargs['pose']`` set to
      ``pose_builder(value)``.
    * **Pattern prim_path** (e.g. ``/Scene_0/blocks/*``) with a ``prim_resolver``:
      applying one drawn pose to every matched prim stacks them all at the same
      spot. Instead, expand the pattern to its concrete prims and draw an
      **independent** pose per prim (recorded under each concrete path, so every
      object is placed — and reproduced via injection — individually).
      ``kwargs['prim_path']`` becomes the concrete list and ``kwargs['pose']`` a
      list of poses in the same order (which ``_cmd_set_local_poses`` applies
      element-wise).

    **Zoning:** when an instruction carries a ``grid`` (a ``Grid``) and ``zone`` is
    given (``>= 0``), the prim equal to ``zone_target`` samples inside that zone's
    cell instead of the full range; every other prim (and non-grid instruction) is
    drawn free. ``zone_target=None`` zones *all* prims of the grid instruction.
    """

    def _dist_for(instr: dict, prim: str | None):
        base = instr["pose_dist"]
        grid = instr.get("grid")
        if grid is not None and zone is not None and zone >= 0:
            if zone_target is None or prim == zone_target:
                return grid.restrict(base, int(zone))
        return base

    for instruction in instructions:
        if instruction.get("pose_dist") is None:
            continue
        kwargs = instruction.setdefault("kwargs", {})

        # Remember the original pattern the first time we expand it. Expanding
        # overwrites kwargs['prim_path'] with the concrete prim list, so on the
        # NEXT randomization it would no longer look like a pattern -- the draw
        # would collapse back to one shared pose and stack every prim into a line.
        # Resolving from the remembered pattern keeps every pass per-prim.
        pattern = instruction.get("_prim_pattern")
        if pattern is None and is_prim_pattern(kwargs.get("prim_path")):
            pattern = kwargs["prim_path"]
            instruction["_prim_pattern"] = pattern

        if prim_resolver is not None and pattern is not None:
            prims = list(prim_resolver(pattern))
            if prims:
                kwargs["prim_path"] = prims
                kwargs["pose"] = [
                    pose_builder(randomizer.draw(p, _dist_for(instruction, p))) for p in prims
                ]
                continue

        # Single literal path (or a pattern that matched nothing): one draw.
        name = draw_name(instruction)
        kwargs["pose"] = pose_builder(randomizer.draw(name, _dist_for(instruction, name)))
