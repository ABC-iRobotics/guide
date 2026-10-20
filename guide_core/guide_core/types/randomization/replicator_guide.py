"""Replicator YAML as a randomization front-end (spike).

A task's ``randomize.yaml`` written in Replicator YAML (no ``instructions:`` key) is parsed
by ``omni.replicator.replicator_yaml`` after the scene's USD is on the stage. Every key in
that file must resolve to a callable under ``omni.replicator.core``; this module is attached
as ``rep.guide`` so a task can name ``guide.zone`` and ``guide.axis_angle``.

Per episode the orchestrator seeds Replicator, points the zone randomizer at the scene's
zone target, fires the trigger event, runs two app updates and reads the named
distributions' samples back into the ``RandomizationRecord``.

Nothing here imports Isaac at module level, so the pure parts are testable.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from guide_core.types.randomization import _quat
from guide_core.types.randomization.grid import Grid, grid_from_yaml


_registered: dict[str, Any] = {}  # side channel from guide.* calls back to the scene
UNIFORM = "omni.replicator.core.OgnSampleUniform"  # the only node type an arrangement can pin


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def is_replicator_yaml(path: Path) -> bool:
    """An instruction file without an ``instructions`` key is the Replicator dialect."""
    doc = yaml.safe_load(path.read_text()) or {}
    return isinstance(doc, dict) and bool(doc) and "instructions" not in doc


def prefixed(doc: dict, scene_prefix: str) -> dict:
    """Return a copy with every ``path_pattern`` made scene-absolute and every
    ``event_name`` made scene-unique, so several scenes can share one task file."""
    suffix = "_" + scene_prefix.strip("/")

    def walk(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "path_pattern" and isinstance(v, str):
                    out[k] = scene_prefix + v
                elif k == "event_name" and isinstance(v, str):
                    out[k] = v + suffix
                else:
                    out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(doc)


def _find(node, key: str):
    """Every value stored under ``key`` anywhere in a parsed YAML document."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _find(v, key)


def zone_grid(path: Path) -> Grid | None:
    """The zone grid a ``randomize.yaml`` declares, in either dialect; ``None`` if none.

    Read from the file, not from a scene, so a solver process can count the zones a
    ``zones: [-1]`` request expands to. Instruction dialect: the position spec with
    ``grid.enabled``. Replicator dialect: ``guide.zone`` over its named uniform position.
    """
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if "instructions" in doc:
        for instruction in doc["instructions"] or []:
            pose = (instruction.get("kwargs") or {}).get("pose") or {}
            grid = grid_from_yaml(pose.get("position"))
            if grid is not None:
                return grid
        return None
    spec = next(_find(doc, "guide.zone"), None)
    if spec is None:
        return None
    uniform = {d.get("name"): d for d in _find(doc, "distribution.uniform") if isinstance(d, dict)}
    region = uniform[spec["distribution"]]
    return Grid(region["lower"], region["upper"], float(spec.get("resolution", 0.1)))


def principal_axis(axis) -> int:
    """Index of the principal axis ``axis`` lies on, or ``ValueError``."""
    a = _quat.as_vec(axis, 3)
    a = a / np.linalg.norm(a)
    idx = int(np.argmax(np.abs(a)))
    if not np.isclose(abs(a[idx]), 1.0):
        raise ValueError(f"axis {a.tolist()} is not a principal axis; that needs a custom node")
    return idx


# --------------------------------------------------------------------------- #
# the ``rep.guide`` namespace (what a YAML file may name)
# --------------------------------------------------------------------------- #
def zone(distribution: str, path_pattern: str, resolution: float = 0.1):
    """Declare the zone grid over a *named* uniform position distribution and re-draw the
    scene's zone target inside its cell. Written inside the trigger block, after the group
    randomizer, so it evaluates after it; per episode ``fire`` narrows ``path_pattern`` to the
    target and the bounds to the cell (or the whole region for a free draw)."""
    import omni.graph.core as og
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.named_nodes import NamedNodes

    node = NamedNodes._named_nodes[distribution]
    low = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:lower")).get(), 3)
    high = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:upper")).get(), 3)
    g = Grid(low, high, float(resolution))
    prims = rep.get.prims(path_pattern=path_pattern, cache_result=False)
    with prims:
        position = rep.distribution.uniform(lower=g.low.tolist(), upper=g.high.tolist())
        rep.modify.pose(position=position, write_to_usd=True)
    _registered.update(grid=g, zone_prims=prims.node, zone_position=position.node, zone_distribution=distribution)
    return prims


def axis_angle(axis, angle):
    """Uniform rotation about a principal axis, ``angle`` = [min, max] degrees, as Euler."""
    import omni.replicator.core as rep

    lo, hi = (float(x) for x in _quat.as_vec(angle, 2))
    if lo > hi:
        raise ValueError("angle must be [min, max]")
    lower, upper = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    idx = principal_axis(axis)
    lower[idx], upper[idx] = lo, hi
    return rep.distribution.uniform(lower=lower, upper=upper)


def attach() -> None:
    import omni.replicator.core as rep
    import sys

    rep.guide = sys.modules[__name__]


# --------------------------------------------------------------------------- #
# graph build + per-episode driving (Isaac side)
# --------------------------------------------------------------------------- #
def build(yaml_path: Path, scene_prefix: str) -> dict[str, Any]:
    """Parse the task's Replicator YAML for one scene; returns what the scene keeps."""
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.named_nodes import NamedNodes
    from omni.replicator.replicator_yaml import parse

    attach()
    _registered.clear()
    _stop_orchestrator()  # a running orchestrator must not have its graph edited
    doc = prefixed(yaml.safe_load(yaml_path.read_text()), scene_prefix)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f, sort_keys=False)  # order is evaluation order
        tmp = f.name
    before = dict(NamedNodes._named_nodes)
    parse(yaml_path=tmp, root_dir=str(yaml_path.parent))

    state = dict(_registered)
    state["event"] = _event_name(doc)
    # this scene's named distributions (the registry is global; names repeat across scenes)
    state["named"] = {k: v for k, v in NamedNodes._named_nodes.items() if before.get(k) is not v}
    return state


def _event_name(doc: dict) -> str:
    """The (already scene-suffixed) event name of the file's custom-event trigger."""
    for group in doc.values():
        if isinstance(group, dict) and "trigger.on_custom_event" in group:
            return group["trigger.on_custom_event"]["event_name"]
    raise ValueError("randomize.yaml needs a trigger.on_custom_event group")


def _stop_orchestrator() -> None:
    import omni.kit.app
    import omni.replicator.core as rep

    if not rep.orchestrator.get_is_started():
        return
    rep.orchestrator.stop()
    app = omni.kit.app.get_app()
    for _ in range(60):  # stop() is asynchronous; let it settle
        app.update()
        if not rep.orchestrator.get_is_started():
            break


def _set(node, attr: str, value) -> None:
    import omni.graph.core as og

    og.AttributeValueHelper(node.get_attribute(attr)).set(value, update_usd=True)


def _get(node, attr: str):
    import omni.graph.core as og

    return og.AttributeValueHelper(node.get_attribute(attr)).get()


def _pin(named: dict, values: dict) -> list:
    """Pin each named uniform distribution an arrangement lists: lower = upper = value. Every name,
    node type and value length is checked before any node is touched. Returns what _unpin restores."""
    unknown = sorted(set(values) - set(named))
    if unknown:
        raise ValueError(f"arrangement: {unknown} name no distribution of this recipe (it has {sorted(named)})")
    plan = []
    for name, value in values.items():
        node = named[name]
        if node.get_type_name() != UNIFORM:
            raise ValueError(f"arrangement: {name} is a {node.get_type_name()}; only distribution.uniform can be pinned")
        lower = np.asarray(_get(node, "inputs:lower"), float).tolist()
        upper = np.asarray(_get(node, "inputs:upper"), float).tolist()
        pinned = np.atleast_1d(np.asarray(value, float)).tolist()
        if len(pinned) != len(lower):
            raise ValueError(f"arrangement: {name} takes {len(lower)} value(s), got {value!r}")
        plan.append((node, pinned, lower, upper))
    for node, pinned, _, _ in plan:
        _set(node, "inputs:lower", pinned)
        _set(node, "inputs:upper", pinned)
    return [(node, lower, upper) for node, _, lower, upper in plan]


def _unpin(pins: list) -> None:
    for node, lower, upper in pins:
        _set(node, "inputs:lower", lower)
        _set(node, "inputs:upper", upper)


def zone_pin(state: dict, values: dict, zone_target) -> list | None:
    """The zone redraw's pinned position under an arrangement: the arrangement's value of the zone's
    distribution (it must name it, or the redraw would move the target), or None (no arrangement, no zone)."""
    if not values or state.get("grid") is None or not zone_target:
        return None
    name = state.get("zone_distribution")
    if name not in values:
        raise ValueError(f"arrangement: this scene re-draws its zone target from {name}; the arrangement must pin it")
    return values[name]


def zone_bounds(g: Grid, zone: int | None, pinned) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) of the zone redraw: the pinned position, the zone's cell, or the whole region."""
    if pinned is not None:
        p = np.asarray(pinned, float)
        return p, p
    if zone is not None and zone >= 0:
        return g.cell_bounds(int(zone))
    return g.low, g.high


def _reseed(rng, seed: int, current: int) -> None:
    """Restart every sampler from ``seed``. Replicator's set_global_seed() ignores a seed equal to
    the current setting (an unset one reads 0), so seed 0 on a fresh app or a repeated seed would
    leave the samplers mid-stream: step through another value so the change subscription fires."""
    if seed == current:
        rng.set_global_seed(seed ^ 1)
    rng.set_global_seed(seed)


def fire(
    state: dict[str, Any],
    seed: int | None = None,
    zone: int | None = None,
    zone_target: str | None = None,
    values: dict | None = None,
) -> dict:
    """Seed (if given), pin an arrangement's ``values`` (if given), aim the zone randomizer, fire the
    file's trigger, let it run; return the named samples. Used for randomize (seeded, zoned,
    arranged) and reset (none). ``values`` = {named distribution: value}, pinned to lower = upper =
    value for this fire only (restored after, on error too); a bad name, node or value raises before
    anything is sent."""
    import time

    import carb.settings
    import omni.kit.app
    import omni.replicator.core as rep
    from omni.replicator.core.utils import rng

    values = values or {}
    pinned_zone = zone_pin(state, values, zone_target)
    pins = _pin(state["named"], values)
    try:
        t0 = time.perf_counter()
        if seed is not None:
            # A global-seed *change* resets every sampler from (seed, node id) -- through a
            # settings subscription that runs on the next app update, so pump one before firing.
            current = carb.settings.get_settings().get_as_int("/omni/replicator/globalSeed")
            _reseed(rng, int(seed) % (2**31 - 1), current)  # the graph's seed slot is 32-bit
            omni.kit.app.get_app().update()
        t1 = time.perf_counter()

        g: Grid | None = state.get("grid")
        if g is not None and zone_target:
            low, high = zone_bounds(g, zone, pinned_zone)
            _set(state["zone_prims"], "inputs:pathPattern", re.escape(zone_target) + "$")
            _set(state["zone_position"], "inputs:lower", np.asarray(low, float).tolist())
            _set(state["zone_position"], "inputs:upper", np.asarray(high, float).tolist())

        # Started once and kept running: a stopped orchestrator re-initialises on every step()
        # and evaluates every trigger, which fires the other scenes too and burns RNG state.
        if not rep.orchestrator.get_is_started():
            rep.orchestrator.run()
        rep.utils.send_og_event(state["event"])
        # With the orchestrator running, the graph evaluates on every app update. The event is
        # queued and consumed on the *next* evaluation: one update delivers it, the second runs
        # the randomizers it triggered (verified with guide_core/scripts/spike_diag.py, removed
        # from the tree; read it at commit 6f8adb5).
        # orchestrator.step() would do the same but stalls ~6 s every other call.
        for _ in range(2):
            omni.kit.app.get_app().update()
        t2 = time.perf_counter()
        print(f"[replicator_guide] {state['event']}: seed {t1 - t0:.3f}s  event+2 updates {t2 - t1:.3f}s", flush=True)

        samples = {}
        for name, node in state["named"].items():
            try:
                samples[name] = np.asarray(_get(node, "outputs:samples")).tolist()
            except Exception:  # a named node without samples
                pass
        return samples
    finally:
        _unpin(pins)
