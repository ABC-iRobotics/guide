"""Replicator YAML as a randomization front-end (spike).

A task's ``randomize.yaml`` written in Replicator YAML (no ``instructions:`` key) is parsed
by ``omni.replicator.replicator_yaml`` after the scene's USD is on the stage. Every key in
that file must resolve to a callable under ``omni.replicator.core``; this module is attached
as ``rep.guide`` so a task can name ``guide.grid`` and ``guide.axis_angle``.

Per episode the orchestrator seeds Replicator, points the zone randomizer at the scene's
zone target, fires the trigger event, steps one frame and reads the named distributions'
samples back into the ``RandomizationRecord``.

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
from guide_core.types.randomization.grid import Grid

EVENT = "guide_randomize"

_registered: dict[str, Any] = {}  # side channel from guide.* calls back to the scene


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def is_replicator_yaml(path: Path) -> bool:
    """A randomize file without an ``instructions`` key is the Replicator dialect."""
    doc = yaml.safe_load(path.read_text()) or {}
    return isinstance(doc, dict) and bool(doc) and "instructions" not in doc


def prefixed(doc: dict, scene_prefix: str) -> dict:
    """Return a copy with every ``path_pattern`` made scene-absolute."""

    def walk(node):
        if isinstance(node, dict):
            return {
                k: (scene_prefix + v if k == "path_pattern" and isinstance(v, str) else walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(doc)


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
def grid(distribution: str, resolution: float = 0.1):
    """Declare the zone grid over a *named* uniform position distribution."""
    import omni.graph.core as og
    from omni.replicator.core.scripts.named_nodes import NamedNodes

    node = NamedNodes._named_nodes[distribution]
    low = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:lower")).get(), 3)
    high = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:upper")).get(), 3)
    g = Grid(low, high, float(resolution))
    _registered["grid"] = g
    _registered["region"] = distribution
    return node


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
    from omni.replicator.replicator_yaml import parse

    attach()
    _registered.clear()
    doc = prefixed(yaml.safe_load(yaml_path.read_text()), scene_prefix)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f)
        tmp = f.name
    parse(yaml_path=tmp, root_dir=str(yaml_path.parent))

    state = dict(_registered)
    if "grid" in state:
        # One extra randomizer for the zone target: its prim pattern and bounds are rewritten
        # every episode; it runs after the group under the same trigger so the target wins.
        with rep.trigger.on_custom_event(event_name=EVENT):
            prims = rep.get.prims(path_pattern="__none__", cache_result=False)
            with prims:
                position = rep.distribution.uniform(lower=[0.0, 0.0, 0.0], upper=[0.0, 0.0, 0.0])
                rep.modify.pose(position=position)
        state["zone_prims"] = prims.node
        state["zone_position"] = position.node
    rep.orchestrator.run()
    return state


def _named(name: str):
    from omni.replicator.core.scripts.named_nodes import NamedNodes

    return NamedNodes._named_nodes[name]


def _set(node, attr: str, value) -> None:
    import omni.graph.core as og

    og.AttributeValueHelper(node.get_attribute(attr)).set(value, update_usd=True)


def _get(node, attr: str):
    import omni.graph.core as og

    return og.AttributeValueHelper(node.get_attribute(attr)).get()


def draw(state: dict[str, Any], seed: int, zone: int | None, zone_target: str | None) -> dict:
    """Seed, zone, fire the trigger, step one frame; return the named samples."""
    import omni.replicator.core as rep
    from omni.replicator.core.utils import rng

    rng.set_global_seed(int(seed))

    g: Grid | None = state.get("grid")
    if g is not None:
        if zone is not None and zone >= 0 and zone_target:
            low, high = g.cell_bounds(int(zone))
            _set(state["zone_prims"], "inputs:pathPattern", re.escape(zone_target))
            _set(state["zone_position"], "inputs:lower", low.tolist())
            _set(state["zone_position"], "inputs:upper", high.tolist())
        else:
            _set(state["zone_prims"], "inputs:pathPattern", "__none__")

    rep.utils.send_og_event(EVENT)
    rep.orchestrator.step(rt_subframes=1, pause_timeline=False, wait_for_render=False)

    from omni.replicator.core.scripts.named_nodes import NamedNodes

    samples = {}
    for name, node in NamedNodes._named_nodes.items():
        try:
            samples[name] = np.asarray(_get(node, "outputs:samples")).tolist()
        except Exception:  # a named node without samples (get.prims)
            pass
    return samples
