"""Replicator YAML as a randomization front-end (spike).

A task's ``randomize.yaml`` written in Replicator YAML (no ``instructions:`` key) is parsed
by ``omni.replicator.replicator_yaml`` after the scene's USD is on the stage. Every key in
that file must resolve to a callable under ``omni.replicator.core``; this module is attached
as ``rep.guide`` so a task can name ``guide.zone`` and ``guide.axis_angle``.

Per episode the orchestrator seeds Replicator, points the zone randomizer at the scene's
zone target, fires the trigger event, runs two app updates and reads the named
distributions' samples back into the ``RandomizationRecord``. ``guide.fixture_pose`` places a
fixture by its world weld and ``guide.joint_state`` sets a drawer or knob (spikes K and J); fire()
writes both after the trigger, the joints JOINT_DELAY_FRAMES physics frames after the poses.

Nothing here imports Isaac at module level, so the pure parts are testable.
"""
from __future__ import annotations

import math
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from guide_core.types.randomization import _quat
from guide_core.types.randomization.grid import Grid, grid_from_yaml


_registered: dict[str, Any] = {}  # side channel from guide.* calls back to the scene
UNIFORM = "omni.replicator.core.OgnSampleUniform"  # the only node type an arrangement can pin
JOINT_EVENT = "guide_randomize_joints"  # the trigger guide.joint_state goes under
JOINT_DELAY_FRAMES = 3  # physics frames between the pose event and the joint event (welds pull, spike K)
NO_MATCH = "^$"  # the path pattern of a switched-off zone redraw: no prim path is empty


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def is_replicator_yaml(path: Path) -> bool:
    """An instruction file without an ``instructions`` key is the Replicator dialect."""
    doc = yaml.safe_load(path.read_text()) or {}
    return isinstance(doc, dict) and bool(doc) and "instructions" not in doc


def prefixed(doc: dict, scene_prefix: str) -> dict:
    """Return a copy with every ``path_pattern`` and every absolute ``parent`` (scene.yaml's create.* calls)
    made scene-absolute and every ``event_name`` made scene-unique, so several scenes can share one task file."""
    suffix = "_" + scene_prefix.strip("/")

    def walk(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "path_pattern" and isinstance(v, str):
                    out[k] = scene_prefix + v
                elif k == "parent" and isinstance(v, str) and v.startswith("/"):
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
    """Declare the zone grid over a *named* uniform position distribution and re-draw zoned
    objects inside their cells. Written inside the trigger block, after the group randomizer,
    so it evaluates after it. One redraw per prim ``path_pattern`` matches on the stage now;
    per episode ``fire`` aims each at its object's cell (or the whole region for a free draw)
    and switches off the ones no zone names, so those keep the group draw."""
    import omni.graph.core as og
    import omni.replicator.core as rep
    import omni.usd
    from omni.replicator.core.scripts.named_nodes import NamedNodes

    node = NamedNodes._named_nodes[distribution]
    low = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:lower")).get(), 3)
    high = _quat.as_vec(og.AttributeValueHelper(node.get_attribute("inputs:upper")).get(), 3)
    g = Grid(low, high, float(resolution))
    pattern = re.compile(path_pattern)
    stage = omni.usd.get_context().get_stage()
    slots = {}
    for path in [str(p.GetPath()) for p in stage.Traverse() if pattern.search(str(p.GetPath()))]:
        prims = rep.get.prims(path_pattern=NO_MATCH, cache_result=False)
        with prims:
            position = rep.distribution.uniform(lower=g.low.tolist(), upper=g.high.tolist())
            rep.modify.pose(position=position, write_to_usd=True)
        slots[path] = (prims.node, position.node)
    _registered.update(grid=g, zone_slots=slots, zone_distribution=distribution)


def build(yaml_path: Path, scene_prefix: str) -> dict[str, Any]:
    """Parse the task's Replicator YAML for one scene; returns what the scene keeps."""
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.named_nodes import NamedNodes
    from omni.replicator.replicator_yaml import parse

    attach()
    _registered.clear()
    _registered["prefix"] = scene_prefix  # guide.fixture_pose reads the scene origin
    _stop_orchestrator()  # a running orchestrator must not have its graph edited
    doc = prefixed(yaml.safe_load(yaml_path.read_text()), scene_prefix)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f, sort_keys=False)  # order is evaluation order
        tmp = f.name
    before = dict(NamedNodes._named_nodes)
    parse(yaml_path=tmp, root_dir=str(yaml_path.parent))

    state = dict(_registered)
    state["event"], state["joint_event"] = events(doc)
    if state.get("joints") and not state["joint_event"]:
        raise ValueError(f"{yaml_path}: guide.joint_state needs a group with "
                         f"trigger.on_custom_event: {{event_name: {JOINT_EVENT}}}")
    # this scene's named distributions (the registry is global; names repeat across scenes)
    state["named"] = {k: v for k, v in NamedNodes._named_nodes.items() if before.get(k) is not v}
    return state


def check_scene_recipe(doc, where) -> None:
    """A scene recipe (config/scene.yaml) runs once, at registration, with no trigger: only create.*
    calls build anything then. Any other call (trigger.*, with.*, modify.*, randomizer.*, guide.*,
    distribution.*) raises, naming it, as does an instruction-list file."""
    if not isinstance(doc, dict) or "instructions" in doc:
        raise ValueError(f"{where}: a scene recipe is Replicator YAML (create.* groups), not an instruction list")

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    for key in keys(doc):
        if isinstance(key, str) and "." in key and not key.startswith("create."):
            raise ValueError(f"{where}: {key} -- a scene recipe runs once at registration and takes create.* "
                             f"calls only; draw per episode in randomize.yaml")


def check_scene_assets(doc: dict, config_dir: Path) -> None:
    """Every ``usd: {property: file_path, value: <path>}`` of a scene recipe must exist with all it depends on (sublayers,
    references, payloads, textures); a USD reference to a missing file otherwise loads as an empty prim and the scene
    looks fine until the render. Raises FileNotFoundError naming each missing file, and the models/ folder a git-ignored
    model (a LIBERO suite's guide_tasks/<suite>/models/README.md) is meant to be put in."""
    from pxr import UsdUtils

    missing = []
    for usd in _find(doc, "usd"):
        path = (config_dir / str(usd["value"])).resolve() if isinstance(usd, dict) and usd.get("property") == "file_path" else None
        if path is None:
            continue
        if not path.is_file():
            missing.append(str(path))
        else:
            # MDL materials are found by the renderer by module name or URL (OmniPBR.mdl, NVIDIA's Base materials), not by the file system
            missing += [u for u in UsdUtils.ComputeAllDependencies(str(path))[2] if not u.endswith(".mdl") and u not in missing]
    if missing:
        raise FileNotFoundError("scene recipe asset files are missing: " + ", ".join(missing) + ". An asset whose model is not "
                                "shipped (TurboSquid wine_rack/wine_bottle) is filled by `guide-bench assets --libero <LIBERO>`: "
                                "see the README.md in the suite's models/ folder (guide_tasks/libero/goal/models/)")


def run_scene_recipe(yaml_path: Path, scene_prefix: str) -> None:
    """Build config/scene.yaml for one scene. Its create.* calls reference fixtures and objects under
    the scene: ``parent: /objects`` -> /Scene_<i>/objects, defined as an Xform if missing (Replicator
    refuses a missing parent); ``usd: {property: file_path, value: <relative path>}`` resolves against
    config/ (Replicator YAML's own rule). create.from_usd(name: N, parent: P) lands at P/N_Xform/N."""
    import omni.usd
    from omni.replicator.replicator_yaml import parse
    from pxr import UsdGeom

    attach()
    _stop_orchestrator()  # create.* adds group nodes to the graph
    doc = prefixed(yaml.safe_load(Path(yaml_path).read_text()) or {}, scene_prefix)
    check_scene_recipe(doc, yaml_path)
    check_scene_assets(doc, Path(yaml_path).parent)
    stage = omni.usd.get_context().get_stage()
    for parent in {p for p in _find(doc, "parent") if isinstance(p, str)}:
        if not stage.GetPrimAtPath(parent).IsValid():
            UsdGeom.Xform.Define(stage, parent)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(doc, f, sort_keys=False)
        tmp = f.name
    parse(yaml_path=tmp, root_dir=str(Path(yaml_path).parent))


def events(doc: dict) -> tuple[str, str | None]:
    """The file's (already scene-suffixed) custom-event names: the pose event and, if the file has
    one, the guide_randomize_joints event."""
    names = {g["trigger.on_custom_event"]["event_name"] for g in doc.values()
             if isinstance(g, dict) and "trigger.on_custom_event" in g}  # reset.yaml repeats its one name
    joint = sorted(n for n in names if re.fullmatch(rf"{JOINT_EVENT}(_Scene_\d+)?", n))
    main = sorted(names - set(joint))
    for kind, found in (("pose", main), ("joint", joint)):
        if len(found) > 1:
            raise ValueError(f"randomize.yaml has several {kind} events: {', '.join(found)}")
    if not main:
        raise ValueError("randomize.yaml needs a trigger.on_custom_event group")
    return main[0], (joint[0] if joint else None)


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


def _pin(named: dict, values: dict, pins: list | None = None) -> list:
    """Pin each named uniform distribution an arrangement lists: lower = upper = value. Every name,
    node type and value length is checked before any node is touched. Returns ``pins`` (a new list unless
    given), what _unpin restores: each node is listed right before its own write, so a failure part
    way leaves every node already touched in it."""
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
    pins = [] if pins is None else pins
    for node, pinned, lower, upper in plan:
        pins.append((node, lower, upper))
        _set(node, "inputs:lower", pinned)
        _set(node, "inputs:upper", pinned)
    return pins


def _unpin(pins: list) -> None:
    """Restore every pinned node; one that fails does not stop the rest, the first error is re-raised."""
    first = None
    for node, lower, upper in pins:
        try:
            _set(node, "inputs:lower", lower)
            _set(node, "inputs:upper", upper)
        except Exception as error:
            first = first or error
    if first:
        raise first


@contextmanager
def pinning(named: dict, values: dict):
    """_pin for the duration of a with-block, restored on the way out (an error included)."""
    pins: list = []
    try:
        _pin(named, values, pins)
        yield
    finally:
        _unpin(pins)


def zone_pin(state: dict, values: dict, zoned) -> list | None:
    """The zone redraws' pinned position under an arrangement: the arrangement's value of the zone's
    distribution (it must name it, or the redraw would move the objects), or None (no arrangement, no zone)."""
    if not values or state.get("grid") is None or not zoned:
        return None
    name = state.get("zone_distribution")
    if name not in values:
        raise ValueError(f"arrangement: this scene re-draws its zone target from {name}; the arrangement must pin it")
    return values[name]


def zone_paths(state: dict, zones: dict) -> dict:
    """{object: zone} -> {prim path: zone}. An object is a zoned prim's path or its trailing part
    ('red_block', 'blocks/red_block'); it must name exactly one prim the zone redraws."""
    slots = state.get("zone_slots", {})
    out = {}
    for name, z in zones.items():
        hits = [p for p in slots if p == name or p.endswith("/" + name.strip("/"))]
        if len(hits) != 1:
            raise ValueError(f"zone object {name!r} names {len(hits)} of the zoned prims {list(slots)}")
        out[hits[0]] = z
    return out


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
    zones: dict | None = None,
) -> dict:
    """Seed (if given), pin an arrangement's ``values`` (if given), aim the zone randomizer, fire the
    file's trigger and let it run; then place the guide.fixture_pose fixtures by their welds and,
    JOINT_DELAY_FRAMES physics frames after the pose event's weld write, fire guide_randomize_joints
    and write the guide.joint_state joints. Returns the named samples. Used for randomize (seeded, zoned,
    arranged) and reset (none). ``values`` = {named distribution: value}, pinned to lower = upper =
    value for this fire only (restored after, on error too); a bad name, node or value raises before
    anything is sent. ``zones`` = {object: zone, -1 or None = anywhere} redraws each named object in its
    cell (see zone_paths); without it, ``zone_target`` alone is redrawn in ``zone``."""
    import time

    import carb.settings
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from omni.replicator.core.utils import rng

    from guide_core.types.randomization import fixtures

    app = omni.kit.app.get_app()
    values = values or {}
    targets = zone_paths(state, zones) if zones else ({zone_target: zone} if zone_target else {})
    pinned_zone = zone_pin(state, values, targets)
    with pinning(state["named"], values):
        t0 = time.perf_counter()
        if seed is not None:
            # A global-seed *change* resets every sampler from (seed, node id) -- through a
            # settings subscription that runs on the next app update, so pump one before firing.
            current = carb.settings.get_settings().get_as_int("/omni/replicator/globalSeed")
            _reseed(rng, int(seed) % (2**31 - 1), current)  # the graph's seed slot is 32-bit
            app.update()
        t1 = time.perf_counter()

        g: Grid | None = state.get("grid")
        if g is not None:
            for path, (prims, position) in state["zone_slots"].items():
                _set(prims, "inputs:pathPattern", re.escape(path) + "$" if path in targets else NO_MATCH)
                if path in targets:
                    low, high = zone_bounds(g, targets[path], pinned_zone)
                    _set(position, "inputs:lower", np.asarray(low, float).tolist())
                    _set(position, "inputs:upper", np.asarray(high, float).tolist())

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
            app.update()

        stage = omni.usd.get_context().get_stage()
        for fixture in state.get("fixtures", []):
            fixtures.write_weld(fixture["weld"], *fixtures.weld_world_pose(
                fixtures.scene_matrix(stage, state["prefix"]), drawn(fixture["position"]),
                drawn(fixture["rotation"]), *fixture["local1"]))
        if state.get("joint_event"):
            # The welds pull, they do not teleport (spike K): set the joints once the bases have landed.
            for _ in range(joint_delay_updates(*_dts())):
                app.update()
            rep.utils.send_og_event(state["joint_event"])
            for _ in range(2):
                app.update()
            for joint in state.get("joints", []):
                fixtures.write_joint_state(joint["joint"], joint["kind"], float(np.ravel(drawn(joint["value"]))[0]))
        t2 = time.perf_counter()
        print(f"[replicator_guide] {state['event']}: seed {t1 - t0:.3f}s  events+updates {t2 - t1:.3f}s", flush=True)

        samples = {}
        for name, node in state["named"].items():
            try:
                samples[name] = np.asarray(_get(node, "outputs:samples")).tolist()
            except Exception:  # a named node without samples
                pass
        return samples
