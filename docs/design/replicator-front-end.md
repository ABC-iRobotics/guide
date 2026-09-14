# Replicator YAML as the randomization front-end — spike findings

Branch `spike/replicator-yaml` (throwaway). Measured on 2026-09-14 with Isaac Sim 6.0.1,
`omni.replicator.core` 1.13.27, `omni.replicator.replicator_yaml` 2.0.12, headless, one
scene, block_bin, 20 free episodes plus seeded and zoned ones. Script:
`guide_core/scripts/spike_replicator.py`; results `spike_replicator.json` / `spike_legacy.json`.

## What was built

- `randomize.yaml` in Replicator YAML: native `distribution.uniform` for position (absolute
  bounds) and Euler rotation, both *named*; prim groups inside the registered randomizers;
  `trigger.on_custom_event`; one GUIDE key, `guide.grid`, declared after the trigger block.
- `guide_core/types/randomization/replicator_guide.py`: attaches `rep.guide` (`grid`,
  `axis_angle`), prefixes `path_pattern`s with `/Scene_<id>`, parses after the USD is on the
  stage, builds one extra zone randomizer (a `get.prims` with `cache_result=False` plus a
  uniform node), and per episode seeds Replicator, rewrites that randomizer's pattern and bounds
  from `Grid.cell_bounds`, fires the event, steps one frame, reads the named samples into the
  `RandomizationRecord`.
- `SceneOrchestrator`: a `randomize.yaml` without `instructions:` takes this path; `reset.yaml`
  and `success.yaml` are untouched.

## Results

| Measurement | Replicator YAML | Instruction executor |
|---|---|---|
| Same seed twice → same poses and same record | yes | yes |
| Four blocks → four independent positions | yes | yes |
| `Randomize` call, mean / p95 | 72.7 ms / 82.7 ms | 12.3 ms / 14.7 ms |
| Zone target inside its cell, zones 0 / 7 / 19 | yes / yes / yes | yes / yes / yes |
| Poses applied when the call returns | yes | yes |
| Record contents | `color`, `side`, `replicator/blocks_position`, `replicator/blocks_rotation` (per-prim sample lists) | `color`, `side`, one 7-vector per prim path |
| Scene registration | 4.8 s | 4.5 s |

The 60 ms difference is one orchestrator step (an app frame) per episode; against an episode
of many seconds it is noise.

## What the spike settled

- **Custom keys work as a `rep.guide` namespace.** The parser resolves anything reachable from
  `omni.replicator.core` by attribute, so a module attached as `rep.guide` is a first-class
  key; `guide.grid` read the named node's bounds and built the existing `Grid`.
- **Zones need no custom OmniGraph node.** Rewriting `inputs:pathPattern` and
  `inputs:lower/upper` between triggers works, provided the prim node is created with
  `cache_result=False`.
- **Determinism** under `rep.utils.rng.set_global_seed` per episode holds for the same seed,
  including the record read back from the named nodes.
- **Recording** the drawn values is native: `name:` on a distribution, then
  `outputs:samples` after the step.
- **Timing**: `rep.orchestrator.step()` from inside the runtime's command handler is safe (same
  thread as the loop, no re-entrant callback).

## What it did not settle / gaps

- **Injection** (reproducing a stored layout by feeding recorded values back) has no Replicator
  path; it would need a node that emits given values instead of sampling.
- **Arbitrary axis-angle rotation** is not expressible with native nodes; `guide.axis_angle`
  covers principal axes only.
- **Sample order**: `outputs:samples` is one list per named node in the order the prim group
  matched; the record does not carry the prim paths next to the values. A `zip` with the
  group's resolved prims is needed before it is as self-describing as today's record.
- **`scene_num_zones()`** in `block_bin/solve_task.py` still reads the grid from the
  instruction dialect; with this file it returns 1, so `zones: [-1]` collapses to one free
  episode. The scene's `_grid` is correct; the solver-side helper would need to ask the scene.
- Pre-existing, seen in both dialects: `/blocks/*` (and `/blocks/.*`) also matches the
  `/blocks/properties` Xform, which is randomized like a block.
- Not measured: multi-scene registration (one graph per scene), and behaviour with cameras
  publishing, which adds render load to the orchestrator step.

## Recommendation

Adopt Replicator YAML for the *randomize* file as an additive dialect, native-first as built
here, keeping the instruction executor for reset and success. Before it leaves the spike:
carry prim paths in the record, teach `scene_num_zones()` to read the scene's grid, and decide
whether value injection is still required (it is the one feature that needs a custom node).
