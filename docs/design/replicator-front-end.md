# Replicator YAML as the randomization front-end — spike findings

Branch `spike/replicator-yaml` (throwaway). Isaac Sim 6.0.1, `omni.replicator.core` 1.13.27,
`omni.replicator.replicator_yaml` 2.0.12. Headless measurements via
`guide_core/scripts/spike_replicator.py` (results: `spike_replicator.json`,
`spike_replicator_2scenes.json`, `spike_legacy.json`); the window demo via
`spike_show.py`; the trigger-latency probe via `spike_diag.py`. Measured 2026-09-14.

## What was built

- `randomize.yaml` in Replicator YAML: native `distribution.uniform` for position (absolute
  bounds, local to the prim's parent) and Euler rotation, both *named*; prim groups declared
  inside their registered randomizers; `trigger.on_custom_event`; one GUIDE key,
  `guide.zone`, written inside the trigger block after `randomizer.place_blocks`.
- `guide_core/types/randomization/replicator_guide.py`: attaches `rep.guide` (`zone`,
  `axis_angle`); prefixes `path_pattern`s with `/Scene_<id>` and suffixes `event_name`s with the
  scene; parses after the scene's USD is on the stage (stopping a running orchestrator first);
  per episode re-seeds Replicator, narrows the zone randomizer to the scene's target and its
  cell, fires the scene's event and reads the named samples into the `RandomizationRecord`.
- `SceneOrchestrator`: a `randomize.yaml` without `instructions:` takes this path; `reset.yaml`
  and `success.yaml` are untouched.

## Results (headless, block_bin, 20 free episodes + seeded + zoned)

| Measurement | Replicator, 1 scene | Replicator, 2 scenes | Executor, 1 scene |
|---|---|---|---|
| Same seed twice → same PhysX poses and same record | yes | yes / yes | yes |
| Four blocks → four independent positions | yes | yes / yes | yes |
| Zone target inside its cell (zones 0, 7, 19) | 3/3 | 3/3 and 3/3 | 3/3 |
| Poses inside the region, in the `/blocks` frame, when the call returns | yes | yes / yes | yes |
| Block scale unchanged (0.0515) | yes | yes / yes | yes |
| Randomizing one scene leaves the other alone | — | yes | — |
| `Randomize` call, mean / p95 | 198 / 204 ms | 337 / 346 ms | 10.5 / 10.7 ms |
| RTF idle (app updating freely) | 0.256 | — | 0.250 |
| RTF with one randomization per sim second | 0.239 | — | 0.247 |

A Replicator randomization costs three app frames — one so the seed reset lands, one to
deliver the event, one to run the randomizers — which is where its ~200 ms per call comes from
(the executor advances no frames). At one call per sim second that is a 3 % RTF cost; per
episode it is noise.

**Reaction speed:** the custom event is consumed on the *next* graph evaluation, and the
randomizers run on the one after — the new layout is in PhysX two frames after the event
(33 ms of sim time at 60 Hz; 130 ms wall with one scene, 220 ms with two).

## What the spike settled — including the wrong turns

- `modify.pose` writes to **Fabric** by default; PhysX never sees it. `write_to_usd: true` is
  required for rigid bodies to teleport (the first demo showed unmoved blocks while the
  read-back said "moved" — it was reading Fabric).
- Do **not** use `relative_to`: it switches `modify.pose` from `UPDATE_*` tokens (update the
  local transform in place — parent-relative, scale-preserving, exactly `set_local_poses`) to
  composing from identity, which drops the prims' scale (the giant cubes).
- `get.prims(path_pattern=…)` is a regex *search*: `/bin_0` also matched the bin's `Visuals`
  and shader prims. Anchor with `$`; use `[^/]+$` for direct children.
- `/Scene/blocks` carries a `rotateZYX (0,0,−90)`: the yaml region is in the `/blocks` frame,
  as the executor's local poses always were. A world-minus-origin read-back is wrong.
- Custom events are consumed one evaluation late; a randomizer written on a *second* trigger
  node under the same event has no ordering guarantee against the first, so the zone re-draw
  must be declared in the YAML's own trigger block after the group.
- The orchestrator must be **started once and kept running**: a stopped orchestrator
  re-initialises on every `step()` and evaluates every trigger (all scenes fire, RNG state
  burns). Its graph must not be edited while running — stop it around a later scene's parse.
  `orchestrator.step()` stalled ~6 s every other call; plain app updates do not.
- A global-seed change resets every sampler from `(seed, node id)` through a settings
  subscription that lands on the next app update; pump one before firing the event. The seed
  slot is 32-bit.
- `yaml.safe_dump` sorts keys by default; evaluation order is file order, so `sort_keys=False`.
- Several scenes share one task file: events get a scene suffix, and each scene keeps the
  handles of its own named nodes (the registry is global and names repeat).
- The ROS 2 camera topics (`/Sim_0/Scene_N/cam_{top,base,wrist}`, `camera_info`,
  `franka/joint_states`, `tf`) publish for every scene during the runs.

## Gaps

- Injecting recorded values to reproduce a layout has no native path (custom node).
- `guide.axis_angle` covers principal axes only.
- Samples are recorded per named node in prim-match order without the prim paths.
- `scene_num_zones()` in `block_bin/solve_task.py` still reads the instruction dialect.
- Pre-existing, both dialects: `/blocks/*` also matches `/blocks/properties`; a second scene
  logs "Failed to add robot /Scene_1/fr3 … name is not unique"; `add_scene`'s filesystem branch
  expects the flat `dummy_scene` layout; a headless `SimulationApp.close()` sometimes leaves
  the interpreter alive.

## Recommendation

Adopt Replicator YAML for the *randomize* file, native-first as built here, keeping the
instruction executor for reset and success. Before it leaves the spike: carry prim paths in
the record, teach `scene_num_zones()` to ask the scene, and decide whether value injection is
still required.
