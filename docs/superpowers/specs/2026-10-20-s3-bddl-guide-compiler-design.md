# S3 — BDDL → GUIDE task compiler (design)

**Roadmap:** `docs/superpowers/plans/2026-10-19-libero-isaac-roadmap.md`, step S3.
**Facts:** `~/libero_assets/s3_inputs/facts.md` (BDDL grammar over all 130 files, demo HDF5 layout,
init files, frames, what a GUIDE task needs, S1 reuse). Section numbers below cite it as F§n.
**Status:** draft for review, 2026-10-20.

## 1. Goal

Compile a LIBERO task (its `.bddl`, plus its demonstration HDF5 when there is one) into a GUIDE
task that GUIDE can register and run with no hand-written code:

- **H1:** start every episode from the exact start state of one LIBERO demonstration, so the FR3
  can replay that demonstration (the replay itself is S5).
- **H2:** start from new layouts drawn from the BDDL regions, so GUIDE-EX can generate new
  demonstrations from the goal (the solver behaviour is S5, the predicates S4).

Both run from the same compiled task. Slice 1 is LIBERO-Spatial: 10 tasks, 50 demos each,
whose assets S1 already converted.

**Done when:** all 10 Spatial tasks compile, register in GUIDE, and each of their 500 demo start
states loads with every movable within 5 mm and 3° of the demo's `states[0]` after 0.25 s of
settling. Fixtures must be within 1 mm and 0.1°, and drawer and knob joints within 1 mm or 0.01 rad.

## 2. Decisions this design follows (user, 2026-10-20, tracker q-s3-scope, q-s3-approach)

1. Each task's scene comes out **both** as a Replicator recipe over the converted assets **and** as
   a standalone USD.
2. The compiler also turns the BDDL behaviour (the `:goal`) into a **GUIDE-EX task description**.
3. **The FR3 replays the demos**, not the Panda. The roadmap's review focus #1 (keep LIBERO's
   Panda for round 1) is overridden. The 2 × 2 in S8 then measures simulator and embodiment
   together. The robot stays a compile option (`--robot fr3|panda`), so a Panda round 1 is
   one flag away if wanted.
4. Fixed start states go through **Replicator**, not a separate state writer: "separate the
   setups to callable arrangements, and let Replicator load the values".
5. **Reuse GUIDE as much as possible; make every piece as reusable as possible.** Anything not
   LIBERO-specific lives in GUIDE core or GUIDE-EX and works for any task.

## 3. Architecture

Three layers. Only the bottom one knows about LIBERO.

```
            GUIDE core (any task)                    GUIDE-EX (any task)
  ┌──────────────────────────────────────┐   ┌───────────────────────────────┐
  │ arrangements: named value sets that  │   │ tasks/pick_and_place.py       │
  │   Replicator loads into a recipe     │   │   (lifted from block_bin)     │
  │ static fixtures: kinematic, placed   │   │ tasks/goal_task.py: builds a  │
  │   by the recipe while playing        │   │   tree from a task description│
  │ joint states in recipes              │   └───────────────▲───────────────┘
  │ bake: recipe + arrangement → USD     │                   │ task.yaml
  │ RecipeScene: data-driven Scene       │                   │
  └──────────────────▲───────────────────┘                   │
                     │ init/randomize/reset/success.yaml,     │
                     │ arrangements/*.yaml, scene USD         │
  ┌──────────────────┴────────────────────────────────────────┴──────┐
  │ guide_libero (beside guide, like guide_dataset_tools)            │
  │  bddl.py → TaskModel   demos.py → arrangements   assets.py (S1)  │
  │  emit/: usd, recipe, arrangements, success, task, init           │
  │  cli: guide-libero compile --suite spatial --out <dir>           │
  └──────────────────────────────────────────────────────────────────┘
```

The output of a compile is a task directory that registers by path, with no colcon build
(F§5): `<out>/<task>/{<task>/scene.py, config/*.yaml, config/arrangements/*.yaml, assets/}`.

## 4. GUIDE core additions (task-agnostic)

### 4.1 Arrangements (the callable setups)

An **arrangement** is a named set of values for a task's *named* Replicator distributions:
`{distribution_name: value}`. Loading it pins each named distribution to its value for one
episode, and the recipe then runs exactly as written. So one recipe serves a fixed start
(arrangement given) and a random one (none given).

- **Recipe convention:** each thing an arrangement can set has its own named distribution.
  `<instance>_position` / `<instance>_rotation` drive a `modify.pose`, `<joint>_state` drives a
  `modify.attribute` (4.3). This is already how `blocks_position` works; the compiler emits one per
  instance.
- **Loading:** `replicator_guide.fire(..., values=None)` gains `values`. For each name, it sets the
  node's `inputs:lower = inputs:upper = value`, fires, and restores the bounds. It's the same
  per-episode node edit `guide.zone` already does (`_set(..., "inputs:lower", ...)`). A name with
  no node fails the call and names it.
- **Selection:** `Randomize.srv` gains `string arrangement` (empty means a free draw).
  `Scene.randomize(seed, inject, zone, arrangement)` reads
  `config/arrangements/<arrangement>.yaml`, and a hook (`arrangement_values()`) can override that.
  `Demonstration.srv` gains `string[] arrangements`, which plans episodes the way `zones` does.
- **Replaying recorded episodes:** a recorded episode's `values` (already written to
  `meta/guide_episodes.jsonl` as `replicator/<name>`) *is* an arrangement. A small tool
  `guide_dataset_tools arrangement <dataset> <episode>` writes one. This closes the roadmap's
  "inject does not replay Replicator draws" gap for every task, not only LIBERO.

### 4.2 Static fixtures (placed while playing)

LIBERO fixtures (table, cabinet, stove) must not move, yet each arrangement places them
somewhere else (their poses differ per demo, F§2). S1 found that welded fixtures snap back during
play.

**Design:** a fixture becomes a **kinematic** rigid body with no weld. `modify.pose` with
`write_to_usd` teleports it, and contacts can't push it. Jointed parts (drawers, knobs) stay
dynamic bodies, jointed to the kinematic base and driven (S1's drive gains).
**Spike K** (before the plan) proves this in Isaac 6.0.1 for the cabinet:
- teleport it during play, with drawers attached;
- the drawers follow;
- the drive holds them.

The fallback is S1's stopped-timeline placement (measured at 203 ms per fire).

These are the same "objects that shouldn't move" as the new OSCBF obstacle list (tracker
q-oscbf-obstacles). A compiled task lists its fixtures there.

### 4.3 Joint states in recipes

Drawer and knob positions are a `modify.attribute` on the joint's drive target, plus its state
attribute (`state:linear:physics:position` / `state:angular:physics:position`, UsdPhysics
JointStateAPI), both fed from one named distribution `<joint>_state`.

**Spike J** proves that writing both during play moves the joint there and holds it. If only the
drive target works, the joint is driven there during the settle window.

### 4.4 Bake (standalone USD)

`guide_core/tools/bake_usd.py <task> <arrangement> <out.usd>` opens the task's scene USD with pxr
and writes the arrangement's poses and joint values as authored xform ops and joint states. It
then flattens the result into one standalone USD with no Replicator or GUIDE needed to view it.

Because the values are constants, baking needs no Replicator. It reads the same recipe file to
find which prim each name drives (the recipe's `get.prims` path for that group).

### 4.5 RecipeScene

`guide_core/scene/recipe_scene.py: RecipeScene(SceneOrchestrator)` is a Scene whose behaviour is
all data, so a compiled task's `scene.py` is one line (`from ... import RecipeScene as Scene`).
- Task text comes from `config/task.yaml: language`.
- `zone_target()` comes from `task.yaml: target`.
- The success hooks specialise `success.yaml` from `task.yaml: goal` (4.6).
- The arrangement comes from 4.1.

block_bin and cube_stack keep their own Scenes. RecipeScene is new and replaces nothing.

### 4.6 Success (slice 1 subset; the exact port is S4)

`success.yaml` is generated from the goal. Slice 1 emits only what Spatial needs: `On obj obj`,
as `get_world_poses` on both prims plus a post-check (above, with xy within 3 cm). Contact and
`In/Open/Close/Turnon/Turnoff` are S4. They need a contact report and joint queries, and must match
LIBERO's thresholds exactly (roadmap review focus #3).

## 5. GUIDE-EX additions (task-agnostic)

- **`guide_ex/tasks/pick_and_place.py`:** block_bin's `PickAndPlace` lifted out of `solve_task.py`
  and parameterised by object prim, place pose source, and grasp and recovery options.
  block_bin's `solve_task.py` then imports it (no behaviour change; its 12-zone sweep is the
  regression).
- **`guide_ex/tasks/goal_task.py`:** builds a TASK from a task description:

  ```yaml
  # config/task.yaml (emitted by the compiler)
  language: pick up the black bowl next to the plate and place it on the plate
  target: /objects/akita_black_bowl_1          # zone target / tracked main object
  goal:                                        # BDDL :goal, one step per fact, in order
    - {on: [/objects/akita_black_bowl_1, /objects/plate_1]}
  ```

  Each `on`/`in` step becomes a `PickAndPlace` whose place pose comes from the support's live
  pose: above its top, at the object's rest height. `open/close/turnon/turnoff` steps refuse at
  build time with a clear message until S5 adds those skills. Spatial uses `on` only.
- A generic `solve_task` entry (`guide_ex.tasks.goal_task.main`) serves every compiled task, so
  compiled tasks ship no solver code.

## 6. guide_libero (LIBERO-specific)

A new repository beside guide (roadmap constraint: tooling lives beside guide), with no remote yet.
It's an installable Python package in the `.venv`.

| Module | Does | Reuses |
|---|---|---|
| `bddl.py` | `.bddl` → `TaskModel`: instances (name, type, fixture/object), regions (target, rect, yaw), init facts, goal facts, language | S1 `parse_bddl` (F§1, F§6) |
| `demos.py` | demo HDF5 → one arrangement per demo: movables from `states[0]` qpos (inner `object` frame), fixtures from the demo XML body pose, drawer and knob joints from qpos, the robot's start joints | F§2 probe code (mujoco + h5py, 2 path patches) |
| `frames.py` | LIBERO world → GUIDE scene frame (table top z = 0: z − 0.9), robot base pose, default movable yaw π/2 | F§2, F§4 |
| `assets.py` | asset type → converted USD (S1 table, extended per suite); applies S1's per-object fixes at build time | `spike_libero_convert.py`, `spike_libero_isaac.py` |
| `emit/usd.py` | scene USD: table, fixtures (kinematic, 4.2), objects at `/objects/<inst>` (the inner `object` frame), robot and cameras | S1 `build_scene` |
| `emit/recipe.py` | `randomize.yaml`: one named position/rotation distribution per instance, bounds = its BDDL region for table-region facts (H2), at most 6 `with` blocks per trigger; `reset.yaml` | S1 `to_replicator_yaml` |
| `emit/arrangements.py` | `config/arrangements/<demo>.yaml` from `demos.py` | — |
| `emit/success.py`, `emit/task.py`, `emit/init.py` | 4.6, 5, and `init.yaml` (robot profile, cameras, dataset section, `tracked_objects` = `:obj_of_interest` + supports) | block_bin/cube_stack `init.yaml` |
| `cli.py` | `guide-libero compile --suite spatial --demos ~/Downloads/libero_spatial --out ~/libero_assets/guide_tasks [--robot fr3] [--standalone]` | — |

**Frames and robot:**
- Scene frame: origin at the table-top centre, z up (F§4).
- The robot's base goes where LIBERO's Panda base is, (−0.66, 0, 0.012). The arrangement's
  robot joints are the demo's Panda start joints, which are within the FR3's limits.
- Cameras:
  - `agentview` at LIBERO's pose and fovy, scene frame (0.6586, 0, 0.7104).
  - The wrist camera at LIBERO's eye-in-hand offset from the hand (0.05, 0, 0).
  - Resolution is a compile option, default 128 × 128 like LIBERO (decision D2 below).

**H2 coverage in slice 1:** randomizing is emitted only for `On obj T-region` facts, which is
110 of Spatial's 160 regions. Instances in other facts (on another object, in or on a fixture
region, F§1b) keep a reference arrangement's values. Region geometry for those comes from the
asset `<site>`s in a later slice.

## 7. Data flow (one task)

```
 .bddl ──bddl.py──► TaskModel ──┬─ emit/usd ────────► assets/scene.usd ──bake──► assets/<demo>.usd (--standalone)
                                ├─ emit/recipe ─────► config/randomize.yaml, reset.yaml
                                ├─ emit/success ────► config/success.yaml
                                ├─ emit/task ───────► config/task.yaml ──► guide_ex goal_task (S5)
                                └─ emit/init ───────► config/init.yaml
 demo .hdf5 ──demos.py──────────── emit/arrangements ► config/arrangements/demo_<k>.yaml
```

At run time:
- `Register {path: <out>/<task>}`;
- `Randomize {arrangement: demo_7}` loads demo 7's start;
- `Randomize {}` draws an H2 layout.

## 8. Error handling

- **Compile time:** an unsupported fact refuses that fact and names it with its file and line,
  e.g. `In obj F-region` in H2. The task still compiles if H1 doesn't need the fact, and a
  `compile_report.json` lists every refusal per task.
- An asset type with no converted USD refuses the task.
- **Run time:**
  - an arrangement naming a distribution the recipe lacks fails `Randomize` with the name;
  - a fixture that isn't kinematic at registration fails `Register`;
  - success steps for an unsupported predicate fail at registration, not at episode end.

## 9. Testing

- **Unit (no Isaac):**
  - `bddl.py` over all 130 BDDLs, with the counts in F§1b as fixtures;
  - `demos.py` against the probed demo_0 numbers in F§2, to 1e-6;
  - `frames.py` round trips;
  - each emitter as golden YAML for one Spatial task;
  - `fire(values=...)` pins and restores bounds, against a fake node;
  - `goal_task` builds the expected tree from `task.yaml`.
- **Isaac:**
  - Spikes K and J;
  - then the Done-when check in §1: a script registers each compiled task, loads all 50
    arrangements, settles 0.25 s and compares against `states[0]`. It writes a JSON report with
    per-instance errors.
  - block_bin's 12-zone sweep stays 12/12 after the `PickAndPlace` lift.

## 10. Out of scope (later slices or steps)

- The other four suites: asset conversion and the remaining table fixtures, slice 2.
- Exact success predicates (S4).
- Replay of demo actions and new solvers (S5).
- Open/Close/Turnon skills (S5).
- Overlap rejection for H2 draws (cube_stack's redraw pattern, slice 2).
- Region geometry for non-table regions.

## 11. Decisions needed from you

- **D1** — the robot for slice 1 is the FR3 per your answer; the confound is recorded in §2. Keep
  Panda as a compile option? Recommended: yes, it costs one profile.
- **D2** — camera resolution: LIBERO's 128 × 128 (same views as the training data), or higher for
  the "better rendering" goal? Recommended: compile option, default 128. Record 256 later if
  wanted.
- **D3** — where compiled tasks live: `~/libero_assets/guide_tasks/<suite>/<task>/` (recommended:
  generated, not committed), or committed under `guide_tasks/`.
- **D4** — the `PickAndPlace` lift into `guide_ex/tasks` touches block_bin's solver. Recommended:
  yes, it's the reuse you asked for, with block_bin's sweep as the regression.

## 12. Before the plan

Spikes K (4.2) and J (4.3) run first, each about half a day in Isaac, and end in a
recommendation. The plan's task order: core 4.1 → 4.3 → 4.2 → 4.4 → 4.5, then GUIDE-EX §5, then
guide_libero §6, then the §1 validation.
