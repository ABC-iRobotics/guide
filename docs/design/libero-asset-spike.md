# LIBERO asset route (S1) — spike findings

Branch `spike/libero-assets` (throwaway). Isaac Sim 6.0.1.0, `mujoco_usd_converter` 0.2.0,
LIBERO commit `8f1084e` (MIT, sparse clone in `~/libero_assets/LIBERO`). Measured 2026-10-19 with
`guide_core/scripts/spike_libero_convert.py`, `spike_libero_isaac.py` and `spike_libero_place.py`.
Results: `spike_libero_convert.json`, `spike_libero_isaac.json` (`_findings`, `_texture_check`),
`spike_libero_place.json` (`summary`, `drawer_recheck`, `_findings`), its three variant runs
`spike_libero_place_offset.json`, `_preplay.json`, `_tensor.json`, and the generated
`spike_libero_randomize.yaml`. Renders, not copied into the repo: `~/libero_assets/spike/renders/`.
Program tracker: https://claude.ai/artifact/HC3YKe5CcfYdCf4wbTLVzE

Question (roadmap S1): which route gives Isaac-loadable USD with sane colliders, working
articulation, acceptable visuals, and a licence we can use? This spike probed `mujoco_usd_converter`
on LIBERO's MJCF, with placement from the BDDL regions by Replicator. The licence criterion was not
checked, so the adoption below is conditional on that check (still open). Scope: the LIBERO-Spatial object set (two `akita_black_bowl`, `plate`,
`glazed_rim_porcelain_ramekin`, `cookies`, `wooden_cabinet`, `flat_stove`).

**Verdict: adopt with fixes, conditional (licence check open).** Details under "Recommendation".

## What was built

1. **Patch + convert** (`spike_libero_convert.py`): each legacy MuJoCo `.msh` becomes an `.obj`
   (texture `v` flipped as MuJoCo's loader expects), `file="*.msh"` references are rewritten, the
   `<mujoco model>` name is set to the asset name (the plate's was `model`), duplicate texture
   names are de-duplicated in document order (the flat stove defines `tex-stove_knob` twice).
   Then `mujoco_usd_converter` with `--no-physics-scene`. A fidelity check compares each USD with
   the patched XML (meshes, materials, textures, joints) and keeps converter warnings.
2. **Isaac checks** (`spike_libero_isaac.py`): `/Scene_0` with a static table (top at z = 0), the
   assets as references, a 3 s drop at 60 Hz with PhysX poses, drive tests of the drawers and the
   knob, RTX renders. `build_scene` carries the workarounds listed below.
3. **Placement** (`spike_libero_place.py`): a BDDL parser, a generator from `:regions` and `:init`
   facts to a GUIDE-dialect Replicator YAML (one `modify.pose` per `on <obj> main_table_<region>`
   fact, uniform x/y over the region, yaw from `:yaw_rotation`, at most 6 `with` blocks per
   trigger), run through `replicator_guide.build` and `fire` for seeds 0 to 9.

## Results

### Conversion (`spike_libero_convert.json`)

| Asset | `.msh` converted | Rigid bodies | Meshes USD / used by source geoms | Materials source / USD | Moving joints | Warnings |
|---|---|---|---|---|---|---|
| akita_black_bowl | 1 | 2 | 1 / 1 | 1 / 2 | none | 0 |
| plate | 1 | 2 | 1 / 1 | 1 / 2 | none | 0 |
| glazed_rim_porcelain_ramekin | 1 | 2 | 1 / 1 | 1 / 2 | none | 0 |
| cookies | 1 | 2 | 1 / 1 | 1 / 2 | none | 0 |
| wooden_cabinet | 7 | 6 | 7 / 7 | 7 / 8 | 3 prismatic | 0 |
| flat_stove | 2 | 6 | 2 / 2 | 1 / 3 | 1 revolute | 2 |

6 of 6 assets convert, after the `.msh` patch (the converter reads only OBJ/STL: `mujoco_usd_converter/_impl/mesh.py:58-64`; MuJoCo 3.8 itself loads the unpatched `.msh`, the converter does not). The 2 stove
warnings: the textured material is bound to Cube prims (`base_vis`, `base`) and the converter says
the texture is discarded at render time. `stove_knob_base_vis` is declared in the XML, but no geom
uses it, so the converter drops it (`unused_mesh_assets`).

### Isaac load, drop, articulation, render (`spike_libero_isaac.json`)

All 7 instances load. After a 3 s drop from 5 cm above rest (with the workarounds below), no
instance fell through, and the final speed is 0 for all but the cabinet (0.0001).

| Instance | BBox (m) | Rest z (m): movables = settled body-origin height above the table; fixtures = origin height with the visible bottom at z = 0 |
|---|---|---|
| akita_black_bowl_1 / _2 | 0.112 x 0.111 x 0.053 | -0.0016 |
| plate_1 | 0.137 x 0.137 x 0.017 | 0.0025 |
| glazed_rim_porcelain_ramekin_1 | 0.089 x 0.089 x 0.043 | -0.0007 |
| cookies_1 | 0.061 x 0.084 x 0.020 | 0.0094 |
| wooden_cabinet_1 (fixture) | 0.256 x 0.222 x 0.217 | -0.0046 |
| flat_stove_1 (fixture) | 0.295 x 0.190 x 0.074 | 0.020 |

The bowl (-0.0016) and ramekin (-0.0007) heights are bounding boxes over visual meshes sitting 1 to 2 mm below the colliders (e.g. the bowl's lowest collision box bottom is at +0.0016 m: pos z 0.00417 minus half-size 0.00258). The cabinet's -0.0046 (4.6 mm): in the converted USD both its lowest visible geometry and its lowest collider sit at +0.0046 m in the asset frame.

| Joint | Type | Limits | Reached open / closed |
|---|---|---|---|
| cabinet top_level | prismatic | -0.16 ... 0.01 m | -0.1599 / 0.0099 |
| cabinet middle_level | prismatic | -0.16 ... 0.01 m | -0.1599 / 0.0099 |
| cabinet bottom_level | prismatic | -0.16 ... 0.01 m | -0.16 / 0.0099 |
| stove knob `button` | revolute | -0.005 ... 2.1 rad | 2.0999 / -0.0049 |

Which end is "open" (cabinet: lower limit; stove: upper limit) is inferred from the limits and the
render. Not verified against LIBERO's code.

Renders (pixel range 250 and 216, so neither is blank): `~/libero_assets/spike/renders/isaac_overview.png`,
`isaac_cabinet_open.png`. Reference: `libero_agentview_next_to_plate_demo0_frame0.png` (LIBERO
agentview, demo 0 frame 0, flipped upright).

Textures (`_texture_check`, bowl, cookies and plate only; the cabinet and stove textures were not checked): the converted textures of those three are byte-identical to the sources. Bowl: a 4096^2
UV atlas, 66.6 % unused black background; the islands are light ceramic with a black floral
pattern, and both LIBERO and Isaac show light bowls with a dark pattern ("black" is the asset's
name). Bowl/table luma is 0.98 in Isaac and 0.70 in LIBERO (lighting, not texture). Cookie label
upright and legible; plate rings correct. The stove base renders plain grey (the converter's
discarded texture, above).

### As-converted defects (`spike_libero_isaac.json` `_findings`)

| # | Defect as converted | Measured without the workaround | Workaround in `build_scene` | Fix belongs in |
|---|---|---|---|---|
| 1 | MJCF's unnamed outer body (sites only) becomes rigid body `Geometry/Body` with no collider and no mass; PhysX falls back to a default mass. In the bowl it sits 0.1 m from the bowl. | Both bowls tipped (76.8 and 78.1 deg), rest z -0.0574, "fell through" by the brief's rule. Plate, ramekin, cookies stayed upright. | 0.1 g mass on colliderless bodies | converter patch step |
| 2 | Every asset is welded to the world (fixed joint, body0 = root Xform without a rigid body). LIBERO object files have no free joint; robosuite adds one on placement. | Every movable hangs at its release height after 3 s (z 0.05 / 0.05 / 0.05 / 0.06 m, speed 0) | disable the joint on movables; fixtures keep it | converter patch step (free joint, or no weld for objects) |
| 3 | A fixture moved by writing its body snaps back to the weld anchor | cabinet and stove returned toward the root pose; the joint-disjoint warning fired | place fixtures through the reference root | placement rule (but see fixtures, below) |
| 4 | Collision-only geoms (MJCF group 0) render as grey cubes over the textures | cabinet front mean RGB 169.6 / 164.8 / 159.6 instead of 33.1 / 28.1 / 23.9 | `purpose = guide` on those prims (hidden; collisions unchanged). Counts: bowl 40, ramekin 25, plate 10, cookies 1, cabinet 36, stove 20 | converter patch step |
| 5 | USD camera default near plane is 1 m | close-ups empty | near plane 0.01 m | GUIDE camera note, not the asset |

### Placement from the BDDL regions (`spike_libero_place.json`, seeds 0 to 9)

Target file: `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl`. All 7 init
facts are `on <instance> main_table_<region>`. Generated YAML: 2 randomizers (6 + 1 `with` blocks),
2 triggers, one event. Mean `fire` 66.8 ms (first call 212.6 ms), one scene, no render product
attached.

As briefed (R5: write `place_prim`; movables `Geometry/Body`, fixtures the reference root):

| Instance | in region | drift < 1 cm | yaw ok | two bodies followed |
|---|---|---|---|---|
| akita_black_bowl_1 | 10/10 | 0/10 | 10/10 | 10/10 |
| akita_black_bowl_2 | 10/10 | 0/10 | 10/10 | 10/10 |
| plate_1 | 10/10 | 2/10 | 10/10 | 10/10 |
| cookies_1 | 10/10 | 10/10 | 10/10 | 10/10 |
| glazed_rim_porcelain_ramekin_1 | 10/10 | 0/10 | 10/10 | 10/10 |
| wooden_cabinet_1 | 1/10 | 0/10 | 1/10 | 10/10 (vacuous) |
| flat_stove_1 | 1/10 | 7/10 (vacuous) | 10/10 (vacuous for 9/10: no yaw range, and the stove never left its anchor in seeds 1-9) | 10/10 (vacuous) |

- **Movables:** each `modify.pose` write during play teleports both welded rigid bodies exactly
  (`err_to_sample_m` 0.0 in every case). Whether it keeps the velocities was not measured (velocities were never read); the one case seen: a body teleported
  back from 7.7 km came in with its weld stretched to 0.1044 m and came to rest one second later (drift 0.0067 m).
- **The drift failures are an R5 error, see "wrong turns".** With bounds shifted by the 0.1 m offset
  and the region checked on the bowl itself (`spike_libero_place_offset.json`):
  in region 50/50, drift < 1 cm in 47/50. The 3 failures: seed 3 bowl_1 0.051 m and plate 0.018 m
  (bowl-plate centre distance 0.111 m is under the radii sum 0.125 m: LIBERO's two regions allow
  touching draws, and this path does not reject them); seed 0 bowl_2 3805 m (inferred to be a strike
  by the cabinet snapping back; not isolated).
- **Fixtures during play fail.** The root write lands exactly on the sample (seed 0, yaw included),
  but the weld keeps the anchor PhysX built at play start: within 1 s the stove snaps back 0.72 m
  (drift 0.7218), the cabinet diverges (post-fire reads in seeds 1-9 are finite, about 4e9 m, NaN only after stepping), and the drawer re-check is null for every
  joint (`drawer_recheck.opens_and_closes` false). A tensor-API teleport (`set_world_poses`) lands on
  the sample every seed and snaps back every seed (stove drift 0.708 to 0.725 m, cabinet NaN).
- **Fixtures with the timeline stopped work.** Stop, fire, reset per seed: fixtures 10/10 in region, yaw and
  drift (PhysX = sample to 0.0001 m; the stove's seed-0 yaw error is 0.004 deg), and the drawers still reach -0.1599 / 0.0099
  after ten root writes with yaw. Cost: mean fire 202.7 ms (see Task 3 report) plus a full stop and
  reset per layout. (Run in `spike_libero_place_preplay.json`.) That run did not compensate the movables' offset, so its 10/10 is the fixtures' result only (e.g. bowl_2 in region 8/10, drift < 1 cm 0/10). Its seed-0 movables did not land on the first stopped fire (cause not found; cookies err 0.305 m at the Task-2 rest, ramekin 0.655, plate 0.666, bowls 0.47-0.49); seeds 1-9 landed (err <= 5 mm). The tensor-API variant is in `spike_libero_place_tensor.json`.
- **Play-time anchors hit seed 0.** The cabinet's play-start weld anchor footprint (x -0.38..-0.12, y 0.19..0.41) covers the bowl_2 and ramekin regions, so seed 0 of every play-time run was hit by the cabinet snapping back; future runs should park anchors outside all regions.
- **Not tried:** writing the weld's `physics:localPos0/localRot0` during play; kinematic fixtures
  without a weld.
- **Unsupported init facts, all 10 Spatial BDDLs:** 12 of 71 are not `on <obj> main_table_<region>`
  (on an object 3, on a fixture region 7, in a fixture region 1, open a fixture region 1); 4 of 10
  files are fully supported.

## What the spike settled

- The route works: 6/6 assets convert once the `.msh` files are converted, the checked properties match the
  source (mesh, material and joint counts, joint limits, unreferenced bboxes; texture bytes for bowl, cookies and plate only), articulation works (drawers and knob reach their
  limits), and the Replicator front-end places LIBERO movables from BDDL regions in one pass.
- Three defects are properties of the converted asset, not of our scene (table above, rows 1, 2
  and 4), and each is expected to be a small change in the patch step (an expectation: only `build_scene` workarounds exist, the converter-side fixes were never implemented). Row 3 (fixtures) needs a placement rule.
- **Wrong turns.**
  - Ruling R5 (controller): place movables through `place_prim` = `Geometry/Body`. It failed. That
    body is the converter's colliderless outer body, 0.1 m off the bowl, so the bowls landed on the
    plate and the ramekin. Cost: a baseline drift of 0/10 for bowls and ramekin and 2/10 for the
    plate, one debugging attempt, and an offset-shifted rerun. R5's fixture half (write the root)
    was necessary but not sufficient.
  - The bowl first looked wrongly black (mean RGB 60); it is a mostly-black atlas, not a bad
    conversion.
  - The first stove binding used the wrong texture (a de-duplication ordering bug, fixed in the Task 1 review fix round, commit 7ac3470).
- **GUIDE defect found and fixed.** `replicator_guide.fire(seed)` did not restart Replicator's
  samplers for seed 0 on a fresh app, or for a seed repeated twice in a row: Replicator's
  `rng.set_global_seed` returns early when the seed equals the stored value (0 while unset). Seed-0
  samples for bowl_1 differed across three runs ([0.0065, 0.3191], [0.0028, 0.3098],
  [0.0092, 0.3142]). Fixed in commit `eb6d160`, merged into dev (0dbe09f) and an ancestor of this branch.
  Verified in a headless Isaac run in two fresh processes: seed 0 first vs later, the same seed
  twice, and seeds 0 and 5 across processes all draw identical samples (before the fix, seed 0 on a fresh app and a repeated seed did not; seeds 1-9 were already identical across processes). Controller-run verification; evidence in the tracker and the commit message of `eb6d160`.

## Gaps

- **Licences not checked.** The S1 roadmap item "check licences of LIBERO's asset families" was not
  done. Only the repo licence (MIT) is recorded. Scanned, HOPE, TurboSquid and articulated assets
  may carry their own terms. Not measured.
- **Physics fidelity is not compared with MuJoCo** (mass, friction, contact parameters). Only
  geometry, joint limits and rest behaviour were checked.
- **Region frame vs LIBERO** was checked only against one agentview frame (layout agrees); the table-centre offset, the yaw sign and the fixture z rule are unchecked.
- **Opening direction** of the cabinet and stove knob is inferred (see above).
- **Robosuite behaviour** (inner `object` body is what LIBERO places; group 0 is never rendered) is
  recalled, not checked: the robosuite/LIBERO env code is not on this machine. The agentview frame
  agrees (bowl beside the plate).
- **Only the Spatial object set** (6 asset types). Other LIBERO suites' assets are not converted.
- **Fixture placement during play** is open (two untried options above).
- **Overlap rejection, velocity zeroing and the drift of 47/50** were not re-run on a converter that
  merges the outer body; the 47/50 is with a placement-side offset.
- **Stove base texture** is lost (grey) by the converter.
- **Isaac renders brighter** than MuJoCo (luma ratio 0.98 vs 0.70): lighting, not tuned.
- **Success predicates** (S4) and initial states from `init_files` (S3/S5) are untouched.
- Deferred minors from the review ledger (spike script hygiene: hard-coded home paths, three
  `--attempt` debugging branches kept in `spike_libero_place.py`, silent drop of instances missing
  from the Task 2 JSON) do not affect the results.

## Recommendation for S1: adopt with fixes

Adopt `mujoco_usd_converter` (plus the `.msh` to `.obj` patch step). The route gave matching
mesh, material and joint counts and joint limits (see "What the spike settled"), works in Isaac 6.0.1, and is expected to cost one patch script, which we expect to be cheaper than
tuning the Isaac MJCF importer per asset or depending on RobotLearningLab / Lightwheel assets
(an expectation: none of the fallbacks was tried, and the converter-side fixes were never
implemented; they are the fallbacks if a fix below fails). Adoption is conditional on the open
licence check (fix 8). Fixes, in order:

1. **Patch step: remove the colliderless outer body.** Merge it into the colliding body, taking the inner `object` frame (merging into the outer frame keeps the 0.1 m offset; MuJoCo
   semantics) or add a free joint on the inner body so that `Geometry/Body` is the object. Then
   defect 1 (fallback mass, tipped bowls) goes away and the brief's region bounds are expected to be right as
   they are (placement on the colliding body). This rests on the unchecked belief that LIBERO places
   the inner `object` body (see Gaps, robosuite behaviour). The committed YAML z values are outer-body heights and must be re-measured after the fix.
2. **Patch step: drop the world weld for movables** (or give them a free joint). Keep it for
   fixtures only if the placement rule below needs it.
3. **Patch step: map MJCF group-0 geoms to non-render** (`purpose = guide`, as `build_scene` does).
4. **Fixture placement rule.** Pick one before S3: (a) place fixtures with the timeline stopped
   (stop, fire, reset: measured 10/10), (b) write the weld's `localPos0/localRot0` during play,
   (c) kinematic fixtures without a weld. (b) and (c) are not tried; (a) costs about 200 ms plus a
   reset per layout. (a) was measured for fixtures only; movables untested under (a) (their seed-0 write did not land on the first stopped fire).
5. **Placement: overlap rejection and velocity zeroing.** Redraw on overlap (cube_stack's 12 cm
   rule) and zero `physics:velocity` / `angularVelocity`, as cube_stack does (`guide_tasks/cube_stack/config/randomize.yaml:50-56`); whether `modify.pose` keeps velocities was not measured.
6. **Camera near plane** 0.01 m wherever cameras are within 1 m of the subject.
7. **For S3, the compiler must handle the unsupported predicate shapes:** `on <object> <object>`,
   `on <object> <fixture region>`, `in <object> <fixture region>`, `open <fixture region>` (12 of
   71 init facts across the 10 Spatial files).
8. **Before relying on the assets:** check the licences of each asset family, and compare mass and
   friction with MuJoCo.
9. **Stove base texture:** the converter drops it on Cube prims; accept grey or re-author.

The reseed fix (`eb6d160`, merged into dev as 0dbe09f) is a precondition for reproducible
seed-0 layouts and is already in this branch.
