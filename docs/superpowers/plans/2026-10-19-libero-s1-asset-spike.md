# S1 — LIBERO asset spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** answer whether LIBERO-Spatial's assets, converted with `mujoco_usd_converter`, load and behave in Isaac Sim 6.0.1, and whether GUIDE's Replicator front-end can place them from the BDDL regions.

**Architecture:** four throwaway scripts on branch `spike/libero-assets`:
1. `.msh` → `.obj` plus patched MJCF copies, then conversion to USD.
2. A headless Isaac check: load, drop test, articulation, render.
3. A BDDL → GUIDE-dialect Replicator YAML generator, run through `guide_core.types.randomization.replicator_guide`.
4. A findings document.

Results are JSON files in `docs/design/`. Converted assets and renders live outside the repo in `~/libero_assets/`.

**Tech stack:** `~/ros2_ws/.venv/bin/python` (Python 3.12, Isaac Sim 6.0.1, `omni.replicator.core`), `mujoco` 3.8.0, `mujoco_usd_converter` 0.2.0 (CLI `~/ros2_ws/.venv/bin/mujoco_usd_converter`), `pxr` (USD), numpy, PyYAML.

**Spec:** the roadmap `docs/superpowers/plans/2026-10-19-libero-isaac-roadmap.md`, section S1. Live tracker: https://claude.ai/artifact/HC3YKe5CcfYdCf4wbTLVzE.

## Global Constraints

- Work in `~/ros2_ws/src/guide` on branch `spike/libero-assets`. No worktrees. Do not touch the user's uncommitted `guide_core/config/init.yaml`, and never stage it.
- Commits authored by the repo's user only: **no `Co-Authored-By`, no Claude/Generated-with lines**. Never push. Never open PRs.
- `~/libero_assets/LIBERO` is a pristine sparse clone of LIBERO (commit 8f1084e, MIT): read-only data. Never modify it, and never run code from inside it.
- Derived data goes to `~/libero_assets/mjcf/` (patched MJCF + `.obj` meshes), `~/libero_assets/usd/` (converted USD) and `~/libero_assets/spike/` (renders, logs).
- Spike code lives only in `guide_core/scripts/spike_libero_*.py`. Do not change any module under `guide_core/guide_core/`, `guide_ex/` or `guide_tasks/`.
- No new Python dependencies. If one seems unavoidable, stop and report `BLOCKED`.
- Isaac runs headless through `~/ros2_ws/.venv/bin/python`:
  - set `OMNI_KIT_ACCEPT_EULA=YES`; unset `CUDA_DEVICE_ORDER` (otherwise camera images come back empty);
  - the first launch compiles shaders for about 2 minutes;
  - before launching, check that no other Isaac/GUIDE process runs (`pgrep -af "isaac|kit/kit|guide_core"`).
- Every non-trivial pure function (parsers, converters) has one runnable self-check (`--selftest`, `assert`-based). No test frameworks are added.
- Asset set = LIBERO-Spatial's objects. Asset names and sources:

| name | MJCF source (under `~/libero_assets/LIBERO/libero/libero/assets/`) | role |
|---|---|---|
| `akita_black_bowl` | `stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml` | object (two instances) |
| `plate` | `stable_scanned_objects/plate/plate.xml` | object |
| `glazed_rim_porcelain_ramekin` | `stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml` | object |
| `cookies` | `stable_hope_objects/cookies/cookies.xml` | object |
| `wooden_cabinet` | `articulated_objects/wooden_cabinet.xml` (parts in `articulated_objects/wooden_cabinet/`) | fixture, articulated (drawers) |
| `flat_stove` | `articulated_objects/flat_stove.xml` (parts in `articulated_objects/flat_stove/`) | fixture, articulated (knob) |

- Table for the spike: a static box 1.0 m (x) × 1.2 m (y) × 0.05 m, LIBERO's `table_full_size`. Its **top surface is z = 0 of the scene frame `/World/Scene_0`**, and its centre is at the scene origin. BDDL region ranges `(x_min y_min x_max y_max)` on `main_table` are read directly as x/y in that frame.

## Review Focus

1. **Scale and units:** an asset arriving 100× too large or too small (MJCF mesh `scale`, USD `metersPerUnit`). Expected: the bowl's world bounding box is about 0.1–0.2 m across. Pinned in Task 2's load check.
2. **Colliders on dynamic bodies:** a triangle-mesh collider on a rigid body is invalid in PhysX, so the object falls through the table. Expected: every movable object rests on the table after the drop test (z within 5 cm of the top, speed < 1 cm/s). Pinned in Task 2's drop test.
3. **MSH texture coordinates:** a flipped `v` makes textures look scrambled. Expected: renders show the bowl black and the cookie box with its printed label. Pinned in Task 1's self-check (the `v` convention) and Task 2's renders.
4. **Region frame:** objects placed outside their BDDL rectangle because of a frame or axis swap. Expected: every placed object's x/y lies inside its region, within 1 mm. Pinned in Task 3's per-seed check.
5. **Fixtures placed by Replicator:** an articulated fixture (cabinet, stove) that does not move when its root pose is written, or whose articulation breaks. Expected: the reported pose equals the drawn pose, and the drawer still opens afterwards. Pinned in Task 3 (pose check) and Task 2 (drawer check, run after placement).

---

### Task 1: MSH → OBJ, patched MJCF, conversion to USD

**Files:**
- Create: `guide_core/scripts/spike_libero_convert.py`
- Create (result): `docs/design/spike_libero_convert.json`

**Interfaces:**
- Produces:
  - `~/libero_assets/usd/<name>/<name>.usda` for each asset name in the table above (converted with `--no-physics-scene`; the Isaac scene supplies one physics scene).
  - `docs/design/spike_libero_convert.json`: `{"<name>": {"ok": bool, "error": str|null, "usd": "<abs path>", "msh_converted": int, "bodies": int, "joints": [{"name": str, "type": str}], "meshes": int}}`.
  - Python functions `read_msh(path) -> dict(vertices, normals, texcoords, faces)` and `msh_to_obj(src, dst) -> None`.

The legacy MuJoCo `.msh` layout (little-endian): `int32 nvertex, nnormal, ntexcoord, nface`, then `float32 vertices[3*nvertex]`, `float32 normals[3*nnormal]`, `float32 texcoords[2*ntexcoord]`, `int32 faces[3*nface]`. The file size must equal `16 + 12*nv + 12*nn + 8*nt + 12*nf`. MuJoCo flips `v` when it loads an OBJ (`v -> 1 - v`) but not when it loads an MSH. So the OBJ written here stores `1 - v`, which renders the same as the original MSH.

- [ ] **Step 1: Write the MSH reader/writer with its self-check**

```python
"""Spike S1: LIBERO-Spatial MJCF -> USD. Throwaway (branch spike/libero-assets).

  ~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py --selftest
  ~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path.home() / "libero_assets/LIBERO/libero/libero/assets"
MJCF = Path.home() / "libero_assets/mjcf"
USD = Path.home() / "libero_assets/usd"
CONVERTER = Path.home() / "ros2_ws/.venv/bin/mujoco_usd_converter"
ASSETS = {
    "akita_black_bowl": "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
    "plate": "stable_scanned_objects/plate/plate.xml",
    "glazed_rim_porcelain_ramekin": "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml",
    "cookies": "stable_hope_objects/cookies/cookies.xml",
    "wooden_cabinet": "articulated_objects/wooden_cabinet.xml",
    "flat_stove": "articulated_objects/flat_stove.xml",
}


def read_msh(path):
    raw = Path(path).read_bytes()
    nv, nn, nt, nf = np.frombuffer(raw[:16], "<i4")
    if len(raw) != 16 + 12 * nv + 12 * nn + 8 * nt + 12 * nf:
        raise ValueError(f"{path}: size {len(raw)} does not match header {nv, nn, nt, nf}")
    o = 16
    def take(n, dtype, width):
        nonlocal o
        a = np.frombuffer(raw[o:o + 4 * n * width], dtype).reshape(n, width)
        o += 4 * n * width
        return a
    return {"vertices": take(nv, "<f4", 3), "normals": take(nn, "<f4", 3),
            "texcoords": take(nt, "<f4", 2), "faces": take(nf, "<i4", 3)}


def msh_to_obj(src, dst):
    m = read_msh(src)
    v, n, t, f = m["vertices"], m["normals"], m["texcoords"], m["faces"] + 1
    lines = [f"v {a:.7g} {b:.7g} {c:.7g}" for a, b, c in v]
    lines += [f"vn {a:.7g} {b:.7g} {c:.7g}" for a, b, c in n]
    lines += [f"vt {a:.7g} {1 - b:.7g}" for a, b in t]  # MuJoCo flips v for OBJ only
    def corner(i):  # i, i/i, i//i or i/i/i
        if not (len(t) or len(n)):
            return str(i)
        return f"{i}/{i if len(t) else ''}" + (f"/{i}" if len(n) else "")
    lines += ["f " + " ".join(corner(i) for i in face) for face in f]
    Path(dst).write_text("\n".join(lines) + "\n")


def _selftest():
    with tempfile.TemporaryDirectory() as d:
        msh = Path(d) / "tri.msh"
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4")
        tex = np.array([[0, 0], [1, 0], [0, 0.25]], "<f4")
        faces = np.array([[0, 1, 2]], "<i4")
        msh.write_bytes(np.array([3, 0, 3, 1], "<i4").tobytes() + verts.tobytes() + tex.tobytes() + faces.tobytes())
        m = read_msh(msh)
        assert m["vertices"].shape == (3, 3) and m["faces"].tolist() == [[0, 1, 2]]
        obj = Path(d) / "tri.obj"
        msh_to_obj(msh, obj)
        text = obj.read_text()
        assert "vt 0 0.75" in text, text          # v flipped
        assert "f 1/1 2/2 3/3" in text, text      # 1-based, texcoords, no normals
    print("selftest ok")
```

- [ ] **Step 2: Run the self-check**

Run: `~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py --selftest`
Expected: `selftest ok`. If the `f` line format assertion fails, fix `corner()` so that a face corner is `i` (no tex, no normals), `i/i` (tex only), `i//i` (normals only) or `i/i/i` (both). Keep the assertion and make it pass.

- [ ] **Step 3: Patch MJCF copies and convert**

Add to the script:
- `patch(name)`: copy the asset's source folder into `MJCF/<name>/`. For the two articulated assets, copy the `.xml` and its part folder (`articulated_objects/<name>/`) side by side. Convert every `.msh` under the copy to `.obj` next to it, and rewrite every `file="…​.msh"` in the copied XML to `.obj`. Return the patched XML path.
- `convert(name)`: run `[CONVERTER, xml, USD/name, "--no-physics-scene"]` with `subprocess.run(..., capture_output=True, text=True, timeout=600)`. Read the result with `pxr.Usd.Stage.Open` (import `usdex.core` first: the `.venv`'s `pxr` needs it to open files standalone) and count bodies (prims with `PhysicsRigidBodyAPI` or `PhysicsArticulationRootAPI`), joints (`UsdPhysics.Joint` subclasses: record name and type name) and meshes (`UsdGeom.Mesh`). Record `ok`, `error` (the converter's last stderr line on failure), `msh_converted`.
- `main()`: run every asset, keep going on failure, and write `docs/design/spike_libero_convert.json` (pretty, sorted keys).

- [ ] **Step 4: Run it**

Run: `~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_convert.py`
Expected: `docs/design/spike_libero_convert.json` with 6 entries. Each failed asset has a concrete `error`. Spend at most 3 debugging attempts per asset on converter errors, then record the error and move on. `wooden_cabinet` should report prismatic joints for its drawers, `flat_stove` a revolute joint for its knob.

- [ ] **Step 5: Commit**

```bash
git add guide_core/scripts/spike_libero_convert.py docs/design/spike_libero_convert.json
git commit -m "spike(libero): convert LIBERO-Spatial assets to USD via mujoco_usd_converter"
```

---

### Task 2: Isaac checks — load, drop, articulation, render

**Files:**
- Create: `guide_core/scripts/spike_libero_isaac.py`
- Create (result): `docs/design/spike_libero_isaac.json`
- Renders: `~/libero_assets/spike/renders/isaac_*.png`

**Interfaces:**
- Consumes: `docs/design/spike_libero_convert.json` (Task 1). Only assets with `"ok": true` are loaded.
- Produces:
  - `build_scene(stage, assets: dict[str, list[str]]) -> dict[str, str]`: creates `/World/Scene_0` with a `PhysicsScene`, a ground-free static table (box as above, collider, top at scene z = 0), a distant light, a dome light, and an `Xform` `/World/Scene_0/objects`. Under it, one reference per instance, e.g. `{"akita_black_bowl": ["akita_black_bowl_1", "akita_black_bowl_2"], …}` → `/World/Scene_0/objects/akita_black_bowl_1` referencing `~/libero_assets/usd/akita_black_bowl/akita_black_bowl.usda`. Returns `{instance: path of the prim that carries PhysicsRigidBodyAPI or PhysicsArticulationRootAPI}`, which may be the reference root or a descendant. Task 3 imports this function.
  - `docs/design/spike_libero_isaac.json`: `{"<instance>": {"loaded": bool, "body_prim": str, "bbox_size": [x, y, z], "dropped_from_z": float, "rest_pose": [x, y, z, qw, qx, qy, qz], "rest_z_above_table": float, "final_speed": float, "fell_through": bool}, "_articulation": {"wooden_cabinet_1": {"joints": {...name: {"type": str, "lower": float, "upper": float, "reached_open": float, "reached_closed": float}}}, "flat_stove_1": {...}}, "_renders": [paths], "_isaac": "<version>"}`.

Template for a headless Isaac script that reads PhysX poses: `git show 6f8adb5:guide_core/scripts/spike_replicator.py` (same Isaac 6.0.1; `RigidPrim`/`XFormPrim` from `isaacsim.core.prims`).

- [ ] **Step 1: Write the script**

```python
"""Spike S1: load the converted LIBERO-Spatial USDs in Isaac 6.0.1 and check them.

  OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python \
      guide_core/scripts/spike_libero_isaac.py
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import json
from pathlib import Path

import numpy as np
# remaining imports (omni.usd, pxr, isaacsim.core.*, omni.replicator.core for the render) after SimulationApp
```

Checks, in order, with the results written to the JSON above:
1. **Load:** every instance resolves; `bbox_size` from `UsdGeom.BBoxCache` (world, `default` + `render` purposes).
2. **Drop test:** place each movable instance (bowls, plate, ramekin, cookies) at its own spot on the table, 5 cm above its resting height. Use the bbox bottom to compute that height. Keep 25 cm between instances. Step 3 s of physics at 60 Hz, then read the PhysX pose and speed. `fell_through` = final z below −0.05. `rest_z_above_table` = rest z of the body prim's origin minus the table top. Fixtures (`wooden_cabinet_1`, `flat_stove_1`) sit at fixed spots on the table, not dropped.
3. **Articulation:** for each fixture's joints, record type and limits. Drive each prismatic or revolute joint to its upper limit, then back to its lower limit:
   - through a drive target (`UsdPhysics.DriveAPI`, stiffness 1e4, damping 1e3) if the joint has none;
   - or through `isaacsim.core.prims.Articulation` joint position targets.

   Record the reached positions after 2 s each.
4. **Render:** one RGB frame from a camera 1.2 m above the table, looking down at 60°. Use `omni.replicator.core` (render product + `rgb` annotator) and step a few frames before reading. Save `~/libero_assets/spike/renders/isaac_overview.png`, plus one close-up of the cabinet with its top drawer open (`isaac_cabinet_open.png`).

- [ ] **Step 2: Run it**

Run: `OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python guide_core/scripts/spike_libero_isaac.py 2>&1 | tail -40`
Expected: the JSON written, both PNGs non-empty (not uniformly black: check `np.ptp(img) > 0` before saving and record it). Record every failure in the JSON with the exception text. Do not paper over a failure.

- [ ] **Step 3: Commit**

```bash
git add guide_core/scripts/spike_libero_isaac.py docs/design/spike_libero_isaac.json
git commit -m "spike(libero): load, drop, articulation and render checks in Isaac 6.0.1"
```

---

### Task 3: BDDL regions → GUIDE Replicator YAML → placement

**Files:**
- Create: `guide_core/scripts/spike_libero_place.py`
- Create (results): `docs/design/spike_libero_randomize.yaml` (generated), `docs/design/spike_libero_place.json`
- Renders: `~/libero_assets/spike/renders/place_seed<N>.png`

**Interfaces:**
- Consumes:
  - `build_scene` from `guide_core/scripts/spike_libero_isaac.py` (Task 2): import it by path, e.g. via `importlib.util.spec_from_file_location`. Its module-level `SimulationApp` must not start twice, so move that file's app start-up under `if __name__ == "__main__":` as part of this task if needed.
  - `rest_z_above_table` per instance from `docs/design/spike_libero_isaac.json`.
  - `guide_core.types.randomization.replicator_guide.build(yaml_path, scene_prefix)` and `.fire(state, seed=…)` with `PYTHONPATH=~/ros2_ws/src/guide/guide_core`. Paths in the YAML are scene-relative: `build` prefixes `/Scene_<id>`. Pass `scene_prefix="/World/Scene_0"`, or build the stage at `/Scene_0` — check `prefixed()` in that module and pick whichever matches it.
  - BDDL: `~/libero_assets/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl`.
- Produces: `parse_bddl(text) -> dict(regions={name: {"target": str, "ranges": [[x0, y0, x1, y1], …], "yaw": [lo, hi] | None}}, objects={instance: category}, fixtures={instance: category}, init=[[pred, *args], …])` and `to_replicator_yaml(bddl, rest_z, body_prims) -> dict`.

- [ ] **Step 1: Write the BDDL parser with its self-check**

```python
import re

def sexpr(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    return stack[0][0]


def section(tree, key):
    return next((s[1:] for s in tree if isinstance(s, list) and s and s[0] == key), [])


def typed(items):
    """`a b - cat c - cat2` -> {a: cat, b: cat, c: cat2}"""
    out, pending = {}, []
    it = iter(items)
    for tok in it:
        if tok == "-":
            cat = next(it)
            out.update({p: cat for p in pending})
            pending = []
        else:
            pending.append(tok)
    return out


def parse_bddl(text):
    tree = sexpr(text)
    regions = {}
    for reg in section(tree, ":regions"):
        name, body = reg[0], {k[0]: k[1:] for k in reg[1:]}
        target = body[":target"][0]
        ranges = [[float(x) for x in r] for r in body.get(":ranges", [[]])[0]] if ":ranges" in body else []
        yaw = [float(x) for x in body[":yaw_rotation"][0][0]] if ":yaw_rotation" in body else None
        regions[f"{target}_{name}"] = {"target": target, "ranges": ranges, "yaw": yaw}
    return {"regions": regions,
            "fixtures": typed(section(tree, ":fixtures")),
            "objects": typed(section(tree, ":objects")),
            "init": [[t.lower() for t in fact] for fact in section(tree, ":init")]}
```

Self-check (`--selftest`), run on the real file:
- `parse_bddl` of the next-to-the-plate BDDL gives `regions["main_table_plate_region"]["ranges"] == [[0.05, 0.19, 0.07, 0.21]]`;
- `regions["main_table_cabinet_region"]["yaw"] == [2.6613777765410678, 2.7242096296128633]`;
- `objects["akita_black_bowl_2"] == "akita_black_bowl"`;
- `["on", "plate_1", "main_table_plate_region"] in init`;
- `regions["wooden_cabinet_1_top_region"]["ranges"] == []`.

Note that the parser lowercases init facts; region and instance names are lowercase already.

- [ ] **Step 2: Generate the Replicator YAML**

`to_replicator_yaml`: for every `on <instance> main_table_<region>` init fact, add one prim group plus `modify.pose`:
- `position`: `distribution.uniform` with `lower: [x0, y0, z]`, `upper: [x1, y1, z]`, where `z = rest_z_above_table + 0.001`;
- `rotation`: `distribution.uniform` in degrees, `[0, 0, deg(yaw_lo)]..[0, 0, deg(yaw_hi)]` when the region has `:yaw_rotation`, otherwise the constant `[0, 0, 0]`;
- `write_to_usd: true`;
- a `name` per distribution (`<instance>_position`, `<instance>_rotation`) so `fire()` records the samples.

Write the `path_pattern` as the scene-relative body prim path from `build_scene`, anchored with `$`. Then:
- follow the layout of `guide_tasks/cube_stack/config/randomize.yaml` (`randomizer.register` groups, one `trigger.on_custom_event`, `with.<trigger>` invoking the randomizers);
- put **at most 6 `with` blocks per registered randomizer and per trigger** (registration time is exponential in blocks per trigger; see `docs/design/replicator-front-end.md`, Part 2). Use several triggers on the same `event_name` as `reset.yaml` does;
- list the facts that are not `on <instance> main_table_<region>` under `unsupported` in the result JSON, and count them across all 10 Spatial BDDLs in that folder.

Save the YAML to `docs/design/spike_libero_randomize.yaml` with `yaml.safe_dump(..., sort_keys=False)`.

- [ ] **Step 3: Run placement in Isaac**

Start SimulationApp headless, `build_scene(...)`, `replicator_guide.build(yaml, prefix)`. Then for `seed in range(10)`: `fire(state, seed=seed)`, step 1 s of physics, read each placed body's PhysX pose in the table frame, and check:
- x/y inside the region rectangle (±1 mm) right after `fire`;
- drift after 1 s < 1 cm (no interpenetration push-out);
- for fixtures, the yaw inside the drawn range (±0.5°).

Render `place_seed<N>.png` for seeds 0–2. After seed 9, re-run Task 2's drawer check on `wooden_cabinet_1` (does the articulation still open after a Replicator pose write?).

Write `docs/design/spike_libero_place.json`: per seed and instance `{in_region, drift_m, yaw_ok}`, plus `"fire_ms"` (mean wall time of `fire`), `"unsupported"` (this task) and `"unsupported_all_spatial"` (counts by predicate shape across the 10 files).

- [ ] **Step 4: Commit**

```bash
git add guide_core/scripts/spike_libero_place.py docs/design/spike_libero_randomize.yaml docs/design/spike_libero_place.json
git commit -m "spike(libero): place LIBERO-Spatial objects from BDDL regions through GUIDE's Replicator front-end"
```

---

### Task 4: Findings and recommendation

**Files:**
- Create: `docs/design/libero-asset-spike.md`
- Modify: `docs/superpowers/plans/2026-10-19-libero-isaac-roadmap.md` (section S1 checkboxes + one-paragraph result; status table row S1)

**Interfaces:**
- Consumes: the three result JSON files and the renders' paths.

- [ ] **Step 1: Write the findings**

Follow the shape of `docs/design/replicator-front-end.md`:
1. what was built;
2. a results table, one row per check, every number taken from the JSON files (cite the file);
3. what the spike settled, including wrong turns;
4. gaps;
5. a recommendation for S1, with one of three verdicts:
   - **adopt** `mujoco_usd_converter` as is;
   - **adopt with fixes** (list them);
   - **reject** (say why, and name the fallback: Isaac MJCF importer, RobotLearningLab USD, Lightwheel assets).

State facts only as measured; mark anything not measured as not measured.

- [ ] **Step 2: Update the roadmap S1 section**

Tick the S1 checkboxes that the results support. Add a one-paragraph result with a link to `docs/design/libero-asset-spike.md`. In the status table, set S1 to `spike done — <verdict>`.

- [ ] **Step 3: Commit**

```bash
git add docs/design/libero-asset-spike.md docs/superpowers/plans/2026-10-19-libero-isaac-roadmap.md
git commit -m "docs(libero): S1 asset spike findings"
```
