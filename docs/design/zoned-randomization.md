# Design: Zoned (grid-based) position randomization

Branch: `feat/zoned-randomization`
Companion: [`zoned-randomization-todo.md`](./zoned-randomization-todo.md) (future work + limitations)

## 1. Context & goal

The randomization system draws each object's pose from a continuous distribution
(`PoseDist` = `UniformVec` position + `AxisAngle` orientation). For VLA training we want to
**stratify** a target object's position range into a regular grid of cells ("zones") and
generate a *chosen number of demonstrations per zone* — from an easy "10 in every zone"
down to per-zone micromanagement ("4 in zone 2, 10 in zone 16"). This gives controlled,
countable coverage of the state space instead of unstructured sampling.

Scope:
1. A **grid/zone model** over a position-randomization region (2D x,y; default 0.1 m cells).
2. **Zoned draws**: the *target* object samples within a requested zone's cell; other
   ("disturbance") objects keep sampling their full range.
3. **Zone-targeted generation API**: `Randomize(use_zone, zone)` + a zoned batch on
   `Demonstration(all_zones, zones[], counts[])`, plus a Python wrapper.
4. Per-episode **zone recorded** in the dataset sidecar (required).
5. **`block_bin`** as the hello-world zoned task: bins fixed, the color-selected cube zoned
   in front of the robot, color/side still vary — a miniature distribution for a first VLA agent.

Design bias (per request): favor the **cleanest structure for future development** over
preserving current code shape. The grid is a first-class, pure, testable type; the draw
pipeline threads a zone; distributions stay pure.

Locked decisions: explicit `bool use_zone` on `Randomize.srv`; `Demonstration` carries
`zones/counts` (a zone of **`-1`** is the all-zones signal — the originally planned separate
`all_zones` flag proved redundant); grid reuses the `position.random` matrix in x,y; a `grid:`
block marks the candidate; zoning lands in the existing `block_bin` (§7); **at most one grid
per scene** for now (see §11).

Note on the sentinels: `use_zone` stays explicit at the `Randomize` layer. At the
`Demonstration` layer an empty `zones` means free (unstratified) draws and a zone of `-1`
means "every zone"; `zone_plan` expands `-1` before anything reaches the draw. Inside the
draw, `draw_instructions` gates on `zone >= 0`, so a `None`/negative zone falls through to
the full range (pinned by `test_negative_zone_is_free`).

## 2. The grid/zone model

New module **`guide_core/types/randomization/grid.py`** — pure NumPy, no Isaac/ROS.

```python
@dataclass(frozen=True, slots=True)
class Grid:
    low: np.ndarray        # (3,) region min  (from position.random low)
    high: np.ndarray       # (3,) region max  (from position.random high)
    resolution: float      # cell size in metres (default 0.1)

    #  ncols = max(1, ceil((high[0]-low[0]) / r))   # x -> columns
    #  nrows = max(1, ceil((high[1]-low[1]) / r))   # y -> rows
    #  num_zones = ncols * nrows
    @property
    def ncols(self) -> int: ...
    @property
    def nrows(self) -> int: ...
    @property
    def num_zones(self) -> int: ...

    def cell_bounds(self, zone: int) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) of the cell for `zone`, clamped to the region.
        Row-major, 0-indexed from the (min-x, min-y) corner:
            col = zone % ncols ;  row = zone // ncols
        x,y restricted to the cell; z left as the region's (usually fixed) z range.
        Raises for zone < 0 or zone >= num_zones."""

    def zone_of(self, point) -> int:
        """Inverse: which zone an (x,y) falls in (clamped). Lets us tag a free draw
        with its containing zone in the record."""

    def restrict(self, pose_dist, zone) -> PoseDist:
        """PoseDist(UniformVec(cell_bounds(zone)), pose_dist.orientation) — keeps
        apply.py free of distribution internals."""
```

- **Numbering:** zone 0 = corner nearest (min-x, min-y); `col` advances with x, then `row`
  with y (row-major). `num_zones` is queryable for validation and API bounds.
- **Partial edge cells** (region not an exact multiple of `resolution`) are clamped to the
  region — no out-of-range placement.
- **Reproducibility:** the zone is an input, not a draw. Within-cell position is drawn from
  `UniformVec(cell_bounds(zone))` on the seeded RNG, so `(seed, zone) → pose` is deterministic;
  injection replays the stored pose verbatim (order-independent — keyed by name).

## 3. Config schema

A `grid:` block under an instruction's `pose.position` marks it as the **grid candidate**.
Which *prim* inside it is actually zoned is decided at runtime by the scene (§4), so the
disturbances in the same instruction stay free.

```yaml
- cmd: set_local_poses
  kwargs:
    prim_path: '/blocks/*'          # all four colored blocks
    pose:
      position:
        random: [[0.30, 0.60], [-0.20, 0.20], [0.025, 0.025]]   # region in front
        grid: { enabled: true, resolution: 0.1 }   # <-- grid candidate
      orientation:
        random: { axis: [0,0,1], angle: 180 }
- cmd: set_local_poses
  kwargs:
    prim_path: '/bin_0'
    pose:
      position: { value: [0.25, -0.4, 0.09] }      # FIXED bin (no random)
```

- `grid.resolution` defaults to 0.1 m when omitted.
- **Validation:** at most **one** `grid:`-enabled instruction per scene (see §11). More than
  one is rejected at parse time with a clear error.

## 4. Randomization pipeline integration

### 4.1 Dynamic target selection (the key structural change)

The zoned object is chosen at runtime: in `block_bin` the target is the
**color-selected block**, which is only known after the scene's discrete draw. So:

- **Reorder `randomize()`** so the scene's `randomize_preprocess(randomizer)` (discrete
  color/side draws, `self.task`) runs **before** the pose draws. This is safe for injection
  reproduction (replay is keyed by draw *name*, not order); only fresh sampling sequences
  change, which is acceptable on this branch.
- The scene exposes an optional **`zone_target() -> Optional[str]`** returning the prim to
  zone (e.g. `/Scene_0/blocks/{color}_block`), or `None` when the scene has no zoning.
- **`apply.draw_instructions(..., zone=None, zone_target=None)`**: for the single grid
  instruction, in the per-prim expansion, the prim equal to `zone_target` is drawn from
  `grid.restrict(base, zone)`; every other prim (and every non-grid instruction) is drawn
  from its **free** range. So the color-picked block lands in the zone and the other three
  blocks are free disturbances.

```python
grid = instruction.get("grid")
base = instruction["pose_dist"]
for prim in prims:                       # per-prim expansion (existing)
    zoned = grid is not None and zone is not None and zone >= 0 and prim == zone_target
    dist = grid.restrict(base, zone) if zoned else base
    kwargs_pose.append(pose_builder(randomizer.draw(prim, dist)))
```

### 4.2 Wiring

- `grid.py` adds `grid_from_yaml(position_spec) -> Optional[Grid]`.
- `SceneOrchestrator.parse_instruction`: attach `instruction["grid"] = grid_from_yaml(...)`;
  enforce the single-grid rule across the instruction list.
- `randomize(zone=None)` resolves `zone_target()` after preprocess and passes both into
  `draw_instructions`.
- Distributions stay pure; the draw is recorded as today (per-prim keys). The zone is
  captured separately (§6), so injection is unchanged.

Free/disturbance draws (`zone is None`/`<0`, or a non-target prim) are exactly current
behaviour — the feature is additive.

## 5. Messages & generation API

**`guide_msgs/srv/Randomize.srv`** (shared with Reset; Reset ignores the fields):
```
uint8 id
bool use_zone   # false -> free (full range, current behaviour); true -> use `zone`
int32 zone      # cell index when use_zone is true
---
string message
bool success
```
`use_zone` defaults false → backward compatible; no caller can silently target zone 0.

**`guide_msgs/srv/Demonstration.srv`**:
```
string path
int32[] zones           # empty -> free draws; -1 -> ALL zones; else the explicit zones to sample
int32[] counts          # per-zone counts (parallel with zones); counts[0] is reused when shorter
---
bool success
string message
```
Semantics:
- `zones` **empty** → `counts[0]` free (unstratified) demos.
- `zones` contains **`-1`** → that count in every zone (a scene with no grid: free demos).
- `zones` **explicit** → `counts[i]` demos in `zones[i]`.

**Not** request-validated, deliberately: `zone_plan` passes explicit zones through, and an
out-of-range *positive* zone fails at draw time when
`Grid.cell_bounds` raises with an explicit `zone N out of range [0, M)`. A short `counts`
reuses `counts[0]` rather than erroring. Adding request-time bounds checking would need
`num_zones` at the service layer, which is scene-side state — see the zone-querying service
in the TODO doc.

**Generation flow** (`block_bin/solve_task.py`):
- Build a per-episode zone plan via `zone_plan()` (in `grid.py`, so future zoned tasks reuse
  it): `zones=[-1]` → `[0]*counts[0] + [1]*counts[0] + …`; `zones=[]` → free draws;
  `zones=[2,16],counts=[4,10] -> [2,2,2,2,16,…]`.
- Each episode: `Randomize.Request(id, use_zone=True, zone=z)` → solve → record.

**Python wrapper** (task-side helpers):
- `zoned_request({2: 4, 16: 10}, path="")` → `zones=[2,16], counts=[4,10]`.
- `all_zones_request(count=10, path="")` → `zones=[-1], counts=[count]`.

## 6. Recording the zone (required)

Every episode's metadata must carry its zone:
- Add `zone: Optional[int] = None` to **`SceneContext`** (`to_dict`/`from_dict`);
  `randomize(zone)` stores it on `_last_context`.
- **`meta/guide_episodes.jsonl`** per-episode line gains: `zone` (the target's requested zone,
  `-1`/`null` for a free draw) and `zone_cell` (`[low_xy, high_xy]` bounds, present only when
  `zone >= 0`).
  > The originally planned `zone_of(target_xy)` **containing-zone label for free draws is not
  > implemented** — free episodes record `zone: -1` with no cell. Deferred, not dropped: see
  > "Containing-zone label for free draws" in the TODO doc.
- **`meta/guide_info.json`** gains a `grid` block: `region` (low/high), `resolution`,
  `ncols`, `nrows`, `num_zones` — the dataset is self-describing about its zone layout.

## 7. `block_bin` gains a zone grid

Zoning lands in the existing `guide_tasks/block_bin/` rather than a parallel package,
because **the grid is inert unless a request asks for a zone** —
`draw_instructions` only restricts to a cell when `zone >= 0`, and `_cmd_randomize_scene`
passes `None` unless `Randomize.use_zone` is set. A gridded task therefore still generates
ordinary free demonstrations, so the split bought nothing and cost a duplicate copy of the
task (including 36 MB of byte-identical USD/STL assets).

Changes to `block_bin`:
- **`config/init.yaml`**: `publish_camera_topics: false` — the recorder captures images via
  annotators, and publishing the large reliable image streams starves service replies (§6).
- **`config/randomize.yaml`**: `/blocks/*` position `grid`-enabled over a region in front of
  the robot (0.1 m cells → 5×4 = 20 zones); yaw still randomized. **Bins fixed** (value only)
  so the place target does not vary independently of the cube's zone — the previous free-bin
  randomization is retained commented-out for anyone who wants the original task back.
- **`scene.py`**: implements `zone_target()` → `/Scene_{id}/blocks/{color}_block`, so the
  color-selected block is the zoned one and the other three stay free disturbances.
- **`solve_task.py`**: zoned generation entry point; `scene_num_zones()` reads the grid from
  the package's own `randomize.yaml` so "all zones" expands to the right count.

Hello-world usage: `{zones: [-1], counts: [10]}` (10 per zone) or a few chosen zones → a small,
countable dataset covering the front region → first VLA agent.

## 8. File-by-file changes

New:
- `guide_core/guide_core/types/randomization/grid.py` — `Grid`, `grid_from_yaml`,
  `Grid.restrict`, plus `single_grid` (scene validation) and `zone_plan` (request expansion).
- `guide_core/test/test_grid.py` — grid math + zoned-draw + dynamic-target unit tests.
- `docs/design/zoned-randomization-todo.md` — future work + limitations (this PR seeds it).

Modified:
- `types/randomization/__init__.py` — export `Grid`, `grid_from_yaml`, `single_grid`, `zone_plan`.
- `types/randomization/apply.py` — `draw_instructions(..., zone, zone_target)` zoned branch.
- `types/scene_context.py` — `zone` field.
- `scene/scene_orchestrator.py` — attach `grid`; `self._grid = single_grid(...)` validation;
  reorder `randomize()` (preprocess → `zone_target()` → pose draws); `randomize(zone=…)`.
- `guide_tasks/block_bin/**` — grid config, `zone_target()`, zoned generation (§7).
- `scene/scene_manager.py` — thread `zone` through `randomize_preprocess`; record `zone`/`zone_cell`.
- `core/commands/_cmd_simulator.py` — `_cmd_randomize_scene(use_zone, zone)`.
- `core/guide_simulator.py` / `ros/guide_ros.py` — pass `use_zone`/`zone` from `Randomize.Request`.
- `guide_msgs/srv/Randomize.srv`, `Demonstration.srv` — new fields (**rebuild guide_msgs**).
- `scene/scene_recorder.py` — `grid` block in `guide_info.json`; `zone`/`zone_cell` in episode lines.

## 9. Verification

- **Unit (no Isaac):**
  - `test_grid.py`: `ncols/nrows/num_zones`; `cell_bounds` tiles the region with no gaps/overlap
    and clamps partial edges; `zone_of(cell_center) == zone` round-trips; `restrict` samples
    **inside** the requested cell over many seeds; out-of-range zone raises.
  - `apply` zoned branch: only `zone_target` uses the cell, other prims free; non-grid
    instruction ignores `zone`; `zone<0`/`None` == free; injection reproduces.
  - `Demonstration` zone-plan expansion (all_zones / zones+counts / free).
  - Single-grid validation rejects a second `grid:` instruction.
- **End-to-end (after rebuild):** run `block_bin`, `all_zones_request(10)` and a
  `{2:4,16:10}` batch; confirm the color-picked cube lands in the requested cells, the other
  blocks are free, `guide_episodes.jsonl` records each episode's `zone`/`zone_cell`, and
  `guide_info.json` carries the grid layout.

## 10. Suggested build order

1. `Grid` + `grid_from_yaml` + `test_grid.py` (Isaac-free, immediately runnable).
2. `apply.draw_instructions` zoned branch + tests; `SceneContext.zone`.
3. `randomize()` reorder + `zone_target()` + single-grid validation.
4. Messages (`use_zone`, `all_zones`, `zones/counts`) + `guide_ros`/simulator wiring + rebuild.
5. Sidecar zone metadata.
6. `block_bin` grid config + `zone_target()` + e2e.

## 11. Limitations (current, deliberate)

- **One grid per scene.** Multiple grid candidates would make zone numbering ambiguous (whose
  grid does `zone=16` mean?). Enforced at parse time; multi-grid design is deferred — see the
  TODO doc. This also means a *single* zoned target per episode.
- **2D grid (x,y) only**; z comes from the region's (usually fixed) z range.
- **Uniform, axis-aligned cells** from the `position.random` box; no rotated/weighted zones.
- **Grid on a wildcard instruction zones exactly one prim** (`zone_target`); the rest are free.
  A grid whose instruction has *no* matching `zone_target` falls back to a fully free draw.
- **Reorder side effect:** fresh (non-injected) sampling sequences for existing scenes shift
  because discrete draws now precede pose draws; recorded datasets still replay by name.
