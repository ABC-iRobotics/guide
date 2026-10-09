# Zoned randomization — TODO & known limitations

Companion to [`zoned-randomization.md`](./zoned-randomization.md). Tracks deferred work and
serious limitations of the grid/zone feature so they are not lost.

## Deferred features

### Multiple grids per scene (deliberately disabled for now)
Only **one** `grid:`-enabled instruction is allowed per scene, enforced at parse time. With
two or more grids (e.g. a grid for the picked object *and* a grid for the placing container),
a single `zone` index is ambiguous — whose grid does `zone=16` address? — and the generation
API (`zones/counts`) has no way to say "cube in zone 2, bin in zone 5".

Future design options to resolve before enabling multi-grid:
- **Named grids**: each grid instruction gets an id; the request carries a `{grid_name: zone}`
  map instead of a scalar `zone`. `Randomize`/`Demonstration` messages grow a structured field.
- **Composite zone index**: a single index decoded across grids (mixed-radix). Compact but
  opaque and brittle if a grid's resolution changes.
- Decision needed on how per-episode metadata records multiple zones (`zones: {name: idx}`).

Until then: `zone` is a single scalar addressing the one grid; a second `grid:` block is an error.

### Containing-zone label for free draws
Free (unstratified) episodes record `zone: -1` and no `zone_cell`. The original design called
for tagging them with `grid.zone_of(target_xy)` — the zone the target *happened* to land in —
so a mixed free/zoned dataset can still be analysed per-zone. Cheap to add: `_capture_episode_meta`
already has both the `Grid` and the target's recorded pose in `meta["main_object"]["pose"]`.
Left out of the initial implementation to keep the sidecar schema minimal; would need a schema
bump (`guide_meta_schema`) or a distinct key (e.g. `zone_landed`) so consumers can tell a
*requested* zone from an *observed* one.

### Other future work
- **3D / non-axis-aligned grids** (z stratification, rotated regions).
- **Weighted / curriculum zone sampling** (non-uniform per-zone counts driven by success rate).
- **Per-zone statistics** surfaced back (success/attempts per zone) for active data collection.
- **Zone querying service** so external clients can discover `num_zones` / layout without
  reading `guide_info.json`.

## Serious limitations (current)

- **One grid, one zoned target per episode** (see above).
- **2D grid (x,y) only**; z is taken from the region's (usually fixed) z range.
- **Uniform axis-aligned cells** derived from the `position.random` box; no rotated or
  variable-size zones.
- **Wildcard grid instruction zones exactly one prim** (the runtime-selected `zone_target`);
  all other matched prims are free. If no prim matches `zone_target`, the draw is fully free
  (no error) — a silent fallback to watch for.
- **`randomize()` reorder side effect**: discrete draws now run before pose draws, so
  fresh-sampling RNG sequences for *existing* scenes change (injection/replay is unaffected,
  since it is keyed by draw name). Datasets recorded before this change replay correctly.
- **Partial edge cells**: when the region isn't an exact multiple of `resolution`, boundary
  zones are smaller (clamped). Zone *count* uses `ceil`, so the last row/column may be thin.
- **Shared `Randomize.srv`**: Reset reuses this message and simply ignores `use_zone`/`zone`.
