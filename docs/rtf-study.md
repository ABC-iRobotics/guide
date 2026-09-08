# Why the GUIDE sim runs at RTF ~0.32

A mapping exercise, not a fix. Dated 2026-09-08, branch `perf/isaac-runtime-frame-budget`.

## The measurement

From `[runtime]` in the GUIDE log, steady state, headless, depth off:

```
51.3 ms/step + 0.0 ms/cmds (budget 16.7 ms), RTF 0.32
```

Three things this pins down before any hypothesis:

- **The whole frame is `_world.step()`.** `ms/cmds` is 0.0, so command handling, the ROS
  service layer and the scene-manager callbacks are free. Everything is physics + render.
- **The budget is 16.7 ms** (`step_freq: 60`). We are 3.1x over it.
- **`realtime: true` cannot be the cause.** That sleep only ever pads a frame that came in
  under budget; at 51 ms it computes to zero and never fires. (Same trap as the original
  4x investigation -- see `guide-isaac-frame-budget` memory.)

## Ruled out by experiment

Each of these was measured, not argued.

| Hypothesis | Test | Result |
|---|---|---|
| Kit GUI viewport is a 4th render target | `headless: True`, restart | 51.8 -> 51.4 ms. **No change.** |
| 3 depth AOVs doubled the frame | `depth: false` x3, restart | 51.4 -> 51.3 ms. **No change.** |
| Slow command handlers / service starvation | `ms/cmds` in the frame log | 0.0 ms. **Not a factor.** |
| The `time.sleep` in `run_loop` | Arithmetic: 51 ms > 16.7 ms budget | Never fires. **Not a factor.** |
| Camera render products at 60 Hz | Render gate to 10 Hz, restart | No measurable RTF change. |

That last row is the important negative. The entire camera-gating effort on this branch
(hydra-texture gating, `frameSkipCount`, H.264) bought **bandwidth**, not frame time:
~166 MB/s of raw rgb8 down to ~1.5 MB/s, which fixed service-reply starvation. It was
never going to move RTF, and it should be judged on that basis alone.

## CONFIRMED 2026-09-08: both workloads were on the weaker GPU

Setting `active_gpu: 1` (the A4000) and restarting, with nothing else changed:

```
before   51.4 ms/step   RTF 0.32     Isaac on A2000 (100% util)
after    29.2 ms/step   RTF 0.57     Isaac on A4000 (100% util, A2000 down to 40%)
```

1.76x, against a ~2.4x core-count ratio -- so most, though not all, of the ratio carries
over. Kit enumerates by PCI bus id, same as nvidia-smi. The remaining 29.2 ms against a
16.7 ms budget is now genuinely the scene: candidates 1, 2 and 4 below are what is left,
and they should be re-measured on the A4000 because every earlier reading was taken on a
saturated card.

### But it does NOT speed up an evaluation

Same 3-episode zone-7 eval, before and after, timed:

```
                    idle             during eval          wall/episode (60s timeout)
A2000   51.4 ms  RTF 0.32    52.5-58.9 ms  RTF 0.28-0.32   215.1 s mean (3)
A4000   29.2 ms  RTF 0.57    56.4-73.7 ms  RTF 0.23-0.30   233.7 s mean (2)
```

Under load the two cards converge on ~57-70 ms despite a ~2.4x shader gap, and the
A4000 run was ~8.6% SLOWER in wall time. So:

- **Idle, the sim is GPU-render-bound** -- the card swap is worth 1.76x there.
- **Under eval load it is not.** Something card-independent sets a ~57 ms floor. The
  A2000 was already at its ceiling idle (100% util), so eval work barely moved it; the
  A4000 had headroom idle and the eval load consumed all of it.

That reframes everything below: the remaining candidates should be re-ranked around what
an evaluation adds that idling does not, and what could be card-independent --

**A. CPU / host-side stall. MEASURED AND RULED OUT.** Per-thread sample under eval load:
408% of 1600% available (16 cores, 25% utilised), busiest thread the main python one at
40-50%, everything else 20-40% spread evenly across carb task threads and TBB workers.
Nothing pegged, no single-thread ceiling. The main thread at 40-50% means it is blocked
about half the time -- waiting inside `_world.step()`, not computing. Concurrent RTF
72-74 ms/step, 0.22-0.23.

**A'. Per-pass overhead / GPU synchronisation. NEW LEADING CANDIDATE.** `utilization.gpu`
reads 100% on whichever card Isaac uses, but that counter is "fraction of time a kernel was
resident", not occupancy. A pipeline of many small serialised operations -- 4 render
products, their annotator copies, 3 NVENC encodes, the readbacks -- pins it at 100% while
leaving most SMs idle, and a 2.4x wider card then buys nothing. This is the only hypothesis
that explains all four observations together: A4000 wins 1.76x idle (few passes,
throughput-bound), both cards converge at 57-74 ms under load (many passes, latency-bound),
CPU is 25% with the main thread half-blocked, and headless/depth-off changed nothing
because neither removed a *pass*, only pixels.

Worth testing: `rep.create.render_product_tiled` renders several cameras into ONE product.
Replicator's own docstring -- "most performant when rendering a large number of sensors at
low resolution" -- describes this case exactly, and it would collapse three passes and
three readbacks into one. It is a real change to `create_render_products`, not a config
flag, so measure the pass count first (candidate C) before building it.

**B. PhysX with an active scene.** Idle is a static stage; an eval has a moving
articulation, contacts, and randomised blocks. `enable_gpu_dynamics(True)` -- but which
device PhysX picks, and whether it is the same one as the renderer, is unverified.

**C. GPU->CPU readback.** The recorder's annotators and the NVENC encode both move data
host-ward. Bandwidth-bound work does not scale with shader count.

The original hypothesis and its evidence follow.

## Leading hypothesis: both workloads are on the weaker GPU

Measured during a live eval:

```
GPU 0  NVIDIA RTX A2000 12GB   100 % util   4920 MiB used
GPU 1  NVIDIA RTX A4000         45 % util    607 MiB used

pid 12245 (Isaac/GUIDE)   1835 MiB on A2000    288 MiB on A4000
pid 13065 (eval policy)   1428 MiB on A2000    160 MiB on A4000
```

The A2000 is **saturated at 100%** while the A4000 — the substantially faster card — sits
at 45% holding 600 MB. Both the renderer and the policy have their primary allocation on
the A2000.

### How it happened: two different GPU orderings

- `nvidia-smi` enumerates by **PCI bus id**: index 0 = A2000, index 1 = A4000.
- CUDA (and therefore torch) defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST`:
  `cuda:0` = A4000, `cuda:1` = A2000.

Two consequences, both live right now:

1. `guide_core/config/init.yaml` has `active_gpu: 0` with the comment *"Force single-GPU
   rendering on GPU 0 (the A4000)"*. **The comment is wrong** — in Kit's enumeration GPU 0
   is the A2000, and the utilisation figures confirm Isaac is rendering there.
2. `eval_smolvla.sh` sets `DEVICE=cuda:1` with the comment *"Keep inference off the GPU
   Isaac is rendering on"*. In torch's ordering `cuda:1` is the A2000 — the log line reads
   `Inference on cuda:1 = NVIDIA RTX A2000 12GB`. **It lands on the same card as Isaac.**

So a setting intended to separate them actually collides them, on the slower of the two.

### Why this fits the evidence better than anything else

- Explains why headless and depth changed nothing: the card is saturated by the base
  render regardless of how many extra targets are removed or gated.
- Explains the magnitude. A4000 is roughly 2.3-2.5x the A2000 in shader throughput
  (6144 vs 3328 CUDA cores). 51 ms / ~2.4 lands near 21 ms — RTF ~0.8.
- Explains the ~26.7 ms / RTF 0.62 seen in a log from ~17 days earlier: plausibly a run
  before the GPU pinning, or before inference shared the card.

### The caveat

`multi_gpu: False` and `active_gpu: 0` were set deliberately, for a documented reason:
Isaac's multi-GPU renderer deadlocks on this box ("Failed to begin render graph ...
semaphore timed out") because the two cards are mismatched with no CUDA peer access.
So the constraint is real; only the *choice of card* looks wrong.

## Remaining candidates, ranked

**1. Scene / asset cost on the pinned card.** Even alone on the right GPU, `block_bin` may
not hit 16.7 ms. The original 4x comparison was against `isaac-sim.sh` rendering one
viewport of a near-empty stage — not the same workload. Untested.

**2. GPU physics.** `enable_gpu_dynamics(True)` at `physics_freq: 120` = 2 substeps per
frame, sharing the saturated card with the renderer. Contribution unknown; the frame log
does not separate physics from render.

**3. Three camera render products, unconditionally.** They exist whenever `dataset.images`
lists cameras, independent of `publish_camera_topics`. The gate reduces *when* they render,
but `set_updates_enabled` stopping the RTX pass was never independently verified — the flat
RTF is weak evidence it does not.

**4. RaytracedLighting settings.** `anti_aliasing: 0` is already minimal, but reflections,
shadow quality and sample counts are untouched Kit defaults.

**5. `isaacsim.exp.base` vs `exp.full` renderer defaults.** Different Kit apps ship
different RTX settings; only `rateLimitEnabled` has been compared.

## What to measure next, in order

1. **Move Isaac to the A4000.** Set `active_gpu: 1` and confirm with `nvidia-smi` that the
   A4000 carries Isaac's allocation. Read one `[runtime]` line. This is the single highest
   expected-value experiment on the board.
2. **Move inference to the A2000 explicitly** — with Isaac on the A4000, `--device cuda:1`
   becomes correct rather than accidental. Verify against the script's own log line.
3. **Baseline the empty stage.** Launch GUIDE with a scene carrying no cameras and no
   assets; that separates "this scene is expensive" from "this runtime is expensive".
4. **Split physics from render** in the frame log — time `_world.step()` around a
   `render=False` step — to size candidate 2.
5. Only then touch RTX quality settings.

## Correction log

Three hypotheses I ranked confidently and were falsified by measurement: the GUI viewport,
the depth AOVs, and (earlier) "7 render targets" — which was 4, because
`rep.create.render_product` reuses an existing product for the same camera and resolution.
The pattern is reasoning ahead of the evidence; the GPU-assignment hypothesis above is
stated with the utilisation figures attached for that reason.


## Possible fixes, explored

Grounding fact, from `isaacsim.exp.base.kit`:

```
omni.replicator.asyncRendering = false   # Async rendering must be disabled for SDG
asyncRendering = false
asyncRenderingLowLatency = false
```

Kit disables async rendering *because* Synthetic Data Generation needs it off: annotators
and writers must read deterministic data for the frame that was just rendered. So a
synchronous render-then-readback stall every frame is architectural for any stage with
replicator annotators attached -- not a setting to flip. That reframes the fixes: the
lever is how many synchronous passes happen and how often, not how fast each one runs.

Ranked by expected value against cost:

**1. Count the passes first (diagnostic, ~30 min).** Everything below is sized by one
number nobody has measured: how many render products exist during an eval, and whether
`set_updates_enabled(False)` actually stops their passes. We know the ROS *writers* keep
firing when the texture is paused (they are ON_DEMAND, per `app.update()`); if the SDG
sync survives the gate too, that alone explains why gating changed nothing. Query
`rep.functional.get.renderproduct()` in-sim and log the count. Do this before writing any
of the below.

**2. Tiled render product (real change, highest ceiling).**
`rep.create.render_product_tiled` renders several cameras into ONE product -- its docstring
says it is "most performant when rendering a large number of sensors at low resolution",
which is exactly three 640x480 cameras. Collapses 3 passes and 3 readbacks into 1.
Costs: the annotator returns one tiled image that `record_step` must slice per camera, and
the ROS 2 camera graphs cannot share it (`IsaacCreateRenderProduct` builds its own), so
this helps dataset generation more than evaluation. Constraint from the source: only one
tiled resolution per session.

**3. Detach annotators when not recording (moderate).** Between episodes and during
evaluation the recorder's annotators are still attached and still synchronising, for frames
nobody reads. `set_updates_enabled` pauses the texture but demonstrably did not buy frame
time; actually detaching (`annotator.detach([rp])`) removes the SDG node from the graph.
Reversible per episode via the existing RECORDING state transitions.

**4. Verify RTSubframes (one line, unknown payoff).** Replicator renders N subframes per
capture; a test config in this build passes `--/omni/replicator/RTSubframes=1` with the
comment "capturing at every frame", implying the default may be higher. GUIDE never sets
it. If it is >1 in RaytracedLighting, every capture is costing N renders. Read it at
runtime before assuming either way.

**5. Fewer or smaller camera products (cheap, but couples to the dataset).** Resolution and
camera count are fixed by what the checkpoints trained on, so this is only available if a
policy uses a subset -- `eval_policy*.py` already supports a camera mapping for exactly
that case.

**6. exp.full vs exp.base renderer defaults (unknown).** Only `rateLimitEnabled` has ever
been compared between the two Kit apps. They ship different RTX defaults, and the original
"4x faster under isaac-sim.sh" observation was never re-examined after the render-target
count turned out to be 4 rather than 7.

### Not worth pursuing

- **Async rendering** -- disabled by Kit deliberately for SDG; turning it on would break
  annotator determinism, which is the dataset.
- **More GPU** -- measured: card-independent under load.
- **CPU work** -- measured: 25% of 16 cores, nothing pegged.
- **Rate limiting / the loop sleep** -- never fires above budget.
