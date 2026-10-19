# LIBERO in GUIDE (Isaac Sim) — Roadmap

> **For agentic workers:** this is a *roadmap*, not a task-level plan. Each main step below gets
> its own brainstorm → spec (`docs/superpowers/specs/`) → plan (`docs/superpowers/plans/`) when it
> starts; spikes end in a recommendation, not kept code. Track progress with the checkboxes and
> the status table. Update this file whenever a step changes state or a decision is made.

**Goal:** an automated pipeline that turns LIBERO (BDDL tasks + recorded demos) into
higher-quality pre-training data in Isaac Sim through GUIDE — photorealistic rendering, more
modalities (depth, segmentation, force/contact), better physics — and an experiment that shows
whether that data boosts a pre-trained policy.

**Hypothesis:** either (H1) *the same demonstrations* as LIBERO's, re-simulated in Isaac, or (H2)
*the same semantics* (tasks, scenes, language) with new GUIDE-generated demonstrations, improves
pre-trained model performance.

**Architecture:** LIBERO assets → USD; a compiler from BDDL to a GUIDE task (scene USD +
`randomize`/`reset`/`success` YAML + prompts); demonstrations by replay (H1) or by GUIDE-EX
solvers generated from the goal predicates (H2); the recorder writes LeRobot v3 with the extra
modalities; `guide_dataset_tools` exports the LIBERO layout; training and a cross-domain
evaluation decide.

**Tech stack:** Isaac Sim 6.0.1 (pip, `.venv`), ROS 2 Jazzy, MoveIt 2, GUIDE 2.0.0, Replicator
YAML, `mujoco` 3.8.0 + `mujoco_usd_converter` 0.2.0 (already in `.venv`), lerobot 0.6.0
(`envs/libero.py` needs the `hf-libero` extra, not installed), SmolVLA first, pi0.5-class later.

**Spec:** none yet — this roadmap is the decomposition agreed in the 2026-10-19 brainstorm.
Each step's spec links back here.

**Live tracker:** [LIBERO Isaac Roadmap](https://claude.ai/artifact/HC3YKe5CcfYdCf4wbTLVzE), a
claude.ai page (private, owner's account) with the steps, ongoing work, problems, open questions,
decisions and a change log. It is updated whenever any of these change; this file keeps the
step contents and candidate solutions.

## Evaluation design (user decision, 2026-10-19)

Round 1 is a 2 × 2 cross-domain test with the same model recipe:

| | eval in MuJoCo LIBERO | eval in GUIDE (Isaac) LIBERO |
|---|---|---|
| **trained on original LIBERO** | baseline (published setting) | cross-domain |
| **trained on GUIDE-LIBERO** | cross-domain | in-domain |

- One training source wins in **both** domains → clear winner.
- Each wins only **in its own domain** → round 2 on the real robot decides.
- Embodiments expand later beyond Panda: FR3, UR5e, and others.

## Global constraints

- Commits authored by the user only: no `Co-Authored-By`, no Claude trailers. Never open PRs; push only when asked.
- Robot-agnostic GUIDE layers stay free of Isaac/sim-specific code; no container-only GUIDE code.
- GUIDE-EX nodes are universal and reusable; task packages hold only the scene + `solve_task`; tooling lives beside guide (like `block_bin_eval`, `guide_dataset_tools`).
- Datasets are generated into `~/dataset/<task>/`, never a job tmp directory.
- `graphify-out/` is never committed.
- Python installs into `.venv` with `-c $PINS` for anything torch-dependent (Isaac 6.0.1 pins).
- LeRobot dataset v3.0. HF uploads go to the `ABC-iRobotics` org via the `hf` CLI, with the `v3.0` tag.

## Review focus (what can silently invalidate the experiment)

1. **Embodiment confound:** if round 1 changes the robot, cameras or controller together with the simulator, the 2 × 2 measures the embodiment, not data quality. Round 1 keeps LIBERO's Panda, camera poses, 20 Hz and OSC-delta action.
2. **Action/state convention mismatch:** LIBERO actions are normalized OSC commands in [−1, 1] (×0.05 m, ×0.5 rad) with a binary gripper (−1 open, +1 close). GUIDE's current "LIBERO layout" (`--eef-delta-action`) is a *metric* delta with an *absolute* gripper width. A mismatch trains fine and fails at eval. Check against the checkpoint normaliser, not constants (`guide-libero-state-frame` lesson).
3. **Success-predicate drift:** GUIDE's `On`/`In`/`Open` must reproduce LIBERO's tests exactly (On = contact + above + xy < 3 cm; In = inside the site box; Open/Close = per-asset joint thresholds). Otherwise the two eval columns measure different things.
4. **Init-state mismatch:** both eval columns must start from LIBERO's 50 fixed initial states per task (`init_files/*.pruned_init`), mapped into the Isaac scene frame.
5. **Statistical noise:** 50 trials × 10 tasks per suite. Differences of a few percent sit inside seed variance, so use several training seeds and report confidence intervals.

## Status

| Step | State | Branch / artifact | Updated |
|---|---|---|---|
| S0 Workspace & tracking | in progress | `feat/libero` (guide), this file | 2026-10-19 |
| S1 Asset route (spike) | spike done — adopt with fixes | `spike/libero-assets`, [findings](../../design/libero-asset-spike.md) | 2026-10-19 |
| S2 Embodiments | not started | — | — |
| S2b Native motion planning + OSC | later | `feat/libero-native-motion` | 2026-10-19 |
| S3 BDDL → GUIDE compiler | not started | — | — |
| S4 Success predicates | not started | — | — |
| S5 Demonstrations (H1 replay, H2 solvers) | not started | — | — |
| S6 Modalities (force/contact) | not started | — | — |
| S7 Dataset export | not started | — | — |
| S8 Round 1: 2 × 2 cross-domain eval | not started | — | — |
| S9 Round 2: real robot | conditional | — | — |
| S10 Embodiment expansion (FR3, UR5e, …) | not started | — | — |

---

## S0 — Workspace & tracking

- [x] Branch `feat/libero` from local `dev` (834bb48, = content of `origin/main`, GUIDE 2.0.0), checked out in `~/ros2_ws/src/guide` (no worktrees, user decision 2026-10-19)
- [x] Live tracker page (link above)
- [x] This roadmap
- [x] Memory (2026-10-19):
  - file memory `guide-libero-isaac`;
  - claude-mem was tried and removed the same day (its search backend was broken);
  - graphify graph of the guide repo in `graphify-out/` (git-ignored): 1,452 nodes, 2,993 edges, 100 labelled communities, 0 dangling edges; the live tracker is a node in it. Refresh after each step: AST extraction with an explicit `root=` (the repo), new docs through an agent that is given the real code ids, then `graphify-out/resolve_ids.py` on the merged extraction before the build.
- [x] `guide_dataset_tools` and `vla_training` stay outside git (user decision 2026-10-19), although S7/S8 change them.

## S1 — Asset route (spike, LIBERO-Spatial)

LIBERO-Spatial is the first slice: all 10 tasks share one object set (two `akita_black_bowl`,
`plate`, `glazed_rim_porcelain_ramekin`, `cookies`, `wooden_cabinet`, `flat_stove`), one skill
(bowl → plate) and 50 demos each (75–197 steps at 20 Hz). Verified in `~/Downloads/libero_spatial`.

Question: which route gives Isaac-loadable USD with sane colliders, working articulation
(cabinet drawers, stove knob), acceptable visuals, and a licence we can use?

| Option | For | Against |
|---|---|---|
| a. `mujoco_usd_converter` (in `.venv`) on LIBERO's MJCF | exact geometry + joints; already installed; handles `.msh` via MuJoCo | needs the LIBERO repo assets; MuJoCo materials, not photoreal |
| b. Isaac MJCF importer (`isaacsim.asset.importer.mjcf`) | Isaac-native physics schemas | same visual ceiling; per-asset tuning |
| c. RobotLearningLab `libero/USD` (Apache-2.0, HF) | possibly done already, includes replay HDF5s | robot, fidelity, completeness unknown |
| d. Lightwheel assets (LW-BenchHub / SDK) | "refreshed", higher visual quality, all 130 tasks | SDK access; asset licence unclear (their simready repo is CC BY-NC); geometry may differ from LIBERO |

Photoreal upgrade (materials/textures on top of exact geometry) is a separate knob. "Same
semantics" constrains geometry and poses, not appearance.

User decision (2026-10-19): probe option **a** (`mujoco_usd_converter`) on branch
`spike/libero-assets`, placing the objects from the BDDL regions with **Replicator**.

- [x] Probe a on the Spatial object set: load in Isaac 6.0.1, Replicator placement, drop test, drawer open/close, render
- [ ] Check licences of LIBERO's asset families (scanned, HOPE, TurboSquid, articulated)
- [x] Recommendation recorded here

**Result (2026-10-19): adopt with fixes.** All 6 Spatial asset types convert once LIBERO's `.msh`
meshes are turned into `.obj`; they load in Isaac 6.0.1, rest on the table, the cabinet drawers and
the stove knob reach their limits, textures are byte-identical, and a Replicator YAML generated
from the BDDL regions places movables (in region 50/50 with the bowl offset compensated). Fixes
needed in the patch step: merge the colliderless outer body, drop the world weld for movables,
hide group-0 geoms. Open: fixture placement during play (works only with the timeline stopped),
overlap rejection, 12 of 71 init facts in the Spatial BDDLs use predicate shapes the generator
does not express, and asset licences were not checked (hence the box above stays open). Details:
[`docs/design/libero-asset-spike.md`](../../design/libero-asset-spike.md).

## S2 — Embodiments

Round 1 needs **Panda** (LIBERO's robot). Later: FR3, UR5e and others.

- Panda: `franka_description` (jazzy fork) already has `fer` (= Panda); Isaac ships a Panda USD; MoveIt config for `fer` to check.
- UR5e: `ur_description` + `ur_moveit_config` are installed in `/opt/ros/jazzy`; gripper choice (Robotiq 2F-85) open.
- LIBERO's robot sits on a pedestal mount at a fixed offset from the table. Reproduce the base pose and both cameras (`agentview`, `robot0_eye_in_hand`) exactly for round 1.
- [ ] Panda in GUIDE: config, MoveIt, gripper, base pose matched to LIBERO
- [ ] An OSC-equivalent controller for **evaluation** (LIBERO's policy outputs OSC deltas at 20 Hz). Options: Isaac Lab differential-IK / OSC, MoveIt Servo, Pink (as `eval_policy_pink`).

## S2b — Native motion planning + OSC (branch `feat/libero-native-motion`, later)

User decision (2026-10-19): a feature branch off `feat/libero` that removes the ROS 2 planning
pipeline (MoveIt through `pymoveit2`/`irob_lerobot_ros`) and plans trajectories natively from
the USD. Operational space control (OSC) goes on top of the planner. Worked on later; it gets
its own brainstorm → spec → plan.

| Candidate | Notes |
|---|---|
| cuRobo (GPU motion generation; cuMotion is its Isaac ROS / MoveIt packaging) | world from the USD stage, robot from URDF/USD; fast batched planning; check Isaac 6.0.1 / torch 2.11 compatibility |
| PINK (Pinocchio differential IK, already used by `eval_policy_pink`) | robot from URDF; no collision-aware global planning by itself; light and runs on the real robot too |
| Isaac Lab controllers (differential IK, OSC) | OSC layer candidate |

Open question: GUIDE-EX solvers drive the real FR3 as well, and robot-agnostic layers must not
hold Isaac-specific code. Planning "from the USD" has to keep that boundary.

## S3 — BDDL → GUIDE task compiler

Input: a `.bddl` (+ optional fixed init states). Output: scene USD (asset references),
`randomize.yaml` / `reset.yaml` (Replicator), `success.yaml`, prompts.

| BDDL | GUIDE | Gap |
|---|---|---|
| `:fixtures`, `:objects` | prims in the scene USD | asset step (S1) |
| `On obj region` (table rectangle) | `get.prims` + `modify.pose`, `distribution.uniform` in the table frame | native |
| `:yaw_rotation` | `distribution.uniform` rotation (rad → deg) | native |
| `On obj obj` | none (`relative_to` drops scale) | `guide.*` node or `scene.py` |
| `In obj region` (fixture region) | Xform region in the fixture USD | `guide.*` node or parenting |
| `Open/Close/TurnOn/TurnOff` | `modify.attribute` on drive target | only with drives |
| non-overlap | `Scene.randomize` redraw (as cube_stack) | `scene.py` |
| fixed init states | instruction dialect `inject`; Replicator has none | biggest gap |
| `:language` | `Scene.task` | native |
| `:obj_of_interest` | `zone_target()`, tracked instance objects | native |

Replicator constraints from the spike: one `get.prims` + `modify` per distinct constant;
registration time exponential in `with` blocks per trigger (split over several triggers on one
event).

Parser options: LIBERO's own (`libero.libero.envs.bddl_utils`, adds the LIBERO dependency) or a
small s-expression parser (no dependency). Init-state injection options: a `guide.inject` node;
or bypass Replicator and write poses directly for fixed-state episodes.

- [ ] Spec + plan
- [ ] Spatial suite compiles; layouts match LIBERO's init states within tolerance

## S4 — Success predicates

Port LIBERO's object-state tests: `on`, `in`, `open`, `close`, `turnon`, `turnoff`, `up`.
`On`/`In` map onto executor commands (`get_world_poses`, `is_prim_contained`) plus
post-processing; `Open`/`TurnOn` need a joint-state query. Contact needs a PhysX contact report.

- [ ] Predicates with tests against LIBERO's own outcomes on replayed demo final states

## S5 — Demonstrations

**H1 (same demonstrations):**

| Option | Fidelity | Physics/force meaningful? |
|---|---|---|
| a. kinematic state replay (robot + objects set from LIBERO `states` per frame) | exact | no (teleported) |
| b. dynamic replay: Panda tracks the recorded joint/eef trajectory, objects simulated | diverges where MuJoCo/PhysX contacts differ | yes |
| c. b + success filter + retry from the same init state | as b, only successful episodes | yes |
| d. retarget to FR3/UR5e via IK, success-filtered (S10) | new embodiment | yes |

**H2 (same semantics, new demonstrations):**

| Option | Notes |
|---|---|
| a. GUIDE-EX solvers generated from the goal predicates (Pick, Place, Open, Close, TurnOn skills) | GUIDE's native path; scripted motion differs in style from human teleop |
| b. MimicGen-style: segment LIBERO demos into object-centric spans and transform them to new layouts | keeps the human style; more machinery |

- [ ] Choose per hypothesis; Spatial first

## S6 — Modalities

GUIDE already records RGB, depth (12-bit HEVC) and instance segmentation. To add:
- wrist force/torque: an articulation joint-force sensor at the hand joint
- finger contact: PhysX contact sensors
- joint efforts

Match the real robots' signals: FR3 `O_F_ext_hat_K`, UR5e built-in F/T.

- [ ] Recorder features + LeRobot schema

## S7 — Dataset export

Same layout as the LIBERO training data the baseline uses:
- images at 256 × 256 (LeRobot LIBERO convention, check against the chosen baseline set), `agentview` + wrist
- 8-dim state
- 7-dim action **in LIBERO's normalized OSC convention**
- 20 Hz
- the same task strings

Extra modalities as extra features, so a baseline recipe can ignore them.

- [ ] Exporter flags in `guide_dataset_tools`; an equivalence test against an original LIBERO episode

## S8 — Round 1: 2 × 2 cross-domain eval

- MuJoCo column: lerobot's LIBERO env (`lerobot-eval --env.type=libero`, needs `hf-libero`), 50 fixed init states per task.
- GUIDE column: closed-loop eval of the compiled tasks from the same init states, through the S2 controller (an eval tool beside guide, like `block_bin_eval`).
- Training: SmolVLA, same recipe and steps for both sources, several seeds.
- [ ] Results table + decision per the rule above

## S9 — Round 2: real robot (only if each source wins its own domain)

Real LIBERO-like setup on the FR3 (bowls, plate, …), runner `vla-inference-ui`.

## S10 — Embodiment expansion

FR3, UR5e and others: robot-agnostic task configs, per-robot MoveIt and gripper, retargeted H1
(S5 d), per-robot recording relative to the robot's base (GUIDE TODO: several robots per scene).

---

## Appendix A — prior art (checked 2026-10-19)

- **Lightwheel LIBERO tasks** (all 130) on Isaac Lab-Arena, in [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub). They are framed as evaluation environments. Training data: four LeRobot v3 sets on HF, Apache-2.0: Double-Piper (130 LIBERO tasks, 6,500 eps), X7S (117 LIBERO + 89 RoboCasa, 10,300 eps), G1-Controller, G1-WBC. All have three 640 × 480 RGB cameras + joint state + action at 50 fps: **no depth, segmentation, force or end-effector pose, and no Franka**. Teleoperated, 50 per task × robot. The raw `lightwheel_tasks` set stores per-episode HDF5 + state-replay video.
- **RobotLearningLab** ([HF](https://huggingface.co/datasets/china-sae-robotics/RobotLearningLab_Dataset), Apache-2.0): `libero/USD`, `libero/assembled_hdf5` (for trajectory replay in Isaac Lab), `libero/replayed_demos`. Robot and suites unverified.

## Appendix B — LIBERO facts verified locally

- Each demo in the HDF5 files has:
  - `model_file`: the full compiled MJCF, ~100 KB, with mesh paths from the authors' machine;
  - `init_state`;
  - `states`: 92 values per step;
  - `actions` (N, 7);
  - `obs/{agentview_rgb, eye_in_hand_rgb}` at 128 × 128;
  - `ee_pos`, `ee_ori`, `joint_states`, `gripper_states`.
- `env_args`: Panda, `OSC_POSE` (delta, output ±0.05 m / ±0.5 rad, kp 150), control 20 Hz.
- Actions span about [−0.94, 0.94] on translation; the gripper is exactly ±1.
- BDDL regions in the Spatial suite are 2 × 2 cm (±1 cm jitter); the benchmark evaluates from fixed init states.
