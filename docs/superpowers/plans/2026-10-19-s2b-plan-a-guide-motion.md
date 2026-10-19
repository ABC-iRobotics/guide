# S2b Plan A — `guide_motion`: targets, planner, OSCBF control, robot descriptions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a pure-Python `guide_motion` package (no Isaac, no ROS imports) holding GUIDE's metric motion targets and the LIBERO converters, a Pink IK planner, the OSCBF controller (torque and velocity), and self-validated FR3 and Panda descriptions generated from Isaac's USD — on a `.venv` that is back on Isaac's exact pins.

**Architecture:** a new top-level directory `guide_motion/` in the guide repo: a plain pyproject distribution installed editable into `~/ros2_ws/.venv` with the pins file. It ships two packages: `guide_motion` and OSCBF, vendored unmodified at its upstream commit as the top-level package `oscbf`. `model` turns URDF + XRDF into Pinocchio kinematics (fingers locked) and OSCBF's JAX dynamics (built lazily, so only the control process imports JAX). `control` builds one jitted QP step per mode. `planner` turns a far goal into 50 ms targets with Pink IK. `description` holds the robot description pipeline: a Kit-free half tested on the XRDF test's raw exporter outputs, and a Kit CLI that writes `guide_motion/guide_motion/robots/<robot>/`.

**Tech Stack:** Python 3.12 · numpy 2.3.1 · JAX/jaxlib 0.4.30 (CPU, float64) · cbfpy 0.1.0 · qpax 0.1.4 · pin 4.1.0 / pin-pink 4.3.0 / quadprog · scipy · PyYAML · h5py (tests) · Isaac Sim 6.0.1 (only for `python -m guide_motion.description`) · pytest 9 · uv.

**Spec:** `docs/superpowers/specs/2026-10-19-s2b-native-motion-design.md`, sections 4.1, 4.2, 5, 7 (unit tests) and 8 (rows 2–4). Plans B (`2026-10-19-s2b-plan-b-isaac-host.md`) and C (`2026-10-19-s2b-plan-c-rosrobot-guide-ex.md`) depend on this plan.

**Sources the code is built from** (read them when a task says so):
- OSCBF at commit `6082329f8e1d61ee8e875a63a5cd80a14d513e12` (`github.com/StanfordASL/oscbf`, MIT; a local clone is at `/home/user/libero_assets/s2b_inputs/oscbf_repo`). Its evaluation: `/home/user/libero_assets/s2b_inputs/oscbf-report.md`.
- The XRDF/URDF pipeline test: `/home/user/libero_assets/s2b_inputs/xrdf_test/report.md`, `make_fr3_xrdf.py`, `check_frames.py` and the raw outputs Task 2 copies as test fixtures. That directory is a job scratch directory: **run Task 2 before it is cleaned up.**
- The Isaac measurement: `/home/user/libero_assets/s2b_inputs/oscbf_isaac/{run.py,oscbf_ctrl.py,report.md}`.
- LIBERO-Spatial demos: `~/Downloads/libero_spatial/*.hdf5` (robomimic layout: `data/demo_<i>/actions` (T, 7), `obs/ee_pos` (T, 3), `obs/ee_ori` (T, 3, axis-angle); `env_args` says `OSC_POSE`, `output_max` [0.05 ×3, 0.5 ×3], `control_freq` 20).

## Spec questions (recorded; the plan follows the reading shown)

1. **"Bring the `.venv` back onto Isaac's pins" cannot end at zero `uv pip check` lines.** lerobot 0.6.0's metadata caps `numpy<2.3.0` and `packaging<26.0`; Isaac pins `numpy==2.3.1` and `packaging==26.0`. Restoring Isaac's 9 exact pins therefore leaves exactly two lerobot lines. Reading: Isaac's pins win (the spec's prerequisite), the two lerobot lines are expected, and `INSTALLATION.md`/`README.md` are corrected (their `lerobot … -c $PINS` line cannot resolve; it becomes `--override $PINS`).
2. **"OSCBF vendored as a pinned dependency at its upstream commit."** As a pip dependency it would pull `cbfpy==0.0.1` (which needs `numpy<2`), pybullet (a source build) and matplotlib, and `uv pip check` would flag it. Reading: the upstream files are copied unmodified at `6082329` (MIT, `LICENSE` kept) into the `guide_motion` distribution as the top-level package `oscbf`. Its URDF parser needs `lxml` (new in `.venv`), `networkx`, `pillow`, `six` and `trimesh` (already in `.venv`). These are listed as dependencies next to the spec's exact list. `h5py` is a test extra (the LIBERO tests).
3. **"No Isaac imports" vs "`description`: the pipeline, run as a CLI".** The pipeline needs Kit. Reading: `guide_motion.description` imports Isaac only inside `main()`. Importing any `guide_motion` module never imports Isaac or ROS.
4. **The exporter URDF has zero inertias** (FR3: inertia 0 on every link, no finger inertials). Reading: the pipeline writes the PhysX mass properties it reads anyway (spec 4.2.3), re-expressed in the URDF link frames, into the committed URDF. The host's readback at registration (spec 4.2.3, Plan B) stays; it recompiles only when the scene's masses differ (a hand-mounted camera, say).
5. **The Panda USD has no `panda_hand_tcp`** (its PhysX links are `panda_link0..7`, `panda_hand` and the two fingers). Reading: the pipeline adds it as a fixed link 0.1034 m along `panda_hand`'s z (franka_description's `franka_hand` `tcp_xyz`), so that `tool_frames: [panda_hand_tcp]` exists.
6. **Self-collision barriers "from the XRDF spheres".** One barrier per sphere pair would put 1404 rows into the FR3's QP. Reading: one barrier per pair of link groups at least 2 joints apart (21 for the FR3). Each barrier is the smooth minimum (log-sum-exp, β = 1000 /m) of that pair's sphere gaps; it is never larger than the true minimum. Measured on this machine with a prototype, standalone, single-threaded XLA: torque 0.30 ms and velocity 0.19 ms per step, against 0.14 / 0.10 ms with no self-collision barrier. Plan B's RTF check is the gate. `barriers: {min_group_gap: 3}` saves about 15 %.
7. **"QP infeasible."** cbfpy's QP is the relaxed (elastic) form and never reports infeasibility. Reading: a step counts as infeasible when its solution breaks a QP row by more than 1e-2, or is not finite. Then only joint damping is applied (spec 8). In torque mode that is `-kd_null · qd` without gravity compensation, exactly as the spec says; a single such step at 120 Hz drops the arm by about 0.3 mm.

## Global Constraints

Copied from the spec:
- `guide_motion`: "No Isaac imports, no ROS imports. It is a plain pyproject package installed into `.venv`, and later vla-inference-ui's env."
- "Actions are metric everywhere in GUIDE (user decision 2026-10-19): Δp in metres, Δω as a rotation vector in radians, gripper width in metres. LeRobot normalizes the recorded data for training, so no [−1, 1] interval is used in recording, the controller or the target topic."
- `delta_action_to_target`: "clamp Δp to ±0.05 m and Δω to ±0.5 rad per axis (LIBERO's per-step bound, kept as a safety limit); apply it as a delta from the **measured** end-effector pose at the policy step (`goal_pos = p + Δp`, `goal_ori = R(Δω)·R`), not from the previous target; hold the result for one 50 ms step."
- `from_libero_normalized` / `to_libero_normalized`: "(×0.05 m, ×0.5 rad, gripper ±1 ↔ width). They are used **only** at the 2×2 evaluation boundary."
- "Dependencies (exact): `jax==0.4.30`, `jaxlib==0.4.30`, `cbfpy==0.1.0`, `qpax==0.1.4`, `ml-dtypes`, `opt-einsum`, `pin` / `pin-pink` (already in `.venv`). OSCBF itself is vendored as a pinned dependency at its MIT-licensed upstream commit. Installed with the repo's pins file (`-c $PINS`)."
- "Prerequisite: the `.venv` must first be brought back onto Isaac's pins."
- Torque gains: "task kp 150 and kd 2√150 ≈ 24.5 on all six axes (ζ = 1); nullspace kp 10, kd 2√10 ≈ 6.32 towards the joint configuration at the episode start." Velocity: "OSCBF velocity gains".
- "Barriers: joint limits, self-collision from the XRDF spheres, singularity, and an end-effector workspace box."
- XRDF: "add `tool_frames: [<robot>_hand_tcp]`; add the ignore line `<robot>_link7: [<robot>_hand, <robot>_leftfinger, <robot>_rightfinger]`; FR3: 10 rad/s² and 5000 rad/s³ on every joint; Panda: [15, 7.5, 10, 12.5, 15, 20, 20] and [7500, 3750, 5000, 6250, 7500, 10000, 10000]; keep link0 self-collision-only; re-express every sphere centre in the exported URDF's link frames."
- URDF: "mimic multiplier → +1; prismatic finger velocity → the USD's value without the degree conversion; add `effort` limits from the USD drives; keep the USD's joint limits." "USD → URDF … timeline stopped."
- "Each run validates itself and fails loudly if a check fails: cuMotion's loader accepts the pair; the home pose is free of self-collision; random-configuration self-collision ≤ 8%; M and g from `guide_motion` agree with PhysX within 1e-3 relative."
- "Outputs (URDF, XRDF, the override layer and a `description.json` with the source asset path and pipeline version) are committed under `guide_motion/robots/<robot>/`."
- "Command outside limits: clamp to the URDF effort and velocity limits." "QP infeasible or non-finite output: apply joint damping only, count the event, keep running."

Process (this workspace):
- Work in `~/ros2_ws/src/guide` on branch `feat/libero-native-motion`. Commits are authored by the repo's configured user only: **no `Co-Authored-By`, no Claude/Generated-with lines.** Never push, never open PRs.
- Never stage `guide_core/config/init.yaml` (the user's uncommitted edit). Never run `git checkout --`, `git restore`, `git reset --hard`, `git stash` or `git clean`. Stage files by name.
- The venv is uv's (no `pip`): `uv pip … --python ~/ros2_ws/.venv/bin/python`, always with `-c $PINS` where `PINS=~/ros2_ws/src/guide/modules/isaac6-safe-pins.txt`.
- Tests: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test` (ROS's launch_testing plugin breaks pytest 9).
- Isaac runs headless through `~/ros2_ws/.venv/bin/python` with `OMNI_KIT_ACCEPT_EULA=YES` and `env -u CUDA_DEVICE_ORDER`. Before launching, `pgrep -af "isaac|kit/kit|guide_core"` must print nothing. The first launch compiles shaders for about 2 minutes.
- Never upload anything to Hugging Face.

## Review Focus

1. **A measured state a hair past a joint limit** (PhysX can report one): the barrier is negative there. Expected: a finite command, and in velocity mode one that pushes the joint back inside. Pinned in Task 6 (`test_a_state_past_a_joint_limit_gives_a_finite_inward_command`).
2. **float32 states** (PhysX's tensors are float32; cbfpy's jvp rejects float32): expected, the same command as for the same values in float64. Pinned in Task 6 (`test_float32_states_give_the_float64_command`).
3. **A goal that turns the hand by almost 180°:** the slerp axis is nearly ambiguous there. Expected: a finite plan that ends exactly at the goal. Pinned in Task 7 (`test_a_half_turn_of_the_hand_is_planned`).
4. **A joint goal outside the URDF limits** (a GUIDE-EX configuration written for another robot): expected, `PlanningError` before anything moves, not a move that stalls at the barrier until it times out. Pinned in Task 7 (`test_joint_goal_outside_the_limits_raises`).
5. **The Panda's exporter output has no Kit-free fixture** (only the FR3's was captured by the XRDF test): its frames, finger mimic and drive comments may differ. Expected: Task 9's run passes all four checks, and the committed Panda loads with every barrier positive at home. Pinned in Task 9 (`test_committed_descriptions_are_safe_at_home`).

---

## File Structure

```
guide_motion/                         new top-level directory (a pyproject distribution, not a colcon package)
  COLCON_IGNORE                       keeps colcon from treating the directory as a package
  pyproject.toml                      dependencies; the oscbf vendoring note
  guide_motion/
    __init__.py                       JAX on CPU, float64, one XLA thread (set before JAX is imported)
    targets.py                        Target, delta_action_to_target, LIBERO converters, TargetHolder
    description.py                    the pipeline: Kit-free fixes and frame maths; main() = Kit CLI
    model.py                          Model: URDF + XRDF -> Pinocchio kinematics, OSCBF dynamics, spheres
    control.py                        Controller: the jitted OSCBF QP step, fallbacks, compilation cache
    planner.py                        plan(): Pink IK straight lines, joint interpolation
    robots/fr3/                       fr3.urdf, fr3.xrdf, description.json (Task 8, generated)
    robots/panda/                     panda.urdf, panda.xrdf, panda.usda (mass override), description.json (Task 9)
  oscbf/                              OSCBF, vendored unmodified at 6082329 (LICENSE inside)
  test/
    data/                             the XRDF test's raw FR3 exporter/editor outputs and references
    fr3_fixture.py                    shared helpers (franka_description stands in for the USD)
    conftest.py                       session fixtures: fr3_dir, fr3_model
    test_vendored_oscbf.py, test_targets.py, test_description.py, test_model.py,
    test_control.py, test_planner.py, test_robots.py
INSTALLATION.md, README.md            pins and install lines (Tasks 1-2)
```

Shared helpers live in `test/fr3_fixture.py`, not in `conftest.py`. Tests import them as `from fr3_fixture import …`: several test directories have a `conftest.py`, so a `from conftest import …` would pick whichever was imported first in a joint run.

---

### Task 1: The `.venv` back on Isaac's exact pins

**Files:**
- Modify (local, git-ignored): `modules/isaac6-safe-pins.txt`
- Modify: `INSTALLATION.md` (section "2. Python environment", lines 49–71), `README.md` (install block, lines 105–122; troubleshooting bullet at line 393)

**Interfaces:**
- Consumes: nothing.
- Produces: a `.venv` with numpy 2.3.1, torch 2.11.0+cu130, torchvision 0.26.0+cu130, packaging 26.0, opencv-python-headless 4.13.0.90, filelock 3.25.2, click 8.1.7, typing-extensions 4.12.2, jinja2 3.1.5; the pins file listing all nine. Every later `uv pip install` in Plans A–C uses `-c $PINS`.

- [ ] **Step 1: Record the drift**

Run:
```bash
cd ~/ros2_ws && uv pip check --python .venv/bin/python
```
Expected: `Found 9 incompatibilities`: isaacsim-core needs packaging 26.0, opencv-python-headless 4.13.0.90, torch 2.11.0, torchvision 0.26.0, filelock 3.25.2; isaacsim-kernel needs numpy 2.3.1, click 8.1.7, typing-extensions 4.12.2, jinja2 3.1.5.

- [ ] **Step 2: Put all nine exact pins in the pins file**

Overwrite `~/ros2_ws/src/guide/modules/isaac6-safe-pins.txt` with:
```text
# Constraints protecting Isaac Sim 6.0.1's EXACT pins (isaacsim-kernel and isaacsim-core) when
# uv-installing into .venv. Pass `-c` on every install. lerobot 0.6.0 caps numpy<2.3.0 and
# packaging<26.0 in its metadata; install it with `--override` this file instead (it runs fine at
# Isaac's versions), and expect `uv pip check` to report exactly those two lerobot lines.
numpy==2.3.1
torch==2.11.0
torchvision==0.26.0
packaging==26.0
opencv-python-headless==4.13.0.90
filelock==3.25.2
click==8.1.7
typing-extensions==4.12.2
jinja2==3.1.5
```

- [ ] **Step 3: Dry-run the restore**

Run:
```bash
cd ~/ros2_ws && PINS=src/guide/modules/isaac6-safe-pins.txt
uv pip install --python .venv/bin/python --dry-run -c $PINS \
  torch==2.11.0 torchvision==0.26.0 numpy==2.3.1 packaging==26.0 opencv-python-headless==4.13.0.90 \
  filelock==3.25.2 click==8.1.7 typing-extensions==4.12.2 jinja2==3.1.5 \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
```
Expected: `Would uninstall 11 packages` / `Would install 11 packages`, including `+ torch==2.11.0+cu130`, `+ torchvision==0.26.0+cu130`, `+ numpy==2.3.1` and the cu13 `cuda-bindings` / `cuda-pathfinder`. No resolution error.

- [ ] **Step 4: Restore**

Run the same command without `--dry-run`.
Expected: `Installed 11 packages`.

- [ ] **Step 5: Verify**

Run:
```bash
cd ~/ros2_ws && uv pip check --python .venv/bin/python
.venv/bin/python -I -c "import numpy, torch, lerobot; from lerobot.robots import Robot; from lerobot.datasets.lerobot_dataset import LeRobotDataset; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```
Expected: `Found 2 incompatibilities`, both for `lerobot` (numpy `<2.3.0`, packaging `<26.0`), and `2.3.1 2.11.0+cu130 True`.

Then run GUIDE's documented check (INSTALLATION.md §7) from a shell with the workspace sourced (`source /opt/ros/jazzy/setup.zsh && source ~/ros2_ws/install/setup.zsh`):
```bash
cd ~/ros2_ws/src/guide
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q \
  guide_core/test guide_ex/test guide_tasks/cube_stack/test \
  --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'
```
Expected: all pass (the same count as before Step 4; run it once before Step 4 if unsure).

- [ ] **Step 6: Correct the install docs**

In `INSTALLATION.md`, replace the line `printf 'numpy==2.3.1\ntorch==2.11.0\ntorchvision==0.26.0\n' > $PINS` with:
```bash
cat > $PINS <<'EOF'
numpy==2.3.1
torch==2.11.0
torchvision==0.26.0
packaging==26.0
opencv-python-headless==4.13.0.90
filelock==3.25.2
click==8.1.7
typing-extensions==4.12.2
jinja2==3.1.5
EOF
```
Replace the two lines
```bash
uv pip install --python .venv/bin/python python-fcl "lerobot==0.6.0" "transformers>=5.4,<5.6" -c $PINS
uv pip check --python .venv/bin/python     # must report no incompatibilities
```
with
```bash
uv pip install --python .venv/bin/python python-fcl "lerobot==0.6.0" "transformers>=5.4,<5.6" --override $PINS
uv pip check --python .venv/bin/python     # exactly 2 incompatibilities, both lerobot's caps
```
Replace the bullet that starts `` - `-c $PINS` on every torch-dependent install`` (through `` `huggingface-hub 1.x`. ``) with:
```markdown
- `$PINS` holds Isaac's nine exact pins (`isaacsim-kernel`, `isaacsim-core`). Pass `-c $PINS` on
  every install. lerobot 0.6.0 caps `numpy<2.3.0` and `packaging<26.0` in its metadata, so `-c`
  cannot resolve it: install it with `--override $PINS`. It runs fine at Isaac's versions, and
  `uv pip check` then reports exactly those two lerobot lines; anything else is drift. lerobot 0.6.0
  also needs `transformers 5.4–5.6` and `huggingface-hub 1.x`.
```
Replace the bullet `` - After a drift (also a cu128 torch pulled in by another project), restore with `` and its command line with:
```markdown
- After a drift (also a cu128 torch pulled in by another project), restore all nine with
  `uv pip install --python .venv/bin/python -c $PINS torch==2.11.0 torchvision==0.26.0 numpy==2.3.1 packaging==26.0 opencv-python-headless==4.13.0.90 filelock==3.25.2 click==8.1.7 typing-extensions==4.12.2 jinja2==3.1.5 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`.
```
In `README.md`, make the same three changes to its install block (the `printf` line, the lerobot line + its `uv pip check` comment) and replace its note paragraph `> Always pass -c $PINS …` with:
```markdown
> Pass `-c $PINS` on every install: it holds Isaac's nine exact pins. lerobot 0.6.0 caps
> `numpy<2.3.0` and `packaging<26.0`, so install it with `--override $PINS`; `uv pip check` then
> reports exactly those two lerobot lines (it runs fine at Isaac's versions). lerobot 0.6.0 also
> needs `transformers 5.4-5.6` + `huggingface-hub 1.x`.
```
and in its troubleshooting bullet ``- **`uv pip check` reports incompatibilities**`` replace the command with the nine-pin restore command above.

- [ ] **Step 7: Commit**

```bash
cd ~/ros2_ws/src/guide
git add INSTALLATION.md README.md
git commit -m "docs(install): pin all nine of Isaac's exact versions; lerobot via --override"
git status --porcelain   # guide_core/config/init.yaml still ' M', unstaged
```

---

### Task 2: The `guide_motion` distribution with OSCBF vendored

**Files:**
- Create: `guide_motion/COLCON_IGNORE` (empty), `guide_motion/pyproject.toml`, `guide_motion/guide_motion/__init__.py`
- Create (copied unmodified): `guide_motion/oscbf/{__init__.py,LICENSE}`, `guide_motion/oscbf/core/{__init__,manipulator,controllers,oscbf_configs,franka_collision_model}.py`, `guide_motion/oscbf/utils/{__init__,urdf_parser}.py`, `guide_motion/oscbf/utils/urdfpy/{__init__,urdf,utils,version}.py`, `guide_motion/oscbf/assets/__init__.py`
- Create (fixtures, copied): `guide_motion/test/data/{fr3_exporter_raw.urdf,fr3_editor_raw.xrdf,fr3_franka_description.urdf,fr3_spheres_reference.xrdf}`
- Create: `guide_motion/test/fr3_fixture.py`, `guide_motion/test/test_vendored_oscbf.py`
- Modify: `INSTALLATION.md` §2 and `README.md` install block (one install line each)

**Interfaces:**
- Consumes: Task 1's pins file.
- Produces: `import guide_motion`, `import oscbf` in `.venv`; `oscbf.core.manipulator.Manipulator.from_urdf(path, ee_offset=…)`; test helpers `fr3_fixture.DATA: Path`, `FD_URDF: str`, `ARM: list[str]`, `fd_dynamics(q) -> (M (7,7), g (7,))`.

- [ ] **Step 1: Copy the fixtures first** (they live in a job scratch directory)

```bash
cd ~/ros2_ws/src/guide && mkdir -p guide_motion/test/data
X=/home/user/libero_assets/s2b_inputs/xrdf_test
cp $X/urdf/fr3.urdf guide_motion/test/data/fr3_exporter_raw.urdf
cp $X/fr3.xrdf guide_motion/test/data/fr3_editor_raw.xrdf
cp $X/fr3_franka_description.urdf guide_motion/test/data/fr3_franka_description.urdf
cp $X/fr3_fixed_usd_export_mimicfix_frames.xrdf guide_motion/test/data/fr3_spheres_reference.xrdf
wc -c guide_motion/test/data/*
```
Expected sizes: 11398, 5776, 20144, 7240 bytes. (`fr3_exporter_raw.urdf` is the USD exporter's output with all its defects, `fr3_editor_raw.xrdf` the editor's export before any fix, `fr3_franka_description.urdf` the xacro of `~/ros2_ws/src/franka_description/robots/fr3/fr3.urdf.xacro`, and `fr3_spheres_reference.xrdf` the XRDF test's spheres re-expressed into the exporter's link frames.) If `$X` is gone, stop and report `BLOCKED: fixtures missing`.

- [ ] **Step 2: Vendor OSCBF at its commit**

```bash
cd ~/ros2_ws/src/guide/guide_motion
git clone --quiet https://github.com/StanfordASL/oscbf /tmp/oscbf-upstream \
  || git clone --quiet /home/user/libero_assets/s2b_inputs/oscbf_repo /tmp/oscbf-upstream
git -C /tmp/oscbf-upstream switch --quiet --detach 6082329f8e1d61ee8e875a63a5cd80a14d513e12
U=/tmp/oscbf-upstream
mkdir -p oscbf/core oscbf/utils/urdfpy oscbf/assets
cp $U/LICENSE $U/oscbf/__init__.py oscbf/
cp $U/oscbf/core/{__init__,manipulator,controllers,oscbf_configs,franka_collision_model}.py oscbf/core/
cp $U/oscbf/utils/{__init__,urdf_parser}.py oscbf/utils/
cp $U/oscbf/utils/urdfpy/{__init__,urdf,utils,version}.py oscbf/utils/urdfpy/
cp $U/oscbf/assets/__init__.py oscbf/assets/
for f in $(cd oscbf && find . -name '*.py'); do cmp oscbf/$f $U/oscbf/$f || echo "DIFFERS $f"; done
head -1 oscbf/LICENSE
```
Expected: no `DIFFERS` line; `MIT License`. (Not vendored: the pybullet demo environment, examples, scripts and the Panda assets; nothing in the copied core imports them.)

- [ ] **Step 3: Write the distribution files**

`guide_motion/COLCON_IGNORE`: an empty file.

`guide_motion/pyproject.toml`:
```toml
# guide_motion: GUIDE's motion layer (S2b). Plain Python: no Isaac and no ROS imports.
# Install: uv pip install --python ~/ros2_ws/.venv/bin/python -c $PINS -e "src/guide/guide_motion[test]"
#
# oscbf/ is OSCBF (github.com/StanfordASL/oscbf, MIT, oscbf/LICENSE), vendored unmodified at commit
# 6082329f8e1d61ee8e875a63a5cd80a14d513e12. As a pip dependency it would pull cbfpy==0.0.1
# (numpy<2), pybullet and matplotlib; its core runs on cbfpy 0.1.0 and numpy 2.3.1.
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "guide_motion"
version = "0.1.0"
description = "Pink IK planning and OSCBF control for GUIDE robots"
requires-python = ">=3.12"
license = "GPL-3.0-only"
dependencies = [
    "numpy",
    "scipy",
    "pyyaml",
    "jax==0.4.30",
    "jaxlib==0.4.30",
    "cbfpy==0.1.0",
    "qpax==0.1.4",
    "ml-dtypes",
    "opt-einsum",
    "pin",
    "pin-pink",
    "quadprog",
    # the vendored OSCBF's URDF parser (oscbf/utils/urdfpy)
    "lxml",
    "networkx",
    "pillow",
    "six",
    "trimesh",
]

[project.optional-dependencies]
test = ["pytest", "h5py"]

[tool.setuptools.packages.find]
include = ["guide_motion*", "oscbf*"]

[tool.setuptools.package-data]
guide_motion = ["robots/*/*"]
oscbf = ["LICENSE"]
```

`guide_motion/guide_motion/__init__.py`:
```python
"""GUIDE's motion layer: Pink IK planning and OSCBF control. No Isaac, no ROS."""

import os

# JAX on GPU is out of scope (S2b spec 1); the CPU wheel otherwise warns at import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# OSCBF and cbfpy need float64 (float32 states raise in cbfpy's jvp); the vendored oscbf sets it too.
os.environ.setdefault("JAX_ENABLE_X64", "1")
# ponytail: one XLA thread per call, measured 15-20 % faster next to PhysX's workers
# (S2b report); unset it here if the controller ever runs alone on a big machine.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
```

- [ ] **Step 4: Install**

```bash
cd ~/ros2_ws && PINS=src/guide/modules/isaac6-safe-pins.txt
uv pip install --python .venv/bin/python -c $PINS -e "src/guide/guide_motion[test]"
uv pip check --python .venv/bin/python
```
Expected: installs `guide-motion`, `jax==0.4.30`, `jaxlib==0.4.30`, `cbfpy==0.1.0`, `qpax==0.1.4`, `ml-dtypes`, `opt-einsum`, `lxml`, `h5py`, and changes none of the nine pinned packages. `uv pip check`: the same 2 lerobot lines as Task 1, nothing else.

- [ ] **Step 5: Write the failing test**

`guide_motion/test/fr3_fixture.py`:
```python
"""The FR3 test fixture: a description built Kit-free from the XRDF test's raw exporter outputs.

franka_description's FR3 stands in for the USD: the XRDF test measured its link frames
equal to the USD's, except the right finger, whose USD frame equals the left finger's at
q = 0. Its inertials stand in for PhysX's.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

DATA = Path(__file__).parent / "data"
FD_URDF = (DATA / "fr3_franka_description.urdf").read_text()
ARM = [f"fr3_joint{i}" for i in range(1, 8)]


def fd_dynamics(q):
    """(M, g) of franka_description's FR3, fingers locked, by Pinocchio: the physical reference."""
    full = pin.buildModelFromXML(FD_URDF)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    M = pin.crba(model, data, q)
    return np.triu(M) + np.triu(M, 1).T, pin.computeGeneralizedGravity(model, data, q).copy()
```
(`ET` is used by the helpers Task 5 appends.)

`guide_motion/test/test_vendored_oscbf.py`:
```python
"""The vendored OSCBF runs on this venv (numpy 2.3.1, jax 0.4.30, cbfpy 0.1.0) and agrees with Pinocchio."""

import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import guide_motion  # noqa: F401  (JAX settings before JAX is imported)
from fr3_fixture import ARM, FD_URDF, fd_dynamics


def test_oscbf_dynamics_match_pinocchio_on_franka_description():
    from oscbf.core.manipulator import Manipulator

    root = ET.fromstring(FD_URDF)
    for joint in root.iter("joint"):  # OSCBF wants every non-arm joint fixed
        if joint.get("type") != "fixed" and joint.get("name") not in ARM:
            joint.set("type", "fixed")
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)
    for link in root.iter("link"):  # and loads every mesh it finds
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
        f.write(ET.tostring(root, encoding="unicode"))
        f.flush()
        man = Manipulator.from_urdf(f.name)
    for q in np.random.default_rng(0).uniform(-1.5, 1.5, size=(3, 7)):
        M, g = fd_dynamics(q)
        np.testing.assert_allclose(np.asarray(man.mass_matrix(q)), M, atol=1e-12)
        np.testing.assert_allclose(np.asarray(man.gravity_vector(q)), g, atol=1e-12)
```

- [ ] **Step 6: Run it**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_vendored_oscbf.py`
Expected: `1 passed` (the prototype measured differences of 1e-15). If it fails with `ModuleNotFoundError`, Step 4's install is incomplete; fix it, not the test.

- [ ] **Step 7: Document the install line**

In `INSTALLATION.md` §2, after the `uv pip install … lerobot …` line, add:
```bash
uv pip install --python .venv/bin/python -c $PINS -e "src/guide/guide_motion[test]"   # S2b motion layer
```
and below the bullets add:
```markdown
- `guide_motion` (S2b) is a plain Python package, not a colcon one (`COLCON_IGNORE`): it is
  installed editable into `.venv`, so the solver and the Isaac process import the checkout. It
  vendors OSCBF unmodified (`guide_motion/oscbf`, MIT) because OSCBF's own pins need numpy<2.
```
Add the same install line to `README.md`'s install block after its lerobot line.

- [ ] **Step 8: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/COLCON_IGNORE guide_motion/pyproject.toml guide_motion/guide_motion/__init__.py \
  guide_motion/oscbf guide_motion/test/data guide_motion/test/fr3_fixture.py guide_motion/test/test_vendored_oscbf.py \
  INSTALLATION.md README.md
git commit -m "feat(guide_motion): package skeleton with OSCBF vendored at 6082329"
git status --porcelain   # init.yaml still ' M'
```

---

### Task 3: Targets, the LIBERO boundary and the target holder

**Files:**
- Create: `guide_motion/guide_motion/targets.py`
- Test: `guide_motion/test/test_targets.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (numpy, scipy, h5py).
- Produces:
  - `POSE, TWIST, JOINTS = 0, 1, 2` (the same values as `guide_msgs/MotionTarget` `MODE_*`, Plan B); `MAX_DPOS = 0.05`, `MAX_DROT = 0.5`, `OPEN_WIDTH = 0.08`.
  - `@dataclass(frozen=True) Target(mode: int = POSE, position: ndarray(3) = 0, rotation: ndarray(3,3) = I, twist: ndarray(6) = 0, joints: ndarray(n) = [], gripper_width: float = nan)`: base frame, metres/radians; `gripper_width` NaN = leave the gripper as it is.
  - `delta_action_to_target(action: array(7), measured_pose: (p (3,), R (3,3))) -> Target`.
  - `from_libero_normalized(action: array(7)) -> ndarray(7)`, `to_libero_normalized(action: array(7)) -> ndarray(7)`.
  - `TargetHolder(stale_after: float = 0.25)`: `.put(target)` (any thread), `.get(now: float) -> (Target | None, stale: bool)`.

- [ ] **Step 1: Write the failing tests**

`guide_motion/test/test_targets.py`:
```python
import threading
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from guide_motion.targets import (
    OPEN_WIDTH,
    POSE,
    Target,
    TargetHolder,
    delta_action_to_target,
    from_libero_normalized,
    to_libero_normalized,
)

LIBERO = sorted((Path.home() / "Downloads" / "libero_spatial").glob("*.hdf5"))
needs_libero = pytest.mark.skipif(not LIBERO, reason="no LIBERO-Spatial demos in ~/Downloads/libero_spatial")


def libero_demo(file_index=0, demo="demo_0"):
    h5py = pytest.importorskip("h5py")
    with h5py.File(LIBERO[file_index], "r") as f:
        d = f[f"data/{demo}"]
        actions, ee_pos, ee_ori = d["actions"][()], d["obs/ee_pos"][()], d["obs/ee_ori"][()]
    assert actions.shape[1] == 7 and ee_pos.shape[1] == 3 and ee_ori.shape[1] == 3, (actions.shape, ee_ori.shape)
    return actions, ee_pos, ee_ori


def test_clamp_holds_at_libero_bounds():
    p, R = np.array([0.4, 0.0, 0.3]), np.eye(3)
    t = delta_action_to_target([0.2, -0.2, 0.01, 1.0, -1.0, 0.1, 0.04], (p, R))
    np.testing.assert_allclose(t.position - p, [0.05, -0.05, 0.01])
    np.testing.assert_allclose(Rotation.from_matrix(t.rotation).as_rotvec(), [0.5, -0.5, 0.1], atol=1e-12)
    assert t.mode == POSE and t.gripper_width == 0.04


def test_libero_gripper_is_open_at_minus_one_and_closed_at_plus_one():
    assert from_libero_normalized([0, 0, 0, 0, 0, 0, -1])[6] == pytest.approx(OPEN_WIDTH)
    assert from_libero_normalized([0, 0, 0, 0, 0, 0, 1])[6] == pytest.approx(0.0)


@needs_libero
def test_libero_actions_round_trip_through_metric():
    for i in range(len(LIBERO)):
        actions, _, _ = libero_demo(i)
        back = np.array([to_libero_normalized(from_libero_normalized(a)) for a in actions])
        np.testing.assert_allclose(back, actions, atol=1e-12)


@needs_libero
def test_delta_is_applied_to_the_measured_pose():
    actions, ee_pos, ee_ori = libero_demo()
    for t in range(0, len(actions), 7):
        metric = from_libero_normalized(actions[t])
        R = Rotation.from_rotvec(ee_ori[t]).as_matrix()
        target = delta_action_to_target(metric, (ee_pos[t], R))
        np.testing.assert_allclose(target.position - ee_pos[t], np.clip(metric[:3], -0.05, 0.05), atol=1e-12)
        np.testing.assert_allclose(target.rotation, Rotation.from_rotvec(np.clip(metric[3:6], -0.5, 0.5)).as_matrix() @ R, atol=1e-12)


@needs_libero
def test_libero_arm_moves_the_way_the_delta_points():
    """Real demos: OSC_POSE's delta lives in the frame ee_pos is reported in, same sign."""
    actions, ee_pos, _ = libero_demo()
    commanded = actions[:-1, :3] * 0.05
    moved = np.diff(ee_pos, axis=0)
    big = np.linalg.norm(commanded, axis=1) > 0.01
    cosine = np.sum(commanded[big] * moved[big], axis=1) / (
        np.linalg.norm(commanded[big], axis=1) * np.linalg.norm(moved[big], axis=1)
    )
    assert big.sum() > 20 and cosine.mean() > 0.7, (big.sum(), cosine.mean())


def test_holder_latest_wins_and_goes_stale():
    holder = TargetHolder(stale_after=0.25)
    assert holder.get(0.0) == (None, True)
    a, b = Target(position=np.ones(3)), Target(position=2 * np.ones(3))
    holder.put(a)
    holder.put(b)
    target, stale = holder.get(1.0)
    assert target is b and not stale
    assert not holder.get(1.25)[1]
    assert holder.get(1.26)[1]
    holder.put(b)  # re-sending the same target refreshes it
    assert not holder.get(1.3)[1]


def test_holder_put_from_another_thread():
    holder = TargetHolder()
    targets = [Target(position=np.full(3, i)) for i in range(1000)]
    worker = threading.Thread(target=lambda: [holder.put(t) for t in targets])
    worker.start()
    worker.join()
    assert holder.get(0.0)[0] is targets[-1]
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_targets.py`
Expected: collection error `ModuleNotFoundError: No module named 'guide_motion.targets'`.

- [ ] **Step 3: Implement**

`guide_motion/guide_motion/targets.py`:
```python
"""What one 20 Hz step hands the controller (S2b spec 4.1, 4.4).

Actions are metric everywhere in GUIDE: dp in metres, dw a rotation vector in radians,
gripper width in metres. LIBERO's [-1, 1] convention exists only at the evaluation
boundary (from_libero_normalized / to_libero_normalized).
"""

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

POSE, TWIST, JOINTS = 0, 1, 2  # == guide_msgs/MotionTarget MODE_*
MAX_DPOS, MAX_DROT = 0.05, 0.5  # LIBERO's per-step bound (m, rad per axis), kept as a safety limit
OPEN_WIDTH = 0.08  # Franka hand fully open, m


@dataclass(frozen=True)
class Target:
    """One motion target in the robot base frame."""

    mode: int = POSE
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [v m/s, w rad/s]
    joints: np.ndarray = field(default_factory=lambda: np.zeros(0))  # rad, arm joints
    gripper_width: float = float("nan")  # m, absolute; NaN leaves the gripper as it is


def delta_action_to_target(action, measured_pose) -> Target:
    """robosuite OSC_POSE's goal, in metres: clamp per axis, then apply to the MEASURED pose.

    action = [dx, dy, dz (m), rx, ry, rz (rotation vector, rad), gripper width (m)];
    measured_pose = (position (3,), rotation (3, 3)) of the end effector at this policy step.
    goal_pos = p + dp, goal_ori = R(dw) @ R. The caller holds the result for one 50 ms step.
    """
    a = np.asarray(action, dtype=float)
    position, rotation = measured_pose
    dp = np.clip(a[:3], -MAX_DPOS, MAX_DPOS)
    dw = np.clip(a[3:6], -MAX_DROT, MAX_DROT)
    return Target(
        POSE,
        np.asarray(position, dtype=float) + dp,
        Rotation.from_rotvec(dw).as_matrix() @ np.asarray(rotation, dtype=float),
        gripper_width=float(a[6]),
    )


def from_libero_normalized(action) -> np.ndarray:
    """LIBERO action -> metric: x0.05 m, x0.5 rad, gripper -1 (open) .. +1 (closed) -> width."""
    a = np.asarray(action, dtype=float)
    return np.r_[a[:3] * MAX_DPOS, a[3:6] * MAX_DROT, (1.0 - a[6]) / 2.0 * OPEN_WIDTH]


def to_libero_normalized(action) -> np.ndarray:
    """Metric action -> LIBERO's normalized one (the inverse of from_libero_normalized)."""
    a = np.asarray(action, dtype=float)
    return np.r_[a[:3] / MAX_DPOS, a[3:6] / MAX_DROT, 1.0 - 2.0 * a[6] / OPEN_WIDTH]


class TargetHolder:
    """Single-slot latest target: the ROS thread puts, the physics callback gets."""

    def __init__(self, stale_after: float = 0.25):
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self._slot = (0, None)  # (sequence number, target)
        self._seen = 0
        self._fresh_at = -np.inf

    def put(self, target: Target) -> None:
        with self._lock:
            self._slot = (self._slot[0] + 1, target)

    def get(self, now: float):
        """(latest target or None, stale): stale = nothing new for stale_after s of `now`'s clock."""
        seq, target = self._slot
        if seq != self._seen:
            self._seen, self._fresh_at = seq, now
        return target, target is None or now - self._fresh_at > self.stale_after
```

- [ ] **Step 4: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q -rs guide_motion/test/test_targets.py`
Expected: `7 passed` and **no skips** (h5py is installed and the demos are in `~/Downloads/libero_spatial`). If `test_libero_arm_moves_the_way_the_delta_points` fails, print its `(count, cosine)` and stop: the assumption that LIBERO's deltas are in `ee_pos`'s frame is wrong, and that changes `delta_action_to_target` for the evaluation boundary. Do not lower the threshold.

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/targets.py guide_motion/test/test_targets.py
git commit -m "feat(guide_motion): metric targets, LIBERO boundary converters, target holder"
```

---

### Task 4: The description pipeline's Kit-free half

**Files:**
- Create: `guide_motion/guide_motion/description.py`
- Modify: `guide_motion/test/fr3_fixture.py` (append `HOME`, `usd_poses`)
- Test: `guide_motion/test/test_description.py`

**Interfaces:**
- Consumes: Task 2's fixtures and `fr3_fixture.DATA`, `FD_URDF`.
- Produces (all pure Python, Pinocchio imported lazily):
  - constants `PIPELINE_VERSION = 1`, `ROBOTS_DIR: Path` (= `guide_motion/guide_motion/robots`), `FRANKA_DESCRIPTION: Path`, `HOME: list[float]` (7), `TCP_Z = 0.1034`, `ROBOTS: dict[str, dict]` with keys `usd`, `accel`, `jerk`, `datasheet`.
  - `fix_urdf(text: str, robot: str) -> str`; `fix_xrdf(xrdf: dict, robot: str) -> dict`.
  - `link_poses(urdf_text: str, root: str | None = None) -> dict[str, ndarray(4,4)]` (every frame at q = 0).
  - `usd_to_urdf(usd_poses: dict[str, (4,4)], urdf_text: str) -> dict[str, ndarray(4,4)]`; `reframe_spheres(xrdf: dict, frames: dict) -> dict`.
  - `to_urdf_frames(masses: dict, coms: dict, inertias: dict, frames: dict) -> dict[str, (mass, com (3,), inertia (3,3))]`.
  - `set_inertials(urdf_text: str, inertials: dict, extra_links: dict[str, (parent: str, T (4,4))] | None = None) -> str`.
  - `franka_inertials(arm: "fer" | "fr3", fd: Path = FRANKA_DESCRIPTION) -> dict[str, (mass, com, inertia)]` (keys `panda_*` / `fr3_*`, hand and fingers included).
  - `build(raw_urdf: str, raw_xrdf: dict, usd_poses: dict, robot: str) -> (urdf: str, xrdf: dict, description: dict)`; `description["usd_to_urdf"]` maps each USD rigid body to its 4×4 (as lists).
  - test helpers `fr3_fixture.HOME: ndarray(7)`, `usd_poses() -> dict`.

- [ ] **Step 1: Extend the fixture helpers**

Replace `guide_motion/test/fr3_fixture.py` with:
```python
"""The FR3 test fixture: a description built Kit-free from the XRDF test's raw exporter outputs.

franka_description's FR3 stands in for the USD: the XRDF test measured its link frames
equal to the USD's, except the right finger, whose USD frame equals the left finger's at
q = 0. Its inertials stand in for PhysX's.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

from guide_motion import description

DATA = Path(__file__).parent / "data"
FD_URDF = (DATA / "fr3_franka_description.urdf").read_text()
ARM = [f"fr3_joint{i}" for i in range(1, 8)]
HOME = np.array(description.HOME)


def usd_poses() -> dict:
    poses = description.link_poses(FD_URDF, root="fr3_link0")
    poses["fr3_rightfinger"] = poses["fr3_leftfinger"]
    return poses


def fd_dynamics(q):
    """(M, g) of franka_description's FR3, fingers locked, by Pinocchio: the physical reference."""
    full = pin.buildModelFromXML(FD_URDF)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    M = pin.crba(model, data, q)
    return np.triu(M) + np.triu(M, 1).T, pin.computeGeneralizedGravity(model, data, q).copy()
```

- [ ] **Step 2: Write the failing tests**

`guide_motion/test/test_description.py`:
```python
"""The pipeline's Kit-free half against the exporter and editor defects the XRDF test found."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest
import yaml

from fr3_fixture import DATA, usd_poses
from guide_motion import description

RAW_URDF = (DATA / "fr3_exporter_raw.urdf").read_text()
RAW_XRDF = yaml.safe_load((DATA / "fr3_editor_raw.xrdf").read_text())


def joints(text):
    return {j.get("name"): j for j in ET.fromstring(text).iter("joint")}


def test_raw_exporter_urdf_is_rejected_and_the_fixed_one_loads():
    with pytest.raises(ValueError):
        pin.buildModelFromXML(RAW_URDF)  # no effort on the limits (defect C)
    assert pin.buildModelFromXML(description.fix_urdf(RAW_URDF, "fr3")).nq == 9


def test_exporter_defects_are_fixed():
    fixed = joints(description.fix_urdf(RAW_URDF, "fr3"))
    assert fixed["fr3_finger_joint2"].find("mimic").get("multiplier") == "1"  # A
    assert float(fixed["fr3_finger_joint1"].find("limit").get("velocity")) == pytest.approx(0.2, abs=1e-6)  # B
    efforts = {n: float(j.find("limit").get("effort")) for n, j in fixed.items() if j.find("limit") is not None}  # C
    assert efforts == {**{f"fr3_joint{i}": 87.0 for i in range(1, 5)}, **{f"fr3_joint{i}": 12.0 for i in range(5, 8)},
                       "fr3_finger_joint1": 100.0, "fr3_finger_joint2": 100.0}
    raw = joints(RAW_URDF)
    assert fixed["fr3_joint1"].find("limit").get("upper") == raw["fr3_joint1"].find("limit").get("upper")  # USD limits kept


def test_geometry_is_dropped():
    text = description.fix_urdf(RAW_URDF, "fr3")
    assert "<visual" not in text and "<collision" not in text and "<mesh" not in text


def test_a_missing_tool_frame_is_added_on_the_hand():
    text = re.sub(r'\s*<link name="fr3_hand_tcp"/>', "", RAW_URDF)
    text = re.sub(r'\s*<joint name="fr3_hand_tcp_joint".*?</joint>', "", text, flags=re.S)
    assert "fr3_hand_tcp" not in text
    poses = description.link_poses(description.fix_urdf(text, "fr3"))
    offset = np.linalg.inv(poses["fr3_hand"]) @ poses["fr3_hand_tcp"]
    expected = np.eye(4)
    expected[2, 3] = 0.1034
    np.testing.assert_allclose(offset, expected, atol=1e-12)


def test_xrdf_fixes():
    x = description.fix_xrdf(RAW_XRDF, "fr3")
    assert x["tool_frames"] == ["fr3_hand_tcp"]
    assert {"fr3_hand", "fr3_leftfinger", "fr3_rightfinger"} <= set(x["self_collision"]["ignore"]["fr3_link7"])
    assert x["cspace"]["acceleration_limits"] == [10.0] * 7 and x["cspace"]["jerk_limits"] == [5000.0] * 7
    world = x["geometry"][x["world_collision"]["geometry"]]["spheres"]
    assert "fr3_link0" not in world and x["geometry"]["fr3_self_only"]["spheres"]["fr3_link0"]
    assert x["geometry"][x["self_collision"]["geometry"]]["clone"] == [x["world_collision"]["geometry"], "fr3_self_only"]
    assert "fr3_link0" in RAW_XRDF["geometry"][RAW_XRDF["world_collision"]["geometry"]]["spheres"]  # input untouched


def test_spheres_land_where_the_xrdf_test_put_them():
    """Re-expressed centres equal the XRDF test's fr3_fixed_usd_export_mimicfix_frames.xrdf."""
    _, xrdf, _ = description.build(RAW_URDF, RAW_XRDF, usd_poses(), "fr3")
    reference = yaml.safe_load((DATA / "fr3_spheres_reference.xrdf").read_text())
    ref = reference["geometry"][reference["world_collision"]["geometry"]]["spheres"]
    ours = {**xrdf["geometry"][xrdf["world_collision"]["geometry"]]["spheres"], **xrdf["geometry"]["fr3_self_only"]["spheres"]}
    assert set(ours) == set(ref)
    for link in ref:
        np.testing.assert_allclose([s["center"] for s in ours[link]], [s["center"] for s in ref[link]], atol=1e-6, err_msg=link)


def test_inertials_move_with_their_frame():
    X = np.eye(4)
    X[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 deg about z
    X[:3, 3] = [0.1, 0, 0]
    I = np.diag([1.0, 2.0, 3.0])
    (m, c, J), = description.to_urdf_frames({"l": 2.0}, {"l": [1.0, 0, 0]}, {"l": I}, {"l": X}).values()
    assert m == 2.0
    np.testing.assert_allclose(c, [0.1, 1.0, 0.0])
    np.testing.assert_allclose(J, np.diag([2.0, 1.0, 3.0]))


@pytest.mark.skipif(not description.FRANKA_DESCRIPTION.exists(), reason="no ~/ros2_ws/src/franka_description")
def test_franka_inertials_cover_the_arm_and_the_hand():
    fer = description.franka_inertials("fer")
    assert set(fer) == {f"panda_link{i}" for i in range(8)} | {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
    assert fer["panda_hand"][0] == pytest.approx(0.6544)
    np.testing.assert_allclose(fer["panda_rightfinger"][1], fer["panda_leftfinger"][1])
```

- [ ] **Step 3: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_description.py`
Expected: collection error `ImportError: cannot import name 'description' from 'guide_motion'` (raised by `fr3_fixture`).

- [ ] **Step 4: Implement**

`guide_motion/guide_motion/description.py`:
```python
"""Robot description pipeline (S2b spec 4.2): Isaac USD -> URDF + XRDF for guide_motion.

Run once per robot, in the Isaac venv, with no other Isaac process running:

    OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER ~/ros2_ws/.venv/bin/python -m guide_motion.description --robot fr3

Only main() starts Kit. Everything above it is plain Python, tested without Isaac.
"""

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PIPELINE_VERSION = 1
ROBOTS_DIR = Path(__file__).parent / "robots"
FRANKA_DESCRIPTION = Path.home() / "ros2_ws" / "src" / "franka_description"
HOME = [0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398]  # block_bin init.yaml
TCP_Z = 0.1034  # franka_description franka_hand.xacro tcp_xyz
ROBOTS = {
    # acceleration / jerk: franka_description for the FR3, Isaac's shipped Panda XRDF for the Panda
    "fr3": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        "accel": [10.0] * 7,
        "jerk": [5000.0] * 7,
        "datasheet": "fr3",
    },
    "panda": {
        "usd": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "accel": [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0],
        "jerk": [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0],
        "datasheet": "fer",
    },
}
_DRIVE = re.compile(r"\s*isaac:source_drive\s+(\{.*\})\s*$", re.S)


def _num(x) -> str:
    return f"{float(x):.10g}"


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text, ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def fix_urdf(text: str, robot: str) -> str:
    """The USD exporter's URDF with the XRDF test's defects fixed.

    A: mimic multiplier -1 -> +1. B: prismatic velocity had the degree conversion -> undone.
    C: no effort -> the USD drive's max force (a mimic joint takes its leader's).
    Joint limits stay the USD's. Visual and collision geometry are dropped: collision is
    the XRDF's spheres, and OSCBF's URDF parser would load every mesh. A missing
    <robot>_hand_tcp (the Panda USD has none) is added 0.1034 m along the hand's z.
    """
    root = _parse(text)
    joints = {j.get("name"): j for j in root.iter("joint")}
    effort = {}
    for name, joint in joints.items():
        for child in joint:
            match = child.tag is ET.Comment and _DRIVE.match(child.text or "")
            if match:
                force = json.loads(match.group(1))["drive"]["max_force"]
                if math.isfinite(force) and force > 0:
                    effort[name] = force
    for name, joint in joints.items():
        mimic, limit = joint.find("mimic"), joint.find("limit")
        if mimic is not None and mimic.get("multiplier") == "-1":
            mimic.set("multiplier", "1")
        if limit is None:
            continue
        if joint.get("type") == "prismatic":
            limit.set("velocity", _num(float(limit.get("velocity")) * 180.0 / math.pi))
        leader = mimic.get("joint") if mimic is not None else name
        limit.set("effort", _num(effort[name] if name in effort else effort[leader]))
    for link in root.iter("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    tcp = f"{robot}_hand_tcp"
    if root.find(f"link[@name='{tcp}']") is None:
        ET.SubElement(root, "link", name=tcp)
        joint = ET.SubElement(root, "joint", name=f"{tcp}_joint", type="fixed")
        ET.SubElement(joint, "origin", xyz=f"0 0 {TCP_Z}", rpy="0 0 0")
        ET.SubElement(joint, "parent", link=f"{robot}_hand")
        ET.SubElement(joint, "child", link=tcp)
    return ET.tostring(root, encoding="unicode")


def fix_xrdf(xrdf: dict, robot: str) -> dict:
    """The editor's XRDF with tool frame, link7 ignore line, real limits, link0 self-only."""
    x = copy.deepcopy(xrdf)
    x["tool_frames"] = [f"{robot}_hand_tcp"]
    ignore = x["self_collision"].setdefault("ignore", {})
    hand = {f"{robot}_hand", f"{robot}_leftfinger", f"{robot}_rightfinger"}
    ignore[f"{robot}_link7"] = sorted(set(ignore.get(f"{robot}_link7", [])) | hand)
    x["cspace"]["acceleration_limits"] = list(ROBOTS[robot]["accel"])
    x["cspace"]["jerk_limits"] = list(ROBOTS[robot]["jerk"])
    # link0's spheres reach below the base plane: self-collision only, as Isaac's Panda XRDF does
    world = x["world_collision"]["geometry"]
    base = f"{robot}_link0"
    x["geometry"][f"{robot}_self_only"] = {"spheres": {base: x["geometry"][world]["spheres"].pop(base)}}
    x["geometry"][f"{robot}_self"] = {"clone": [world, f"{robot}_self_only"]}
    x["self_collision"]["geometry"] = f"{robot}_self"
    return x


def link_poses(urdf_text: str, root: str | None = None) -> dict:
    """Every frame's 4x4 pose at q = 0, relative to `root` (default: the URDF's root link)."""
    import pinocchio as pin

    model = pin.buildModelFromXML(urdf_text)
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    origin = data.oMf[model.getFrameId(root)].inverse() if root else pin.SE3.Identity()
    return {frame.name: (origin * data.oMf[i]).homogeneous for i, frame in enumerate(model.frames)}


def usd_to_urdf(usd_poses: dict, urdf_text: str) -> dict:
    """Per link, the fixed transform taking USD body-frame coordinates into the URDF link frame.

    usd_poses: each link prim's pose at q = 0 relative to the root link prim (timeline stopped).
    """
    urdf = link_poses(urdf_text)
    return {name: np.linalg.inv(urdf[name]) @ np.asarray(T, float) for name, T in usd_poses.items() if name in urdf}


def reframe_spheres(xrdf: dict, frames: dict) -> dict:
    """Sphere centres authored in USD body frames, re-expressed in the URDF's link frames."""
    x = copy.deepcopy(xrdf)
    for group in x["geometry"].values():
        for link, spheres in group.get("spheres", {}).items():
            X = frames[link]
            for sphere in spheres:
                sphere["center"] = [float(v) for v in (X @ np.r_[sphere["center"], 1.0])[:3]]
    return x


def to_urdf_frames(masses: dict, coms: dict, inertias: dict, frames: dict) -> dict:
    """{link: (mass, com, inertia)} with COM and inertia (about the COM) moved by frames[link]."""
    out = {}
    for name, mass in masses.items():
        X = np.asarray(frames.get(name, np.eye(4)), float)
        R = X[:3, :3]
        out[name] = (float(mass), R @ np.asarray(coms[name], float) + X[:3, 3], R @ np.asarray(inertias[name], float) @ R.T)
    return out


def set_inertials(urdf_text: str, inertials: dict, extra_links: dict | None = None) -> str:
    """Each named link's <inertial> replaced; extra_links {name: (parent, T_parent_child)} added as fixed links."""
    root = _parse(urdf_text)
    for name, (parent, T) in (extra_links or {}).items():
        T = np.asarray(T, float)
        ET.SubElement(root, "link", name=name)
        joint = ET.SubElement(root, "joint", name=f"{name}_fixed", type="fixed")
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        ET.SubElement(joint, "origin", xyz=" ".join(map(_num, T[:3, 3])), rpy=" ".join(map(_num, rpy)))
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
    links = {link.get("name"): link for link in root.iter("link")}
    for name, (mass, com, I) in inertials.items():
        link = links[name]
        for element in link.findall("inertial"):
            link.remove(element)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(_num, com)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=_num(mass))
        ET.SubElement(
            inertial, "inertia",
            ixx=_num(I[0, 0]), ixy=_num(I[0, 1]), ixz=_num(I[0, 2]),
            iyy=_num(I[1, 1]), iyz=_num(I[1, 2]), izz=_num(I[2, 2]),
        )
    return ET.tostring(root, encoding="unicode")


def franka_inertials(arm: str, fd: Path = FRANKA_DESCRIPTION) -> dict:
    """franka_description's {'<prefix>_<link>': (mass, com, inertia)}, arm 'fer' (Panda) or 'fr3', plus the hand."""
    prefix = {"fer": "panda", "fr3": "fr3"}[arm]
    out = {}
    for path in (fd / "robots" / arm / "inertials.yaml", fd / "end_effectors" / "franka_hand" / "inertials.yaml"):
        for name, entry in yaml.safe_load(path.read_text()).items():
            assert entry["origin"]["rpy"].split() == ["0", "0", "0"], (path, name)
            i = entry["inertia"]
            out[f"{prefix}_{name}"] = (
                float(entry["mass"]),
                np.array(entry["origin"]["xyz"].split(), float),
                np.array([[i["xx"], i["xy"], i["xz"]], [i["xy"], i["yy"], i["yz"]], [i["xz"], i["yz"], i["zz"]]], float),
            )
    return out


def build(raw_urdf: str, raw_xrdf: dict, usd_poses: dict, robot: str):
    """Kit-free half of the pipeline: (urdf text, xrdf dict, description dict)."""
    urdf = fix_urdf(raw_urdf, robot)
    frames = usd_to_urdf(usd_poses, urdf)
    xrdf = reframe_spheres(fix_xrdf(raw_xrdf, robot), frames)
    description = {
        "robot": robot,
        "pipeline_version": PIPELINE_VERSION,
        "usd_to_urdf": {name: X.tolist() for name, X in frames.items()},
    }
    return urdf, xrdf, description
```

- [ ] **Step 5: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_description.py`
Expected: `8 passed`. (The prototype reproduced the reference sphere centres to 9e-8 m.)

- [ ] **Step 6: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/description.py guide_motion/test/fr3_fixture.py guide_motion/test/test_description.py
git commit -m "feat(guide_motion): description fixes for the USD exporter and the XRDF editor"
```

---

### Task 5: The robot model

**Files:**
- Create: `guide_motion/guide_motion/model.py`, `guide_motion/test/conftest.py`
- Modify: `guide_motion/test/fr3_fixture.py` (append `simulator_mass_properties`)
- Test: `guide_motion/test/test_model.py`

**Interfaces:**
- Consumes: Task 4's `description.build`, `to_urdf_frames`, `set_inertials`, `ROBOTS_DIR`; the vendored `oscbf.core.manipulator.Manipulator`.
- Produces `guide_motion.model.Model`:
  - `Model.load(robot: str | Path) -> Model` (a name under `ROBOTS_DIR` or a directory holding `<name>.urdf`, `<name>.xrdf`, optional `description.json`); attribute `robot_dir: Path`.
  - attributes `urdf_text: str`, `xrdf: dict`, `usd_to_urdf: dict[str, (4,4)]`, `arm_joints: list[str]` (7), `gripper_joint: str` (`<prefix>_finger_joint1`), `tool_frame: str`, `base_link: str` (the URDF root), `lower`, `upper`, `max_effort`, `max_velocity`, `home`: `ndarray(7)`; `ee_offset: (4,4)`; `pin_model`, `pin_data` (Pinocchio, fingers locked at 0); `sphere_centers (S,3)` in their parent joint's frame, `sphere_radii (S,)`, `sphere_groups (S,)` (0 = fixed base, 1..7 = arm joints), `sphere_links: list[str]`, `ignore: set[frozenset[str]]`.
  - `manipulator` (cached property, OSCBF `Manipulator`; JAX is imported here, not at module import).
  - `ee_pose(q) -> (position (3,), rotation (3,3))` and `ee_speed(q, qd) -> float` (Pinocchio; base frame).
  - `set_mass_properties(masses: dict, coms: dict, inertias: dict, extra_links: dict | None = None) -> bool` (simulator values, COM and inertia in USD body frames; False and nothing rebuilt when unchanged).
- Produces test fixtures `fr3_dir` (a robot directory built by `description.build` from the raw fixtures) and `fr3_model` (`Model.load(fr3_dir)` with franka_description's inertials handed over as the simulator's).

- [ ] **Step 1: Extend the helpers, add the fixtures**

Append to `guide_motion/test/fr3_fixture.py`:
```python
def simulator_mass_properties():
    """franka_description's inertials as PhysX would report them: in the (stand-in) USD frames."""
    raw = {}
    for link in ET.fromstring(FD_URDF).iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        raw[link.get("name")] = (
            float(inertial.find("mass").get("value")),
            np.array(inertial.find("origin").get("xyz").split(), float),
            np.array([[i["ixx"], i["ixy"], i["ixz"]], [i["ixy"], i["iyy"], i["iyz"]], [i["ixz"], i["iyz"], i["izz"]]]),
        )
    fd, usd = description.link_poses(FD_URDF, root="fr3_link0"), usd_poses()
    moved = description.to_urdf_frames(
        {n: v[0] for n, v in raw.items()}, {n: v[1] for n, v in raw.items()}, {n: v[2] for n, v in raw.items()},
        {n: np.linalg.inv(usd[n]) @ fd[n] for n in raw},
    )
    return ({n: v[0] for n, v in moved.items()}, {n: v[1] for n, v in moved.items()}, {n: v[2] for n, v in moved.items()})
```

`guide_motion/test/conftest.py`:
```python
import json

import pytest
import yaml

from fr3_fixture import DATA, simulator_mass_properties, usd_poses
from guide_motion import description


@pytest.fixture(scope="session")
def fr3_dir(tmp_path_factory):
    urdf, xrdf, meta = description.build(
        (DATA / "fr3_exporter_raw.urdf").read_text(), yaml.safe_load((DATA / "fr3_editor_raw.xrdf").read_text()), usd_poses(), "fr3"
    )
    d = tmp_path_factory.mktemp("robots") / "fr3"
    d.mkdir()
    (d / "fr3.urdf").write_text(urdf)
    (d / "fr3.xrdf").write_text(yaml.safe_dump(xrdf, sort_keys=False))
    (d / "description.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="session")
def fr3_model(fr3_dir):
    from guide_motion.model import Model

    model = Model.load(fr3_dir)
    model.set_mass_properties(*simulator_mass_properties())
    return model
```

- [ ] **Step 2: Write the failing tests**

`guide_motion/test/test_model.py`:
```python
import numpy as np
import pinocchio as pin

from fr3_fixture import ARM, HOME, fd_dynamics, simulator_mass_properties

RNG = np.random.default_rng(0)


def random_q(model, n=10):
    return RNG.uniform(model.lower, model.upper, size=(n, len(model.lower)))


def test_description_basics(fr3_model):
    assert fr3_model.arm_joints == ARM and fr3_model.gripper_joint == "fr3_finger_joint1"
    assert fr3_model.tool_frame == "fr3_hand_tcp" and fr3_model.base_link == "fr3_link0"
    np.testing.assert_allclose(fr3_model.home, HOME, atol=1e-4)  # the XRDF keeps 4 decimals
    np.testing.assert_allclose(fr3_model.max_effort, [87, 87, 87, 87, 12, 12, 12])
    assert set(np.unique(fr3_model.sphere_groups)) == set(range(8))  # base + 7 joints, hand on joint 7


def test_oscbf_matches_pinocchio_on_the_same_urdf(fr3_model):
    full = pin.buildModelFromXML(fr3_model.urdf_text)
    model = pin.buildReducedModel(full, [full.getJointId(n) for n in full.names[1:] if n not in ARM], pin.neutral(full))
    data = model.createData()
    man = fr3_model.manipulator
    tool = model.getFrameId("fr3_hand_tcp")
    for q in random_q(fr3_model):
        M = pin.crba(model, data, q)
        np.testing.assert_allclose(np.asarray(man.mass_matrix(q)), np.triu(M) + np.triu(M, 1).T, atol=1e-9)
        np.testing.assert_allclose(np.asarray(man.gravity_vector(q)), pin.computeGeneralizedGravity(model, data, q), atol=1e-9)
        pin.framesForwardKinematics(model, data, q)
        np.testing.assert_allclose(np.asarray(man.ee_transform(q)), data.oMf[tool].homogeneous, atol=1e-9)
        p, R = fr3_model.ee_pose(q)
        np.testing.assert_allclose(np.c_[R, p], data.oMf[tool].homogeneous[:3], atol=1e-12)


def test_simulator_mass_properties_give_the_physical_dynamics(fr3_model):
    """Inertials handed over in USD frames reproduce franka_description's M and g."""
    for q in random_q(fr3_model):
        M_ref, g_ref = fd_dynamics(q)
        M, g = np.asarray(fr3_model.manipulator.mass_matrix(q)), np.asarray(fr3_model.manipulator.gravity_vector(q))
        assert np.linalg.norm(M - M_ref) / np.linalg.norm(M_ref) < 1e-6
        assert np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref) < 1e-6


def test_same_mass_properties_keep_the_compiled_model(fr3_model):
    man = fr3_model.manipulator
    assert fr3_model.set_mass_properties(*simulator_mass_properties()) is False
    assert fr3_model.manipulator is man


def test_an_extra_simulator_link_is_folded_into_its_parent(fr3_dir):
    from guide_motion.model import Model

    model = Model.load(fr3_dir)
    masses, coms, inertias = simulator_mass_properties()
    T = np.eye(4)
    T[:3, 3] = [0.05, 0.0, 0.0]
    model.set_mass_properties({**masses, "camera": 0.2}, {**coms, "camera": np.zeros(3)},
                              {**inertias, "camera": 1e-4 * np.eye(3)}, extra_links={"camera": ("fr3_hand", T)})
    g, g_ref = np.asarray(model.manipulator.gravity_vector(HOME)), fd_dynamics(HOME)[1]
    assert np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref) > 1e-3  # 0.2 kg on the hand shows in g
```

- [ ] **Step 3: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_model.py`
Expected: 5 errors, `ModuleNotFoundError: No module named 'guide_motion.model'`.

- [ ] **Step 4: Implement**

`guide_motion/guide_motion/model.py`:
```python
"""A robot for guide_motion: URDF + XRDF -> kinematics, dynamics and collision spheres."""

import json
import tempfile
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from guide_motion import description


class Model:
    """Arm kinematics (Pinocchio, fingers locked) and dynamics (OSCBF, built lazily)."""

    def __init__(self, urdf_text: str, xrdf: dict, usd_to_urdf: dict | None = None):
        self.urdf_text, self.xrdf = urdf_text, xrdf
        self.usd_to_urdf = {name: np.asarray(X, float) for name, X in (usd_to_urdf or {}).items()}
        self.arm_joints = list(xrdf["cspace"]["joint_names"])
        self.gripper_joint = self.arm_joints[0].rsplit("_joint", 1)[0] + "_finger_joint1"
        self.tool_frame = xrdf["tool_frames"][0]
        root = ET.fromstring(urdf_text)
        children = {j.find("child").get("link") for j in root.iter("joint")}
        self.base_link = next(link.get("name") for link in root.iter("link") if link.get("name") not in children)

        full = pin.buildModelFromXML(urdf_text)
        locked = [full.getJointId(n) for n in full.names[1:] if n not in self.arm_joints]
        self.pin_model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.pin_data = self.pin_model.createData()
        assert list(self.pin_model.names[1:]) == self.arm_joints, list(self.pin_model.names[1:])
        self.lower = self.pin_model.lowerPositionLimit.copy()
        self.upper = self.pin_model.upperPositionLimit.copy()
        self.max_effort = self.pin_model.effortLimit.copy()
        self.max_velocity = self.pin_model.velocityLimit.copy()
        self.home = np.array([xrdf["default_joint_positions"][j] for j in self.arm_joints], float)
        assert self.pin_model.existFrame(self.tool_frame), self.tool_frame
        self._tool = self.pin_model.getFrameId(self.tool_frame)
        assert self.pin_model.frames[self._tool].parentJoint == len(self.arm_joints), "TCP must be rigid on the last joint"
        self.ee_offset = self.pin_model.frames[self._tool].placement.homogeneous

        # Self-collision spheres, each in its parent joint's frame; group 0 = the fixed base.
        centers, radii, groups, links = [], [], [], []
        for link, sphere in self._spheres(xrdf["self_collision"]["geometry"]):
            assert self.pin_model.existFrame(link), link
            frame = self.pin_model.frames[self.pin_model.getFrameId(link)]
            centers.append((frame.placement.homogeneous @ np.r_[sphere["center"], 1.0])[:3])
            radii.append(float(sphere["radius"]))
            groups.append(int(frame.parentJoint))
            links.append(link)
        self.sphere_centers, self.sphere_radii = np.array(centers), np.array(radii)
        self.sphere_groups, self.sphere_links = np.array(groups), links
        ignore = xrdf["self_collision"].get("ignore", {})
        self.ignore = {frozenset((a, b)) for a, others in ignore.items() for b in others}

    @classmethod
    def load(cls, robot) -> "Model":
        """`robot`: a name under guide_motion/robots ('fr3', 'panda') or a directory."""
        d = Path(robot)
        if not d.is_dir():
            d = description.ROBOTS_DIR / str(robot)
        meta = json.loads((d / "description.json").read_text()) if (d / "description.json").exists() else {}
        model = cls((d / f"{d.name}.urdf").read_text(), yaml.safe_load((d / f"{d.name}.xrdf").read_text()), meta.get("usd_to_urdf"))
        model.robot_dir = d
        return model

    def _spheres(self, name: str) -> list:
        group = self.xrdf["geometry"][name]
        cloned = [s for other in group.get("clone", []) for s in self._spheres(other)]
        return cloned + [(link, s) for link, spheres in group.get("spheres", {}).items() for s in spheres]

    @cached_property
    def manipulator(self):
        """OSCBF's JAX model (fingers fixed at 0). Lazy: only the control process imports JAX."""
        from oscbf.core.manipulator import Manipulator

        root = ET.fromstring(self.urdf_text)
        for joint in root.iter("joint"):
            if joint.get("type") != "fixed" and joint.get("name") not in self.arm_joints:
                joint.set("type", "fixed")
                for mimic in joint.findall("mimic"):
                    joint.remove(mimic)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
            f.write(ET.tostring(root, encoding="unicode"))
            f.flush()
            return Manipulator.from_urdf(f.name, ee_offset=self.ee_offset)

    def ee_pose(self, q):
        """(position (3,), rotation (3, 3)) of the tool frame in the base frame."""
        pin.framesForwardKinematics(self.pin_model, self.pin_data, np.asarray(q, float))
        T = self.pin_data.oMf[self._tool]
        return T.translation.copy(), T.rotation.copy()

    def ee_speed(self, q, qd) -> float:
        """Linear tool speed, m/s."""
        J = pin.computeFrameJacobian(self.pin_model, self.pin_data, np.asarray(q, float), self._tool, pin.LOCAL_WORLD_ALIGNED)
        return float(np.linalg.norm(J[:3] @ np.asarray(qd, float)))

    def set_mass_properties(self, masses: dict, coms: dict, inertias: dict, extra_links: dict | None = None) -> bool:
        """Use the simulator's mass properties (COM and inertia in USD body frames, keyed by link).

        extra_links {name: (parent, T_parent_child)}: simulator links the URDF lacks (a camera
        mount, say), added as fixed links so OSCBF folds them into their parent.
        Returns False, and keeps the compiled dynamics, when nothing changed.
        """
        inertials = description.to_urdf_frames(masses, coms, inertias, self.usd_to_urdf)
        text = description.set_inertials(self.urdf_text, inertials, extra_links)
        if text == self.urdf_text:
            return False
        self.urdf_text = text
        self.__dict__.pop("manipulator", None)
        return True
```

- [ ] **Step 5: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_model.py`
Expected: `5 passed` (the prototype: M and g within 2e-7 relative of franka_description's, through two different URDFs' link frames).

- [ ] **Step 6: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/model.py guide_motion/test/conftest.py guide_motion/test/fr3_fixture.py guide_motion/test/test_model.py
git commit -m "feat(guide_motion): robot model from URDF + XRDF, simulator mass properties"
```

---

### Task 6: The OSCBF controller

**Files:**
- Create: `guide_motion/guide_motion/control.py`
- Test: `guide_motion/test/test_control.py`

**Interfaces:**
- Consumes: Task 5's `Model` (`manipulator`, `ee_pose`, sphere arrays, `ignore`, `home`, limits); Task 3's `Target`, `JOINTS`, `TWIST`; the vendored `oscbf.core.controllers.PoseTaskTorqueController` / `PoseTaskVelocityController`, `oscbf.core.oscbf_configs.OSCBFTorqueConfig` / `OSCBFVelocityConfig`; `cbfpy.CBF`, `qpax.solve_qp_elastic_primal`.
- Produces `guide_motion.control`:
  - `GAINS = {"torque": {"kp": 150.0, "kd": 2√150, "kp_null": 10.0, "kd_null": 2√10}, "velocity": {"kp_pos": 50.0, "kp_rot": 20.0, "kp_null": 10.0}}`.
  - `BARRIERS = {"workspace": [[-1, 1], [-1, 1], [-0.05, 1.3]], "singularity": 1e-3, "min_group_gap": 2, "softmin_beta": 1000.0}`; `INFEASIBLE_TOL = 1e-2`; `CACHE_DIR = ~/.cache/guide_motion/jax`.
  - `enable_compilation_cache(path=CACHE_DIR) -> None`.
  - `Controller(model, mode="torque" | "velocity", gains: dict | None = None, barriers: dict | None = None, q_rest: array(7) | None = None)` (`q_rest` defaults to `model.home`: the posture every episode starts from). Attributes `mode`, `gains`, `barriers`, `q_rest`, `limit` (URDF effort or velocity limits), `events = {"infeasible": int}`. Methods `warm_up() -> float` (seconds; resets the event count), `step(target: Target, q, qd) -> ndarray(7)` (torques or joint velocities, clamped to `limit`), `barrier_values(q) -> ndarray`.

- [ ] **Step 1: Write the failing tests**

`guide_motion/test/test_control.py`:
```python
import time

import numpy as np
import pytest

from fr3_fixture import HOME
from guide_motion.control import Controller
from guide_motion.targets import JOINTS, TWIST, Target


@pytest.fixture(scope="module", params=["torque", "velocity"])
def controller(request, fr3_model):
    c = Controller(fr3_model, request.param)
    c.warm_up()
    return c


def hold(model, q=HOME):
    p, R = model.ee_pose(q)
    return Target(position=p, rotation=R)


def test_every_barrier_is_satisfied_at_home(controller):
    assert controller.barrier_values(HOME).min() > 0


def test_hold_is_gravity_compensation_or_rest(controller):
    u = controller.step(hold(controller.model), HOME, np.zeros(7))
    if controller.mode == "torque":
        np.testing.assert_allclose(u, np.asarray(controller.model.manipulator.gravity_vector(HOME)), atol=0.02)
    else:
        assert np.abs(u).max() < 0.05


def test_recorded_inputs_give_the_same_command_twice(controller):
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, qd = HOME + 0.1 * rng.standard_normal(7), 0.1 * rng.standard_normal(7)
        target = hold(controller.model, HOME + 0.05 * rng.standard_normal(7))
        assert np.array_equal(controller.step(target, q, qd), controller.step(target, q, qd))


def test_a_target_ahead_pulls_the_tool_forward(controller):
    model = controller.model
    p, R = model.ee_pose(HOME)
    u = controller.step(Target(position=p + [0.01, 0, 0], rotation=R), HOME, np.zeros(7))
    J = np.asarray(model.manipulator.ee_jacobian(HOME))[:3]
    if controller.mode == "torque":
        man = model.manipulator
        u = np.linalg.solve(np.asarray(man.mass_matrix(HOME)), u - np.asarray(man.gravity_vector(HOME)))
    assert (J @ u)[0] > 0 and abs((J @ u)[0]) > 5 * np.abs((J @ u)[1:]).max()


def test_twist_and_joint_targets(controller):
    u = controller.step(Target(mode=TWIST, twist=np.r_[0.05, 0, 0, 0, 0, 0]), HOME, np.zeros(7))
    assert np.all(np.isfinite(u))
    u = controller.step(Target(mode=JOINTS, joints=HOME), HOME, np.zeros(7))
    assert np.all(np.isfinite(u))


def test_infeasible_step_falls_back_to_damping(controller, monkeypatch):
    before = controller.events["infeasible"]
    monkeypatch.setattr(controller, "_fn", lambda *a: (np.full(7, np.nan), 0.0))
    qd = np.full(7, 0.1)
    u = controller.step(hold(controller.model), HOME, qd)
    assert controller.events["infeasible"] == before + 1
    np.testing.assert_allclose(u, -controller.gains["kd_null"] * qd if controller.mode == "torque" else 0.0)


def test_command_is_clamped_to_the_urdf_limits(controller, monkeypatch):
    monkeypatch.setattr(controller, "_fn", lambda *a: (np.full(7, 1e3), 0.0))
    np.testing.assert_allclose(controller.step(hold(controller.model), HOME, np.zeros(7)), controller.limit)


def test_step_cost_regression(controller):
    """Standalone guard only; the RTF gate is the Isaac check (Plan B)."""
    target, rng = hold(controller.model), np.random.default_rng(2)
    times = []
    for _ in range(200):
        q = HOME + 0.01 * rng.standard_normal(7)
        start = time.perf_counter()
        controller.step(target, q, np.zeros(7))
        times.append(time.perf_counter() - start)
    assert np.mean(times[20:]) < (0.5e-3 if controller.mode == "torque" else 0.3e-3), np.mean(times[20:])


def test_a_state_past_a_joint_limit_gives_a_finite_inward_command(controller):
    q = HOME.copy()
    q[3] = controller.model.upper[3] + 0.01  # joint 4 a hair past its upper limit (a measurement can be)
    u = controller.step(hold(controller.model, HOME), q, np.zeros(7))
    assert np.all(np.isfinite(u))
    if controller.mode == "velocity":
        assert u[3] < 0


def test_float32_states_give_the_float64_command(controller):
    q32 = (HOME + 0.01).astype(np.float32)
    target = hold(controller.model)
    np.testing.assert_array_equal(controller.step(target, q32, np.zeros(7, np.float32)),
                                  controller.step(target, q32.astype(np.float64), np.zeros(7)))
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_control.py`
Expected: collection error `ModuleNotFoundError: No module named 'guide_motion.control'`.

- [ ] **Step 3: Implement**

The step is the measurement's `oscbf_ctrl.make_control` (`/home/user/libero_assets/s2b_inputs/oscbf_isaac/oscbf_ctrl.py`) with this spec's gains, barriers and fallbacks. `guide_motion/guide_motion/control.py`:
```python
"""OSCBF control (S2b spec 4.1, 5): one small QP per step, torque or joint velocity."""

import math
import time
from pathlib import Path

import numpy as np

from guide_motion.targets import JOINTS, TWIST, Target

GAINS = {
    # LIBERO-like: robosuite OSC_POSE as LIBERO configures it (kp 150, zeta 1, all six axes);
    # nullspace PD towards the posture at the episode start.
    "torque": {"kp": 150.0, "kd": 2 * math.sqrt(150.0), "kp_null": 10.0, "kd_null": 2 * math.sqrt(10.0)},
    # OSCBF's own velocity gains (its dynamic_motion.py).
    "velocity": {"kp_pos": 50.0, "kp_rot": 20.0, "kp_null": 10.0},
}
BARRIERS = {
    "workspace": [[-1.0, 1.0], [-1.0, 1.0], [-0.05, 1.3]],  # tool box in the base frame, m
    "singularity": 1e-3,  # prod of the Jacobian's singular values
    "min_group_gap": 2,  # self-collision between link groups at least this many joints apart
    "softmin_beta": 1000.0,  # 1/m; the soft minimum undershoots the true gap by <= ln(pairs)/beta
}
INFEASIBLE_TOL = 1e-2  # a QP row the relaxed solution breaks by more than this = infeasible
CACHE_DIR = Path.home() / ".cache" / "guide_motion" / "jax"


def enable_compilation_cache(path=CACHE_DIR) -> None:
    """JAX's persistent cache: a restart loads the compiled step instead of recompiling."""
    import jax

    Path(path).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


class Controller:
    """step(target, q, qd) -> joint torques (torque mode) or joint velocities (velocity mode)."""

    def __init__(self, model, mode: str = "torque", gains=None, barriers=None, q_rest=None):
        assert mode in GAINS, mode
        self.model, self.mode = model, mode
        self.gains = {**GAINS[mode], **(gains or {})}
        self.barriers = {**BARRIERS, **(barriers or {})}
        self.q_rest = np.asarray(model.home if q_rest is None else q_rest, float)
        self.limit = model.max_effort if mode == "torque" else model.max_velocity
        self.events = {"infeasible": 0}
        self._fn, self._h = _build(model, mode, self.gains, self.barriers)

    def warm_up(self) -> float:
        """Compile the step (seconds taken). Without it the first physics step stalls."""
        start = time.perf_counter()
        position, rotation = self.model.ee_pose(self.q_rest)
        self.step(Target(position=position, rotation=rotation), self.q_rest, np.zeros_like(self.q_rest))
        self.events["infeasible"] = 0
        return time.perf_counter() - start

    def step(self, target: Target, q, qd) -> np.ndarray:
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        des_q, twist = self.q_rest, np.zeros(6)
        if target.mode == JOINTS:
            des_q = np.asarray(target.joints, float)
            position, rotation = self.model.ee_pose(des_q)
        elif target.mode == TWIST:
            position, rotation = self.model.ee_pose(q)
            twist = np.asarray(target.twist, float)
        else:
            position, rotation = target.position, target.rotation
        u, violation = self._fn(q, qd, position, rotation, twist[:3], twist[3:], des_q)
        u = np.asarray(u)
        if not np.all(np.isfinite(u)) or float(violation) > INFEASIBLE_TOL:
            self.events["infeasible"] += 1
            u = -self.gains["kd_null"] * qd if self.mode == "torque" else np.zeros_like(q)
        return np.clip(u, -self.limit, self.limit)

    def barrier_values(self, q) -> np.ndarray:
        """Every barrier h(q); all > 0 means q is inside the safe set."""
        return np.asarray(self._h(np.asarray(q, float)))


def _barrier_fn(model, man, b):
    import jax
    import jax.numpy as jnp

    C = np.c_[model.sphere_centers, np.ones(len(model.sphere_radii))]
    r, group, links = model.sphere_radii, model.sphere_groups, model.sphere_links
    pairs = {}
    for a in range(len(r)):
        for c in range(a + 1, len(r)):
            if abs(group[a] - group[c]) >= b["min_group_gap"] and frozenset((links[a], links[c])) not in model.ignore:
                pairs.setdefault((min(group[a], group[c]), max(group[a], group[c])), []).append((a, c))
    index = [(np.array([p[0] for p in v]), np.array([p[1] for p in v])) for v in pairs.values()]
    lo, hi = np.array(man.joint_lower_limits), np.array(man.joint_upper_limits)
    box, beta = np.asarray(b["workspace"], float), float(b["softmin_beta"])

    def h(q):
        T = jnp.concatenate([jnp.eye(4)[None], man.joint_to_world_transforms(q)])  # [base, joint 1..7]
        p = jnp.einsum("sij,sj->si", T[group], C)[:, :3]
        # One barrier per pair of link groups: a smooth minimum of their sphere gaps (never above the true one).
        selfcol = [-jax.nn.logsumexp(-beta * (jnp.linalg.norm(p[i] - p[j], axis=1) - r[i] - r[j])) / beta for i, j in index]
        tool = man.ee_position(q)
        sigma = jax.lax.linalg.svd(man.ee_jacobian(q), compute_uv=False)
        return jnp.concatenate([
            jnp.stack(selfcol) if selfcol else jnp.zeros(0),
            hi - q, q - lo,
            tool - box[:, 0], box[:, 1] - tool,
            jnp.array([jnp.prod(sigma) - b["singularity"]]),
        ])

    return h


def _build(model, mode, gains, barriers):
    import jax
    import jax.numpy as jnp
    import qpax
    from cbfpy import CBF
    from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController
    from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig

    man = model.manipulator
    n = man.num_joints
    h = _barrier_fn(model, man, barriers)
    z3 = jnp.zeros(3)
    if mode == "torque":

        @jax.tree_util.register_static
        class Config(OSCBFTorqueConfig):
            def __init__(self):
                super().__init__(man, rot_obj_weight=0.1)

            def h_2(self, z, **kwargs):
                return h(z[:n])

            def alpha(self, x):
                return 10.0 * x

            def alpha_2(self, x):
                return 10.0 * x

        task = PoseTaskTorqueController(n, gains["kp"], gains["kd"], gains["kp_null"], gains["kd_null"], None, None)

        def nominal(q, qd, p, R, v, w, des_q):
            M, M_inv, g, c, J, T = man.torque_control_matrices(q, qd)
            u = task(q, qd, T[:3, 3], T[:3, :3], p, R, v, w, z3, z3, des_q, jnp.zeros(n), J, M, M_inv, g, c)
            return jnp.concatenate([q, qd]), u

    else:

        @jax.tree_util.register_static
        class Config(OSCBFVelocityConfig):
            def __init__(self):
                super().__init__(man, rot_obj_weight=0.1)

            def h_1(self, z, **kwargs):
                return h(z)

            def alpha(self, x):
                return 10.0 * x

        kp = np.r_[np.full(3, gains["kp_pos"]), np.full(3, gains["kp_rot"])]
        task = PoseTaskVelocityController(n, kp, gains["kp_null"], None, None)

        def nominal(q, qd, p, R, v, w, des_q):
            M_inv, J, T = man.dynamically_consistent_velocity_control_matrices(q)
            return q, task(q, T[:3, 3], T[:3, :3], p, R, v, w, des_q, J, M_inv)

    cbf = CBF.from_config(Config())

    @jax.jit
    def fn(q, qd, p, R, v, w, des_q):
        z, u_nom = nominal(q, qd, p, R, v, w, des_q)
        # CBF.safety_filter's relaxed QP, spelled out so the step also reports how far the
        # solution breaks a row (the relaxed QP never says "infeasible" by itself).
        P, c, _, _, G, hq = cbf.qp_data(z, u_nom)
        penalty = jnp.asarray(cbf.constraint_relaxation_penalties)
        u = qpax.solve_qp_elastic_primal(P, c, G, hq, penalty=penalty, solver_tol=cbf.solver_tol)
        return u, jnp.max(G @ u - hq)

    return fn, jax.jit(h)
```
Note: `Controller.step` reads `gains["kd_null"]` in velocity mode too; `GAINS["velocity"]` has no `kd_null`, so the velocity branch must not evaluate it — the conditional expression above only evaluates `-self.gains["kd_null"] * qd` in torque mode.

- [ ] **Step 4: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_control.py`
Expected: `20 passed` (10 tests × 2 modes) in about 30 s, most of it the two compilations. The prototype measured torque 0.30 ms and velocity 0.19 ms per step. If `test_every_barrier_is_satisfied_at_home` fails, print `controller.barrier_values(HOME)` and stop: a self-collision pair is too close at home, and the fix is in the description (Task 8), not the threshold.

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/control.py guide_motion/test/test_control.py
git commit -m "feat(guide_motion): OSCBF torque and velocity control with LIBERO-like gains"
```

---

### Task 7: The Pink IK planner

**Files:**
- Create: `guide_motion/guide_motion/planner.py`
- Test: `guide_motion/test/test_planner.py`

**Interfaces:**
- Consumes: Task 5's `Model` (`pin_model`, `tool_frame`, `ee_pose`, `lower`, `upper`); Task 3's `Target`, `POSE`, `JOINTS`.
- Produces `guide_motion.planner`: constants `DT = 0.05`, `SPEED = 0.1` (m/s), `ANGULAR_SPEED = 1.0`, `JOINT_SPEED = 1.0` (rad/s); `class PlanningError(RuntimeError)`; `plan(model, goal: Target, q) -> list[Target]` (≥ 1 target, 50 ms apart; the last one is `goal` itself for POSE goals; every target carries `goal.gripper_width`).

- [ ] **Step 1: Write the failing tests**

`guide_motion/test/test_planner.py`:
```python
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fr3_fixture import HOME
from guide_motion.planner import DT, SPEED, PlanningError, plan
from guide_motion.targets import JOINTS, POSE, Target


def test_one_step_goal_is_sent_as_is(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p + [0.002, 0, 0], rotation=R, gripper_width=0.08)
    assert plan(fr3_model, goal, HOME) == [goal]


def test_pose_goal_is_a_reachable_line_ending_at_the_goal(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p + [0.1, 0.15, -0.2], rotation=Rotation.from_euler("z", 0.6).as_matrix() @ R, gripper_width=0.03)
    targets = plan(fr3_model, goal, HOME)
    assert targets[-1] is goal and all(t.mode == POSE and t.gripper_width == 0.03 for t in targets)
    steps = np.linalg.norm(np.diff([t.position for t in targets], axis=0), axis=1)
    assert steps.max() < 1.2 * SPEED * DT
    assert len(targets) >= np.linalg.norm(goal.position - p) / (SPEED * DT)


def test_unreachable_goal_raises(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    with pytest.raises(PlanningError):
        plan(fr3_model, Target(position=p + [1.5, 0, 0], rotation=R), HOME)


def test_joint_goal_is_interpolated(fr3_model):
    goal = HOME + np.r_[0.5, 0, 0, 0, 0, 0, -0.2]
    targets = plan(fr3_model, Target(mode=JOINTS, joints=goal), HOME)
    assert len(targets) == 10 and all(t.mode == JOINTS for t in targets)
    np.testing.assert_allclose(targets[-1].joints, goal)


def test_joint_goal_outside_the_limits_raises(fr3_model):
    with pytest.raises(PlanningError):
        plan(fr3_model, Target(mode=JOINTS, joints=fr3_model.upper + 0.1), HOME)


def test_a_half_turn_of_the_hand_is_planned(fr3_model):
    p, R = fr3_model.ee_pose(HOME)
    goal = Target(position=p, rotation=Rotation.from_euler("z", np.radians(179)).as_matrix() @ R)
    targets = plan(fr3_model, goal, HOME)
    assert targets[-1] is goal and all(np.all(np.isfinite(t.rotation)) for t in targets)
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_planner.py`
Expected: collection error `ModuleNotFoundError: No module named 'guide_motion.planner'`.

- [ ] **Step 3: Implement**

The Pink calls follow `~/ros2_ws/src/block_bin_eval/block_bin_eval/eval_policy_pink.py` (`ArmIK.solve`: quadprog, `safety_break=False`, fingers locked out of the model). `guide_motion/guide_motion/planner.py`:
```python
"""Pink IK planner (S2b spec 4.1): a goal more than one 20 Hz step away -> a list of Targets."""

import math

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation, Slerp

from guide_motion.targets import JOINTS, POSE, Target

DT = 0.05  # one target step, s (20 Hz)
SPEED = 0.1  # m/s along the line; the send_action timeout assumes this speed
ANGULAR_SPEED = 1.0  # rad/s
JOINT_SPEED = 1.0  # rad/s, joint goals
POS_TOL, ROT_TOL = 1e-3, math.radians(0.5)  # IK has reached the goal
EXTRA_STEPS = 20  # IK steps allowed at the goal before giving up


class PlanningError(RuntimeError):
    pass


def plan(model, goal: Target, q) -> list:
    """Targets 50 ms apart from the configuration q to `goal`, the last one the goal itself.

    POSE goals follow a straight line (slerp for the orientation) at SPEED, each waypoint
    tracked by Pink IK under the URDF joint limits, so every target is reachable. JOINTS
    goals are interpolated in joint space. TWIST goals pass through unchanged.
    """
    q = np.asarray(q, float)
    width = goal.gripper_width
    if goal.mode == JOINTS:
        goal_q = np.asarray(goal.joints, float)
        if np.any(goal_q < model.lower) or np.any(goal_q > model.upper):
            raise PlanningError(f"joint goal outside the URDF limits: {np.round(goal_q, 4).tolist()}")
        dq = goal_q - q
        n = max(1, math.ceil(np.abs(dq).max() / (JOINT_SPEED * DT)))
        return [Target(JOINTS, joints=q + dq * k / n, gripper_width=width) for k in range(1, n + 1)]
    if goal.mode != POSE:
        return [goal]
    p0, R0 = model.ee_pose(q)
    n = math.ceil(max(
        np.linalg.norm(goal.position - p0) / (SPEED * DT),
        Rotation.from_matrix(goal.rotation @ R0.T).magnitude() / (ANGULAR_SPEED * DT),
    ))
    if n <= 1:
        return [goal]
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R0, goal.rotation]))
    configuration = pink.Configuration(model.pin_model, model.pin_model.createData(), q.copy())
    task = FrameTask(model.tool_frame, position_cost=1.0, orientation_cost=1.0)
    posture = PostureTask(cost=1e-3)
    posture.set_target(q)
    targets = []
    for k in range(1, n + EXTRA_STEPS + 1):
        s = min(1.0, k / n)
        task.set_target(pin.SE3(slerp(s).as_matrix(), (1.0 - s) * p0 + s * goal.position))
        for _ in range(3):
            velocity = pink.solve_ik(configuration, [task, posture], DT, solver="quadprog", safety_break=False)
            configuration.integrate_inplace(velocity, DT)
        T = configuration.get_transform_frame_to_world(model.tool_frame)
        if s < 1.0:
            targets.append(Target(POSE, T.translation.copy(), T.rotation.copy(), gripper_width=width))
        elif (np.linalg.norm(T.translation - goal.position) < POS_TOL
              and Rotation.from_matrix(T.rotation @ goal.rotation.T).magnitude() < ROT_TOL):
            return targets + [goal]
    raise PlanningError(f"IK did not reach the goal within {EXTRA_STEPS} steps (reach or joint limits)")
```
The model's root is the base link (`fr3_link0` / `panda_link0`), so Pink's world frame is the base frame the targets are given in: no root correction as in `eval_policy_pink._to_root` is needed.

- [ ] **Step 4: Run them to see them pass**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_planner.py`
Expected: `6 passed` (the prototype planned a 27 cm move with a 34° turn as 54 targets, ≤ 5 mm apart, in 31 ms).

- [ ] **Step 5: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/planner.py guide_motion/test/test_planner.py
git commit -m "feat(guide_motion): Pink IK planner for pose and joint goals"
```

---

### Task 8: The pipeline's Kit CLI, run for the FR3

**Files:**
- Modify: `guide_motion/guide_motion/description.py` (append `principal_inertia`, `_write_mass_override`, `main`)
- Modify: `guide_motion/test/test_description.py` (append one test)
- Create (generated, committed): `guide_motion/guide_motion/robots/fr3/{fr3.urdf,fr3.xrdf,description.json}`

**Interfaces:**
- Consumes: Tasks 4–5 (`build`, `set_inertials`, `to_urdf_frames`, `franka_inertials`, `Model`); Isaac's `UsdToUrdfConverter`, `isaacsim.robot_setup.xrdf_editor` (`EditorState`, `articulation_discovery`, `sphere_generation`), `isaacsim.robot_motion.cumotion.load_cumotion_robot`, `cumotion.create_robot_world_inspector`, `isaacsim.core.experimental.prims.Articulation` (getters as in the measurement's `run.py`).
- Produces: `principal_inertia(inertia (3,3)) -> (diagonal (3,), quaternion wxyz (4,))`; `python -m guide_motion.description --robot {fr3,panda} [--spheres 6] [--radius-offset 0.01] [--franka-description PATH]`, which writes `ROBOTS_DIR/<robot>/` only when every check passes, else exits 1 naming the check. `description.json` keys: `robot`, `pipeline_version`, `usd_to_urdf`, `source_usd`, `spheres_per_link`, `radius_offset`, `checks` (`cumotion_load`, `home_self_collision`, `random_self_collision`, `mass_matrix_rel`, `gravity_rel`), `mass_check_kg` (`{link: [PhysX, franka_description]}`).

- [ ] **Step 1: Write the failing test for the pure helper**

Append to `guide_motion/test/test_description.py`:
```python
def test_principal_inertia_rebuilds_the_tensor():
    from scipy.spatial.transform import Rotation

    I = np.array([[0.0090, 0.0, 0.0020], [0.0, 0.0115, 0.0], [0.0020, 0.0, 0.0085]])  # franka_description fer link0
    diagonal, (w, x, y, z) = description.principal_inertia(I)
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    np.testing.assert_allclose(R @ np.diag(diagonal) @ R.T, I, atol=1e-12)
```
Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_description.py -k principal`
Expected: FAIL, `AttributeError: module 'guide_motion.description' has no attribute 'principal_inertia'`.

- [ ] **Step 2: Implement the Kit half**

Append to `guide_motion/guide_motion/description.py` (the Kit calls are `make_fr3_xrdf.py`'s, the getters `run.py`'s):
```python
def principal_inertia(inertia):
    """USD's (diagonalInertia, principalAxes wxyz) for a full 3x3 inertia about the COM."""
    diagonal, axes = np.linalg.eigh(np.asarray(inertia, float))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    return diagonal, np.array([w, x, y, z])


def _write_mass_override(path: str, source: str, inertials: dict) -> None:
    """A wrapper USD that references `source` and overrides its links' mass properties (Kit's pxr)."""
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    root = stage.DefinePrim("/" + Path(path).stem, "Xform")
    root.GetReferences().AddReference(source)
    stage.SetDefaultPrim(root)
    links = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.MassAPI)}
    for name, (mass, com, inertia) in inertials.items():
        diagonal, (w, x, y, z) = principal_inertia(inertia)
        api = UsdPhysics.MassAPI(links[name])
        api.CreateMassAttr().Set(float(mass))
        api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*(float(v) for v in com)))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v) for v in diagonal)))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
    stage.GetRootLayer().Save()


def main(argv=None) -> None:
    import argparse
    import os
    import shutil
    import sys
    import tempfile

    parser = argparse.ArgumentParser(description="Build guide_motion/robots/<robot> from Isaac's robot USD.")
    parser.add_argument("--robot", choices=sorted(ROBOTS), required=True)
    parser.add_argument("--spheres", type=int, default=6, help="collision spheres per link (the XRDF test used 6)")
    parser.add_argument("--radius-offset", type=float, default=0.01, help="the editor UI's default")
    parser.add_argument("--franka-description", type=Path, default=FRANKA_DESCRIPTION)
    args = parser.parse_args(argv)
    robot = args.robot

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from isaacsim.core.utils.extensions import enable_extension

    for extension in ("isaacsim.robot_setup.xrdf_editor", "isaacsim.asset.exporter.urdf",
                      "isaacsim.robot_motion.cumotion", "isaacsim.storage.native"):
        enable_extension(extension)
    app.update()

    import cumotion
    import isaacsim.core.experimental.utils.xform as xform_utils
    import isaacsim.robot_motion.cumotion as cu
    import omni.timeline
    import omni.usd
    from isaacsim.asset.exporter.urdf.converter import UsdToUrdfConverter
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.robot_setup.xrdf_editor import EditorState, articulation_discovery, sphere_generation
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdPhysics

    from guide_motion.model import Model

    def fail(check: str, detail) -> None:
        print(f"[guide_motion.description] {robot}: check '{check}' FAILED: {detail}; nothing written", flush=True)
        os._exit(1)  # skip Kit's teardown: nothing to save

    tmp = Path(tempfile.mkdtemp(prefix=f"guide_motion_{robot}_"))
    source = get_assets_root_path() + ROBOTS[robot]["usd"]
    stage_path = source
    if robot == "panda":  # the shipped inertias are ~1e-7 kg m^2: torque control chattered (spec 4.2.3)
        stage_path = str(tmp / "panda.usda")
        _write_mass_override(stage_path, source, franka_inertials("fer", args.franka_description))
    context = omni.usd.get_context()
    context.open_stage(stage_path)
    while context.get_stage_loading_status()[2] > 0:
        app.update()
    stage = context.get_stage()
    while any(prim.IsInstanceable() for prim in stage.Traverse()):  # instanced meshes cannot feed the sphere generator
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
    app.update()
    base = articulation_discovery.find_all_articulation_base_paths(stage)[0]
    bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(base)) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    link0 = str(bodies[f"{robot}_link0"].GetPath())

    # Timeline stopped: authored link poses, and an export that does not bake the current pose (defect D).
    usd_poses = {name: np.asarray(xform_utils.get_relative_transform(prim, link0)) for name, prim in bodies.items()}
    raw = tmp / "raw"
    raw.mkdir()
    UsdToUrdfConverter(stage=stage, root_prim_path=base, mesh_dir_name="meshes", mesh_path_prefix="./").convert(str(raw / f"{robot}.urdf"))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        app.update()
    editor = EditorState()
    editor.select_articulation(base)
    for link, meshes in editor.link_to_meshes.items():  # one mesh per link, as the UI's "Generate Spheres"
        mesh = "/collisions" if "/collisions" in meshes else meshes[0]
        path = editor.link_path(link)
        points, faces, counts = sphere_generation.compute_link_frame_mesh(path, path + mesh)
        editor.collision_sphere_editor.generate_spheres(path, points, faces, counts, args.spheres, args.radius_offset, False)
    editor.active_joints[:] = [name.startswith(f"{robot}_joint") for name in editor.dof_names]
    editor.joint_positions = np.array((HOME + [0.0, 0.0])[: editor.num_dof], dtype=float)
    editor.export_xrdf(str(raw / f"{robot}.xrdf"), format_version=2.0)
    urdf, xrdf, meta = build((raw / f"{robot}.urdf").read_text(), yaml.safe_load((raw / f"{robot}.xrdf").read_text()), usd_poses, robot)

    # The simulator's mass properties go into the URDF (in its link frames).
    arm = Articulation(base)
    app.update()
    dofs = list(arm.dof_names)
    arm_dofs = [dofs.index(j) for j in xrdf["cspace"]["joint_names"]]

    def settle(q):
        full = np.zeros(arm.num_dofs)
        full[arm_dofs] = q
        arm.set_dof_positions(full[None])
        arm.set_dof_position_targets(full[None])
        arm.set_dof_velocities(np.zeros((1, arm.num_dofs)))
        app.update()
        return arm.get_dof_positions().numpy()[0][arm_dofs].astype(float)

    settle(HOME)
    names = list(arm.link_names)
    unknown = sorted(set(names) - set(meta["usd_to_urdf"]))
    if unknown:
        fail("links", f"PhysX links the URDF lacks: {unknown}")
    masses = arm.get_link_masses().numpy()[0]
    coms = arm.get_link_coms()[0].numpy()[0]
    inertias = arm.get_link_inertias().numpy()[0].reshape(-1, 3, 3)
    frames = {name: np.array(X) for name, X in meta["usd_to_urdf"].items()}
    urdf = set_inertials(urdf, to_urdf_frames(dict(zip(names, masses)), dict(zip(names, coms)), dict(zip(names, inertias)), frames))
    out = tmp / robot
    out.mkdir()
    (out / f"{robot}.urdf").write_text(urdf)
    (out / f"{robot}.xrdf").write_text(yaml.safe_dump(xrdf, sort_keys=False))
    model = Model.load(out)

    # Self-validation (spec 4.2.5): fail loudly, write nothing.
    checks = {}
    try:
        description_robot = cu.load_cumotion_robot(str(out), f"{robot}.urdf", str(out / f"{robot}.xrdf")).robot_description
    except Exception as error:
        fail("cumotion_load", error)
    checks["cumotion_load"] = "ok"
    inspector = cumotion.create_robot_world_inspector(description_robot, cumotion.create_world().add_world_view())
    home_hits = inspector.frames_in_self_collision(description_robot.default_cspace_configuration())
    if home_hits:
        fail("home_self_collision", home_hits)
    checks["home_self_collision"] = "free"
    kinematics = description_robot.kinematics()
    limits = np.array([[kinematics.cspace_coord_limits(i).lower, kinematics.cspace_coord_limits(i).upper]
                       for i in range(kinematics.num_cspace_coords())])
    samples = np.random.default_rng(0).uniform(limits[:, 0], limits[:, 1], size=(1000, len(limits)))
    fraction = float(np.mean([bool(inspector.frames_in_self_collision(q.reshape(-1, 1))) for q in samples]))
    if fraction > 0.08:
        fail("random_self_collision", f"{fraction:.3f} > 0.08")
    checks["random_self_collision"] = round(fraction, 3)
    worst_M = worst_g = 0.0
    for q in [model.home] + list(np.random.default_rng(1).uniform(model.lower, model.upper, size=(5, len(arm_dofs)))):
        q = settle(q)
        M_physx = arm.get_mass_matrices().numpy()[0][np.ix_(arm_dofs, arm_dofs)]
        g_physx = arm.get_dof_gravity_compensation_forces().numpy()[0][arm_dofs]
        M, g = np.asarray(model.manipulator.mass_matrix(q)), np.asarray(model.manipulator.gravity_vector(q))
        worst_M = max(worst_M, float(np.linalg.norm(M - M_physx) / np.linalg.norm(M_physx)))
        worst_g = max(worst_g, float(np.linalg.norm(g - g_physx) / np.linalg.norm(g_physx)))
    if max(worst_M, worst_g) > 1e-3:
        fail("dynamics", f"M {worst_M:.2e}, g {worst_g:.2e} relative > 1e-3")
    checks["mass_matrix_rel"], checks["gravity_rel"] = worst_M, worst_g

    # Spec 9: the FR3's masses come from PhysX's colliders; compare once with franka_description's.
    datasheet = franka_inertials(ROBOTS[robot]["datasheet"], args.franka_description)
    meta["mass_check_kg"] = {n: [float(m), datasheet[n][0]] for n, m in zip(names, masses) if n in datasheet}
    for name, (physx, sheet) in meta["mass_check_kg"].items():
        print(f"[guide_motion.description] {name}: PhysX {physx:.3f} kg, franka_description {sheet:.3f} kg")
    meta.update(source_usd=source, spheres_per_link=args.spheres, radius_offset=args.radius_offset, checks=checks)
    (out / "description.json").write_text(json.dumps(meta, indent=1))
    if robot == "panda":
        shutil.copy(stage_path, out / "panda.usda")
    timeline.stop()
    app.update()
    destination = ROBOTS_DIR / robot
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(out, destination)
    print(f"[guide_motion.description] {robot}: wrote {destination}: {checks}", flush=True)
    app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the pure tests**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test`
Expected: everything passes (`principal_inertia` included); importing `guide_motion.description` still imports no Isaac module: `~/ros2_ws/.venv/bin/python -I -c "import sys, guide_motion.description, guide_motion.model, guide_motion.control, guide_motion.planner; print([m for m in sys.modules if m.startswith(('isaacsim', 'omni', 'pxr', 'rclpy'))])"` prints `[]`.

- [ ] **Step 4: Run the pipeline for the FR3**

```bash
pgrep -af "isaac|kit/kit|guide_core"   # must print nothing
cd ~/ros2_ws && OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python -m guide_motion.description --robot fr3 2>&1 | tee /tmp/s2b_description_fr3.log | grep "guide_motion.description"
```
Expected (about 1–3 minutes; the first Kit start can take longer): one `PhysX … kg, franka_description … kg` line per link, then `fr3: wrote …/guide_motion/guide_motion/robots/fr3: {'cumotion_load': 'ok', 'home_self_collision': 'free', 'random_self_collision': 0.04…, 'mass_matrix_rel': …, 'gravity_rel': …}`. The XRDF test measured 4.5 % random self-collision for this sphere set. If a check FAILS, nothing is written: fix the cause named (for example an Isaac API that moved — compare with `make_fr3_xrdf.py`, which ran on this Isaac) and rerun. If the dynamics check fails by a constant on M's diagonal, the asset has joint armature: stop and report it, because OSCBF's model has none.

- [ ] **Step 5: Check the output**

```bash
cd ~/ros2_ws/src/guide/guide_motion/guide_motion/robots/fr3 && ls && grep -c "<inertial>" fr3.urdf
~/ros2_ws/.venv/bin/python -c "import json; d = json.load(open('description.json')); print(d['robot'], d['pipeline_version'], d['source_usd'], sorted(d['usd_to_urdf']))"
cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -c "
from guide_motion.model import Model; from guide_motion.control import Controller
m = Model.load('fr3'); c = Controller(m, 'velocity'); print(m.robot_dir, m.base_link, m.tool_frame, round(float(c.barrier_values(m.home).min()), 4))"
```
Expected: `description.json fr3.urdf fr3.xrdf`; an `<inertial>` per PhysX link (11 for the FR3 asset); `fr3 1 https://…/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd [...]`; the last line ends with a positive number.

- [ ] **Step 6: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/description.py guide_motion/test/test_description.py guide_motion/guide_motion/robots/fr3
git commit -m "feat(guide_motion): Kit description pipeline; FR3 description generated and validated"
```

---

### Task 9: The Panda description with its mass override

**Files:**
- Create (generated, committed): `guide_motion/guide_motion/robots/panda/{panda.urdf,panda.xrdf,panda.usda,description.json}`
- Test: `guide_motion/test/test_robots.py`

**Interfaces:**
- Consumes: Task 8's CLI; `~/ros2_ws/src/franka_description/robots/fer/inertials.yaml` and `end_effectors/franka_hand/inertials.yaml`.
- Produces: `Model.load("panda")`, `Model.load("fr3")` from the package; `ROBOTS_DIR / "panda" / "panda.usda"`: a USD that references Isaac's `FrankaPanda/franka.usd` and overrides its links' mass, centre of mass and inertia with franka_description's. Plan B's Isaac checks reference it instead of `franka.usd`.

- [ ] **Step 1: Write the failing test**

`guide_motion/test/test_robots.py`:
```python
"""The committed descriptions (Tasks 8-9) load, and every barrier holds at their home pose."""

import json

import numpy as np
import pytest

from guide_motion import description
from guide_motion.control import Controller
from guide_motion.model import Model


@pytest.mark.parametrize("robot", ["fr3", "panda"])
def test_committed_descriptions_are_safe_at_home(robot):
    model = Model.load(robot)
    meta = json.loads((model.robot_dir / "description.json").read_text())
    assert meta["pipeline_version"] == description.PIPELINE_VERSION and meta["checks"]["home_self_collision"] == "free"
    assert meta["checks"]["random_self_collision"] <= 0.08 and max(meta["checks"]["mass_matrix_rel"], meta["checks"]["gravity_rel"]) <= 1e-3
    assert model.tool_frame == f"{robot}_hand_tcp" and model.base_link == f"{robot}_link0"
    assert Controller(model, "velocity").barrier_values(model.home).min() > 0
    assert np.all(model.max_effort > 0) and np.all(model.max_velocity > 0)


def test_the_panda_override_references_isaacs_asset():
    text = (description.ROBOTS_DIR / "panda" / "panda.usda").read_text()
    assert "FrankaPanda/franka.usd" in text and "physics:diagonalInertia" in text
```
Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test/test_robots.py`
Expected: `fr3` passes; `panda` and the override test fail with `FileNotFoundError` (no `robots/panda` yet).

- [ ] **Step 2: Run the pipeline for the Panda**

```bash
pgrep -af "isaac|kit/kit|guide_core"   # must print nothing
cd ~/ros2_ws && OMNI_KIT_ACCEPT_EULA=YES env -u CUDA_DEVICE_ORDER .venv/bin/python -m guide_motion.description --robot panda 2>&1 | tee /tmp/s2b_description_panda.log | grep "guide_motion.description"
```
Expected: per-link `PhysX … kg, franka_description … kg` lines with equal values (the override wrote franka_description's masses), then `panda: wrote …/robots/panda: {…}` with all five checks. The Isaac Panda reference XRDF gave 5.3 % random self-collision. If `links` fails (a PhysX link the URDF lacks), or the exporter's Panda has no `isaac:source_drive` comment on a joint (`KeyError` in `fix_urdf`), stop and report the joint: the Panda's export has not been inspected before (Review Focus 5).

- [ ] **Step 3: Run the tests**

Run: `cd ~/ros2_ws/src/guide && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_motion/test`
Expected: every guide_motion test passes, none skipped except `test_franka_inertials_cover_the_arm_and_the_hand` on a machine without franka_description (not this one).

- [ ] **Step 4: Commit**

```bash
cd ~/ros2_ws/src/guide
git add guide_motion/guide_motion/robots/panda guide_motion/test/test_robots.py
git commit -m "feat(guide_motion): Panda description with franka_description mass override"
git status --porcelain   # init.yaml still ' M', unstaged; nothing else
```

---

## Self-review (run by the plan's author)

- **Spec coverage.** 4.1 prerequisite → Task 1; package, exact deps, vendored OSCBF → Task 2; `model` + `set_mass_properties` → Task 5; `control` (modes, `warm_up`, `step`, barriers) → Task 6; `targets` (dataclass, metric deltas, LIBERO converters) → Task 3; `planner` → Task 7; `description` CLI → Task 8. 4.2 steps 1–2 → Tasks 4 and 8; step 3 → Task 8 (PhysX readback into the URDF), Task 9 (Panda override; the host's registration readback is Plan B); step 4 → Tasks 8–9; step 5 → Task 8's checks. Section 5 gains → Task 6's `GAINS`. Section 7 unit tests → Pinocchio comparison (Task 5), controller determinism (Task 6), LIBERO HDF5 tests and clamp (Task 3), post-processor vs exporter defects (Task 4), target holder (Task 3). Section 8: infeasible / non-finite → damping and a counter, clamping to URDF limits (Task 6); "description validation fails" → the pipeline writes nothing and names the check (Task 8; the host's refusal is Plan B). The recorder change of 4.1 belongs to S6/S7 (spec) and is not here.
- **Placeholders.** None: every code step has the code, every run step its command and expected output.
- **Type consistency.** `Target` fields, `POSE/TWIST/JOINTS`, `Model.load/ee_pose/ee_speed/set_mass_properties`, `Controller(model, mode, gains, barriers, q_rest).warm_up/step/events/limit`, `plan(model, goal, q)` and `PlanningError` are used with the same names in Plans B and C.
- **Review Focus.** Each line has its test in the task that owns the code (Tasks 6, 7, 9).
