# GUIDE — Installation & Porting Log

This document records **every** step taken to install and run the GUIDE framework on
a machine whose stack differs from the one the framework was originally built for.
Each step lists the command(s) and a short **Why** explaining the rationale, so the
whole process is reproducible.

> **Reproducibility guardrails**
> - **No system packages are modified.** All Python installs go into the Isaac Sim
>   virtualenv (`env_isaaclab`); no `apt`, no changes to `/opt/ros/jazzy`.
> - Isaac Sim's pinned dependencies are protected with a pip **constraints file**
>   (`modules/isaac-safe-pins.txt`) on every install.
> - The shell here is **zsh** — always source the `.zsh` ROS setup files.

---

## ⭐ Current setup: Isaac Sim 6.0.1 on Python 3.12 (native ROS 2 Jazzy)

**This supersedes most of the 5.1 workarounds below.** Isaac Sim 6.0 runs on **Python
3.12 — the same interpreter as ROS 2 Jazzy** — so ROS 2 works *natively* (NVIDIA's
headline 6.0 feature). The entire cross-version stack from the 5.1 era is **obsolete**:
no bundled cp311 rclpy, no `/opt/ros` scrubbing, no `guide_msgs` cp311 overlay / dual
build, no `libpython` juggling. Env: `~/ros2_ws/.venv`.

**Install (uv, from system python3.12):**
```bash
uv venv --python /usr/bin/python3.12 .venv
# torch first (CUDA 13 here), matched torchvision:
uv pip install --python .venv/bin/python torch==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu130
# Isaac Sim 6.0.1 — --prerelease=allow because isaacsim-core needs the pre-release
# tinyobjloader==2.0.0rc13 (via mujoco-usd-converter), which uv skips by default:
uv pip install --python .venv/bin/python "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
# guide deps (protect Isaac's numpy2/torch pins so they aren't clobbered):
uv pip install --python .venv/bin/python python-fcl -c modules/isaac6-safe-pins.txt
uv pip install --python .venv/bin/python "lerobot==0.6.0" "transformers>=5.4,<5.6" -c modules/isaac6-safe-pins.txt   # lerobot 0.6.0 caps numpy<2.3.0 -> MUST pass -c (see Gotcha)
```
**Gotcha:** installing torch-dependent pkgs (lerobot/transformers) without `-c` re-resolves
the CUDA stack. lerobot 0.6.0 caps `numpy<2.3.0`, which conflicts with Isaac 6.0.1's exact
`numpy==2.3.1`: without `-c`, uv downgrades numpy->2.2.6, dragging torchvision->0.25.0 /
torch->2.10.0 and breaking Isaac (`uv pip check` then reports 9 incompatibilities). The
`-c modules/isaac6-safe-pins.txt` forces Isaac's numpy==2.3.1 / torch==2.11.0 / torchvision==0.26.0
over lerobot's cap (numpy 2.3.1 runs fine for lerobot). lerobot 0.6.0 also needs
`transformers 5.4-5.6` + `huggingface-hub 1.x`. Restore after drift (also fixes cu128 breakage):
`uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 numpy==2.3.1 -c modules/isaac6-safe-pins.txt`.

**Running** needs `export OMNI_KIT_ACCEPT_EULA=YES` (the launchers set it).

**5.1 → 6.0 API port (only two changes):**
- `is_file`: `isaacsim.core.utils.nucleus` (removed) → `isaacsim.storage.native` (`_cmd_stage.py`).
- OmniGraph ROS 2 shortcuts: `isaacsim.ros2.bridge.impl.og_shortcuts` → `isaacsim.ros2.ui`
  (`_cmd_robot.py`, `runtime.py`).
- Bonus: `isaacsim.util.clash_detection.ClashDetector` is **back** in 6.0 → re-enabled in
  `runtime.py`, so `_cmd_clash.py` uses full mesh clash again (not just the bbox fallback).

**Build & run (simple now — both sides are 3.12):**
```bash
source /opt/ros/jazzy/setup.zsh && cd ~/ros2_ws
colcon build                 # single cp312 build serves GUIDE *and* rqt
source install/setup.zsh
export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
ros2 launch guide_core bringup.launch.py
```
The launchers just set `python_executable=~/ros2_ws/.venv/bin/python` +
`additional_env={OMNI_KIT_ACCEPT_EULA, CYCLONEDDS_URI}` — all the `_isaac_ros_env`
scrubbing and the `install_isaac` overlay are gone. The **DDS localhost config (§5.4)**
is still needed (host networking, unrelated to Python).

> The sections below document the earlier **Isaac Sim 5.1** effort. They're kept for
> history; on 6.0 the cross-version items (bundled rclpy in §5.1, the `guide_msgs`
> dual-build in §4.4, the launcher scrubbing) no longer apply.

---

## 0. Target environment (as found on this machine)

| Component | GUIDE originally targeted | This machine |
|-----------|---------------------------|--------------|
| Isaac Sim | 4.5.0 (standalone `~/isaacsim/python.sh`) | **5.1.0**, pip-installed into `~/ros2_ws/env_isaaclab` (Python **3.11**) |
| ROS 2 | Humble (Python 3.10) | **Jazzy** (Python **3.12**) |
| MoveIt 2 | development / non-standard build | standard Jazzy MoveIt |

These three version gaps drive most of the changes below. `env_isaaclab` is a
`uv`-created venv; activate it with `source env_isaaclab/bin/activate` (or call
`env_isaaclab/bin/python` / `env_isaaclab/bin/pip` directly).

### 0.1 ⚠️ DO NOT run `isaaclab --generate-vscode-settings` in this venv

```bash
# python -m isaaclab --generate-vscode-settings   # <-- DESTRUCTIVE, do not run
```

**Why (learned the hard way):** on a **pip-installed** Isaac Sim, Isaac Lab's
`--generate-vscode-settings` runs `_mock_python_modules`, which **writes 33-byte
`# Generated by 'isaaclab' package` stub `__init__.py` files into ~1470 Isaac Sim
extension directories** — including the PEP 420 namespace dirs (`omni/`,
`omni/kit/`, …) that must NOT have an `__init__.py`. This converts the `omni.*`
namespace into broken regular packages, so `omni.usd` loads without `UsdContext`
and Isaac Sim segfaults in `omni.kit.raycast.query` the moment the RTX renderer
loads. (This is a known Isaac Sim issue: the exact `omni.usd has no attribute
'UsdContext'` symptom is documented on the NVIDIA forums.)

**If it was already run — recovery (venv-only, no re-download):** delete every
mock stub; the originals were namespace dirs with no `__init__.py`, so deletion
alone restores them:

```bash
EXT=~/ros2_ws/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache
grep -rl "Generated by 'isaaclab' package" "$EXT" --include="__init__.py" | xargs rm -f
```

(The `__main__.py` `omni.kit_app` → `isaacsim.kit.kit_app` patch from earlier is
harmless and unrelated; just never invoke the vscode-settings generator here.)

---

## 1. Clone the repository and all auxiliary modules

The repo lives under the conventional `src/` layout:

```bash
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/ABC-iRobotics/guide.git
```

**Why:** the workspace root (`~/ros2_ws`) also holds the Isaac venv, so packages go
under `src/` and colcon is pointed there (see §4).

The `.gitmodules` lists submodules but **no gitlinks were ever committed**, so
`--recurse-submodules` pulls nothing. Clone them manually:

```bash
cd ~/ros2_ws/src/guide
mkdir -p modules && cd modules

# Declared in .gitmodules
git clone https://github.com/AndrejOrsula/pymoveit2.git
git clone --branch humble https://github.com/ros2/rclpy.git
git clone https://github.com/huggingface/lerobot.git

# Referenced by the packages/imports but MISSING from .gitmodules
git clone https://github.com/ABC-iRobotics/irob_lerobot_ros.git      # hard dep of block_bin
git clone https://github.com/ABC-iRobotics/irob_franka_ros2.git      # robot description/config

# irob_lerobot_ros has its own pinned lerobot submodule (the tested version)
git -C irob_lerobot_ros submodule update --init --recursive

# Keep colcon from treating the vendored trees as workspace packages
touch COLCON_IGNORE
```

**Why:** `irob_lerobot_ros` is imported 8× and is a `block_bin` dependency, yet it is
not in `.gitmodules`; `irob_franka_ros2` provides the robot config. The pinned lerobot
inside `irob_lerobot_ros` (`0b067df…`, torch>=2.2,<2.8, numpy-1.x compatible) is the
version GUIDE was validated against — the **top-level** lerobot HEAD wants `numpy>=2.0`
which would break Isaac Sim, so we use the pinned one.

---

## 2. Python dependencies (into the Isaac venv only)

### 2.1 Constraints file to protect Isaac Sim

```bash
cat > ~/ros2_ws/src/guide/modules/isaac-safe-pins.txt <<'EOF'
numpy==1.26.0
torch==2.7.0
torchvision==0.22.0
imageio==2.37.0
packaging==23.0
EOF
```

**Why:** Isaac Sim 5.1 pins these exactly (`isaacsim` requires `numpy==1.26.0`,
`isaacsim-core` requires `packaging==23.0`, torch 2.7.0+cu126 is preinstalled).
Passing `-c isaac-safe-pins.txt` on every install stops transitive deps (lerobot,
datasets, accelerate…) from upgrading them and breaking Isaac.

### 2.2 Install lerobot (pinned) + its runtime subset

```bash
cd ~/ros2_ws/src/guide
PINS=modules/isaac-safe-pins.txt

# Editable, no-deps: GUIDE only uses lerobot's dataset format + Robot base class,
# not the full training stack (which drags a heavy/conflicting dependency tree).
env_isaaclab/bin/pip install -e modules/irob_lerobot_ros/modules/lerobot --no-deps

# The lightweight libs actually needed at import time:
env_isaaclab/bin/pip install draccus pyserial deepdiff accelerate datasets av -c $PINS
```

**Why:** torch/torchvision/numpy/scipy/pyyaml/trimesh are already in the venv (Isaac
ships them). `--no-deps` avoids the lerobot resolver pulling `imageio[ffmpeg]`/
`torchcodec`/`wandb`/etc. GUIDE imports only `from lerobot.robots import Robot` and the
LeRobot dataset format.

### 2.3 FCL for bounding-box collision (replaces hand-rolled SAT — see §3.4)

```bash
env_isaaclab/bin/pip install python-fcl -c ~/ros2_ws/src/guide/modules/isaac-safe-pins.txt
```

**Why:** `trimesh` (already present) + `python-fcl` (the engine MoveIt uses) provide
robust box collision/distance queries. The manylinux cp311 wheel is numpy-safe (Isaac's
numpy 1.26 is preserved).

---

## 3. Source-code compatibility changes

All edits are inside `src/guide/…`; none touch installed system code.

### 3.1 Replace `cv_bridge` with a numpy-only image decoder
- **File:** `modules/irob_lerobot_ros/irob_lerobot_ros/ros2camera.py`
- **What:** removed `from cv_bridge import CvBridge`; added `imgmsg_to_rgb(msg)` (pure
  numpy) and dropped `cv_bridge` from `package.xml`.
- **Why:** `cv_bridge` is a compiled extension linked against a specific numpy ABI (and
  the one on this machine is a Py-3.12 build that would crash in the 3.11 venv). A
  `sensor_msgs/Image` is just a byte buffer, so numpy decodes it with no dependency.

### 3.2 Isaac Sim 4.5 → 5.1 API relocations
- **`_cmd_robot.py`, `runtime.py`:** `isaacsim.ros2.bridge.scripts.og_shortcuts.*`
  → `isaacsim.ros2.bridge.impl.og_shortcuts.*` (Kit moved these).
- **`scene_orchestrator.py`:** legacy `omni.isaac.core.prims.XFormPrimView`
  → `isaacsim.core.prims.XFormPrim` (batched view unified; same `prim_paths_expr` ctor).
- **`scene_orchestrator.py`, `_cmd_clash.py`:** collapsed the confusing nested
  `try/except` import ladders (whose fallbacks pointed at the **removed**
  `omni.isaac.core.*` namespace) into single, clearly-commented deferred imports.
- **Why:** verified against the installed 5.1 tree; most `isaacsim.*` symbols are
  unchanged, these are the ones that actually moved/were removed.

### 3.3 `ClashDetector` removed in 5.1
- **Files:** `_cmd_clash.py`, `runtime.py`.
- **What:** made `from isaacsim.util.clash_detection import ClashDetector` optional
  (`try/except → None`); the clash commands degrade to a bounding-box overlap check;
  removed the mandatory `clash_detection` entry from the runtime extension list.
- **Why:** `isaacsim.util.clash_detection` was deleted in 5.1 with no replacement. This
  keeps `guide_core` importable/runnable (precise mesh contact is the only thing lost).

### 3.4 Bounding-volume geometry class (backs §3.3)
- **New file:** `guide_core/guide_core/types/bounding.py` — `AABB`/`OBB` dataclasses
  (with `from_isaac(...)` matching `compute_aabb`/`compute_obb`) and `BoundingVolumeOps`
  with logical ops for one/two/many objects, backed by FCL.
- **`_cmd_clash.py`:** replaced ~155 lines of inline separating-axis math with
  `BoundingVolumeOps`.
- **Why:** delegates collision to a maintained library (FCL) instead of hand-rolled SAT;
  overlap rule is `signed_distance <= tolerance`.

---

## 4. Build the workspace

### 4.1 One-time colcon configuration (makes a plain `colcon build` work)

```bash
# Keep colcon from descending into the Isaac venv (it contains CMake test artifacts
# and stray setup.py files that break package discovery).
touch ~/ros2_ws/env_isaaclab/COLCON_IGNORE

# Force the system Python 3.12 for CMake/rosidl. FindPython3 otherwise grabs the
# uv-managed ~/.local/bin/python3.11 (ahead in PATH), which lacks `empy` and breaks
# message generation ("No module named 'em'").
mkdir -p ~/.colcon
cat > ~/.colcon/defaults.yaml <<'EOF'
build:
  cmake-args:
    - -DPython3_EXECUTABLE=/usr/bin/python3
EOF
```

**Why:** encodes the two non-obvious requirements so the setup command stays a plain
`colcon build` (Jazzy's rosidl must run under 3.12, not the Isaac 3.11).

### 4.2 Packaging fix — namespace subpackages

- **Files:** `setup.py` in `guide_core`, `guide_ex`, `guide`, `guide_tasks/block_bin`.
- **What:** `find_packages(exclude=["test"])` → `find_namespace_packages(include=[package_name, f"{package_name}.*"])`.
- **Why:** the subpackages (`ros`, `core`, `scene`, `types`, `steps`, …) have **no
  `__init__.py`** (PEP 420 namespace packages). `find_packages` silently drops them, so
  a copy install produced `ModuleNotFoundError: No module named 'guide_core.ros'`. This
  only "worked" under `--symlink-install`; the fix makes a plain copy install correct.

### 4.3 Build

```bash
source /opt/ros/jazzy/setup.zsh          # NOTE: .zsh, not .bash (shell is zsh)
cd ~/ros2_ws
colcon build                             # copy install of all 5 packages
source install/setup.zsh
```

**Why:** plain `colcon build` (copy install) also avoids the fragile symlinked install
of the CMake message package `guide_msgs`. Result: `guide_msgs` exposes 13 services +
1 action; all subpackages resolve.

> **zsh note:** sourcing the colcon-generated `setup.bash` under zsh silently fails
> (`$BASH_SOURCE` is empty), leaving `ros2 pkg list` empty. Always use `setup.zsh`.

### 4.4 Build `guide_msgs` for Isaac's Python 3.11 (custom-message ABI)
GUIDE runs under Isaac's Python **3.11**, so its message package's C extensions must be
cp311. The default build (§4.1, forced to system 3.12) produces
`libguide_msgs__rosidl_generator_py.so` linked to `libpython3.12` → **`SIGABRT` in
`..._convert_from_py`** the moment a `guide_msgs` message is (de)serialized (e.g. the
`RegisterScene` response when registering a task). The original framework avoided this
because Isaac 4.5 + ROS Humble were **both Python 3.10**; Isaac 5.1 (3.11) + Jazzy (3.12)
are not, and there is no stock ROS 2 built for 3.11.

**Fix — build `guide_msgs` TWICE** (both ends of the mismatch need it, from the same
`.idl` so the DDS type hash matches and they still interoperate):
- **cp312** in the main `install/` — for the system tools (rqt, `ros2 service call`,
  Python 3.12). Without this, rqt reports it *cannot import the message class*.
- **cp311** in a separate `install_isaac/` overlay — for GUIDE (Isaac's Python 3.11).

First give Isaac's interpreter the pure-Python ROS build tools (`rosidl_*`/`ament` come
from `/opt/ros`; `catkin_pkg`/`empy`/`lark` go in the venv), then build both:
```bash
env_isaaclab/bin/pip install catkin_pkg empy==3.3.4 lark==1.1.9
source /opt/ros/jazzy/setup.zsh
# (a) pure-Python pkgs + cp312 guide_msgs for system tools:
colcon build
# (b) cp311 guide_msgs overlay for GUIDE:
colcon build --packages-select guide_msgs \
  --build-base build_isaac --install-base install_isaac \
  --cmake-args -DPython3_EXECUTABLE=$HOME/ros2_ws/env_isaaclab/bin/python
```
(Do NOT add `/usr/lib/python3/dist-packages` to `PYTHONPATH` for the cp311 build — it
shadows the venv's numpy 1.26 with the system 3.12 numpy and CMake's `FindPython3 NumPy`
then fails.) The launchers prepend `install_isaac/guide_msgs` to GUIDE's PYTHONPATH/
LD_LIBRARY_PATH so GUIDE uses the cp311 copy while system tools use the cp312 main install.
Pure-Python `guide_core`/`guide_ex`/`block_bin`/`guide` stay on the 3.12 build (they import
fine under 3.11).

**Runtime consequence:** the venv's uv-built interpreter is *statically* linked, so the
cp311 `guide_msgs` `.so` needs `libpython3.11.so.1.0` found at load time. The launchers'
`_isaac_ros_env` adds the base interpreter's `lib/` (`…/uv/python/cpython-3.11.15…/lib`)
to `LD_LIBRARY_PATH` automatically.

---

## 5. Runtime setup

### 5.1 rclpy for the Isaac Python (3.11)

- **Files:** `guide_core/launch/bringup.launch.py`, `isaac_sim.launch.py`.
- **What:** the launchers prepend Isaac Sim's **bundled** rclpy to the spawned process's
  `PYTHONPATH`:
  `env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/$ROS_DISTRO/rclpy`
  (via `ExecuteProcess(additional_env=...)`, distro from `$ROS_DISTRO`, venv from the
  interpreter path).
- **Why:** the guide nodes run under Isaac's Python 3.11, but the system Jazzy rclpy is
  compiled for 3.12 (`_rclpy_pybind11.cpython-312*.so`) → `ModuleNotFoundError: No
  module named 'rclpy._rclpy_pybind11'`. Isaac Sim ships a **cp311** rclpy (+ full ROS 2
  Python stack) for the active distro; using it needs no rebuild of the bundled
  `modules/rclpy` submodule.
- **Also (critical):** the launchers now **scrub every `/opt/ros` entry** from the
  spawned process's `PYTHONPATH` **and** `LD_LIBRARY_PATH` (via `_isaac_ros_env`, which
  also prepends Isaac's bundled `rclpy/` and `lib/`). Otherwise the system cp312
  `rcl_interfaces` collides with Isaac's cp311 build and rclpy **aborts on the first
  publish**: `rcl_interfaces__msg__parameter_event__convert_from_py: Assertion
  'strncmp("...ParameterEvent", full_classname_dest, 50) == 0' failed`. With mismatched
  Python versions, Isaac's **internal** ROS 2 libraries must be used exclusively; Isaac's
  bundled RMW (cyclonedds/fastrtps) still interoperates with system ROS 2 nodes over DDS.
- **Consequence — install `lark` + `empy` into the venv:** Isaac's bundled `rosidl_parser`
  imports `lark` (used when rclpy parses a message `.idl` at runtime — which happens because
  `guide_msgs` is built for cp312 and re-parsed under Isaac's cp311); other ROS Python bits
  import `em` (empy). System ROS normally provides both, but the `/opt/ros` scrub removes
  that access → `ModuleNotFoundError: No module named 'lark'` / `'em'`. Fix:
  `env_isaaclab/bin/pip install lark==1.1.9 empy==3.3.4` (match system Jazzy; pure-Python).

### 5.4 DDS discovery on this host (localhost cyclonedds config)
- **Symptom:** GUIDE's node (`/Sim_0/GUIDE`, 17 services) is alive and its executor spins
  (heartbeats fire), but `ros2 node list` / rqt_graph can't see it and hang.
- **Cause:** this box's loopback `lo` has **no `MULTICAST` flag**, and it has several NICs
  (`eno1`, `tailscale0` VPN, a `NO-CARRIER` USB NIC). Cyclonedds' default multicast
  discovery can't work on `lo` and stalls probing the wrong interfaces — so processes on the
  same host never discover each other.
- **Fix:** `guide_core/config/cyclonedds_localhost.xml` forces the `lo` interface with
  **unicast SPDP to `localhost`** (no multicast). The launchers set `CYCLONEDDS_URI` to it
  for the GUIDE process automatically (respecting an existing override). **Any shell you run
  `ros2`/rqt_graph from must export the same file:**
  ```bash
  export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
  ```
  Verified: with this, a second process's `ros2 node list` discovers a running node instantly.

For a **manual** run (invoking the venv python directly, not via `ros2 launch`):

```bash
export PYTHONPATH=~/ros2_ws/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/$ROS_DISTRO/rclpy:$PYTHONPATH
```

### 5.2 Simulator path in the launchers

- **What:** `python_executable` default changed from the old 4.5 `~/isaacsim/python.sh`
  to the venv interpreter `~/ros2_ws/env_isaaclab/bin/python` (override via
  `ISAACSIM_PYTHON`).
- **Why:** Isaac Sim 5.1 is pip-installed into the venv; there is no `python.sh`.

### 5.3 Launch

```bash
source /opt/ros/jazzy/setup.zsh && source ~/ros2_ws/install/setup.zsh
ros2 launch guide_core bringup.launch.py
```

---

## 6. Issues encountered

### 6.1 Isaac Sim crash on RTX startup — **RESOLVED** (root cause: §0.1)
The `raycast.query` / `omni.usd::UsdContext` segfault documented below was caused by
the `isaaclab --generate-vscode-settings` run in §0.1 stubbing 1470 extension
`__init__.py` files. **Fix:** delete the stubs (see §0.1 recovery). After that, the
RTX rendering experience boots cleanly (`exit 0`). The diagnosis trail is kept below
for reference.

<details><summary>Original investigation (resolved)</summary>

#### Isaac Sim crashes when the RTX rendering stack loads (`raycast.query` / `omni.usd`)
- **Symptom:** the GUIDE process boots Isaac Sim and segfaults (exit `-11`) ~1 s in.
  The backtrace is in `libomni.kit.raycast.query.plugin.so` calling
  `omni::usd::UsdContext::getNameSv()`, preceded by a cascade of
  `module 'omni.usd' has no attribute 'UsdContext'` / pybind11 `type not registered`.
- **Isolated to the RTX renderer.** Systematic experience-file testing (bare
  `SimulationApp`, minimal `env -i`):
  - Isaac Lab's **non-rendering** experience `isaaclab.python.headless.kit` → **boots OK**.
  - Isaac Lab's **rendering** experience, Isaac Sim's `isaacsim.exp.base.kit` and
    `isaacsim.exp.full.kit` (GUIDE's implicit default) → **crash**.
  - The only extensions the rendering experience adds are `omni.kit.viewport.rtx`,
    `omni.renderer.core`, `omni.kit.material.library`; `omni.kit.viewport.rtx` pulls in
    `omni.kit.raycast.query`, the crashing extension.
- **Ruled out:** GUIDE code, the ROS overlay/`ros2 launch` env (reproduces under
  `env -i`), the rclpy change, `numpy` (1.26.0 = Isaac's exact pin), `libstdc++`
  (system 3.4.33 ≥ Isaac's 3.4.30), `opencv` (matches pin), `pip check` (Isaac deps
  intact), isaacsim wheel version mix (uniform 5.1.0.0, one build hash), and multi-GPU
  (`CUDA_VISIBLE_DEVICES=0` still crashes).
- **Likely cause:** an RTX-renderer / GPU-driver / extscache issue in the Isaac Sim 5.1
  pip install. Note Isaac's own log recommends driver `535.x`; this box runs `580.159.03`.
- **Options (not yet applied — pending decision, kept reproducible & venv-only):**
  1. Repair the RTX extscache: `env_isaaclab/bin/pip install --force-reinstall --no-deps
     isaacsim-extscache-kit isaacsim-extscache-kit-sdk -c modules/isaac-safe-pins.txt`
     (multi-GB; may not help if it is a driver issue).
  2. GPU driver is a **system** change — excluded by the reproducibility guardrail.
  3. Interim: GUIDE can boot with a non-rendering experience, but that disables the
     camera-based demonstration recording.

</details>

### 6.2 Optional: lean experience to speed GUIDE startup
GUIDE's `SimulationApp(startup_config)` loads Isaac's default **full editor**
experience (`isaacsim.exp.full.kit`); first run spends ~2 min compiling RTX shaders.
Not a bug — but passing a leaner Kit experience (e.g. `isaaclab.python.rendering.kit`)
via `SimulationApp(cfg, experience=<path>)` would cut startup. Left as-is for now.

---

## Appendix — reproducibility artifacts
- `modules/isaac-safe-pins.txt` — constraints protecting Isaac's pinned deps.
- A full `env_isaaclab` `pip freeze` snapshot can be regenerated with
  `env_isaaclab/bin/pip freeze`.
