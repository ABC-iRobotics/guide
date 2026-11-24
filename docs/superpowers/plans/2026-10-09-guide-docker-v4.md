# GUIDE Docker Images (design v4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GUIDE demonstration generation in Docker. Each container holds one simulator on one A4000 with several scenes, started as a swarm service. It either runs a plan by itself or is driven, launched and shut down by a master over ROS 2. Datasets are delivered per dataset to a folder or to Ceph.

**Architecture:**
- **GUIDE gets general features only:**
  - Register fetches and builds a task and, on request, launches its bringup.
  - Task launches follow `sim_id`, and the clock is per simulator (`/Sim_N/clock`).
  - One shutdown path for `/Sim_N/shutdown`, Ctrl-C and SIGTERM.
  - A `/Sim_N/dataset_finalized` topic.
- **Container glue lives under `docker/`:**
  - the runner (`docker/runner/guide_container.py`), which writes the DDS config, starts `GUIDE --id N`, drives a plan, and delivers each finalized dataset;
  - the mock simulator and the scripts.
- **The Dockerfile runs the README's install blocks verbatim**, so every build failure is a docs fix.

**Tech Stack:** Docker (BuildKit, swarm services, overlay networks), NVIDIA Container Toolkit, ROS 2 Jazzy (rclpy, Cyclone DDS), Isaac Sim 6.0.1 (pip, Python 3.12), uv, colcon, rosdep, boto3 (Ceph/MinIO), pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-guide-docker-design.md` (v4). The spec stays unchanged for a later recheck; this plan records where it deviates.

## Deviations from the spec (decided while planning)

1. **Register launches the bringup only on request.**
   - `RegisterScene.srv` gains `bool bringup` (default false).
   - Why: the spec's "always" would start `block_bin`'s solver and MoveIt for a manual
     Register before a `block_bin_eval` run, fighting the policy for the arm. With the
     field, existing workflows stay exactly the same, and masters and the runner send
     `bringup: true`.
2. **`RegisterScene` answers with `string package`.** The runner needs the task's package
   name to read its zone grid when the task came as an `s3://` bundle.
3. **Ctrl-C today was misdescribed.** The spec says nothing ends Isaac's loop. Measured:
   - SIGINT interrupts Isaac's main thread mid-frame (`KeyboardInterrupt`) while the ROS
     thread finalizes;
   - SIGTERM never ends the main loop.

   The fix stays the same, one shutdown path. GUIDE handles both signals itself
   (`rclpy.init(signal_handler_options=SignalHandlerOptions.NO)`, verified with a probe),
   so ROS is still up to announce the finalized datasets.
4. **Crash reporter:** off through `SimulationApp`'s own `enable_crashreporter: false`
   config key, not through `extra_args`.
5. **Zones are dealt, not ranged:** a job's zones go heaviest first to the lightest scene.
   The scene sizes match the spec's example (35/35/30); the zone sets interleave.
6. **No stack file.** A templated volume name per replica doesn't fit stack files.
   `docker/launch_sim.sh <id>` is the one reference spec, for the master and for manual
   fleets.
7. **One image per node until a registry exists.** GitHub Actions is off and there is no
   registry, so services use `--no-resolve-image`, and each node needs the image
   (`docker save | ssh <node> docker load`).

## Global Constraints

- Python 3.12; ROS 2 Jazzy; `isaacsim==6.0.1.0`; torch 2.11.0 (cu130), torchvision 0.26.0, numpy 2.3.1; `lerobot==0.6.0`.
- Every torch-dependent install passes `-c <pins>` (`numpy==2.3.1`, `torch==2.11.0`, `torchvision==0.26.0`).
- **GUIDE packages get no container-only code.** Every change under `guide_core/`, `guide_msgs/` and `guide_tasks/` must be used the same way outside Docker. Container glue lives under `docker/`.
- `guide_core/guide_core/ros/task_bringup.py` and `docker/runner/guide_container.py` import nothing from Isaac (`pxr`, `omni`, `isaacsim`).
- The simulator container runs as root (decided; the non-root path is deferred in spec §12). Image paths: `/root/ros2_ws/.venv`, `/root/ros2_ws/install`.
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. One `ROS_DOMAIN_ID` for all containers; the `Sim_<id>` namespace separates simulators.
- **Never set `CUDA_DEVICE_ORDER`** in the simulator environment: annotators return nothing.
- Commit messages carry **no** `Co-Authored-By`, Claude or "Generated with" lines. The user is the sole author. Never open a PR; push `feat/docker` only.
- Never build into `~/ros2_ws/install`. Local test runs use a private install:
  - `PRIV=${CLAUDE_JOB_DIR:-/tmp/guide-docker}/tmp`
  - From the worktree, with `/opt/ros/jazzy/setup.zsh` and `~/ros2_ws/install/setup.zsh` sourced, run `colcon build --base-paths . --build-base $PRIV/build --install-base $PRIV/install --packages-select <pkgs>`.
  - Then `source $PRIV/install/setup.zsh`.
- Test command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q <paths>`, from the worktree with the private install sourced.
- Datasets a person keeps go to `~/dataset/...`, never a job tmp dir. Don't run two GUIDE simulators on the host at once (recorder port 50050).
- Shell: zsh locally (source `.zsh` files), bash in the images.

## Review Focus

1. **`docker stop` / `docker service rm` in the middle of a plan.** Every dataset GUIDE finalizes on the way down should be delivered, and nothing undelivered deleted. The container should exit within the 180 s grace period. Pinned by Task 11 Step 5 and Task 14 Step 5.
2. **Ceph unreachable or bad credentials.** The dataset stays in scratch and is announced `complete: false`, `target: null`; the runner keeps serving and doesn't crash. Pinned by Task 8 `test_a_failed_upload_keeps_the_dataset` and Task 9 `test_an_unreachable_bucket_keeps_the_dataset`.
3. **A task bundle that can't be built** (no task package, two task packages, a missing rosdep key, a colcon error). Register should answer `success: false` with the reason and never stop the simulator, so the other scenes keep stepping. Pinned by Task 6 `test_a_bundle_holds_at_most_one_task` and Task 7 `test_a_failed_build_fails_register_and_keeps_the_simulator_running`.
4. **A malformed plan or too few scenes.** The container should exit 1 before Isaac starts, naming the problem. Pinned by Task 8 `test_bad_plans_are_rejected` and Task 9 `test_a_bad_plan_never_starts_guide`.
5. **Two simulators on `guide-net`.** Each has its own `/Sim_N/clock`, there is no global `/clock`, one publisher per clock, and the slaves don't see each other. Pinned by Task 10 Step 3.

## File Structure

| File | Responsibility |
|---|---|
| `guide_tasks/{block_bin,cube_stack}/launch/bringup.launch.py` | `sim_id`, `first_scene`, `num_env`; `/clock` → `/Sim_N/clock` |
| `guide_core/guide_core/scene/scene_recorder.py` | report each finalize (event + dataset dir) |
| `guide_core/guide_core/scene/scene_manager.py` | `wait_finalized`; `finalize_all_recordings` returns what it wrote |
| `guide_core/guide_core/core/runtime.py` | leave the loop right after a shutdown command |
| `guide_core/guide_core/ros/guide_ros.py` | namespaced clock, `dataset_finalized`, blocking finalize, one shutdown path, Register through `TaskBringup` |
| `guide_core/guide_core/ros/task_bringup.py` (new) | fetch (dir / `s3://`), build (rosdep, pip, colcon), activate, launch |
| `guide_core/launch/bringup.launch.py` | give GUIDE time to finalize when `ros2 launch` stops it |
| `guide_core/package.xml` | `std_msgs`, `std_srvs` |
| `guide_msgs/srv/RegisterScene.srv` | `bool bringup`; `string package` |
| `guide_core/test/conftest.py` | shared `isaac_import` fixture (moved from `test_command_fixes.py`) |
| `guide_core/test/test_multi_sim.py`, `test_task_bringup.py` (new) | tests for the above |
| `docker/runner/guide_container.py`, `test_guide_container.py` (new) | the runner and its tests |
| `docker/mock/mock_sim.py` (new) | stand-in GUIDE (also the runner's plan-mode test double) |
| `docker/Dockerfile`, `docker/readme_steps.sh`, `docker/entrypoint.sh`, `.dockerignore` | images |
| `docker/trim.sh`, `docker/kit_exts_keep.txt`, `docker/warm.sh`, `docker/warm_plan.yaml`, `docker/e2e_plan.yaml` | deploy image size, warm cache, end-to-end plan |
| `docker/test_comms.sh`, `docker/launch_sim.sh` | overlay communication test; reference service spec |
| `docker/CONVENTIONS.md`, `docker/AUTHORITIES.md`, `.gitignore`, `README.md`, `INSTALLATION.md` | notes for the team, secrets out of git, docs |

**Order:**
- Tasks 2–9 need no Docker; start them now.
- Task 0 (user, sudo) gates Tasks 1 and 10–14.

---

### Task 0: Host setup (user, sudo)

**Files:** none.

- [ ] **Step 1: Docker Engine** (Docker's apt repository)

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # log out and back in
```

- [ ] **Step 2: NVIDIA Container Toolkit as the default runtime.** Swarm services can't request GPUs, so every container on a GPU node runs under it.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

- [ ] **Step 3: Swarm, node label, networks, secret**

```bash
docker swarm init
docker node update --label-add guide.gpu=a4000 $(docker node ls -q --filter role=manager)
docker network create -d overlay --attachable --internal --opt encrypted --subnet 10.42.0.0/24 guide-net
docker network create -d overlay --attachable guide-egress
mkdir -p docker/secrets   # git-ignored (Task 14)
printf '[default]\naws_access_key_id = guide\naws_secret_access_key = guidesecret\n' > docker/secrets/guide_s3
docker secret create guide_s3 docker/secrets/guide_s3
```
On more nodes: open ESP (IP protocol 50), 2377/tcp, 7946/tcp+udp and 4789/udp between them, and give every A4000 node the label. The `guide_s3` content above is the MinIO test key. Replace it with the Ceph key (scoped write-only to one bucket/prefix) for real runs.

- [ ] **Step 4: Verify**

Run: `docker run --rm ubuntu:24.04 nvidia-smi -L && docker network ls --filter name=guide`
Expected: both GPUs are listed without `--gpus`, which proves the default runtime is nvidia. `guide-net` and `guide-egress` are listed as overlays.

---

### Task 1: Docs-driven build (iterative)

**Files:**
- Create: `.dockerignore`, `docker/readme_steps.sh`, `docker/Dockerfile` (stages `base`, `build`, `test`)
- Modify: `README.md` (runnable Prerequisites block; every fix the loop finds), `INSTALLATION.md` (the same fix with its "why")

**Interfaces:**
- Produces: images `guide:build` (README steps done, sources in `/root/ros2_ws/src/guide`) and `guide:test`. `docker/readme_steps.sh README.md <Section>` runs the bash blocks under `## <Section>`.

- [ ] **Step 1: Write `docker/readme_steps.sh`**

```bash
#!/usr/bin/env bash
# Run the ```bash blocks of one README section verbatim: the image build IS the install test.
# usage: readme_steps.sh README.md <Section>     ("## <Section>" up to the next "## " heading)
# The GUIDE clone is skipped: the build context is the checkout being built.
set -eo pipefail
awk -v h="## $2" '
  $0 == h                {on = 1; next}
  on && !code && /^## /  {on = 0}
  on && /^```bash/       {code = 1; next}
  code && /^```/         {code = 0; next}
  on && code' "$1" | grep -v 'ABC-iRobotics/guide.git' > /tmp/readme_steps.sh
cat /tmp/readme_steps.sh
[ -n "${DRY_RUN:-}" ] && exit 0
bash -exo pipefail /tmp/readme_steps.sh   # no -u: ROS setup scripts read unset variables
```
Run: `chmod +x docker/readme_steps.sh && DRY_RUN=1 docker/readme_steps.sh README.md Installation | head -3`
Expected: `mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src`, then the `irob_franka_ros2` clone. There is no `guide.git` line.

- [ ] **Step 2: Give README "Prerequisites" a runnable block** (under the existing bullets), and add `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` to "Usage" next to `CYCLONEDDS_URI`:

```bash
# Ubuntu 24.04 with ROS 2 Jazzy (ros-base) installed:
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git curl build-essential cmake psmisc iproute2 \
  python3-colcon-common-extensions python3-vcstool python3-rosdep \
  ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-xacro ros-jazzy-robot-state-publisher \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
  ros-jazzy-moveit-ros-move-group ros-jazzy-moveit-planners-ompl ros-jazzy-moveit-kinematics \
  ros-jazzy-moveit-simple-controller-manager ros-jazzy-moveit-configs-utils \
  ros-jazzy-moveit-ros-planning-interface ros-jazzy-pick-ik ros-jazzy-std-srvs \
  libglu1-mesa libvulkan1 libegl1 libxt6 libxrandr2 libxi6 libsm6 libice6
sudo rosdep init 2>/dev/null || true   # Register installs a task's system dependencies with rosdep
rosdep update
curl -LsSf https://astral.sh/uv/0.11.26/install.sh | sh
```
This list is the starting point; the loop adds what's missing and drops what nothing needs.

- [ ] **Step 3: Write `.dockerignore`**

```
.git
.claude
modules
docs
build
install
log
**/__pycache__
docker/secrets
```

- [ ] **Step 4: Write `docker/Dockerfile` (first three stages)**

```dockerfile
# syntax=docker/dockerfile:1.7
# GUIDE images, built from the guide repo root (the build context IS the GUIDE source):
#   docker build -f docker/Dockerfile --target test   -t guide:test   .
#   docker build -f docker/Dockerfile --target deploy -t guide:deploy .
#   docker build -f docker/Dockerfile --target mock   -t guide:mock   .
# base and build run the README's install blocks verbatim (docker/readme_steps.sh): a
# failing build is a docs bug -- fix README.md (+ INSTALLATION.md), not this file.

FROM ros:jazzy-ros-base AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PATH=/root/.local/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends sudo curl
COPY README.md docker/readme_steps.sh /tmp/guide/
RUN /tmp/guide/readme_steps.sh /tmp/guide/README.md Prerequisites && rm -rf /var/lib/apt/lists/*

FROM base AS build
COPY . /root/ros2_ws/src/guide
RUN /tmp/guide/readme_steps.sh /root/ros2_ws/src/guide/README.md Installation

FROM build AS test
RUN uv pip install --python /root/ros2_ws/.venv/bin/python pytest \
      -c /root/ros2_ws/src/guide/modules/isaac6-safe-pins.txt
WORKDIR /root/ros2_ws/src/guide
# INSTALLATION.md §7 plus the runner's tests. No GPU: the tests stub Isaac.
CMD ["bash", "-c", "source /root/ros2_ws/install/setup.bash && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/ros2_ws/.venv/bin/python -m pytest -q guide_core/test guide_ex/test guide_tasks/cube_stack/test docker/runner --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'"]
```

- [ ] **Step 5: First build**

Run: `docker build -f docker/Dockerfile --target build -t guide:build . 2>&1 | tee $PRIV/build.log`
Expected: FAIL somewhere. That is the point of the loop.

- [ ] **Step 6: The loop. Repeat until the build passes:**
  1. Find the failing README line: it is the last `+ …` xtrace line before the error in `$PRIV/build.log`.
  2. Diagnose with superpowers:systematic-debugging. Reproduce in the last good layer: `docker run --rm -it <layer id> bash`.
  3. Fix **README.md**, and the same fact in **INSTALLATION.md** with its "why". Never work around it in the Dockerfile.
  4. Commit: `git commit -am "docs: <the fix> (found by the docker build)"`.
  5. Rebuild (Step 5).

  Known defects and the docs fix to try first:

  | Symptom | Cause | Docs fix |
  |---|---|---|
  | colcon builds `franka_hardware` and fails on libfranka | the `franka_ros2` COLCON_IGNOREs were local-only | fixed in irob_franka_ros2 aa5fd9d (`jazzy`). If it isn't pushed yet, ask the user to push. |
  | recorder `ModuleNotFoundError: datasets`/`av` | `lerobot==0.6.0` lacks the dataset extra | step 3: `"lerobot[dataset]==0.6.0"` |
  | `isaacsim[all]` resolution error | `[all]` is broken in 6.0.1.0 GA (`modules/isaac6-install.md`) | step 3: that file's subset (see Task 12 Step 3 for the exact command) |
  | `fuser: not found` | `psmisc` undocumented | already in the Step 2 block |
  | INSTALLATION §6.2 says `isaacsim.exp.full.kit` | stale | `isaacsim.exp.base.python.kit` (`runtime.py:265` passes no experience) |

  Exit when `docker build --target build` passes from the docs alone.

- [ ] **Step 7: Tests in the image**

Run: `docker build -f docker/Dockerfile --target test -t guide:test . && docker run --rm guide:test`
Expected: everything passes: 519 at dev 834bb48, plus what Tasks 2–9 add once they're merged. A failure is a docs or environment gap: handle it as in Step 6.

- [ ] **Step 8: Commit**

```bash
git add .dockerignore docker/Dockerfile docker/readme_steps.sh README.md INSTALLATION.md
git commit -m "docker: build and test stages run the README install verbatim"
```

---

### Task 2: Task launches follow the simulator; clock remapped

**Files:**
- Modify: `guide_tasks/block_bin/launch/bringup.launch.py`, `guide_tasks/cube_stack/launch/bringup.launch.py`
- Create: `guide_core/test/test_multi_sim.py`

**Interfaces:**
- Produces: launch arguments `sim_id` (default `0`), `first_scene` (default `0`) and `num_env` (default `1`), which launch scenes `first_scene … first_scene+num_env-1` of `/Sim_<sim_id>`. Every node of the launch listens on `/Sim_<sim_id>/clock`.

- [ ] **Step 1: Write the failing test** (`guide_core/test/test_multi_sim.py`)

```python
"""Several simulators on one ROS domain: launches, clocks, finalized datasets, shutdown, Register."""

import importlib.util
from pathlib import Path

import pytest


def load_task_launch(package):
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory(package)) / "launch" / "bringup.launch.py"
    spec = importlib.util.spec_from_file_location(f"{package}_bringup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("package", ["block_bin", "cube_stack"])
def test_a_task_launch_follows_its_simulator_and_clock(package):
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node, SetRemap

    context = LaunchContext()
    context.launch_configurations.update({"sim_id": "3", "first_scene": "2", "num_env": "1"})

    (group,) = load_task_launch(package).generate_nodes(context)
    remap, *rest = group.get_sub_entities()

    assert isinstance(remap, SetRemap)
    assert perform_substitutions(context, remap.src) == "/clock"
    assert perform_substitutions(context, remap.dst) == "/Sim_3/clock"
    solvers = [a for a in rest if isinstance(a, Node)]
    assert [s._Node__arguments for s in solvers] == [["--namespace", "/Sim_3/Scene_2"]]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `colcon build --base-paths . --build-base $PRIV/build --install-base $PRIV/install --packages-select block_bin cube_stack && source $PRIV/install/setup.zsh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py`
Expected: FAIL: `generate_nodes` returns a flat list, with no `SetRemap`.

- [ ] **Step 3: Implement in `block_bin/launch/bringup.launch.py`.** Imports:

```python
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch_ros.actions import Node, SetRemap
```
At the top of `generate_nodes`:
```python
    num_env = int(LaunchConfiguration("num_env").perform(context))
    first = int(LaunchConfiguration("first_scene").perform(context))
    sim = f"/Sim_{LaunchConfiguration('sim_id').perform(context)}"
```
In both loops:
- `for i in range(num_env):` becomes `for i in range(first, first + num_env):`;
- `f"/Sim_0/Scene_{i}/franka"` becomes `f"{sim}/Scene_{i}/franka"`;
- `f"/Sim_0/Scene_{i}"` becomes `f"{sim}/Scene_{i}"`.

The return becomes:
```python
    # Each simulator publishes its own clock (/Sim_N/clock); use_sim_time nodes listen on
    # /clock, so point every node of this launch, the included MoveIt ones too, at it.
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *move_groups, *testers])]
```
`generate_launch_description` declares:
```python
            DeclareLaunchArgument(
                "num_env", default_value="1", description="Number of scenes to launch"
            ),
            DeclareLaunchArgument(
                "first_scene", default_value="0", description="Id of the first scene to launch"
            ),
            DeclareLaunchArgument(
                "sim_id", default_value="0", description="Simulator id: the Sim_<id> namespace"
            ),
```
`cube_stack/launch/bringup.launch.py` gets the same imports, arguments and `first`/`sim` lines. Its single loop (line 38) becomes `for i in range(first, first + num_env):`, with the same two `Sim_0` replacements. It collects everything in `nodes`, so its return is:
```python
    # Each simulator publishes its own clock (/Sim_N/clock); use_sim_time nodes listen on
    # /clock, so point every node of this launch, the included MoveIt ones too, at it.
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *nodes])]
```

- [ ] **Step 4: Rebuild and run the test**

Run: the Step 2 command again.
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add guide_tasks/block_bin/launch/bringup.launch.py guide_tasks/cube_stack/launch/bringup.launch.py guide_core/test/test_multi_sim.py
git commit -m "Task launches take sim_id and first_scene and listen on /Sim_N/clock"
```

> **Outside this repo (with the user):** `block_bin_eval` uses sim time. After Task 4 its
> nodes need `("/clock", "/Sim_0/clock")`: in `launch/eval_pink.launch.py:72-81`'s
> remappings, and as `--ros-args -r /clock:=/Sim_0/clock` for `eval_policy_pink`.

---

### Task 3: The recorder reports each finalize

**Files:**
- Modify: `guide_core/test/conftest.py` (receives `STUBBED` and `isaac_import`), `guide_core/test/test_command_fixes.py` (drops them)
- Modify: `guide_core/guide_core/scene/scene_recorder.py` (`__init__`, `run` lines 264-273, `_finalize_dataset`)
- Modify: `guide_core/guide_core/scene/scene_manager.py` (`finalize_recording`, `finalize_all_recordings` lines 512-528)
- Test: `guide_core/test/test_multi_sim.py`

**Interfaces:**
- Produces:
  - `SceneRecorder.clear_finalized()`; `SceneRecorder.wait_finalized(timeout=None) -> str | None` (the dataset dir, `""` if nothing was recorded, `None` on timeout); `_finalize_dataset() -> str`.
  - `SceneManager.wait_finalized(scene_id, timeout=None) -> str | None`.
  - `SceneManager.finalize_all_recordings() -> list[tuple[int, str]]`: the scenes that finished, with their dataset dir or `""`.
  - All of these go through the existing BaseManager proxy, which exposes every public method.

- [ ] **Step 1: Move the `isaac_import` fixture to `conftest.py`.** Cut `STUBBED` and the `isaac_import` fixture (`test_command_fixes.py:18-54`) and append them to `guide_core/test/conftest.py`, adding `import importlib`, `from unittest.mock import MagicMock` and `import pytest`. `command_module` stays in `test_command_fixes.py` and resolves `isaac_import` from conftest.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_command_fixes.py`
Expected: the same pass count as before.

- [ ] **Step 2: Write the failing tests** (append to `test_multi_sim.py`; add `import logging`, `import threading`, `from types import MethodType, SimpleNamespace` at the top)

```python
def recorder():
    from guide_core.scene.scene_recorder import SceneRecorder

    r = SceneRecorder("pkg", "dataset_0_0", {"dataset": {"fps": 10}})
    r._logger = logging.getLogger("test_multi_sim")
    return r


def test_finalizing_reports_the_written_dataset(tmp_path):
    r = recorder()
    r.dataset = SimpleNamespace(root=tmp_path, finalize=lambda: None)

    assert r._finalize_dataset() == str(tmp_path)
    assert r._finalize_dataset() == ""  # nothing recorded since


def test_the_recorder_thread_reports_every_finalize(monkeypatch):
    r = recorder()
    monkeypatch.setattr(r, "_attach_file_log", lambda: None)  # no log file in ~/.ros
    r.start()
    assert r.wait_finalized(0) is None

    for control in ("FINALIZE", "SHUTDOWN"):
        r.clear_finalized()
        r.put_record_data(control)
        r.set_start_recording()
        assert r.wait_finalized(10) == ""  # reported: nothing was recorded
    assert r.wait_shutdown(10)


class FakeRecorder:
    def __init__(self, path):
        self.path, self.controls = path, []

    def clear_stop_recording(self):
        pass

    def clear_finalized(self):
        self.controls.append("clear")

    def put_record_data(self, item):
        self.controls.append(item)

    def set_start_recording(self):
        pass

    def wait_shutdown(self, timeout=None):
        return True

    def wait_finalized(self, timeout=None):
        return self.path


def test_shutdown_finalizes_every_scene_and_says_what_it_wrote(isaac_import):
    manager = isaac_import("guide_core.scene.scene_manager").SceneManager
    recorders = [FakeRecorder("/scratch/d0"), FakeRecorder("")]
    me = SimpleNamespace(
        _scenes=[SimpleNamespace(state=None, recorder=r) for r in recorders],
        _locks=[threading.Lock(), threading.Lock()],
    )
    me.wait_finalized = MethodType(manager.wait_finalized, me)

    assert manager.finalize_all_recordings(me) == [(0, "/scratch/d0"), (1, "")]
    assert recorders[0].controls == ["clear", "SHUTDOWN"]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py -k "finaliz or reports"`
Expected: FAIL: `wait_finalized`/`clear_finalized` don't exist, and `_finalize_dataset` returns `None`.

- [ ] **Step 4: Implement the recorder.** In `SceneRecorder.__init__`, after `self.shutdown_event = Event()`:

```python
        # Set when a FINALIZE or SHUTDOWN has been written; _finalized_path is that dataset's
        # directory, "" when nothing was recorded since the previous one.
        self.finalized_event = Event()
        self._finalized_path = ""
```
New methods, next to `wait_shutdown`:
```python
    def clear_finalized(self):
        self.finalized_event.clear()

    def wait_finalized(self, timeout=None):
        """The finalized dataset's directory ("" = nothing recorded), or None on timeout."""
        return self._finalized_path if self.finalized_event.wait(timeout) else None
```
In `run`, the two control branches become:
```python
                    elif item == "FINALIZE":
                        self._logger.info("Received FINALIZE indicator. Finalizing dataset...")
                        self._finalized_path = self._finalize_dataset()
                        self.finalized_event.set()
                        self.idle_event.set()
                        self.start_recording_event.clear()
                        break
                    elif item == "SHUTDOWN":
                        self._logger.info("Received SHUTDOWN indicator. Finalizing and exiting...")
                        self._finalized_path = self._finalize_dataset()
                        self.finalized_event.set()
                        self.stop_flag.set()
                        break
```
In `_finalize_dataset`:
- add `-> str` to the signature and `written = ""` as its first line;
- inside `if self.dataset is not None:`, right after `dataset_root = self.dataset.root`, add `written = str(dataset_root)`;
- after the three event lines at the end, add `return written`.

- [ ] **Step 5: Implement the scene manager**

```python
    def finalize_recording(self, scene_id: int):
        with self._locks[scene_id]:
            self._scenes[scene_id].state = SceneState.FINALIZING
            self._scenes[scene_id].recorder.clear_stop_recording()
            self._scenes[scene_id].recorder.clear_finalized()
            self._scenes[scene_id].recorder.put_record_data("FINALIZE")
            self._scenes[scene_id].recorder.set_start_recording()

    def wait_finalized(self, scene_id: int, timeout=None):
        """The dataset the last finalize wrote ("" = nothing recorded), or None on timeout."""
        return self._scenes[scene_id].recorder.wait_finalized(timeout)

    def finalize_all_recordings(self) -> list:
        """Finalize every scene for shutdown; [(scene_id, dataset dir or "")] of those that finished."""
        for scene_id in range(len(self._scenes)):
            with self._locks[scene_id]:
                self._scenes[scene_id].state = SceneState.FINALIZING
                self._scenes[scene_id].recorder.clear_stop_recording()
                self._scenes[scene_id].recorder.clear_finalized()
                self._scenes[scene_id].recorder.put_record_data("SHUTDOWN")
                self._scenes[scene_id].recorder.set_start_recording()

        for scene_id in range(len(self._scenes)):
            self._scenes[scene_id].recorder.wait_shutdown(15.0)
        finished = [(i, self.wait_finalized(i, 0)) for i in range(len(self._scenes))]
        return [(i, path) for i, path in finished if path is not None]
```

- [ ] **Step 6: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py guide_core/test/test_discarded_frames.py guide_core/test/test_recorder_control_delivery.py guide_core/test/test_command_fixes.py`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add guide_core/guide_core/scene/scene_recorder.py guide_core/guide_core/scene/scene_manager.py guide_core/test/conftest.py guide_core/test/test_command_fixes.py guide_core/test/test_multi_sim.py
git commit -m "recorder: report each finalize (dataset dir) through the recorder proxy"
```

---

### Task 4: `/Sim_N/clock`, `/Sim_N/dataset_finalized`, finalize returns when written

**Files:**
- Modify: `guide_core/guide_core/ros/guide_ros.py` (imports, `__init__`, `_register_callback` clock lines 227-229, `_finalize_recording_callback`)
- Modify: `guide_core/package.xml` (`<depend>std_msgs</depend>`)
- Test: `guide_core/test/test_multi_sim.py`

**Interfaces:**
- Consumes: `SceneManager.finalize_recording`, `wait_finalized` (Task 3).
- Produces:
  - topic `/Sim_<id>/clock`;
  - topic `/Sim_<id>/dataset_finalized`: `std_msgs/String` JSON `{"scene": int, "path": str}`, transient-local, depth 100;
  - `GUIDEROS2Interface._announce_finalized(scene_id: int, path: str)`;
  - `finalize_recording` answers `message = <dataset dir>` (or `"Nothing was recorded."`) only after the recorder has written the dataset.

- [ ] **Step 1: Write the failing tests** (append; add `import json` and `from unittest.mock import MagicMock` at the top)

```python
def ros_class(isaac_import):
    return isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface


def test_the_clock_is_created_in_the_simulator_namespace(isaac_import):
    from guide_msgs.srv import RegisterScene

    backend = SimpleNamespace(
        stop=MagicMock(), play=MagicMock(), call=MagicMock(),
        register_scene=MagicMock(return_value=(0, (0.0, 0.0, 0.0))),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=False, _tasks=None,
        get_namespace=lambda: "/Sim_3",
    )

    reply = ros_class(isaac_import)._register_callback(me, RegisterScene.Request(path="block_bin"), None)

    assert reply.success
    backend.call.assert_called_once_with("create_clock", namespace="Sim_3")


def test_finalize_answers_with_the_written_dataset_and_announces_it(isaac_import):
    from guide_msgs.srv import FinalizeRecording

    scenes = SimpleNamespace(finalize_recording=MagicMock(), wait_finalized=lambda id, timeout=None: "/s/d1")
    announced = []
    me = SimpleNamespace(
        _backend=SimpleNamespace(_scene_manager=scenes), _logger=MagicMock(),
        _announce_finalized=lambda i, p: announced.append((i, p)),
    )

    reply = ros_class(isaac_import)._finalize_recording_callback(me, FinalizeRecording.Request(id=2), None)

    assert (reply.success, reply.message) == (True, "/s/d1")
    assert announced == [(2, "/s/d1")]


def test_a_recorder_that_never_finishes_fails_finalize(isaac_import):
    from guide_msgs.srv import FinalizeRecording

    scenes = SimpleNamespace(finalize_recording=MagicMock(), wait_finalized=lambda id, timeout=None: None)
    announced = []
    me = SimpleNamespace(
        _backend=SimpleNamespace(_scene_manager=scenes), _logger=MagicMock(),
        _announce_finalized=lambda i, p: announced.append((i, p)),
    )

    reply = ros_class(isaac_import)._finalize_recording_callback(me, FinalizeRecording.Request(id=2), None)

    assert not reply.success and "did not finish" in reply.message
    assert announced == []


def test_announcements_are_json_with_scene_and_path(isaac_import):
    published = []
    me = SimpleNamespace(_finalized_pub=SimpleNamespace(publish=published.append))

    ros_class(isaac_import)._announce_finalized(me, 1, "/s/d2")

    assert json.loads(published[0].data) == {"scene": 1, "path": "/s/d2"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py -k "clock_is or finalize or announce"`
Expected: FAIL. `create_clock` is called without a namespace, finalize doesn't wait, and `_announce_finalized` doesn't exist.

- [ ] **Step 3: Implement.** Add these imports to `guide_ros.py`: `import json`, `from rclpy.qos import DurabilityPolicy, QoSProfile` and `from std_msgs.msg import String`. Add a module constant below the imports:

```python
# Writing a long dataset's videos takes minutes; past this, finalize reports a stuck recorder.
FINALIZE_TIMEOUT_S = 600.0
```
At the end of `__init__`:
```python
        # Every dataset GUIDE finalizes, for whoever drives this simulator (a master, a
        # script): {"scene": id, "path": dir}; path "" when the scene recorded nothing.
        self._finalized_pub = self.create_publisher(
            String,
            "dataset_finalized",
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
```
New method:
```python
    def _announce_finalized(self, scene_id: int, path: str) -> None:
        self._finalized_pub.publish(String(data=json.dumps({"scene": scene_id, "path": path})))
```
In `_register_callback`:
```python
            if not self._has_clock:
                # /Sim_N/clock: every simulator runs at its own speed. Task launches remap
                # their nodes' /clock to it (SetRemap in <task>/launch/bringup.launch.py).
                self._backend.call("create_clock", namespace=self.get_namespace().strip("/"))
                self._has_clock = True
```
`_finalize_recording_callback`'s `try` body:
```python
            id = request.id
            self._logger.info(f"Finalizing recording for scene {id}...")

            self._backend._scene_manager.finalize_recording(id)
            # Answer once the recorder has written the dataset: it adds the language columns
            # after LeRobot's finalize, and only then is the dataset complete.
            path = self._backend._scene_manager.wait_finalized(id, FINALIZE_TIMEOUT_S)
            if path is None:
                raise TimeoutError(f"the recorder did not finish scene {id} in {FINALIZE_TIMEOUT_S:.0f} s")
            self._announce_finalized(id, path)

            response.message = path or "Nothing was recorded."
            response.success = True
```
In `guide_core/package.xml`, after `<depend>guide_msgs</depend>`: `<depend>std_msgs</depend>`.

- [ ] **Step 4: Run all guide_core tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/ros/guide_ros.py guide_core/package.xml guide_core/test/test_multi_sim.py
git commit -m "guide_core: /Sim_N/clock, /Sim_N/dataset_finalized; finalize_recording returns when written"
```

---

### Task 5: One shutdown path: `/Sim_N/shutdown`, Ctrl-C, SIGTERM

**Files:**
- Modify: `guide_core/guide_core/ros/guide_ros.py` (imports, `__init__`, new `shutdown` and `_shutdown_callback`, `launch_ros_interface`, `ros_entry_point`)
- Modify: `guide_core/guide_core/core/runtime.py` (`run_loop`, after `_process_commands`)
- Modify: `guide_core/launch/bringup.launch.py` (ExecuteProcess timeouts), `guide_core/package.xml` (`<depend>std_srvs</depend>`)
- Test: `guide_core/test/test_multi_sim.py`

**Interfaces:**
- Consumes: `finalize_all_recordings() -> list[tuple[int, str]]` (Task 3); `_announce_finalized` (Task 4); the runtime's existing `shutdown` command (`_cmd_shutdown`: stop, close Isaac, state `UNINITIALIZED`).
- Produces:
  - service `/Sim_<id>/shutdown` (`std_srvs/srv/Trigger`);
  - `GUIDEROS2Interface.shutdown()`, idempotent: finalize all → announce the non-empty ones → `tasks.shutdown()` → Isaac `shutdown` → `rclpy.try_shutdown()`;
  - `GUIDEROS2Interface(backend, node_name, namespace, tasks=None)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shutdown_runs_once_finalize_announce_tasks_isaac_ros(isaac_import, monkeypatch):
    module = isaac_import("guide_core.ros.guide_ros")
    order = []
    monkeypatch.setattr(module.rclpy, "try_shutdown", lambda: order.append("ros"))
    scenes = SimpleNamespace(
        finalize_all_recordings=lambda: order.append("finalize") or [(0, "/s/d0"), (1, "")]
    )
    me = SimpleNamespace(
        _backend=SimpleNamespace(_scene_manager=scenes, call=lambda name, timeout=None: order.append(name)),
        _logger=MagicMock(),
        _tasks=SimpleNamespace(shutdown=lambda: order.append("tasks")),
        _shutdown_lock=threading.Lock(),
        _shutting_down=False,
        _announce_finalized=lambda i, p: order.append(("announce", i, p)),
    )

    module.GUIDEROS2Interface.shutdown(me)
    module.GUIDEROS2Interface.shutdown(me)  # a second Ctrl-C or request changes nothing

    assert order == ["finalize", ("announce", 0, "/s/d0"), "tasks", "shutdown", "ros"]


def test_the_shutdown_service_answers_before_shutting_down(isaac_import):
    from std_srvs.srv import Trigger

    started = threading.Event()
    me = SimpleNamespace(shutdown=started.set)

    reply = ros_class(isaac_import)._shutdown_callback(me, Trigger.Request(), Trigger.Response())

    assert reply.success
    assert started.wait(2)


def test_the_loop_leaves_as_soon_as_a_command_shut_isaac_down(isaac_import):
    runtime = isaac_import("guide_core.core.runtime")
    me = SimpleNamespace(state=runtime.RUNNING, _gate_render=MagicMock())
    me._process_commands = lambda max_per_cycle: setattr(me, "state", runtime.UNINITIALIZED)

    runtime.IsaacSimRuntime.run_loop(me)

    me._gate_render.assert_not_called()  # Isaac is closed: touch nothing more
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py -k "shutdown or loop_leaves"`
Expected: FAIL. `shutdown` and `_shutdown_callback` don't exist, and `run_loop` calls `_gate_render` after the close.

- [ ] **Step 3: Implement in `runtime.py`.** In `run_loop`, right after `self._process_commands(max_per_cycle=50)`:

```python
            if self.state in (SHUTTING_DOWN, UNINITIALIZED):
                break  # a shutdown command closed Isaac: touch nothing more
```

- [ ] **Step 4: Implement in `guide_ros.py`.** Imports: `import signal`, `from threading import Lock, Thread` (replacing `from threading import Thread`), `from rclpy.signals import SignalHandlerOptions` and `from std_srvs.srv import Trigger`.

The `__init__` signature and its first lines:
```python
    def __init__(
        self,
        backend: GUIDESimulator,
        node_name: Optional[str],
        namespace: Optional[str],
        tasks=None,
    ):
        super().__init__(node_name=node_name, namespace=namespace)

        self._backend = backend
        # Fetches, builds and launches tasks for Register (guide_core.ros.task_bringup).
        self._tasks = tasks
        self._shutdown_lock = Lock()
        self._shutting_down = False
```
At the end of `__init__`:
```python
        # Stop this simulator. Ctrl-C and SIGTERM take the same path (ros_entry_point).
        self._shutdown_service = self.create_service(
            srv_type=Trigger,
            srv_name="shutdown",
            callback=self._shutdown_callback,
            callback_group=self._reentrant_group,
        )
```
New methods:
```python
    def shutdown(self) -> None:
        """The one way this simulator stops -- /shutdown, Ctrl-C and SIGTERM alike: finalize
        every scene (announcing what was written), stop the task launches, close Isaac, end ROS."""
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        self._logger.info("Shutting down: finalizing every scene...")
        for scene_id, path in self._backend._scene_manager.finalize_all_recordings():
            if path:
                self._announce_finalized(scene_id, path)
        if self._tasks:
            self._tasks.shutdown()
        try:
            self._backend.call("shutdown", 60.0)  # closes Isaac; run_runtime_loop returns
        finally:
            rclpy.try_shutdown()

    def _shutdown_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        Thread(target=self.shutdown).start()  # answer first: the shutdown ends this node
        response.success = True
        response.message = "Shutting down."
        return response
```
`launch_ros_interface` becomes:
```python
def launch_ros_interface(node: GUIDEROS2Interface):
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except ExternalShutdownException:
        pass  # GUIDEROS2Interface.shutdown ended ROS after finalizing every scene
```
In `ros_entry_point`, `rclpy.init(args=None)` becomes:
```python
    # 2. Initialize ROS 2. GUIDE handles the signals itself so that Ctrl-C, SIGTERM (docker stop,
    #    ros2 launch) and /shutdown all take ros_interface.shutdown: rclpy's own handler interrupts
    #    Isaac's loop mid-frame on SIGINT and never ends it on SIGTERM.
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
```
After `ros_interface` is created and its loggers are set:
```python
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: Thread(target=ros_interface.shutdown).start())
```
Remove the `ExternalShutdownException`/`KeyboardInterrupt` finalize block that `launch_ros_interface` had: `shutdown` does that now.

- [ ] **Step 5: Give `ros2 launch` time to let GUIDE finalize.** In `guide_core/launch/bringup.launch.py`, the GUIDE `ExecuteProcess` gets `sigterm_timeout="180", sigkill_timeout="10",` next to `output="both",`. Without it, `ros2 launch` escalates to SIGKILL after 5 s, in the middle of finalizing. Add `<depend>std_srvs</depend>` to `guide_core/package.xml`.

- [ ] **Step 6: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'`
Expected: all pass.

- [ ] **Step 7: Live check on this host (GPU, private install).** Build `guide_msgs guide_core block_bin cube_stack` into `$PRIV`, source it, and export the localhost `CYCLONEDDS_URI` from README "Usage". For each stop method in turn: Ctrl-C in the launch terminal; `ros2 service call /Sim_0/shutdown std_srvs/srv/Trigger`; `kill -TERM <GUIDE pid>`. Each time:

```bash
ros2 launch guide_core bringup.launch.py &
until ros2 service list | grep -q /Sim_0/Register; do sleep 5; done
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
# stop it one of the three ways
```
Expected:
- the log shows `Shutting down: finalizing every scene...`;
- the GUIDE process exits with code 0 within about 30 s;
- no traceback from `run_runtime_loop`.

- [ ] **Step 8: Commit**

```bash
git add guide_core/guide_core/ros/guide_ros.py guide_core/guide_core/core/runtime.py guide_core/launch/bringup.launch.py guide_core/package.xml guide_core/test/test_multi_sim.py
git commit -m "guide_core: /Sim_N/shutdown; Ctrl-C and SIGTERM take the same finalize-and-close path"
```

---

### Task 6: TaskBringup: fetch, build, activate, launch

**Files:**
- Create: `guide_core/guide_core/ros/task_bringup.py`, `guide_core/test/test_task_bringup.py`

**Interfaces:**
- Produces, in `guide_core.ros.task_bringup`:
  - `split_s3(url) -> tuple[str, str]`
  - `s3_client()` (boto3, path-style addressing; endpoint and credentials from the `AWS_*` env)
  - `is_installed(name) -> bool`
  - `has_bringup(name) -> bool`
  - `task_package(bundle: Path) -> str | None`
  - `class TaskBringup(sim_id: int, workdir: Path = ~/.guide/tasks, run=_run, popen=subprocess.Popen, s3=None)` with:
    - `.prepare(path) -> tuple[str, str | None]`: what to register, and the package whose bringup can be launched;
    - `.activate()`;
    - `.launch(pkg, scene_id)`;
    - `.shutdown(timeout=30.0)`.
- Environment: `GUIDE_PINS` (pip constraints file; optional).

- [ ] **Step 1: Write the failing tests** (`guide_core/test/test_task_bringup.py`)

```python
"""Register a task by name, directory or S3 archive: fetch it, build it, launch its bringup."""

import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from guide_core.ros import task_bringup as tb


def make_bundle(root: Path, name="my_task", reqs=False) -> Path:
    pkg = root / name
    (pkg / name).mkdir(parents=True)
    (pkg / name / "scene.py").write_text("class Scene: pass\n")
    (pkg / "package.xml").write_text(f"<package><name>{name}</name></package>\n")
    if reqs:
        (root / "requirements.txt").write_text("six\n")
    return root


@pytest.fixture
def nothing_installed(monkeypatch):
    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    monkeypatch.setattr(tb, "has_bringup", lambda name: True)
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_an_installed_package_is_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(tb, "is_installed", lambda name: name == "block_bin")
    monkeypatch.setattr(tb, "has_bringup", lambda name: True)
    ran = []
    assert tb.TaskBringup(0, tmp_path, run=ran.append).prepare("block_bin") == ("block_bin", "block_bin")
    assert ran == []


def test_a_bundle_is_built_with_its_dependencies(nothing_installed, monkeypatch, tmp_path):
    bundle = make_bundle(tmp_path / "bundle", reqs=True)
    monkeypatch.setenv("GUIDE_PINS", "/pins.txt")
    ran = []

    assert tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(bundle)) == ("my_task", "my_task")

    assert [cmd[:2] for cmd in ran] == [["rosdep", "install"], ["uv", "pip"], ["colcon", "--log-base"]]
    assert ran[1][-2:] == ["-c", "/pins.txt"]
    assert ran[2][-2:] == ["--packages-up-to", "my_task"]


def test_a_bundle_without_requirements_skips_pip(nothing_installed, tmp_path):
    ran = []
    tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(make_bundle(tmp_path / "b")))
    assert [cmd[0] for cmd in ran] == ["rosdep", "colcon"]


def test_a_bundle_holds_at_most_one_task(tmp_path):
    make_bundle(tmp_path, "a")
    make_bundle(tmp_path, "b")
    with pytest.raises(ValueError, match="more than one task package"):
        tb.task_package(tmp_path)


def test_a_plain_scene_directory_is_registered_without_building(nothing_installed, tmp_path):
    (tmp_path / "scene.py").write_text("class Scene: pass\n")  # like guide_core/dummy_scene
    ran = []
    assert tb.TaskBringup(0, tmp_path / "w", run=ran.append).prepare(str(tmp_path)) == (str(tmp_path), None)
    assert ran == []


def test_an_unknown_path_fails_with_the_reason(nothing_installed, tmp_path):
    with pytest.raises(FileNotFoundError, match="neither an installed package nor a directory"):
        tb.TaskBringup(0, tmp_path).prepare("no_such_task")


def test_an_s3_bundle_is_downloaded_and_unpacked(nothing_installed, tmp_path):
    src = make_bundle(tmp_path / "src")
    archive = tmp_path / "my_task.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=".")

    class FakeS3:
        def download_file(self, bucket, key, dest):
            assert (bucket, key) == ("tasks", "v1/my_task.tar.gz")
            Path(dest).write_bytes(archive.read_bytes())

    work = tmp_path / "work"
    tasks = tb.TaskBringup(0, work, run=lambda cmd: None, s3=FakeS3())

    assert tasks.prepare("s3://tasks/v1/my_task.tar.gz") == ("my_task", "my_task")
    assert (work / "src" / "my_task" / "my_task" / "my_task" / "scene.py").is_file()


def test_activation_puts_the_overlay_first(monkeypatch, tmp_path):
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
    monkeypatch.setattr(sys, "path", list(sys.path))

    tb.TaskBringup(0, tmp_path).activate()

    assert os.environ["AMENT_PREFIX_PATH"] == f"{tmp_path / 'install'}{os.pathsep}/opt/ros/jazzy"
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    assert sys.path[0] == str(tmp_path / "install" / "lib" / py / "site-packages")


def test_launch_starts_one_scene_in_this_simulator(tmp_path):
    started = []

    def popen(cmd, **kwargs):
        started.append(cmd)
        return SimpleNamespace(pid=1, poll=lambda: 0)

    tb.TaskBringup(3, tmp_path, popen=popen).launch("my_task", 2)

    assert started == [[
        "bash", "-c",
        "exec ros2 launch my_task bringup.launch.py sim_id:=3 first_scene:=2 num_env:=1",
    ]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_task_bringup.py`
Expected: FAIL, `ImportError: cannot import name 'task_bringup'`.

- [ ] **Step 3: Implement** `guide_core/guide_core/ros/task_bringup.py`

```python
"""Make a task runnable from a Register path: fetch it, build it with its dependencies, launch it.

A Register path is one of
  - an installed package name ("block_bin")          -> used as it is
  - a directory                                      -> a task bundle, or one of GUIDE's own
                                                        scene layouts (e.g. guide_core/dummy_scene)
  - s3://bucket/key.tar.gz holding a task bundle     -> downloaded and unpacked first
A task bundle holds exactly one task package (<pkg>/<pkg>/scene.py beside its package.xml), the
packages it depends on, and optionally requirements.txt (pip). System dependencies come from
the package.xml files through rosdep. Bundles are built into one overlay, <workdir>/install,
which this process and every launch then use.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_WORKDIR = Path.home() / ".guide" / "tasks"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def split_s3(url: str) -> tuple[str, str]:
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return bucket, key


def s3_client():
    import boto3  # in the venv already: an isaacsim dependency
    from botocore.config import Config

    # Endpoint and credentials from the environment (AWS_ENDPOINT_URL,
    # AWS_SHARED_CREDENTIALS_FILE, ...); path-style, which Ceph RGW and MinIO serve without
    # wildcard DNS.
    return boto3.client("s3", config=Config(s3={"addressing_style": "path"}))


def is_installed(name: str) -> bool:
    from ament_index_python.packages import get_package_share_directory

    try:
        get_package_share_directory(name)
        return True
    except (LookupError, ValueError):  # PackageNotFoundError is a KeyError
        return False


def has_bringup(name: str) -> bool:
    from ament_index_python.packages import get_package_share_directory

    return (Path(get_package_share_directory(name)) / "launch" / "bringup.launch.py").is_file()


def task_package(bundle: Path) -> str | None:
    """The bundle's task package (<pkg>/<pkg>/scene.py beside a package.xml); None if it has none."""
    found = [
        d.name
        for d in [bundle, *sorted(p for p in bundle.iterdir() if p.is_dir())]
        if (d / "package.xml").is_file() and (d / d.name / "scene.py").is_file()
    ]
    if len(found) > 1:
        raise ValueError(f"{bundle} holds more than one task package: {found}")
    return found[0] if found else None


class TaskBringup:
    def __init__(self, sim_id: int, workdir: Path = DEFAULT_WORKDIR, run=_run,
                 popen=subprocess.Popen, s3=None):
        self.sim_id = sim_id
        self.workdir = Path(workdir)
        self.install = self.workdir / "install"
        self._run = run
        self._popen = popen
        self._s3 = s3
        self._launches = []

    def prepare(self, path: str) -> tuple[str, str | None]:
        """What to register, and the package whose bringup can be launched for it (or None)."""
        if not path.startswith("s3://") and is_installed(path):
            return path, path if has_bringup(path) else None
        bundle = self._fetch(path)
        pkg = task_package(bundle)
        if pkg is None:
            return str(bundle), None  # one of GUIDE's own scene layouts: nothing to build
        if not is_installed(pkg):
            self._build(bundle, pkg)
        return pkg, pkg if has_bringup(pkg) else None

    def _fetch(self, path: str) -> Path:
        if not path.startswith("s3://"):
            bundle = Path(path).expanduser()
            if not bundle.is_dir():
                raise FileNotFoundError(f"{path} is neither an installed package nor a directory")
            return bundle
        bucket, key = split_s3(path)
        archive = self.workdir / "downloads" / Path(key).name
        archive.parent.mkdir(parents=True, exist_ok=True)
        (self._s3 or s3_client()).download_file(bucket, key, str(archive))
        name = archive.name.removesuffix(".gz").removesuffix(".tar").removesuffix(".tgz")
        bundle = self.workdir / "src" / name
        with tarfile.open(archive) as tar:
            tar.extractall(bundle, filter="data")  # no absolute paths, no escaping links
        return bundle

    def _build(self, bundle: Path, pkg: str) -> None:
        self._run(["rosdep", "install", "--from-paths", str(bundle), "--ignore-src", "-y"])
        reqs = bundle / "requirements.txt"
        if reqs.is_file():
            pins = os.environ.get("GUIDE_PINS")
            self._run(
                ["uv", "pip", "install", "--python", sys.executable, "-r", str(reqs)]
                + (["-c", pins] if pins else [])
            )
        self._run([
            "colcon", "--log-base", str(self.workdir / "log"), "build", "--merge-install",
            "--base-paths", str(bundle), "--build-base", str(self.workdir / "build"),
            "--install-base", str(self.install), "--packages-up-to", pkg,
        ])
        self.activate()

    def activate(self) -> None:
        """Make the overlay visible here: the scene's ament lookups and its imports."""
        prefix = str(self.install)
        paths = os.environ.get("AMENT_PREFIX_PATH", "")
        if prefix not in paths.split(os.pathsep):
            os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(p for p in (prefix, paths) if p)
        py = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = str(self.install / "lib" / py / "site-packages")
        if site not in sys.path:
            sys.path.insert(0, site)

    def launch(self, pkg: str, scene_id: int) -> None:
        """Start the task's MoveIt + solver for one scene: <pkg>/launch/bringup.launch.py."""
        cmd = (
            f"exec ros2 launch {pkg} bringup.launch.py "
            f"sim_id:={self.sim_id} first_scene:={scene_id} num_env:=1"
        )
        setup = self.install / "setup.bash"
        if setup.is_file():
            cmd = f"source {setup} && {cmd}"
        self._launches.append(self._popen(["bash", "-c", cmd], start_new_session=True))

    def shutdown(self, timeout: float = 30.0) -> None:
        for p in self._launches:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        for p in self._launches:
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_task_bringup.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/ros/task_bringup.py guide_core/test/test_task_bringup.py
git commit -m "guide_core: TaskBringup fetches (dir/S3), builds (rosdep, pip, colcon) and launches tasks"
```

---

### Task 7: Register fetches, builds and (on request) launches

**Files:**
- Modify: `guide_msgs/srv/RegisterScene.srv`
- Modify: `guide_core/guide_core/ros/guide_ros.py` (`_register_callback`, `ros_entry_point`)
- Modify: `README.md` ("Usage": the `bringup` field)
- Test: `guide_core/test/test_multi_sim.py`

**Interfaces:**
- Consumes: `TaskBringup.prepare/launch` (Task 6).
- Produces:
  - `RegisterScene.Request`: `path`, `bool bringup`;
  - `RegisterScene.Response`: `id`, `offset`, `string package` (the registered task package, `""` for a plain scene dir), `message`, `success`.

- [ ] **Step 1: Change the interface** (`guide_msgs/srv/RegisterScene.srv`)

```
# A package name, a directory, or s3://bucket/key.tar.gz holding a task bundle.
string path
# Also launch the task's launch/bringup.launch.py (MoveIt + solver) for the new scene.
bool bringup
---
uint8 id
float32[] offset
# The registered task package ("" for a plain scene directory).
string package
string message
bool success
```
Run: `colcon build --base-paths . --build-base $PRIV/build --install-base $PRIV/install --packages-select guide_msgs guide_core && source $PRIV/install/setup.zsh`

- [ ] **Step 2: Write the failing tests**

```python
def register(isaac_import, prepare, bringup=True):
    from guide_msgs.srv import RegisterScene

    order = []
    backend = SimpleNamespace(
        stop=lambda: order.append("stop"),
        play=lambda: order.append("play"),
        call=MagicMock(),
        register_scene=lambda path: order.append(("register", path)) or (1, (0.0, 2.0, 0.0)),
    )
    tasks = SimpleNamespace(
        prepare=lambda path: order.append("prepare") or prepare(path),
        launch=lambda pkg, scene_id: order.append(("launch", pkg, scene_id)),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=True, _tasks=tasks,
        get_namespace=lambda: "/Sim_3",
    )
    request = RegisterScene.Request(path="s3://t/my_task.tar.gz", bringup=bringup)
    return ros_class(isaac_import)._register_callback(me, request, None), order


def test_register_builds_first_and_launches_the_scene_last(isaac_import):
    reply, order = register(isaac_import, lambda path: ("my_task", "my_task"))

    assert (reply.success, reply.id, reply.package) == (True, 1, "my_task")
    assert order == ["prepare", "stop", ("register", "my_task"), "play", ("launch", "my_task", 1)]


def test_register_without_bringup_launches_nothing(isaac_import):
    reply, order = register(isaac_import, lambda path: ("my_task", "my_task"), bringup=False)

    assert reply.success
    assert [o for o in order if o[0] == "launch"] == []


def test_a_failed_build_fails_register_and_keeps_the_simulator_running(isaac_import):
    def broken(path):
        raise RuntimeError("rosdep: cannot resolve key 'libfoo'")

    reply, order = register(isaac_import, broken)

    assert not reply.success and "libfoo" in reply.message
    assert order == ["prepare"]  # never stopped: the other scenes kept stepping


def test_bringup_needs_a_bringup_launch(isaac_import):
    reply, order = register(isaac_import, lambda path: ("/scenes/flat", None))

    assert not reply.success and "bringup.launch.py" in reply.message
    assert "stop" not in order
```

- [ ] **Step 3: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_multi_sim.py -k "register or bringup"`
Expected: FAIL. `prepare` is never called and `reply.package` is empty.

- [ ] **Step 4: Implement.** `_register_callback`'s `try` body:

```python
            # A task given as a directory or an s3://….tar.gz is fetched and built with its
            # dependencies first: that can take minutes, and the other scenes keep stepping.
            path, package = (
                self._tasks.prepare(request.path) if self._tasks else (request.path, None)
            )
            if request.bringup and package is None:
                raise ValueError(f"{request.path} has no launch/bringup.launch.py to start")

            self._backend.stop()

            id, offset = self._backend.register_scene(path)

            self._logger.info(f"Registered scene with id {id} at offset {offset}")

            if not self._has_clock:
                # /Sim_N/clock: every simulator runs at its own speed. Task launches remap
                # their nodes' /clock to it (SetRemap in <task>/launch/bringup.launch.py).
                self._backend.call("create_clock", namespace=self.get_namespace().strip("/"))
                self._has_clock = True

            self._backend.play()
            if request.bringup:
                self._tasks.launch(package, id)  # the task's MoveIt + solver for this scene
            response.id = id
            response.offset = list(offset)
            response.package = package or ""
            response.message = ""
            response.success = True
```
In `ros_entry_point`, add the import `from guide_core.ros.task_bringup import TaskBringup` and pass the tasks in:
```python
    ros_interface = GUIDEROS2Interface(
        sim, node_name="GUIDE", namespace=NAMESPACE, tasks=TaskBringup(args.id)
    )
```

- [ ] **Step 5: README "Usage".** After the Register example, add:

```bash
# Register can also fetch a task (a directory or s3://bucket/task.tar.gz), build it with its
# dependencies, and start its MoveIt + solver for the new scene -- one call instead of two:
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin', bringup: true}"
```

- [ ] **Step 6: Run all tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test guide_ex/test guide_tasks/cube_stack/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add guide_msgs/srv/RegisterScene.srv guide_core/guide_core/ros/guide_ros.py guide_core/test/test_multi_sim.py README.md
git commit -m "Register fetches and builds tasks (dir/S3) and, with bringup: true, launches their bringup"
```

---

### Task 8: Runner: plan, splitting, DDS config, delivery (pure parts)

**Files:**
- Create: `docker/runner/guide_container.py` (the parts below; `main` comes in Task 9), `docker/runner/test_guide_container.py`

**Interfaces:**
- Consumes: `split_s3`, `s3_client` from `guide_core.ros.task_bringup` (Task 6).
- Produces, in `guide_container`:
  - `load_plan(text) -> {"output", "jobs": [{"task", "zones", "counts"}]}` (raises `ValueError`)
  - `work(job, num_zones) -> list[tuple[int | None, int]]`
  - `capacity(items) -> int`
  - `allocate(loads, caps, scenes) -> list[int]`
  - `deal(items, k) -> list[{"zones", "counts"}]`
  - `overlay_ip(subnet, ip_json=None) -> str`
  - `dds_config(ip, peers) -> str`
  - `zone_counts(dataset) -> Counter`
  - `complete(scene, counts) -> bool`
  - `deliver(dataset, output, ns, s3=None) -> str`
  - `read_text(location, s3=None) -> str`

- [ ] **Step 1: Write the failing tests** (`docker/runner/test_guide_container.py`)

```python
"""The container runner's pure parts: plan, splitting, DDS config, completeness, delivery."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import guide_container as c

PLAN = """
output: s3://bucket/guide
jobs:
  - {task: block_bin, zones: [-1], counts: [5]}
  - {task: s3://tasks/cube_stack.tar.gz, counts: [30]}
"""


def test_a_plan_is_read_with_its_defaults():
    plan = c.load_plan(PLAN)
    assert plan["output"] == "s3://bucket/guide"
    assert plan["jobs"][1] == {"task": "s3://tasks/cube_stack.tar.gz", "zones": [], "counts": [30]}


@pytest.mark.parametrize("jobs, reason", [
    ("[]", "non-empty"),
    ("[{zones: [1], counts: [2]}]", "'task'"),
    ("[{task: t, zones: [1, 2], counts: [3]}]", "one count per distinct zone"),
    ("[{task: t, zones: [2, 2], counts: [1, 1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-2], counts: [1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-1], counts: [1, 2]}]", "exactly one count"),
    ("[{task: t, counts: [0]}]", "positive"),
])
def test_bad_plans_are_rejected(jobs, reason):
    with pytest.raises(ValueError, match=reason):
        c.load_plan(f"jobs: {jobs}")


def test_work_expands_every_zone_and_keeps_free_draws_whole():
    assert c.work({"zones": [-1], "counts": [5]}, 3) == [(0, 5), (1, 5), (2, 5)]
    assert c.work({"zones": [-1], "counts": [5]}, 1) == [(None, 5)]
    assert c.work({"zones": [], "counts": [30]}, 20) == [(None, 30)]
    assert c.work({"zones": [2, 16], "counts": [4, 10]}, 20) == [(2, 4), (16, 10)]


def test_spare_scenes_go_to_the_heaviest_job():
    assert c.allocate([100, 30], [20, 30], 4) == [3, 1]
    assert c.allocate([100, 30], [2, 30], 4) == [2, 2]  # a zoned job can't use more scenes than zones
    assert c.allocate([5], [5], 1) == [1]


def test_a_job_is_dealt_evenly():
    parts = c.deal(c.work({"zones": [-1], "counts": [5]}, 20), 3)
    assert sorted(sum(p["counts"]) for p in parts) == [30, 35, 35]
    assert sorted(z for p in parts for z in p["zones"]) == list(range(20))
    assert c.deal([(None, 10)], 3) == [
        {"zones": [], "counts": [4]}, {"zones": [], "counts": [3]}, {"zones": [], "counts": [3]},
    ]


IP_JSON = json.dumps([
    {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]},
    {"ifname": "eth0", "addr_info": [{"local": "10.42.0.7"}]},
    {"ifname": "eth1", "addr_info": [{"local": "10.0.9.3"}]},
])


def test_the_overlay_address_is_picked_by_subnet():
    assert c.overlay_ip("10.42.0.0/24", IP_JSON) == "10.42.0.7"
    with pytest.raises(RuntimeError, match="guide-net"):
        c.overlay_ip("10.99.0.0/24", IP_JSON)


def test_dds_stays_on_the_overlay_and_peers_with_the_master():
    ns = {"c": "https://cdds.io/config"}
    root = ET.fromstring(c.dds_config("10.42.0.7", ["guide-master"]))
    assert [i.get("address") for i in root.iterfind(".//c:NetworkInterface", ns)] == ["10.42.0.7"]
    assert root.find(".//c:AllowMulticast", ns).text == "false"
    assert [p.get("address") for p in root.iterfind(".//c:Peer", ns)] == ["10.42.0.7", "guide-master"]


def dataset(tmp_path, zones):
    d = tmp_path / "scratch" / "Sim_0" / "scene_0" / "dataset_0_0_x"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "guide_episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "zone": z}) + "\n" for i, z in enumerate(zones)))
    (d / "data").mkdir()
    (d / "data" / "file-000.parquet").write_bytes(b"x")
    return d


def test_completeness_per_scene(tmp_path):
    counts = c.zone_counts(dataset(tmp_path, [2, 2, 16]))
    assert c.complete({"zones": [2, 16], "counts": [2, 1]}, counts)
    assert not c.complete({"zones": [2, 16], "counts": [2, 2]}, counts)
    assert c.complete({"zones": [], "counts": [3]}, counts)


def test_delivery_to_a_folder_moves_it_under_the_simulator(tmp_path):
    d = dataset(tmp_path, [0])
    out = tmp_path / "out"
    out.mkdir()
    assert c.deliver(d, str(out), "Sim_3") == str(out / "Sim_3" / "dataset_0_0_x")
    assert (out / "Sim_3" / "dataset_0_0_x" / "data" / "file-000.parquet").stat().st_uid == out.stat().st_uid
    assert not d.exists()


class FakeS3:
    def __init__(self, fail=False):
        self.keys, self.fail = [], fail

    def upload_file(self, path, bucket, key):
        if self.fail:
            raise ConnectionError("endpoint unreachable")
        self.keys.append((bucket, key))


def test_delivery_to_s3_uploads_every_file_then_frees_scratch(tmp_path):
    d = dataset(tmp_path, [0])
    s3 = FakeS3()
    assert c.deliver(d, "s3://bucket/guide", "Sim_3", s3=s3) == "s3://bucket/guide/Sim_3/dataset_0_0_x"
    assert sorted(s3.keys) == [
        ("bucket", "guide/Sim_3/dataset_0_0_x/data/file-000.parquet"),
        ("bucket", "guide/Sim_3/dataset_0_0_x/meta/guide_episodes.jsonl"),
    ]
    assert not d.exists()


def test_a_failed_upload_keeps_the_dataset(tmp_path):
    d = dataset(tmp_path, [0])
    with pytest.raises(ConnectionError):
        c.deliver(d, "s3://bucket/guide", "Sim_3", s3=FakeS3(fail=True))
    assert (d / "meta" / "guide_episodes.jsonl").is_file()


def test_a_plan_is_read_from_a_file_or_s3(tmp_path):
    f = tmp_path / "plan.yaml"
    f.write_text(PLAN)
    assert c.read_text(str(f)) == PLAN

    class Body:
        def read(self):
            return PLAN.encode()

    class S3:
        def get_object(self, Bucket, Key):
            assert (Bucket, Key) == ("plans", "a/plan.yaml")
            return {"Body": Body()}

    assert c.read_text("s3://plans/a/plan.yaml", s3=S3()) == PLAN
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q docker/runner`
Expected: FAIL, `ModuleNotFoundError: No module named 'guide_container'`.

- [ ] **Step 3: Implement** `docker/runner/guide_container.py` (pure parts)

```python
"""GUIDE container runner: one simulator, driven by a plan or by a master over ROS 2.

Container glue only -- GUIDE itself runs unchanged as `GUIDE --id N`. Environment:
  GUIDE_SIM_ID      simulator id -> /Sim_<id> (default 0)
  GUIDE_PLAN        plan file or s3:// URL; unset = slave mode (a master drives GUIDE)
  GUIDE_OUTPUT      directory or s3://bucket/prefix for finished datasets (overrides the plan's)
  GUIDE_MAX_SCENES  scenes a plan may use (default: one per job)
  GUIDE_MASTER      the master's name or address on guide-net: the DDS peer besides us
  GUIDE_DDS_NET     guide-net's subnet, e.g. 10.42.0.0/24 (unset: DDS on localhost only)
  GUIDE_SCRATCH     recording directory (default /scratch)
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from guide_core.ros.task_bringup import s3_client, split_s3


def read_text(location: str, s3=None) -> str:
    if not location.startswith("s3://"):
        return Path(location).read_text()
    bucket, key = split_s3(location)
    return (s3 or s3_client()).get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def load_plan(text: str) -> dict:
    plan = yaml.safe_load(text) or {}
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan: 'jobs' must be a non-empty list")
    return {"output": plan.get("output"), "jobs": [_job(i, job) for i, job in enumerate(jobs)]}


def _job(i: int, job: dict) -> dict:
    """Demonstration.srv's rules: [] = free draws and [-1] = every zone, one count each;
    otherwise one count per distinct zone >= 0."""
    where = f"plan: job {i}"
    task, zones, counts = job.get("task"), job.get("zones", []), job.get("counts")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{where}: 'task' must name a package, a directory or an s3:// bundle")
    if not isinstance(counts, list) or not counts or not all(
        isinstance(n, int) and n > 0 for n in counts
    ):
        raise ValueError(f"{where}: 'counts' must be positive integers")
    if not isinstance(zones, list) or not all(isinstance(z, int) for z in zones):
        raise ValueError(f"{where}: 'zones' must be a list of integers")
    if zones in ([], [-1]):
        if len(counts) != 1:
            raise ValueError(f"{where}: zones {zones} take exactly one count")
    elif len(zones) != len(counts) or len(set(zones)) != len(zones) or min(zones) < 0:
        raise ValueError(f"{where}: one count per distinct zone >= 0")
    return {"task": task, "zones": zones, "counts": counts}


def work(job: dict, num_zones: int) -> list:
    """(zone, count) pairs; zone None = free draws. [-1] becomes every zone of the task's grid."""
    zones, counts = job["zones"], job["counts"]
    if not zones or (zones == [-1] and num_zones <= 1):
        return [(None, counts[0])]
    if zones == [-1]:
        return [(z, counts[0]) for z in range(num_zones)]
    return list(zip(zones, counts))


def capacity(items: list) -> int:
    """How many scenes a job's work can be spread over."""
    return items[0][1] if items[0][0] is None else len(items)


def allocate(loads: list, caps: list, scenes: int) -> list:
    """Scenes per job: one each, then every spare scene to the job with the most episodes per scene."""
    k = [1] * len(loads)
    for _ in range(scenes - len(loads)):
        open_jobs = [i for i in range(len(loads)) if k[i] < caps[i]]
        if not open_jobs:
            break
        i = max(open_jobs, key=lambda j: loads[j] / k[j])
        k[i] += 1
    return k


def deal(items: list, k: int) -> list:
    """A job's work cut into k scenes: free draws split by count, zones dealt whole, heaviest first."""
    if items[0][0] is None:
        n = items[0][1]
        return [{"zones": [], "counts": [n // k + (i < n % k)]} for i in range(k)]
    parts = [{"zones": [], "counts": []} for _ in range(k)]
    for zone, count in sorted(items, key=lambda zc: -zc[1]):
        part = min(parts, key=lambda p: sum(p["counts"]))
        part["zones"].append(zone)
        part["counts"].append(count)
    return parts


def overlay_ip(subnet: str, ip_json: str | None = None) -> str:
    net = ipaddress.ip_network(subnet)
    if ip_json is None:
        ip_json = subprocess.run(
            ["ip", "-j", "-4", "addr"], capture_output=True, text=True, check=True
        ).stdout
    for iface in json.loads(ip_json):
        for addr in iface.get("addr_info", []):
            if ipaddress.ip_address(addr["local"]) in net:
                return addr["local"]
    raise RuntimeError(f"no interface on {subnet}: is the container attached to guide-net?")


def dds_config(ip: str, peers: list) -> str:
    peer_xml = "".join(f'<Peer address="{p}"/>' for p in [ip, *peers])
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- Written by docker/runner/guide_container.py: DDS on the guide-net interface only, unicast
     discovery (overlay networks carry no multicast) of this container and the master. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface address="{ip}"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
      <Peers>{peer_xml}</Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def zone_counts(dataset: Path) -> Counter:
    meta = dataset / "meta" / "guide_episodes.jsonl"
    lines = meta.read_text().splitlines() if meta.is_file() else []
    return Counter(json.loads(line).get("zone") for line in lines if line.strip())


def complete(scene: dict, counts: Counter) -> bool:
    if not scene["zones"]:
        return sum(counts.values()) == scene["counts"][0]
    return all(counts[z] == n for z, n in zip(scene["zones"], scene["counts"]))


def deliver(dataset: Path, output: str, ns: str, s3=None) -> str:
    """Move or upload one finished dataset; the scratch copy goes only once all of it is out."""
    if output.startswith("s3://"):
        bucket, prefix = split_s3(output)
        base = "/".join(p for p in (prefix.strip("/"), ns, dataset.name) if p)
        client = s3 or s3_client()
        for f in sorted(p for p in dataset.rglob("*") if p.is_file()):
            client.upload_file(str(f), bucket, f"{base}/{f.relative_to(dataset).as_posix()}")
        shutil.rmtree(dataset)
        return f"s3://{bucket}/{base}"
    target = Path(output) / ns / dataset.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dataset), str(target))
    owner = Path(output).stat()  # the container runs as root: hand the files to the folder's owner
    for p in [target.parent, target, *target.rglob("*")]:
        os.chown(p, owner.st_uid, owner.st_gid)
    return str(target)
```

- [ ] **Step 4: Run the tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q docker/runner`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add docker/runner/guide_container.py docker/runner/test_guide_container.py
git commit -m "docker: runner parts: plan jobs, scene splitting, overlay DDS config, delivery"
```

---

### Task 9: Runner main and the mock simulator

**Files:**
- Modify: `docker/runner/guide_container.py` (append the runtime part), `docker/runner/test_guide_container.py`
- Create: `docker/mock/mock_sim.py`, `docker/entrypoint.sh`

**Interfaces:**
- Consumes:
  - Task 8 functions;
  - GUIDE's `--id`;
  - `/Sim_N/Register` with `bringup` and `package` (Task 7);
  - `Scene_i/generate_demonstration`;
  - `/Sim_N/dataset_finalized` (Task 4);
  - `/Sim_N/shutdown` (Task 5).
- Produces:
  - `main(env=None) -> int`, run as `python guide_container.py`;
  - topic `/Sim_N/dataset_delivered` (`std_msgs/String` JSON `{"dataset", "target", "complete"}`, transient-local);
  - env `GUIDE_CMD` replaces the simulator command (the mock image, tests);
  - `docker/mock/mock_sim.py` serves the same names and types as GUIDE.

- [ ] **Step 1: Write `docker/mock/mock_sim.py`**

```python
#!/usr/bin/env python3
"""A stand-in GUIDE for communication tests: the same namespace, services and topics, no Isaac.

Register adds a scene and serves its Scene_<i>/generate_demonstration (in the real system the
task's solver does). A request writes a LeRobot-shaped dataset (meta/guide_episodes.jsonl, one
line per episode) and announces it on /Sim_<id>/dataset_finalized, as GUIDE does.
/Sim_<id>/shutdown exits. /Sim_<id>/clock runs at 10x real time, so two mocks' clocks differ.
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from guide_msgs.srv import Demonstration, RegisterScene
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger


def episodes(zones: list, counts: list) -> list:
    if not zones:
        return [None] * counts[0]
    if zones == [-1]:
        return [z for z in (0, 1) for _ in range(counts[0])]  # the mock's grid: two zones
    return [z for z, n in zip(zones, counts) for _ in range(n)]


class MockSim(Node):
    def __init__(self, sim_id: int):
        super().__init__("GUIDE", namespace=f"Sim_{sim_id}")
        self.sim_id, self.scenes, self.t0 = sim_id, 0, time.monotonic()
        self.finalized = self.create_publisher(
            String, "dataset_finalized",
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.clock = self.create_publisher(Clock, "clock", 10)
        self.create_timer(0.01, self.tick)
        self.create_service(RegisterScene, "Register", self.register)
        self.create_service(Trigger, "shutdown", self.shutdown)

    def tick(self):
        t = (time.monotonic() - self.t0) * 10.0
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = int(t), int(t % 1 * 1e9)
        self.clock.publish(msg)

    def register(self, request, response):
        i = self.scenes
        self.scenes += 1
        self.create_service(Demonstration, f"Scene_{i}/generate_demonstration",
                            lambda req, res: self.generate(i, req, res))
        response.id, response.offset, response.success = i, [0.0, 2.0 * i, 0.0], True
        response.package = Path(request.path).name.removesuffix(".tar.gz")
        return response

    def generate(self, scene: int, request, response):
        threading.Thread(target=self.record, args=(scene, request), daemon=True).start()
        response.success, response.message = True, "Started generating."
        return response

    def record(self, scene: int, request):
        time.sleep(1.0)
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        root = Path(request.path or "~/dataset").expanduser() / f"dataset_{self.sim_id}_{scene}_{stamp}"
        (root / "meta").mkdir(parents=True)
        eps = episodes(list(request.zones), list(request.counts))
        (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(eps)}))
        (root / "meta" / "guide_episodes.jsonl").write_text("".join(
            json.dumps({"episode_index": k, "zone": z}) + "\n" for k, z in enumerate(eps)))
        self.finalized.publish(String(data=json.dumps({"scene": scene, "path": str(root)})))

    def shutdown(self, request, response):
        threading.Timer(1.0, lambda: os._exit(0)).start()  # answer first
        response.success, response.message = True, "Shutting down."
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    args, _ = parser.parse_known_args()
    rclpy.init()
    executor = MultiThreadedExecutor()
    executor.add_node(MockSim(args.id))
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing tests** (append to `test_guide_container.py`; add `import os`, `import sys`, `import time` at the top)

```python
MOCK = Path(__file__).resolve().parents[1] / "mock" / "mock_sim.py"

FAKE_GUIDE = """
import json, pathlib, sys, time
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
d = pathlib.Path(sys.argv[1]); (d / "meta").mkdir(parents=True)
(d / "meta" / "info.json").write_text("{}")
rclpy.init()
node = rclpy.create_node("GUIDE", namespace="Sim_7")
pub = node.create_publisher(String, "dataset_finalized",
                            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
pub.publish(String(data=json.dumps({"scene": 0, "path": str(d)})))
time.sleep(5)  # long enough for the runner to discover us and read the latched message
"""


def env_for(tmp_path, monkeypatch, **extra):
    monkeypatch.setenv("CYCLONEDDS_URI", "")  # main() sets it; monkeypatch restores it
    return {**os.environ, "GUIDE_SCRATCH": str(tmp_path / "scratch"), **extra}


def test_slave_mode_delivers_what_guide_finalizes(tmp_path, monkeypatch):
    fake = tmp_path / "fake_guide.py"
    fake.write_text(FAKE_GUIDE)
    made = tmp_path / "scratch" / "Sim_7" / "dataset_7_0_x"
    out = tmp_path / "out"
    out.mkdir()

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_SIM_ID="7", GUIDE_OUTPUT=str(out),
                          GUIDE_CMD=f"{sys.executable} {fake} {made}"))

    assert code == 0
    assert (out / "Sim_7" / "dataset_7_0_x" / "meta" / "info.json").is_file()


def test_an_unreachable_bucket_keeps_the_dataset(tmp_path, monkeypatch):
    fake = tmp_path / "fake_guide.py"
    fake.write_text(FAKE_GUIDE)
    made = tmp_path / "scratch" / "Sim_7" / "dataset_7_0_x"
    # boto3 reads the process environment, not the runner's env argument.
    for key, value in {"AWS_ENDPOINT_URL": "http://127.0.0.1:9", "AWS_ACCESS_KEY_ID": "x",
                       "AWS_SECRET_ACCESS_KEY": "y", "AWS_MAX_ATTEMPTS": "1"}.items():
        monkeypatch.setenv(key, value)

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_SIM_ID="7", GUIDE_OUTPUT="s3://guide/out",
                          GUIDE_CMD=f"{sys.executable} {fake} {made}"))

    assert code == 0  # GUIDE was fine; the dataset just could not leave
    assert (made / "meta" / "info.json").is_file()


def test_a_plan_runs_to_the_end_on_the_mock(tmp_path, monkeypatch):
    plan = tmp_path / "plan.yaml"
    plan.write_text("jobs:\n  - {task: block_bin, zones: [1, 2], counts: [2, 1]}\n"
                    "  - {task: s3://t/cube_stack.tar.gz, counts: [2]}\n")
    out = tmp_path / "out"
    out.mkdir()

    started = time.monotonic()
    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_PLAN=str(plan), GUIDE_OUTPUT=str(out),
                          GUIDE_MAX_SCENES="3", GUIDE_CMD=f"{sys.executable} {MOCK}"))

    assert code == 0 and time.monotonic() - started < 120
    assert len(list((out / "Sim_0").iterdir())) == 3  # block_bin split in two, cube_stack whole


def test_a_bad_plan_never_starts_guide(tmp_path, monkeypatch):
    plan = tmp_path / "plan.yaml"
    plan.write_text("jobs:\n  - {task: block_bin, zones: [1, 1], counts: [2, 2]}\n")
    marker = tmp_path / "started"

    code = c.main(env_for(tmp_path, monkeypatch, GUIDE_PLAN=str(plan),
                          GUIDE_CMD=f"touch {marker}"))

    assert code == 1 and not marker.exists()
```

- [ ] **Step 3: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q docker/runner -k "slave or bucket or mock or never"`
Expected: FAIL, `AttributeError: module 'guide_container' has no attribute 'main'`.

- [ ] **Step 4: Implement** (append to `guide_container.py`; add `itertools, queue, shlex, signal, socket, sys, tempfile, threading, time` to its imports)

```python
def guide_cmd(sim_id: int, env) -> list:
    if env.get("GUIDE_CMD"):  # the mock image's stand-in simulator; tests
        return shlex.split(env["GUIDE_CMD"]) + ["--id", str(sim_id)]
    from ament_index_python.packages import get_package_prefix

    exe = Path(get_package_prefix("guide_core")) / "lib" / "guide_core" / "GUIDE"
    return [env.get("ISAACSIM_PYTHON", sys.executable), str(exe), "--id", str(sim_id)]


def dds_uri(net, master) -> str:
    if not net:  # sealed: the localhost config GUIDE ships
        from ament_index_python.packages import get_package_share_directory

        return f"file://{get_package_share_directory('guide_core')}/config/cyclonedds_localhost.xml"
    path = Path(tempfile.gettempdir()) / "cyclonedds.xml"
    path.write_text(dds_config(overlay_ip(net), [master] if master else []))
    return f"file://{path}"


def wait_for_name(host: str) -> None:
    """Cyclone resolves peer names once, at start: wait until the master's name resolves."""
    for attempt in itertools.count():
        try:
            socket.gethostbyname(host)
            return
        except OSError:
            if attempt % 30 == 0:
                print(f"[container] waiting for {host} to resolve...", flush=True)
            time.sleep(1.0)


def wait(client, guide, timeout: float = 1800.0) -> None:
    # A cold start compiles shaders (minutes); a solver waits for MoveIt and joint states.
    deadline = time.monotonic() + timeout
    while not client.wait_for_service(timeout_sec=1.0):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{client.srv_name} did not come up in {timeout:.0f} s")


def call(client, request, guide, timeout: float):
    done = threading.Event()
    future = client.call_async(request)
    future.add_done_callback(lambda _: done.set())
    deadline = time.monotonic() + timeout
    while not done.wait(1.0):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{client.srv_name} did not answer in {timeout:.0f} s")
    return future.result()


def zone_count(package: str) -> int:
    """The task's zone grid, read the way its solver reads it (block_bin solve_task.scene_num_zones)."""
    from ament_index_python.packages import get_package_share_directory

    from guide_core.ros.task_bringup import TaskBringup
    from guide_core.types.randomization.replicator_guide import zone_grid

    TaskBringup(0).activate()  # tasks GUIDE fetched live in its overlay
    grid = zone_grid(str(Path(get_package_share_directory(package)) / "config" / "randomize.yaml"))
    return grid.num_zones if grid is not None else 1


def run_plan(node, plan, cap, scratch, finalized, handle, guide) -> bool:
    from guide_msgs.srv import Demonstration, RegisterScene
    from std_srvs.srv import Trigger

    register = node.create_client(RegisterScene, "Register")
    wait(register, guide)

    def add(task):
        # Register fetches and builds an unknown task first: allow for a long build.
        reply = call(register, RegisterScene.Request(path=task, bringup=True), guide, 3600)
        if not reply.success:
            raise RuntimeError(f"Register {task!r}: {reply.message}")
        return reply.id, reply.package

    jobs = plan["jobs"]
    firsts = [add(job["task"]) for job in jobs]
    items = [work(job, zone_count(pkg) if job["zones"] == [-1] else 1)
             for job, (_, pkg) in zip(jobs, firsts)]
    shares = allocate([sum(n for _, n in it) for it in items], [capacity(it) for it in items], cap)
    scenes = {}
    for job, (first, _), it, k in zip(jobs, firsts, items, shares):
        parts = deal(it, k)
        scenes[first] = parts[0]
        for part in parts[1:]:
            scenes[add(job["task"])[0]] = part
    for sid, part in scenes.items():
        client = node.create_client(Demonstration, f"Scene_{sid}/generate_demonstration")
        wait(client, guide)
        request = Demonstration.Request(
            path=str(scratch / f"scene_{sid}"), zones=part["zones"], counts=part["counts"])
        reply = call(client, request, guide, 60)
        if not reply.success:
            raise RuntimeError(f"scene {sid}: {reply.message}")
    results = {}
    while len(results) < len(scenes):
        if guide.poll() is not None:
            raise RuntimeError("GUIDE exited before the plan finished")
        try:
            event = finalized.get(timeout=1.0)
        except queue.Empty:
            continue
        results[event["scene"]] = handle(event, scenes.get(event["scene"]))
    # The same exit a master uses.
    call(node.create_client(Trigger, "shutdown"), Trigger.Request(), guide, 60)
    return all(results.values())


def main(env=None) -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    env = os.environ if env is None else env
    sim_id = int(env.get("GUIDE_SIM_ID") or 0)
    ns = f"Sim_{sim_id}"
    try:
        plan = load_plan(read_text(env["GUIDE_PLAN"])) if env.get("GUIDE_PLAN") else None
        cap = int(env.get("GUIDE_MAX_SCENES") or (len(plan["jobs"]) if plan else 0))
        if plan and len(plan["jobs"]) > cap:
            raise ValueError(f"plan: {len(plan['jobs'])} jobs need as many scenes; GUIDE_MAX_SCENES is {cap}")
    except (OSError, ValueError) as e:
        print(f"[container] {e}", flush=True)
        return 1  # before Isaac starts
    output = env.get("GUIDE_OUTPUT") or (plan or {}).get("output")
    scratch = Path(env.get("GUIDE_SCRATCH") or "/scratch") / ns
    scratch.mkdir(parents=True, exist_ok=True)
    master = env.get("GUIDE_MASTER")
    if master:
        wait_for_name(master)
    os.environ["CYCLONEDDS_URI"] = dds_uri(env.get("GUIDE_DDS_NET"), master)  # GUIDE, its launches, us

    guide = subprocess.Popen(guide_cmd(sim_id, env))
    # docker stop / service rm: hand SIGTERM to GUIDE, which shuts down as /shutdown does; we
    # keep delivering what it finalizes until it is gone.
    previous = {s: signal.signal(s, lambda *_: guide.send_signal(signal.SIGTERM))
                for s in (signal.SIGTERM, signal.SIGINT)}
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("guide_container", namespace=ns)
    latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    finalized: queue.Queue = queue.Queue()
    node.create_subscription(String, "dataset_finalized",
                             lambda m: finalized.put(json.loads(m.data)), latched)
    announce = node.create_publisher(String, "dataset_delivered", latched)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    def handle(event: dict, scene: dict | None = None) -> bool:
        if not event["path"]:
            print(f"[container] scene {event['scene']} recorded nothing", flush=True)
            return False
        dataset = Path(event["path"])
        ok = complete(scene, zone_counts(dataset)) if scene else True
        try:
            target = deliver(dataset, output, ns) if output else str(dataset)
        except Exception as e:  # keep policy: the dataset stays in scratch
            print(f"[container] delivering {dataset} failed, kept in scratch: {e}", flush=True)
            target, ok = None, False
        announce.publish(String(data=json.dumps(
            {"dataset": dataset.name, "target": target, "complete": ok})))
        return ok

    try:
        ok = run_plan(node, plan, cap, scratch, finalized, handle, guide) if plan else True
    except Exception as e:
        print(f"[container] {e}", flush=True)
        ok = False
        if guide.poll() is None:
            guide.send_signal(signal.SIGTERM)
    # Slave mode serves until GUIDE exits (a master's /shutdown, or SIGTERM); a finished plan has
    # asked it to shut down already. Deliver whatever it finalizes on the way out.
    quiet_since = None
    while True:
        try:
            handle(finalized.get(timeout=1.0))
            quiet_since = None
        except queue.Empty:
            if guide.poll() is None:
                continue
            quiet_since = quiet_since or time.monotonic()
            if time.monotonic() - quiet_since > 3.0:  # the last announcements are in
                break
    for s, h in previous.items():
        signal.signal(s, h)
    rclpy.try_shutdown()
    return 0 if ok and guide.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `docker/entrypoint.sh`**

```bash
#!/usr/bin/env bash
# GUIDE container entry point: ROS 2 + the workspace, then the runner (docker/runner).
source /opt/ros/jazzy/setup.bash
source "${GUIDE_WS:-/root/ros2_ws/install}/setup.bash"
exec "${GUIDE_PYTHON:-/root/ros2_ws/.venv/bin/python}" /opt/guide/guide_container.py "$@"
```

- [ ] **Step 6: Run the tests**

Run: `chmod +x docker/entrypoint.sh docker/mock/mock_sim.py && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q docker/runner`
Expected: all pass (22). If `test_a_plan_runs_to_the_end_on_the_mock` times out, check that `$PRIV/install` (with Task 7's `guide_msgs`) is sourced. The mock needs the `bringup` and `package` fields.

- [ ] **Step 7: Commit**

```bash
git add docker/runner docker/mock/mock_sim.py docker/entrypoint.sh
git commit -m "docker: runner main (plan and slave modes, delivery announcements) and the mock simulator"
```

---

### Task 10: Mock image and the overlay communication test

**Files:**
- Modify: `docker/Dockerfile` (stages `mock-build`, `mock`)
- Create: `docker/test_comms.sh`

**Interfaces:**
- Consumes: runner (Task 9), mock (Task 9), `guide_msgs` (Task 7).
- Produces: image `guide:mock`; `docker/test_comms.sh` exits 0 when every check passes.

- [ ] **Step 1: Add the stages**

```dockerfile
FROM ros:jazzy-ros-base AS mock-build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY guide_msgs /ws/src/guide_msgs
COPY guide_core /ws/src/guide_core
RUN source /opt/ros/jazzy/setup.bash && cd /ws \
 && colcon build --merge-install --install-base /opt/guide --packages-select guide_msgs guide_core

FROM ros:jazzy-ros-core AS mock
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-std-srvs python3-boto3 python3-yaml iproute2 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=mock-build /opt/guide /opt/guide
COPY docker/runner/guide_container.py docker/mock/mock_sim.py /opt/guide/
COPY docker/entrypoint.sh /usr/local/bin/guide-entrypoint
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp GUIDE_WS=/opt/guide GUIDE_PYTHON=python3 \
    GUIDE_CMD="python3 /opt/guide/mock_sim.py"
ENTRYPOINT ["guide-entrypoint"]
```
Run: `docker build -f docker/Dockerfile --target mock -t guide:mock . && docker image ls guide:mock`
Expected: built, under 1 GB.

- [ ] **Step 2: Write `docker/test_comms.sh`**

```bash
#!/usr/bin/env bash
# Communication test on guide-net: two mock slaves, a stand-in master (fixed IP), MinIO.
# Needs Task 0 (swarm, guide-net 10.42.0.0/24) and guide:mock. Exit 0 = every check passed.
set -euo pipefail
NET=guide-net SUBNET=10.42.0.0/24 MASTER=10.42.0.2
S3=(-e AWS_ENDPOINT_URL=http://guide-minio:9000 -e AWS_ACCESS_KEY_ID=guide
    -e AWS_SECRET_ACCESS_KEY=guidesecret -e AWS_DEFAULT_REGION=us-east-1)
SLAVE=(-e GUIDE_MASTER=$MASTER -e GUIDE_DDS_NET=$SUBNET -e GUIDE_OUTPUT=s3://guide/out)
names=(guide-master guide-sim-1 guide-sim-2 guide-minio)
cleanup() { docker rm -f "${names[@]}" >/dev/null 2>&1 || true; }
trap cleanup EXIT; cleanup
check() { echo "CHECK: $1"; }
fail() { echo "FAILED: $1"; exit 1; }   # explicit: set -e ignores `! cmd` and && lists

docker run -d --name guide-minio --network $NET -e MINIO_ROOT_USER=guide \
  -e MINIO_ROOT_PASSWORD=guidesecret minio/minio:RELEASE.2025-04-22T22-12-26Z server /data >/dev/null
docker run -d --name guide-master --network $NET --ip $MASTER "${S3[@]}" --entrypoint bash guide:mock -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && cd /opt/guide && python3 -c '
from guide_container import dds_config, overlay_ip
open(\"/tmp/cdds.xml\", \"w\").write(dds_config(overlay_ip(\"$SUBNET\"), []))' && sleep infinity" >/dev/null
m() { docker exec -e CYCLONEDDS_URI=file:///tmp/cdds.xml guide-master bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && $1"; }
sleep 3
m "python3 -c 'import boto3; boto3.client(\"s3\").create_bucket(Bucket=\"guide\")'"
for id in 1 2; do
  docker run -d --name guide-sim-$id --network $NET "${S3[@]}" "${SLAVE[@]}" -e GUIDE_SIM_ID=$id guide:mock >/dev/null
done

check "the master discovers both slaves"
for _ in $(seq 60); do
  services=$(m "ros2 service list")
  grep -q /Sim_1/Register <<<"$services" && grep -q /Sim_2/Register <<<"$services" && break
  sleep 2
done
grep -q /Sim_1/Register <<<"$services" && grep -q /Sim_2/Register <<<"$services" \
  || fail "the master does not see both slaves"

check "a slave registers, generates, delivers to S3, announces it"
m "ros2 service call /Sim_1/Register guide_msgs/srv/RegisterScene \"{path: block_bin, bringup: true}\"" \
  | grep -q "success=True" || fail "Register"
m "ros2 service call /Sim_1/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \"{path: /scratch/Sim_1/s0, zones: [2], counts: [3]}\"" \
  | grep -q "success=True" || fail "generate_demonstration"
m "timeout 60 ros2 topic echo --once --qos-durability transient_local /Sim_1/dataset_delivered" \
  | grep -q '"complete": true' || fail "no complete delivery announced"
m "python3 -c '
import boto3
keys = [o[\"Key\"] for o in boto3.client(\"s3\").list_objects_v2(Bucket=\"guide\", Prefix=\"out/Sim_1/\")[\"Contents\"]]
assert any(k.endswith(\"meta/guide_episodes.jsonl\") for k in keys), keys'" || fail "dataset not in S3"

check "each simulator has its own clock; there is no global /clock"
topics=$(m "ros2 topic list")
grep -qx /Sim_1/clock <<<"$topics" && grep -qx /Sim_2/clock <<<"$topics" || fail "a simulator clock is missing"
grep -qx /clock <<<"$topics" && fail "a global /clock exists"
m "ros2 topic info /Sim_1/clock" | grep -q "Publisher count: 1" || fail "/Sim_1/clock has other publishers"

check "slaves do not see each other"
docker exec -e CYCLONEDDS_URI=file:///tmp/cyclonedds.xml guide-sim-1 bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && ros2 node list" \
  | grep -q /Sim_2/ && fail "Sim_1 sees Sim_2"

check "the host sees no simulator"
(source /opt/ros/jazzy/setup.bash && timeout 10 ros2 node list 2>/dev/null) | grep -q /Sim_ \
  && fail "the host sees a simulator"

check "/shutdown ends a slave cleanly"
m "ros2 service call /Sim_2/shutdown std_srvs/srv/Trigger" | grep -q "success=True" || fail "shutdown call"
[ "$(timeout 30 docker wait guide-sim-2)" = 0 ] || fail "Sim_2 did not exit 0"

echo "ALL CHECKS PASSED"
```

- [ ] **Step 3: Run it**

Run: `chmod +x docker/test_comms.sh && docker/test_comms.sh`
Expected: `ALL CHECKS PASSED`.
- If "the master discovers both slaves" fails, Cyclone isn't answering unicast discovery from unlisted peers. Fall back to the master listing the slaves (`dds_config(ip, [slave IPs])` on the master side) and note it in `docker/CONVENTIONS.md`.
- Prove a check can fail: drop `GUIDE_DDS_NET` for `guide-sim-2`, expect `FAILED: the master does not see both slaves`, then restore it.

- [ ] **Step 4: Plan mode, sealed, and a bad plan**

```bash
tmp=$(mktemp -d)
printf 'jobs:\n  - {task: block_bin, zones: [1, 2], counts: [2, 1]}\n  - {task: cube_stack, counts: [2]}\n' > $tmp/plan.yaml
docker run --rm --network none -v $tmp:/io -e GUIDE_PLAN=/io/plan.yaml -e GUIDE_OUTPUT=/io/out \
  -e GUIDE_MAX_SCENES=3 guide:mock; echo "exit $?"
ls $tmp/out/Sim_0
printf 'jobs:\n  - {task: block_bin, zones: [1, 1], counts: [2, 2]}\n' > $tmp/bad.yaml
docker run --rm --network none -v $tmp:/io -e GUIDE_PLAN=/io/bad.yaml guide:mock; echo "exit $?"
```
Expected:
- `exit 0`, three `dataset_0_*` folders, owned by you, not root;
- then `plan: job 0: one count per distinct zone >= 0` and `exit 1` within seconds.

- [ ] **Step 5: Commit**

```bash
git add docker/Dockerfile docker/test_comms.sh
git commit -m "docker: mock image + guide-net communication test (discovery, S3, clocks, isolation, shutdown)"
```

---

### Task 11: Deploy image and the GPU end-to-end runs

**Files:**
- Modify: `docker/Dockerfile` (stages `trim`, `deploy`)
- Create: `docker/trim.sh` (version 1), `docker/e2e_plan.yaml`

**Interfaces:**
- Consumes: `guide:build` (Task 1), the runner (Task 9), GUIDE features (Tasks 2–7).
- Produces: image `guide:deploy` (`ENTRYPOINT ["guide-entrypoint"]`). Its `init.yaml` defaults are `render_device: cuda:0`, `headless: true`, `enable_crashreporter: false`, `extra_args: ["--/log/fileLogLevel=warning"]`.

- [ ] **Step 1: Write `docker/trim.sh` (version 1)**

```bash
#!/usr/bin/env bash
# Delete what headless GUIDE generation never loads from the venv the deploy image copies.
set -euo pipefail
ISAAC=/root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim
rm -rf "$ISAAC"/kit/cache "$ISAAC"/kit/logs "$ISAAC"/kit/data   # build-time Kit state
```

- [ ] **Step 2: Add the stages**

```dockerfile
FROM build AS trim
RUN /root/ros2_ws/src/guide/docker/trim.sh

FROM base AS deploy
COPY --from=trim /root/ros2_ws/.venv /root/ros2_ws/.venv
COPY --from=trim /root/ros2_ws/install /root/ros2_ws/install
COPY --from=build /root/ros2_ws/src/guide/modules/isaac6-safe-pins.txt /root/ros2_ws/pins.txt
COPY docker/runner/guide_container.py /opt/guide/guide_container.py
COPY docker/entrypoint.sh /usr/local/bin/guide-entrypoint
# Deployment defaults for the one-A4000 nodes; a swarm config (guide_init) replaces the file
# per deployment (the dev host's A4000 is cuda:1).
RUN python3 - <<'EOF'
import yaml
path = "/root/ros2_ws/install/guide_core/share/guide_core/config/init.yaml"
config = yaml.safe_load(open(path))
config["startup"].update(render_device="cuda:0", headless=True, enable_crashreporter=False,
                         extra_args=["--/log/fileLogLevel=warning"])
yaml.safe_dump(config, open(path, "w"), sort_keys=False)
EOF
ENV ISAACSIM_PYTHON=/root/ros2_ws/.venv/bin/python GUIDE_PINS=/root/ros2_ws/pins.txt
RUN mkdir -p /scratch
ENTRYPOINT ["guide-entrypoint"]
```
Run: `docker build -f docker/Dockerfile --target deploy -t guide:deploy . && docker image ls guide`
Expected: built; record the size as Task 12's baseline.

- [ ] **Step 3: Write `docker/e2e_plan.yaml`**

```yaml
# Two tasks in one simulator (one scene each), small counts.
jobs:
  - {task: block_bin, zones: [2], counts: [1]}
  - {task: cube_stack, zones: [], counts: [1]}
```

- [ ] **Step 4: Plan mode on the A4000** (`device=1` on this host shows up inside as `cuda:0`)

```bash
mkdir -p ~/dataset/docker_e2e
docker run --rm --gpus device=1 --stop-timeout 180 \
  -v $PWD/docker/e2e_plan.yaml:/plan.yaml:ro -v ~/dataset/docker_e2e:/output \
  -e GUIDE_PLAN=/plan.yaml -e GUIDE_OUTPUT=/output guide:deploy; echo "exit $?"
```
Expected:
- `Renderer on cuda:0.`, `Shutting down: finalizing every scene...`, and `exit 0`;
- two folders in `~/dataset/docker_e2e/Sim_0/`, owned by you, each with one `guide_episodes.jsonl` line;
- loading one with `LeRobotDataset(..., root=...)` in `~/ros2_ws/.venv` gives frames equal to video frames;
- no `carb.crashreporter` in the Kit log.

- [ ] **Step 5: `docker stop` in the middle** (Review Focus 1)

Copy the plan with `counts: [20]` for block_bin, run it detached (`-d --name guide-stop`), and after the third saved episode in `docker logs`, run `docker stop -t 180 guide-stop`.
Expected:
- the container exits within the timeout;
- `~/dataset/docker_e2e/Sim_0/` holds the finalized datasets, which load with `LeRobotDataset`;
- the logs show `Shutting down` and no "delivering … failed".

- [ ] **Step 6: Slave mode with an S3 task that is not installed**

```bash
S3=(-e AWS_ENDPOINT_URL=http://guide-minio:9000 -e AWS_ACCESS_KEY_ID=guide
    -e AWS_SECRET_ACCESS_KEY=guidesecret -e AWS_DEFAULT_REGION=us-east-1)
docker run -d --name guide-minio --network guide-net -e MINIO_ROOT_USER=guide \
  -e MINIO_ROOT_PASSWORD=guidesecret minio/minio:RELEASE.2025-04-22T22-12-26Z server /data
docker run -d --name guide-master --network guide-net --ip 10.42.0.2 "${S3[@]}" --entrypoint bash guide:mock -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && cd /opt/guide && python3 -c '
from guide_container import dds_config, overlay_ip
open(\"/tmp/cdds.xml\", \"w\").write(dds_config(overlay_ip(\"10.42.0.0/24\"), []))' && sleep infinity"
m() { docker exec -e CYCLONEDDS_URI=file:///tmp/cdds.xml guide-master bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && $1"; }
tar -czf $PRIV/cube_stack.tar.gz -C guide_tasks cube_stack
docker cp $PRIV/cube_stack.tar.gz guide-master:/tmp/
m "python3 -c 'import boto3; s=boto3.client(\"s3\"); [s.create_bucket(Bucket=b) for b in (\"guide\",\"tasks\")]; s.upload_file(\"/tmp/cube_stack.tar.gz\",\"tasks\",\"cube_stack.tar.gz\")'"
# guide-net is internal: guide-egress gives Isaac's asset download and rosdep/pip their internet.
docker run -d --name guide-e2e --gpus device=1 --stop-timeout 180 --network guide-net --network guide-egress "${S3[@]}" \
  -e GUIDE_SIM_ID=3 -e GUIDE_MASTER=10.42.0.2 -e GUIDE_DDS_NET=10.42.0.0/24 -e GUIDE_OUTPUT=s3://guide/out \
  --entrypoint bash guide:deploy -c 'rm -rf /root/ros2_ws/install/cube_stack && exec guide-entrypoint'
m "until ros2 service list | grep -q /Sim_3/Register; do sleep 5; done"
m "ros2 service call /Sim_3/Register guide_msgs/srv/RegisterScene \"{path: 's3://tasks/cube_stack.tar.gz', bringup: true}\""
m "until ros2 service list | grep -q /Sim_3/Scene_0/generate_demonstration; do sleep 5; done"
m "ros2 service call /Sim_3/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \"{path: /scratch/Sim_3/s0, zones: [], counts: [1]}\""
m "timeout 1200 ros2 topic echo --once --qos-durability transient_local /Sim_3/dataset_delivered"
m "ros2 service call /Sim_3/shutdown std_srvs/srv/Trigger"
docker wait guide-e2e
docker rm -f guide-e2e guide-master guide-minio
```
Expected:
- Register answers `package='cube_stack'`, and `docker logs guide-e2e` shows rosdep, colcon `Finished <<< cube_stack` and the bringup launch;
- the delivery JSON says `"complete": true`;
- `out/Sim_3/dataset_3_0_*` is in MinIO;
- `docker wait` prints `0`.

- [ ] **Step 7: Commit**

```bash
git add docker/Dockerfile docker/trim.sh docker/e2e_plan.yaml
git commit -m "docker: deploy image (init.yaml defaults); GPU end-to-end plan, stop and slave runs"
```

---

### Task 12: Trim the deploy image, measuring each cut

**Files:**
- Modify: `docker/trim.sh`, `README.md`/`INSTALLATION.md` (the isaacsim subset is a docs change, through the Task 1 loop)
- Create: `docker/kit_exts_keep.txt`

**Interfaces:**
- Consumes: Task 11 Step 4 is the gate after every cut.

- [ ] **Step 1: Generate the extension keep list** from an untrimmed run's Kit log

```bash
docker run --name guide-keep --gpus device=1 -v $PWD/docker/e2e_plan.yaml:/plan.yaml:ro \
  -e GUIDE_PLAN=/plan.yaml guide:deploy
docker cp guide-keep:/root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim/kit/logs $PRIV/kitlogs
docker rm guide-keep
grep -rhoP '\[ext: \K[^\] ]+(?=\] startup)' $PRIV/kitlogs | sed 's/-[0-9].*//' | sort -u > docker/kit_exts_keep.txt
wc -l docker/kit_exts_keep.txt
```
Expected: about 308 names, matching the host's measurement.

- [ ] **Step 2: Cut never-loaded extensions and their test/doc folders.** Append to `docker/trim.sh`:

```bash
KEEP=/root/ros2_ws/src/guide/docker/kit_exts_keep.txt   # regenerate (Task 12 Step 1) when a task needs more
for dir in "$ISAAC"/extscache/* "$ISAAC"/exts/* "$ISAAC"/extsDeprecated/*; do
  name=$(basename "$dir"); name=${name%%-[0-9]*}       # extscache dirs are <name>-<version>
  grep -qxF "$name" "$KEEP" || rm -rf "$dir"
done
find "$ISAAC" -depth -type d \( -name tests -o -name docs \) -exec rm -rf {} +
```
Rebuild `deploy`, then run Task 11 Step 4. Expected: `exit 0`, about 10 GB smaller. If Kit reports a missing extension, add it to the keep list.

- [ ] **Step 3: isaacsim subset instead of `[all]`, through the Task 1 loop.** README step 3's Isaac line becomes the subset from `modules/isaac6-install.md`:

```bash
uv pip install --python .venv/bin/python \
  "isaacsim[extscache]==6.0.1.0" isaacsim-core==6.0.1.0 isaacsim-ros2==6.0.1.0 \
  isaacsim-sensor==6.0.1.0 isaacsim-robot==6.0.1.0 isaacsim-storage==6.0.1.0 \
  isaacsim-asset==6.0.1.0 \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
```
Rebuild everything, then run Task 11 Step 4. Expected: `exit 0`; record the size. If Kit is missing an extension that the keep list names (e.g. `isaacsim.util.clash_detection`, the Replicator YAML extension), add the `isaacsim-*` package that ships it.

- [ ] **Step 4: Strip shared libraries.** Append `find /root/ros2_ws/.venv -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true` to `trim.sh`. Rebuild, then run Task 11 Step 4. Expected: `exit 0`. If anything fails to load, revert this step.

- [ ] **Step 5: Commit** (sizes go into `INSTALLATION.md` in Task 14)

```bash
git add docker/trim.sh docker/kit_exts_keep.txt README.md INSTALLATION.md
git commit -m "docker: trim the deploy image (unused Kit extensions, isaacsim subset, stripped libraries)"
```

---

### Task 13: Warm the shader and asset caches, commit `-warm`

**Files:**
- Create: `docker/warm.sh`, `docker/warm_plan.yaml`

- [ ] **Step 1: Write `docker/warm_plan.yaml`**

```yaml
# One episode of each shipped task: compiles every shader and fetches every asset they use.
jobs:
  - {task: block_bin, zones: [], counts: [1]}
  - {task: cube_stack, zones: [], counts: [1]}
```

- [ ] **Step 2: Write `docker/warm.sh`**

```bash
#!/usr/bin/env bash
# Bake Isaac's shader and asset caches into an image: run one episode of each task, commit.
# usage: docker/warm.sh [image=guide:deploy] [gpu=1]   -> <image>-warm
set -euo pipefail
IMAGE=${1:-guide:deploy} GPU=${2:-1}
HERE=$(dirname "$(realpath "$0")")
docker rm -f guide-warm >/dev/null 2>&1 || true
docker run --name guide-warm --gpus "device=$GPU" -v "$HERE/warm_plan.yaml:/warm_plan.yaml:ro" \
  -e GUIDE_PLAN=/warm_plan.yaml --entrypoint bash "$IMAGE" -c 'guide-entrypoint \
  && rm -rf /scratch/* /root/.ros/log /root/.nvidia-omniverse/logs \
            /root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim/kit/logs'
# commit records the container's -e too: clear GUIDE_PLAN, or every warm container reruns the warm-up
docker commit --change 'ENTRYPOINT ["guide-entrypoint"]' --change 'CMD []' --change 'ENV GUIDE_PLAN=' \
  guide-warm "${IMAGE}-warm"
docker rm guide-warm
```

- [ ] **Step 3: Run it, then measure cold vs warm**

```bash
chmod +x docker/warm.sh && docker/warm.sh guide:deploy 1
for img in guide:deploy guide:deploy-warm; do
  /usr/bin/time -f "$img %e s" docker run --rm --gpus device=1 \
    -v $PWD/docker/warm_plan.yaml:/p.yaml:ro -e GUIDE_PLAN=/p.yaml $img >/dev/null
done
docker history guide:deploy-warm | head -2
```
Expected: the warm image is clearly faster. Record both times and the warm layer's size for `INSTALLATION.md` (Task 14).

- [ ] **Step 4: Commit**

```bash
git add docker/warm.sh docker/warm_plan.yaml
git commit -m "docker: warm.sh bakes shader/asset caches into <image>-warm"
```

---

### Task 14: Swarm launch, notes for the team, docs

**Files:**
- Create: `docker/launch_sim.sh`, `docker/CONVENTIONS.md`, `docker/AUTHORITIES.md`
- Modify: `.gitignore`, `README.md` (a "Docker" section after "Usage"), `INSTALLATION.md` (Docker notes)

- [ ] **Step 1: `.gitignore`.** Append:

```
# Docker secrets (Ceph/MinIO keys) and env files never enter git
docker/secrets/
*.env
```

- [ ] **Step 2: Write `docker/launch_sim.sh`**, the reference spec of one slave simulator. The master's launcher creates exactly this.

```bash
#!/usr/bin/env bash
# One GUIDE simulator as a swarm service: docker/launch_sim.sh <id> [image]
# The image must be on the node (no registry yet: docker save | ssh <node> docker load).
set -euo pipefail
ID=$1 IMAGE=${2:-guide:deploy-warm}
docker service create --name "guide-sim-$ID" --no-resolve-image \
  --constraint node.labels.guide.gpu==a4000 --replicas-max-per-node 1 \
  --network guide-net --network guide-egress \
  --env GUIDE_SIM_ID="$ID" --env GUIDE_MASTER=guide-master --env GUIDE_DDS_NET=10.42.0.0/24 \
  --env GUIDE_OUTPUT="${GUIDE_OUTPUT:?s3://bucket/prefix}" \
  --env AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:?Ceph endpoint}" \
  --env AWS_SHARED_CREDENTIALS_FILE=/run/secrets/guide_s3 --secret guide_s3 \
  --mount type=volume,source="guide-scratch-$ID",target=/scratch \
  --stop-grace-period 180s --restart-condition on-failure \
  "$IMAGE"
```

- [ ] **Step 3: Swarm end-to-end on this node.** The master is a **service** named `guide-master`, so slaves reach it through its stable virtual IP. It is the stand-in from the mock image, holding MinIO's test key (Task 0's `guide_s3`).

```bash
docker run -d --name guide-minio --network guide-egress -e MINIO_ROOT_USER=guide \
  -e MINIO_ROOT_PASSWORD=guidesecret minio/minio:RELEASE.2025-04-22T22-12-26Z server /data
docker service create --name guide-master --no-resolve-image --network guide-net --network guide-egress \
  -e AWS_ENDPOINT_URL=http://guide-minio:9000 -e AWS_ACCESS_KEY_ID=guide \
  -e AWS_SECRET_ACCESS_KEY=guidesecret -e AWS_DEFAULT_REGION=us-east-1 \
  --entrypoint bash guide:mock -c "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash \
  && cd /opt/guide && python3 -c '
from guide_container import dds_config, overlay_ip
open(\"/tmp/cdds.xml\", \"w\").write(dds_config(overlay_ip(\"10.42.0.0/24\"), []))' && sleep infinity"
m() { docker exec -e CYCLONEDDS_URI=file:///tmp/cdds.xml "$(docker ps -q -f name=guide-master)" bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && $1"; }
m "python3 -c 'import boto3; boto3.client(\"s3\").create_bucket(Bucket=\"guide\")'"
GUIDE_OUTPUT=s3://guide/out AWS_ENDPOINT_URL=http://guide-minio:9000 docker/launch_sim.sh 5
m "until ros2 service list | grep -q /Sim_5/Register; do sleep 5; done"
m "ros2 service call /Sim_5/Register guide_msgs/srv/RegisterScene \"{path: block_bin, bringup: true}\""
m "until ros2 service list | grep -q /Sim_5/Scene_0/generate_demonstration; do sleep 5; done"
m "ros2 service call /Sim_5/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \"{path: /scratch/Sim_5/s0, zones: [], counts: [1]}\""
m "timeout 1200 ros2 topic echo --once --qos-durability transient_local /Sim_5/dataset_delivered"
```
Expected: the delivery says `"complete": true`, and `out/Sim_5/` is in MinIO.

- [ ] **Step 4: Restart keeps the id.** Run `docker kill $(docker ps -q -f name=guide-sim-5)`, then `m "until ros2 service list | grep -q /Sim_5/Register; do sleep 5; done"`. Expected: swarm restarts the task and `/Sim_5/Register` returns under the same id. Register `block_bin` again with `bringup: true`.

- [ ] **Step 5: Removing the service mid-run** (Review Focus 1). Generate `counts: [20]` on `/Sim_5/Scene_0/generate_demonstration`. After a few `Finished episode` lines in `docker service logs guide-sim-5`, run `docker service rm guide-sim-5`.
Expected:
- the service logs show `Shutting down: finalizing every scene...`;
- every finalized dataset is in MinIO under `out/Sim_5/`;
- nothing is left in the `guide-scratch-5` volume that wasn't delivered: `docker run --rm -v guide-scratch-5:/s ubuntu:24.04 ls /s/Sim_5` lists no dataset folders.

Clean up with `docker service rm guide-master && docker rm -f guide-minio`.

- [ ] **Step 6: Write `docker/CONVENTIONS.md`.** Copy the table from spec §11, then apply these edits:
  - "GUIDE interface" row: `Register` takes `{path, bringup}` and answers `{id, offset, package, message, success}`;
  - "Environment (runner)" row: add `GUIDE_DDS_NET` and `GUIDE_SCRATCH`;
  - "Services" row: add `--no-resolve-image`, and "the image must be on every node";
  - a new row "Readiness": a simulator is up when `/Sim_<id>/Register` appears in the ROS graph;
  - a new row "Scratch layout": `/scratch/Sim_<id>/scene_<i>/<dataset>`;
  - a closing paragraph: the master starts simulators with `docker/launch_sim.sh`'s spec, discovers them through `/Sim_*/Register`, stops them with `/Sim_<id>/shutdown` or by removing the service, and learns about datasets from `/Sim_<id>/dataset_finalized` (GUIDE) and `/Sim_<id>/dataset_delivered` (runner).

- [ ] **Step 7: Write `docker/AUTHORITIES.md`.** Copy spec §12 (the table and the deferred non-root section), then make these edits:
  - the "Anyone who can reach `Register`" row: say that it executes task code whether or not `bringup` is set, because a bundle's packages are built;
  - a new row: `Holder: the stand-in or real master service`; `Authority: calls /Sim_<id>/shutdown`; `Why: stop a simulator without Docker access`; `Risk: any guide-net peer can stop any simulator`; `Limited by: guide-net membership`.

- [ ] **Step 8: README "Docker" section and INSTALLATION notes.**
  - The README section holds only commands that were run in Tasks 0, 10, 11, 13 and 14: host setup in short, the three builds, plan mode `docker run`, `docker/launch_sim.sh`, `docker/test_comms.sh`, `docker/warm.sh`, and links to `docker/CONVENTIONS.md` and `docker/AUTHORITIES.md`.
  - INSTALLATION.md gets:
    - the image sizes (Task 12) and warm timings (Task 13);
    - why DDS is unicast on the overlay;
    - why the image builds from the checkout (`origin/dev` lags local `dev`);
    - the `block_bin_eval` `/clock` remap (Task 2 note).

- [ ] **Step 9: Full check**

Run: `docker build -f docker/Dockerfile --target test -t guide:test . && docker run --rm guide:test && docker/test_comms.sh`
Expected: all tests pass, then `ALL CHECKS PASSED`.

- [ ] **Step 10: Commit and push the branch** (no PR)

```bash
git add .gitignore docker/launch_sim.sh docker/CONVENTIONS.md docker/AUTHORITIES.md README.md INSTALLATION.md
git commit -m "docker: swarm launch spec, conventions and authorities notes, README Docker section"
git push
```
