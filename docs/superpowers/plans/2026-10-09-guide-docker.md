# GUIDE Docker Images Implementation Plan

> **SUPERSEDED (2026-10-09)** by design v2 (`docs/superpowers/specs/2026-10-09-guide-docker-design.md`).
> It is kept for its code: the plan will be regenerated from v2 once v2 is approved. Known v1 errors:
> - The `response.id = -1` "fix" is unnecessary: the uint8 range check only runs with `ROS_PYTHON_CHECK_FIELDS=1`.
> - `deps.repos` is dropped.
> - The plan file's "scenes" are now "jobs", split across scenes by `GUIDE_MAX_SCENES`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerised GUIDE demonstration generation. This covers:
- `deploy`, `test`, `mock` and `deploy-warm` images, built by running the README's own install steps;
- one simulator per container (one container per GPU), run by a plan or as a slave to a master over ROS 2 on the `guide-net` overlay;
- datasets recorded locally, then delivered to a folder or S3.

**Architecture:**
- **Runner:** a small runner (`guide_core/ros/container.py`) is the image's entry point. It writes the DDS config, starts GUIDE as a child process, reads its stdout for dataset markers, drives a plan if one is given, and delivers finished datasets.
- **GUIDE flags:** `--set` overrides `init.yaml`, `--bringup` makes Register fetch, build and launch the task, and the clock is per simulator.
- **Docs loop:** the Dockerfile executes the README's install blocks verbatim, so every build failure becomes a docs fix.

**Tech Stack:** Docker (BuildKit, swarm overlay), NVIDIA Container Toolkit, ROS 2 Jazzy, Cyclone DDS, Isaac Sim 6.0.1 (pip, Python 3.12), uv, colcon, vcstool, rosdep, boto3, MinIO (S3 tests), pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-guide-docker-design.md`

## Global Constraints

- Python 3.12; ROS 2 Jazzy; `isaacsim==6.0.1.0`; torch 2.11.0 (cu130), torchvision 0.26.0, numpy 2.3.1, `lerobot==0.6.0`.
- Every torch-dependent install passes `-c <pins>`: `numpy==2.3.1`, `torch==2.11.0`, `torchvision==0.26.0`.
- Image paths stay `/root/ros2_ws/.venv` and `/root/ros2_ws/install`; the venv and colcon setup files hard-code their prefixes.
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. One `ROS_DOMAIN_ID` for all containers; `Sim_<id>` namespaces separate them.
- **Never set `CUDA_DEVICE_ORDER`** in the simulator environment: every annotator returns empty.
- `task_bringup.py` and `container.py` import nothing from Isaac (`pxr`, `omni`, `isaacsim`); they are robot- and sim-agnostic.
- Commit messages carry **no** `Co-Authored-By`, Claude or "Generated with" lines. The user is the sole author. Never open a PR; push the branch only.
- Never build or run from `~/ros2_ws/install`. Local test runs use a private install:
  - `PRIV=${CLAUDE_JOB_DIR:-/tmp/guide-docker}/tmp`
  - build: `colcon build --base-paths . --build-base $PRIV/build --install-base $PRIV/install --packages-select <pkgs>`, run from the worktree with `/opt/ros/jazzy/setup.zsh` and `~/ros2_ws/install/setup.zsh` sourced;
  - then `source $PRIV/install/setup.zsh`.
- Local test command (from the worktree): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q <files>`.
- Datasets a person will keep go to `~/dataset/...`, never a job tmp dir.
- The shell is zsh locally (source `.zsh` setup files); the images use bash.

## Review Focus

1. **`docker stop` in the middle of a plan.** Every finalized dataset should still be delivered, and nothing deleted locally that wasn't delivered. The stop grace period must cover finalize (15 s/scene) plus upload, so use `--stop-timeout 180`. Pinned by Task 10, Step 6.
2. **S3 unreachable or bad credentials.** The dataset must stay in scratch, the failure must be announced (`complete: false`), and the runner must keep serving. Pinned by Task 7, test `test_a_failed_upload_keeps_the_dataset`.
3. **A task bundle that fails to build** (no `scene.py`, missing rosdep key, a colcon error). Register should answer `success: false` with the reason, and the simulator should keep stepping its other scenes. Pinned by Task 6, test `test_a_failed_build_fails_register_and_keeps_the_simulator_running`.
4. **A malformed plan** (zones/counts mismatch, duplicate or negative zones, no scenes). The container should exit 1 before starting Isaac, naming the bad scene. Pinned by Task 7 (`test_bad_plans_are_rejected`) and Task 9, Step 6.
5. **Two simulators on `guide-net`.** No cross-talk: each has its own `/Sim_N/clock`, there is no global `/clock`, and the slaves don't see each other. Pinned by Task 9, Step 5.

## File Structure

| File | Responsibility |
|---|---|
| `.dockerignore` | Keep `.git`, `.claude`, `modules/`, `docs/`, build output out of the build context |
| `docker/Dockerfile` | Targets `base`, `build`, `test`, `trim`, `deploy`, `mock-build`, `mock` |
| `docker/readme_steps.sh` | Run one README section's bash blocks verbatim (the docs loop) |
| `docker/entrypoint.sh` | Source ROS + workspace, exec the runner |
| `docker/trim.sh` | Delete what generation never loads from the venv |
| `docker/kit_exts_keep.txt` | Kit extensions a generation run loads (generated in Task 11) |
| `docker/warm.sh`, `docker/warm_plan.yaml` | Warm the shader/asset caches and commit `-warm` |
| `docker/e2e_plan.yaml` | Two-scene, two-task plan for the GPU end-to-end run |
| `docker/mock/mock_sim.py` | Stand-in GUIDE (services, clock, markers, fake datasets) |
| `docker/test_comms.sh` | Overlay communication test: mock slaves, stand-in master, MinIO |
| `guide_core/guide_core/ros/task_bringup.py` | Fetch a task bundle (dir or S3), build it with its dependencies, activate it, launch its bringup |
| `guide_core/guide_core/ros/container.py` | Container runner: flags, plan, DDS config, GUIDE child, markers, delivery |
| `guide_core/guide_core/ros/guide_ros.py` | `--set`/`--bringup`/`--tasks-dir`, `parse_known_args`, Register through TaskBringup, namespaced clock |
| `guide_core/guide_core/scene/scene_recorder.py` | `GUIDE_DATASET_READY`/`GUIDE_DATASET_EMPTY` markers |
| `guide_tasks/{block_bin,cube_stack}/launch/bringup.launch.py` | `sim_id`, `first_scene`, `/clock` remap |
| `guide_core/test/conftest.py` | Shared `isaac_import` fixture (moved from `test_command_fixes.py`) |
| `guide_core/test/test_sim_flags.py` | Flags, Register, clock, markers, task launch files |
| `guide_core/test/test_task_bringup.py` | TaskBringup |
| `guide_core/test/test_container.py` | Runner |
| `README.md`, `INSTALLATION.md` | Docs-loop fixes, Prerequisites block, Docker section |

---

### Task 0: Host prerequisites (user, needs sudo)

**Files:** none.

- [ ] **Step 1: Install Docker Engine** (Docker's apt repository, Ubuntu 24.04)

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

- [ ] **Step 2: Install the NVIDIA Container Toolkit**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

- [ ] **Step 3: Single-node swarm and the overlay**

```bash
docker swarm init
docker network create -d overlay --attachable --subnet 10.42.0.0/24 guide-net
```

- [ ] **Step 4: Verify**

Run: `docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L && docker network inspect guide-net -f '{{.Driver}} {{(index .IPAM.Config 0).Subnet}}'`
Expected: both GPUs listed (RTX A2000, RTX A4000) and `overlay 10.42.0.0/24`.

---

### Task 1: Docs-driven build (the iterative install loop)

**Files:**
- Create: `.dockerignore`, `docker/readme_steps.sh`, `docker/Dockerfile` (stages `base`, `build`, `test`)
- Modify: `README.md` (Prerequisites gets a runnable block; each fix the loop finds), `INSTALLATION.md` (the same fix, with its "why")

**Interfaces:**
- Produces: image `guide:build` (README steps done, sources at `/root/ros2_ws/src/guide`) and `guide:test`. `readme_steps.sh README.md <Section>` runs the bash blocks under `## <Section>`.

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
Expected: `mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src` first, then the `irob_franka_ros2` clone. There is no `guide.git` line.

- [ ] **Step 2: Give README "Prerequisites" a runnable block** (keep the existing bullets above it)

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
  ros-jazzy-moveit-ros-planning-interface ros-jazzy-pick-ik \
  libglu1-mesa libvulkan1 libegl1 libxt6 libxrandr2 libxi6 libsm6 libice6
curl -LsSf https://astral.sh/uv/0.11.26/install.sh | sh
```
This list is the starting point. The loop (Step 6) adds what is missing and removes what nothing needs. Also add to README "Usage", next to `CYCLONEDDS_URI`: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. Today it is only in the author's `~/.zshrc`.

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
```

- [ ] **Step 4: Write `docker/Dockerfile` (first three stages)**

```dockerfile
# syntax=docker/dockerfile:1.7
# GUIDE images, built from the guide repo root (the build context IS the GUIDE source):
#   docker build -f docker/Dockerfile --target test   -t guide:test   .
#   docker build -f docker/Dockerfile --target deploy -t guide:deploy .
#   docker build -f docker/Dockerfile --target mock   -t guide:mock   .
# base and build run the README's install blocks verbatim (docker/readme_steps.sh), so a
# failing build is a docs bug: fix README.md (+ INSTALLATION.md), not this file.

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
# INSTALLATION.md §7, verbatim. No GPU needed: the tests stub Isaac.
CMD ["bash", "-c", "source /root/ros2_ws/install/setup.bash && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/ros2_ws/.venv/bin/python -m pytest -q guide_core/test guide_ex/test guide_tasks/cube_stack/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'"]
```

- [ ] **Step 5: First build**

Run: `docker build -f docker/Dockerfile --target build -t guide:build . 2>&1 | tee $PRIV/build.log`
Expected: it FAILS somewhere. That is the point of the loop.

- [ ] **Step 6: The loop. Repeat until the build passes:**
  1. Find the failing README line: it is the last `+ ...` xtrace line before the error in `$PRIV/build.log`.
  2. Diagnose the root cause, using superpowers:systematic-debugging. Reproduce it in the last good layer: `docker run --rm -it <last layer id> bash`.
  3. Fix **README.md**, and the same fact in **INSTALLATION.md** with its "why". Never patch around the docs in the Dockerfile.
  4. Commit: `git commit -am "docs: <the fix> (found by the docker build)"`.
  5. Rebuild (Step 5). Layer caching replays the unchanged steps. A fix to an earlier step invalidates the cache from that step on, which is what we want.

  Defects already known, each with the docs fix to try first:

  | Symptom | Cause | Docs fix |
  |---|---|---|
  | colcon builds `franka_hardware`, fails on libfranka | the 19 `COLCON_IGNORE`s in `franka_ros2` are local-only (its `.gitignore` ignores them) | step 4: `colcon build --packages-up-to block_bin cube_stack` |
  | recorder `ModuleNotFoundError: datasets`/`av` at runtime | `lerobot==0.6.0` lacks the dataset extra | step 3: `"lerobot[dataset]==0.6.0"` |
  | `isaacsim[all]` resolution error | `[all]` is broken in 6.0.1.0 GA (`modules/isaac6-install.md`) | step 3: the subset that file names, `[extscache]` plus core/ros2/sensor/robot/storage/asset |
  | GUIDE `fuser: not found` on a busy port | `psmisc` undocumented | already in the Step 2 block |
  | INSTALLATION §6.2 says `isaacsim.exp.full.kit` | stale | it is `isaacsim.exp.base.python.kit` (`runtime.py:265` passes no experience) |

  Exit condition: `docker build --target build` passes from the docs alone.

- [ ] **Step 7: Tests in the image**

Run: `docker build -f docker/Dockerfile --target test -t guide:test . && docker run --rm guide:test`
Expected: `519 passed` (the count at dev 834bb48), 0 failed. A failure is a docs or environment gap: fix it per Step 6.

- [ ] **Step 8: GPU smoke test, following README "Usage"**

```bash
docker run --rm -it --gpus all guide:build bash   # all GPUs: init.yaml still says render_device: cuda:1
# inside:
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
ros2 launch guide_core bringup.launch.py > /tmp/guide.log 2>&1 &
until ros2 service list | grep -q /Sim_0/Register; do sleep 5; done
ros2 service call /Sim_0/Register guide_msgs/srv/RegisterScene "{path: 'block_bin'}"
ros2 launch block_bin bringup.launch.py > /tmp/task.log 2>&1 &
until ros2 service list | grep -q /Sim_0/Scene_0/generate_demonstration; do sleep 5; done
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration "{path: '/tmp/ds', zones: [], counts: [1]}"
until grep -qs "Dataset finalized successfully" ~/.ros/log/guide_recorder_*.log; do sleep 10; done
wc -l /tmp/ds/*/meta/guide_episodes.jsonl
```
Expected: `1 /tmp/ds/<dataset>/meta/guide_episodes.jsonl`. Each failure on the way is a docs fix (Step 6). The first likely one is a missing Vulkan/GL library: add it to the Prerequisites block.

- [ ] **Step 9: Commit**

```bash
git add .dockerignore docker/Dockerfile docker/readme_steps.sh README.md INSTALLATION.md
git commit -m "docker: build, test stages run the README install verbatim"
```

---

### Task 2: GUIDE flags — `--set` overrides, ROS arguments ignored

**Files:**
- Modify: `guide_core/test/conftest.py` (receives `STUBBED` and `isaac_import`), `guide_core/test/test_command_fixes.py` (drops them)
- Modify: `guide_core/guide_core/ros/guide_ros.py` (`create_arguments`, new `apply_overrides`, `ros_entry_point`)
- Create: `guide_core/test/test_sim_flags.py`

**Interfaces:**
- Produces: `apply_overrides(config: dict, pairs: list[str]) -> dict` in `guide_core.ros.guide_ros`, and the GUIDE CLI flag `--set KEY=VALUE` (repeatable).

- [ ] **Step 1: Move the `isaac_import` fixture to `conftest.py`.** Cut `STUBBED` and the `isaac_import` fixture (`test_command_fixes.py:18-54`) and append them to `guide_core/test/conftest.py`, adding `import importlib`, `from unittest.mock import MagicMock` and `import pytest` there. `command_module` stays in `test_command_fixes.py` and now resolves `isaac_import` from conftest.

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_command_fixes.py`
Expected: same pass count as before the move.

- [ ] **Step 2: Write the failing tests** (`guide_core/test/test_sim_flags.py`)

```python
"""GUIDE's own command line: init.yaml overrides, and ROS 2's arguments left alone."""

import argparse

import pytest


def test_set_overrides_startup_keys_and_dotted_sections(isaac_import):
    ros = isaac_import("guide_core.ros.guide_ros")
    config = {"startup": {"render_device": "cuda:1", "headless": True}}

    ros.apply_overrides(
        config, ["render_device=cuda:0", "headless=false", "world.stage_units_in_meters=0.5"]
    )

    assert config == {
        "startup": {"render_device": "cuda:0", "headless": False},
        "world": {"stage_units_in_meters": 0.5},
    }


def test_set_rejects_a_pair_without_a_value(isaac_import):
    ros = isaac_import("guide_core.ros.guide_ros")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        ros.apply_overrides({}, ["headless"])


def test_ros_arguments_are_left_for_ros(isaac_import):
    ros = isaac_import("guide_core.ros.guide_ros")
    parser = argparse.ArgumentParser()
    ros.create_arguments(parser)

    args, rest = parser.parse_known_args(
        ["--id", "3", "--set", "headless=false", "--ros-args", "-r", "__node:=x"]
    )

    assert (args.id, args.set) == (3, ["headless=false"])
    assert rest == ["--ros-args", "-r", "__node:=x"]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py`
Expected: FAIL, `AttributeError: module 'guide_core.ros.guide_ros' has no attribute 'apply_overrides'` (and `args.set` missing).

- [ ] **Step 4: Implement** in `guide_ros.py`. Add `import yaml` and `from guide_core.core.runtime import IsaacSimRuntime` to the imports, then:

```python
def create_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--id",
        type=int,
        help="Id number of the initialized simulation. Used for namespacing.",
        default=0,
    )
    parser.add_argument("-d", "--debug", type=str2bool, help="Debug simulator.", default=False)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an init.yaml value, e.g. --set render_device=cuda:0 --set headless=false. "
        "A bare key is under 'startup'; other sections take a dot: world.stage_units_in_meters=1.0",
    )


def apply_overrides(config: dict, pairs: list[str]) -> dict:
    """Apply --set KEY=VALUE pairs to an init.yaml config; values are YAML (false, 0.5, cuda:0)."""
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        *sections, leaf = key.split(".") if "." in key else ("startup", key)
        node = config
        for section in sections:
            node = node.setdefault(section, {})
        node[leaf] = yaml.safe_load(value)
    return config
```

In `ros_entry_point`, replace `args = parser.parse_args()` and the `init_runtime` call:

```python
    # ROS 2 appends its own arguments (--ros-args ...); they are rclpy's, not ours.
    args, _ = parser.parse_known_args()
    config = apply_overrides(IsaacSimRuntime._load_config(), args.set)
```
```python
    sim.init_runtime(config=config, debug=args.debug, logger=None)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py guide_core/test/test_command_fixes.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add guide_core/guide_core/ros/guide_ros.py guide_core/test/conftest.py guide_core/test/test_command_fixes.py guide_core/test/test_sim_flags.py
git commit -m "guide_core: --set KEY=VALUE overrides init.yaml; ROS arguments are left to ROS"
```

---

### Task 3: One clock per simulator; task launches follow the simulator id

**Files:**
- Modify: `guide_core/guide_core/ros/guide_ros.py:227-229` (`create_clock` in the namespace)
- Modify: `guide_tasks/block_bin/launch/bringup.launch.py`, `guide_tasks/cube_stack/launch/bringup.launch.py`
- Test: `guide_core/test/test_sim_flags.py`

**Interfaces:**
- Produces: the topic `/Sim_<id>/clock`; task launch arguments `sim_id` (default `0`), `first_scene` (default `0`) and `num_env` (default `1`), launching scenes `first_scene .. first_scene+num_env-1`.

- [ ] **Step 1: Write the failing tests** (append to `test_sim_flags.py`; this task's and later tasks' imports go to the file's top import block)

```python
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_the_clock_is_created_in_the_simulator_namespace(isaac_import):
    from guide_msgs.srv import RegisterScene

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    backend = SimpleNamespace(
        stop=MagicMock(), play=MagicMock(), call=MagicMock(),
        register_scene=MagicMock(return_value=(0, (0.0, 0.0, 0.0))),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=False, _tasks=None,
        get_namespace=lambda: "/Sim_3",
    )

    assert ros._register_callback(me, RegisterScene.Request(path="block_bin"), None).success
    backend.call.assert_called_once_with("create_clock", namespace="Sim_3")


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

- [ ] **Step 2: Run to verify they fail**

Run: build the private install first (`colcon build ... --packages-select guide_core block_bin cube_stack`, see Global Constraints), `source $PRIV/install/setup.zsh`, then `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py -k "clock"`
Expected: FAIL. `create_clock` is called without `namespace`, and `generate_nodes` returns a flat list with no `SetRemap`.

- [ ] **Step 3: Implement the namespaced clock** (`guide_ros.py`, in `_register_callback`)

```python
            if not self._has_clock:
                # /Sim_N/clock: every simulator runs at its own speed. Task launches remap
                # their nodes' /clock to it (SetRemap in <task>/launch/bringup.launch.py).
                self._backend.call("create_clock", namespace=self.get_namespace().strip("/"))
                self._has_clock = True
```

- [ ] **Step 4: Implement the launch arguments.** Apply this to both task launch files. In `block_bin/launch/bringup.launch.py`:

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
In both loops, `for i in range(num_env):` becomes `for i in range(first, first + num_env):`; `f"/Sim_0/Scene_{i}/franka"` becomes `f"{sim}/Scene_{i}/franka"`; and `f"/Sim_0/Scene_{i}"` becomes `f"{sim}/Scene_{i}"`. The return becomes:
```python
    # Each simulator publishes its own clock (/Sim_N/clock); use_sim_time nodes listen on
    # /clock, so point every node of this launch, the included MoveIt ones too, at it.
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *move_groups, *testers])]
```
and `generate_launch_description` declares:
```python
            DeclareLaunchArgument(
                "num_env", default_value="1", description="Number of scenes to launch"
            ),
            DeclareLaunchArgument(
                "first_scene", default_value="0",
                description="Id of the first scene (GUIDE --bringup launches one scene at a time)",
            ),
            DeclareLaunchArgument(
                "sim_id", default_value="0", description="Simulator id: the Sim_<id> namespace"
            ),
```
`cube_stack/launch/bringup.launch.py` gets the same imports, arguments and `first`/`sim` lines.
- Its single loop (line 38) becomes `for i in range(first, first + num_env):`.
- `f"/Sim_0/Scene_{i}/franka"` becomes `f"{sim}/Scene_{i}/franka"`, and `f"/Sim_0/Scene_{i}"` becomes `f"{sim}/Scene_{i}"`.
- It collects everything in one list, `nodes`, so its return becomes:
```python
    # Each simulator publishes its own clock (/Sim_N/clock); use_sim_time nodes listen on
    # /clock, so point every node of this launch, the included MoveIt ones too, at it.
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *nodes])]
```

- [ ] **Step 5: Rebuild and run the tests**

Run: `colcon build --base-paths . --build-base $PRIV/build --install-base $PRIV/install --packages-select guide_core block_bin cube_stack && source $PRIV/install/setup.zsh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py`
Expected: all pass.

- [ ] **Step 6: Update README "Usage".** Change "Several scenes" to `num_env:=N [first_scene:=K] [sim_id:=N]`, and add one line saying the clock topic is `/Sim_<id>/clock`; tools that use sim time remap `/clock:=/Sim_0/clock`.

- [ ] **Step 7: Commit**

```bash
git add guide_core/guide_core/ros/guide_ros.py guide_tasks/block_bin/launch/bringup.launch.py guide_tasks/cube_stack/launch/bringup.launch.py guide_core/test/test_sim_flags.py README.md
git commit -m "Per-simulator clock /Sim_N/clock; task launches take sim_id and first_scene"
```

> **Outside this repo:** `block_bin_eval` paces evaluation on sim time
> (`launch/eval_pink.launch.py:72-81`, `block_bin_eval/eval_policy.py:623-640`). After this
> task it needs `("/clock", "/Sim_0/clock")` in that Node's remappings, and the same remap for
> the `eval_policy_pink` node. That is its own repo and its own commit, done with the user.

---

### Task 4: Dataset markers on the simulator's stdout

**Files:**
- Modify: `guide_core/guide_core/scene/scene_recorder.py` (`_finalize_dataset`, and the `FINALIZE` branch of `run` at line 264)
- Test: `guide_core/test/test_sim_flags.py`

**Interfaces:**
- Produces: stdout lines `GUIDE_DATASET_READY <dataset dir>` (after the language columns and the verification) and `GUIDE_DATASET_EMPTY <task_name>` (a `FINALIZE` with nothing recorded). Task 7 parses them.

- [ ] **Step 1: Write the failing tests**

```python
import logging


def recorder():
    from guide_core.scene.scene_recorder import SceneRecorder

    r = SceneRecorder("pkg", "dataset_0_0", {"dataset": {"fps": 10}})
    r._logger = logging.getLogger("test_sim_flags")
    return r


def test_a_finalized_dataset_is_announced_on_stdout(tmp_path, capsys):
    r = recorder()
    r.dataset = SimpleNamespace(root=tmp_path, finalize=lambda: None)

    r._finalize_dataset()

    assert f"GUIDE_DATASET_READY {tmp_path}\n" in capsys.readouterr().out


def test_a_finalize_with_nothing_recorded_is_announced_too(capsys):
    recorder()._finalize_dataset(announce_empty=True)
    assert "GUIDE_DATASET_EMPTY dataset_0_0\n" in capsys.readouterr().out


def test_the_shutdown_sweep_stays_quiet(capsys):
    recorder()._finalize_dataset()
    assert "GUIDE_DATASET" not in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py -k "announced or quiet"`
Expected: FAIL. No marker is printed, and `announce_empty` is an unexpected keyword.

- [ ] **Step 3: Implement.** Change the signature and add both prints in `_finalize_dataset`:

```python
    def _finalize_dataset(self, announce_empty: bool = False):
        if self.dataset is None and announce_empty:
            # A FINALIZE that recorded nothing still ends a generation request; say so, or a
            # plan waiting for this scene's dataset would wait forever.
            print(f"GUIDE_DATASET_EMPTY {self.task_name}", flush=True)
        if self.dataset is not None:
```
After the `try/except` verification block, still inside `if self.dataset is not None:`, add:
```python
            # Machine-readable, on the simulator's stdout (the recorder's log is a file): the
            # container runner (guide_core.ros.container) delivers the dataset when it reads this.
            print(f"GUIDE_DATASET_READY {dataset_root}", flush=True)
```
In `run`, the `FINALIZE` branch calls `self._finalize_dataset(announce_empty=True)`. The `SHUTDOWN` and `finally` calls stay as they are.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py guide_core/test/test_discarded_frames.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/scene/scene_recorder.py guide_core/test/test_sim_flags.py
git commit -m "recorder: GUIDE_DATASET_READY/EMPTY markers on stdout when a dataset is finalized"
```

---

### Task 5: TaskBringup — fetch, build with dependencies, activate, launch

**Files:**
- Create: `guide_core/guide_core/ros/task_bringup.py`
- Test: `guide_core/test/test_task_bringup.py`

**Interfaces:**
- Produces, in `guide_core.ros.task_bringup`:
  - `split_s3(url: str) -> tuple[str, str]`
  - `s3_client()` (a boto3 client; endpoint and credentials come from the `AWS_*` environment)
  - `is_installed(name: str) -> bool`
  - `task_package(bundle: Path) -> str`
  - `class TaskBringup(sim_id: int, workdir: Path, run=_run, popen=subprocess.Popen, s3=None)`, with:
    - `.prepare(path: str) -> str`
    - `.activate() -> None`
    - `.launch(pkg: str, scene_id: int) -> None`
    - `.shutdown(timeout: float = 30.0) -> None`
- Environment: `GUIDE_PINS` (constraints file for pip; optional).

- [ ] **Step 1: Write the failing tests** (`guide_core/test/test_task_bringup.py`)

```python
"""Register with --bringup: a task given by name, directory or S3 archive becomes runnable."""

import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from guide_core.ros import task_bringup as tb


def make_bundle(root: Path, name="my_task", repos=False, reqs=False) -> Path:
    pkg = root / name
    (pkg / name).mkdir(parents=True)
    (pkg / name / "scene.py").write_text("class Scene: pass\n")
    (pkg / "package.xml").write_text(f"<package><name>{name}</name></package>\n")
    if repos:
        (root / "deps.repos").write_text("repositories: {}\n")
    if reqs:
        (root / "requirements.txt").write_text("six\n")
    return root


def test_an_installed_package_is_used_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(tb, "is_installed", lambda name: name == "block_bin")
    ran = []
    assert tb.TaskBringup(0, tmp_path, run=ran.append).prepare("block_bin") == "block_bin"
    assert ran == []


def test_a_bundle_is_built_with_all_its_dependencies(monkeypatch, tmp_path):
    bundle = make_bundle(tmp_path / "bundle", repos=True, reqs=True)
    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    monkeypatch.setenv("GUIDE_PINS", "/pins.txt")
    monkeypatch.setattr(sys, "path", list(sys.path))
    ran = []

    assert tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(bundle)) == "my_task"

    assert [cmd[:2] for cmd in ran] == [
        ["vcs", "import"], ["rosdep", "install"], ["uv", "pip"], ["colcon", "--log-base"],
    ]
    assert ran[2][-2:] == ["-c", "/pins.txt"]
    assert ran[3][-2:] == ["--packages-up-to", "my_task"]


def test_a_bundle_without_extras_needs_only_rosdep_and_colcon(monkeypatch, tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ran = []
    tb.TaskBringup(0, tmp_path / "work", run=ran.append).prepare(str(bundle))
    assert [cmd[0] for cmd in ran] == ["rosdep", "colcon"]


def test_a_bundle_holds_exactly_one_task(tmp_path):
    make_bundle(tmp_path, "a")
    make_bundle(tmp_path, "b")
    with pytest.raises(ValueError, match="exactly one task package"):
        tb.task_package(tmp_path)


def test_an_unknown_path_fails_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    with pytest.raises(FileNotFoundError, match="neither an installed package nor a directory"):
        tb.TaskBringup(0, tmp_path).prepare("no_such_task")


def test_an_s3_bundle_is_downloaded_and_unpacked(monkeypatch, tmp_path):
    src = make_bundle(tmp_path / "src")
    archive = tmp_path / "my_task.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=".")

    class FakeS3:
        def download_file(self, bucket, key, dest):
            assert (bucket, key) == ("tasks", "v1/my_task.tar.gz")
            Path(dest).write_bytes(archive.read_bytes())

    monkeypatch.setattr(tb, "is_installed", lambda name: False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    work = tmp_path / "work"
    pkg = tb.TaskBringup(0, work, run=lambda cmd: None, s3=FakeS3()).prepare(
        "s3://tasks/v1/my_task.tar.gz"
    )

    assert pkg == "my_task"
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

Used when GUIDE runs with --bringup (the container does). A Register path is one of
  - an installed package name ("block_bin")          -> used as it is
  - a directory holding the task (a "bundle")        -> built from there
  - s3://bucket/key.tar.gz holding such a directory  -> downloaded, unpacked, built
A bundle holds exactly one task package (<pkg>/<pkg>/scene.py beside its package.xml) and,
optionally, deps.repos (source dependencies, vcstool) and requirements.txt (pip). System
dependencies come from the package.xml files through rosdep. Everything is built into one
overlay, <workdir>/install, which this process and every launch then use.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tarfile
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def split_s3(url: str) -> tuple[str, str]:
    bucket, _, key = url.removeprefix("s3://").partition("/")
    return bucket, key


def s3_client():
    import boto3  # in the venv already: an isaacsim dependency

    # Endpoint and credentials from the environment (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, ...).
    return boto3.client("s3")


def is_installed(name: str) -> bool:
    from ament_index_python.packages import get_package_share_directory

    try:
        get_package_share_directory(name)
        return True
    except (LookupError, ValueError):  # PackageNotFoundError is a KeyError
        return False


def task_package(bundle: Path) -> str:
    found = [
        d.name
        for d in [bundle, *sorted(p for p in bundle.iterdir() if p.is_dir())]
        if (d / "package.xml").is_file() and (d / d.name / "scene.py").is_file()
    ]
    if len(found) != 1:
        raise ValueError(
            f"{bundle} must hold exactly one task package (<pkg>/<pkg>/scene.py), found {found}"
        )
    return found[0]


class TaskBringup:
    def __init__(self, sim_id: int, workdir: Path, run=_run, popen=subprocess.Popen, s3=None):
        self.sim_id = sim_id
        self.workdir = Path(workdir)
        self.install = self.workdir / "install"
        self._run = run
        self._popen = popen
        self._s3 = s3
        self._launches = []

    def prepare(self, path: str) -> str:
        """The package name to register, after fetching and building the task if needed."""
        if not path.startswith("s3://") and is_installed(path):
            return path
        bundle = self._fetch(path)
        pkg = task_package(bundle)
        if not is_installed(pkg):
            self._build(bundle, pkg)
        return pkg

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
        bundle = self.workdir / "src" / archive.name.removesuffix(".gz").removesuffix(".tar").removesuffix(".tgz")
        with tarfile.open(archive) as tar:
            tar.extractall(bundle, filter="data")  # no absolute paths, no escaping links
        return bundle

    def _build(self, bundle: Path, pkg: str) -> None:
        repos = bundle / "deps.repos"
        if repos.is_file():
            self._run(["vcs", "import", "--input", str(repos), str(bundle / "deps")])
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
The S3 test expects `work/src/my_task`, so the archive name `my_task.tar.gz` must lose `.gz`, then `.tar`. The chained `removesuffix` above does that, and `.tgz` is handled too.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_task_bringup.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/ros/task_bringup.py guide_core/test/test_task_bringup.py
git commit -m "guide_core: TaskBringup fetches (dir/S3), builds (vcs, rosdep, pip, colcon) and launches tasks"
```

---

### Task 6: Register fetches, builds and launches with `--bringup`

**Files:**
- Modify: `guide_core/guide_core/ros/guide_ros.py` (`__init__`, `_register_callback`, `create_arguments`, `ros_entry_point`)
- Test: `guide_core/test/test_sim_flags.py`

**Interfaces:**
- Consumes: `TaskBringup` (Task 5): `prepare(path) -> str`, `launch(pkg, scene_id)`, `shutdown()`.
- Produces: GUIDE flags `--bringup BOOL` (default false) and `--tasks-dir DIR` (default `~/.guide/tasks`), and `GUIDEROS2Interface(backend, node_name, namespace, tasks=None)`.

- [ ] **Step 1: Write the failing tests**

```python
def register_with(isaac_import, tasks):
    from guide_msgs.srv import RegisterScene

    ros = isaac_import("guide_core.ros.guide_ros").GUIDEROS2Interface
    order = []
    backend = SimpleNamespace(
        stop=lambda: order.append("stop"),
        play=lambda: order.append("play"),
        call=MagicMock(),
        register_scene=lambda path: order.append(("register", path)) or (1, (0.0, 2.0, 0.0)),
    )
    me = SimpleNamespace(
        _backend=backend, _logger=MagicMock(), _has_clock=True, _tasks=tasks(order),
        get_namespace=lambda: "/Sim_3",
    )
    reply = ros._register_callback(me, RegisterScene.Request(path="s3://t/my_task.tar.gz"), None)
    return reply, order


def test_register_builds_first_and_launches_the_scene_last(isaac_import):
    reply, order = register_with(isaac_import, lambda order: SimpleNamespace(
        prepare=lambda path: order.append("prepare") or "my_task",
        launch=lambda pkg, scene_id: order.append(("launch", pkg, scene_id)),
    ))

    assert reply.success and reply.id == 1
    assert order == ["prepare", "stop", ("register", "my_task"), "play", ("launch", "my_task", 1)]


def test_a_failed_build_fails_register_and_keeps_the_simulator_running(isaac_import):
    def broken(path):
        raise RuntimeError("rosdep: cannot resolve key 'libfoo'")

    reply, order = register_with(isaac_import, lambda order: SimpleNamespace(
        prepare=broken, launch=None,
    ))

    assert not reply.success and "libfoo" in reply.message
    assert order == []  # never stopped: the other scenes kept stepping


def test_bringup_flags(isaac_import):
    ros = isaac_import("guide_core.ros.guide_ros")
    parser = argparse.ArgumentParser()
    ros.create_arguments(parser)
    args, _ = parser.parse_known_args(["--bringup", "true", "--tasks-dir", "/t"])
    assert (args.bringup, args.tasks_dir) == (True, "/t")
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_sim_flags.py -k "register or bringup"`
Expected: FAIL. `prepare` is never called, and the failure reply's message is empty. The existing `except` sets `response.id = -1` on a `uint8` field: that raises inside the handler, and `finally: return` swallows it before `message` is set.

- [ ] **Step 3: Implement.** In `__init__`, add the parameter and keep it:

```python
    def __init__(
        self,
        backend: GUIDESimulator,
        node_name: Optional[str],
        namespace: Optional[str],
        tasks: Optional["TaskBringup"] = None,
    ):
        super().__init__(node_name=node_name, namespace=namespace)

        self._backend = backend
        # --bringup: Register also fetches, builds and launches the task (TaskBringup).
        self._tasks = tasks
```
Replace `_register_callback`'s body:
```python
        response = RegisterScene.Response()
        try:
            # With --bringup, fetch and build the task with its dependencies first: it can take
            # minutes, and the scenes already registered keep stepping meanwhile.
            path = self._tasks.prepare(request.path) if self._tasks else request.path

            self._backend.stop()

            id, offset = self._backend.register_scene(path)

            self._logger.info(f"Registered scene with id {id} at offset {offset}")

            if not self._has_clock:
                # /Sim_N/clock: every simulator runs at its own speed. Task launches remap
                # their nodes' /clock to it (SetRemap in <task>/launch/bringup.launch.py).
                self._backend.call("create_clock", namespace=self.get_namespace().strip("/"))
                self._has_clock = True

            self._backend.play()
            if self._tasks:
                self._tasks.launch(path, id)  # MoveIt + solver for this scene
            response.id = id
            response.offset = list(offset)
            response.message = ""
            response.success = True

        except Exception as e:
            err_msg = f"{e}\n{traceback.format_exc()}"
            self._logger.error(f"Failed to register scene: {err_msg}")
            # No response.id: -1 does not fit the uint8, and setting it raised here, which
            # lost the message below. success=False is the failure signal.
            response.offset = [0.0, 0.0, 0.0]
            response.message = str(e)
            response.success = False
        finally:
            return response
```
Add to `create_arguments`:
```python
    parser.add_argument(
        "--bringup",
        type=str2bool,
        default=False,
        help="Register also fetches and builds the task with its dependencies and launches its "
        "bringup (MoveIt + solver) for the new scene. The container runs GUIDE this way.",
    )
    parser.add_argument(
        "--tasks-dir",
        default=str(Path.home() / ".guide" / "tasks"),
        help="Where --bringup downloads and builds tasks (one overlay workspace).",
    )
```
In `ros_entry_point`, add `from pathlib import Path` and `from guide_core.ros.task_bringup import TaskBringup` (top-level imports), then:
```python
    tasks = TaskBringup(args.id, Path(args.tasks_dir)) if args.bringup else None
    ros_interface = GUIDEROS2Interface(sim, node_name="GUIDE", namespace=NAMESPACE, tasks=tasks)
```
and after `ros_t.join()`:
```python
    if tasks:
        tasks.shutdown()
```

- [ ] **Step 4: Run all guide_core tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/ros/guide_ros.py guide_core/test/test_sim_flags.py
git commit -m "guide_core: --bringup makes Register fetch, build and launch the task; keep Register's failure reason"
```

---

### Task 7: Container runner — plan, DDS config, markers, delivery (pure parts)

**Files:**
- Create: `guide_core/guide_core/ros/container.py` (the functions below; `main` comes in Task 8)
- Test: `guide_core/test/test_container.py`

**Interfaces:**
- Consumes: `split_s3`, `s3_client` from `guide_core.ros.task_bringup`.
- Produces, in `guide_core.ros.container`:
  - `load_plan(text: str) -> dict` returning `{"output": str | None, "scenes": [{"task", "zones", "counts"}]}`; raises `ValueError`
  - `parse_marker(line: str) -> tuple[str, str] | None` (`("READY", dir)` / `("EMPTY", task_name)`)
  - `overlay_ip(subnet: str, ip_json: str | None = None) -> str`
  - `dds_config(ip: str, peers: list[str]) -> str`
  - `zone_counts(dataset: Path) -> Counter`
  - `complete(scene: dict, counts: Counter) -> bool`
  - `deliver(dataset: Path, output: str, ns: str, s3=None) -> str`
  - `read_text(location: str, s3=None) -> str`

- [ ] **Step 1: Write the failing tests** (`guide_core/test/test_container.py`)

```python
"""The container runner's pure parts: plan, DDS config, markers, completeness, delivery."""

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

from guide_core.ros import container as c

PLAN = """
output: s3://bucket/guide
scenes:
  - {task: block_bin, zones: [2, 16], counts: [4, 10]}
  - {task: s3://tasks/cube_stack.tar.gz, zones: [-1], counts: [5]}
  - {task: block_bin, counts: [3]}
"""


def test_a_plan_is_read_with_its_defaults():
    plan = c.load_plan(PLAN)
    assert plan["output"] == "s3://bucket/guide"
    assert plan["scenes"][2] == {"task": "block_bin", "zones": [], "counts": [3]}


@pytest.mark.parametrize("scenes, reason", [
    ("[]", "non-empty"),
    ("[{zones: [1], counts: [2]}]", "'task'"),
    ("[{task: t, zones: [1, 2], counts: [3]}]", "one count per distinct zone"),
    ("[{task: t, zones: [2, 2], counts: [1, 1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-2], counts: [1]}]", "one count per distinct zone"),
    ("[{task: t, zones: [-1], counts: [1, 2]}]", "exactly one count"),
    ("[{task: t, counts: [0]}]", "positive"),
])
def test_bad_plans_are_rejected(scenes, reason):
    with pytest.raises(ValueError, match=reason):
        c.load_plan(f"scenes: {scenes}")


def test_markers_are_found_in_prefixed_lines():
    assert c.parse_marker("[GUIDE-1] GUIDE_DATASET_READY /scratch/scene_0/d_1\n") == (
        "READY", "/scratch/scene_0/d_1")
    assert c.parse_marker("GUIDE_DATASET_EMPTY dataset_0_1") == ("EMPTY", "dataset_0_1")
    assert c.parse_marker("Dataset finalized successfully.") is None


IP_JSON = json.dumps([
    {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]},
    {"ifname": "eth0", "addr_info": [{"local": "10.42.0.7"}]},
    {"ifname": "eth1", "addr_info": [{"local": "172.18.0.3"}]},
])


def test_the_overlay_address_is_picked_by_subnet():
    assert c.overlay_ip("10.42.0.0/24", IP_JSON) == "10.42.0.7"
    with pytest.raises(RuntimeError, match="guide-net"):
        c.overlay_ip("10.99.0.0/24", IP_JSON)


def test_dds_stays_on_the_overlay_and_peers_with_the_master():
    ns = {"c": "https://cdds.io/config"}
    root = ET.fromstring(c.dds_config("10.42.0.7", ["10.42.0.2"]))
    assert [i.get("address") for i in root.iterfind(".//c:NetworkInterface", ns)] == ["10.42.0.7"]
    assert root.find(".//c:AllowMulticast", ns).text == "false"
    assert [p.get("address") for p in root.iterfind(".//c:Peer", ns)] == ["10.42.0.7", "10.42.0.2"]


def dataset(tmp_path, zones):
    d = tmp_path / "scratch" / "scene_0" / "dataset_0_0_x"
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "guide_episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "zone": z}) + "\n" for i, z in enumerate(zones)))
    (d / "data").mkdir()
    (d / "data" / "file-000.parquet").write_bytes(b"x")
    return d


def test_completeness_follows_the_zone_rules(tmp_path):
    counts = c.zone_counts(dataset(tmp_path, [2, 2, 16]))
    assert c.complete({"zones": [2, 16], "counts": [2, 1]}, counts)
    assert not c.complete({"zones": [2, 16], "counts": [2, 2]}, counts)
    assert c.complete({"zones": [], "counts": [3]}, counts)
    assert not c.complete({"zones": [-1], "counts": [2]}, counts)  # zone 16 has 1


def test_delivery_to_a_folder_moves_it_under_the_simulator(tmp_path):
    d = dataset(tmp_path, [0])
    out = tmp_path / "out"
    out.mkdir()
    target = c.deliver(d, str(out), "Sim_3")
    assert target == str(out / "Sim_3" / "dataset_0_0_x")
    assert (out / "Sim_3" / "dataset_0_0_x" / "data" / "file-000.parquet").is_file()
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

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_container.py`
Expected: FAIL, `ImportError: cannot import name 'container'`.

- [ ] **Step 3: Implement** `guide_core/guide_core/ros/container.py` (pure parts)

```python
"""The GUIDE container's entry point: one simulator, run by a plan or by a master over ROS 2.

Plan mode (--plan): register every scene, generate, deliver every dataset, exit 0 only if every
scene recorded its counts. Slave mode (no plan): start the simulator and deliver whatever the
master has it record, until stopped. Datasets are recorded in --scratch and delivered after
generation to --output (a directory or s3://bucket/prefix) as <output>/Sim_<id>/<dataset>;
each delivery is announced on /Sim_<id>/dataset_delivered.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from guide_core.ros.task_bringup import s3_client, split_s3

MARKER = re.compile(r"GUIDE_DATASET_(READY|EMPTY) (\S+)")


def read_text(location: str, s3=None) -> str:
    if not location.startswith("s3://"):
        return Path(location).read_text()
    bucket, key = split_s3(location)
    return (s3 or s3_client()).get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def load_plan(text: str) -> dict:
    plan = yaml.safe_load(text) or {}
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("plan: 'scenes' must be a non-empty list")
    return {"output": plan.get("output"), "scenes": [_scene(i, s) for i, s in enumerate(scenes)]}


def _scene(i: int, scene: dict) -> dict:
    """Demonstration.srv's rules: [] = free draws and [-1] = every zone, one count each;
    otherwise one count per distinct zone >= 0."""
    where = f"plan: scene {i}"
    task, zones, counts = scene.get("task"), scene.get("zones", []), scene.get("counts")
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


def parse_marker(line: str) -> tuple[str, str] | None:
    m = MARKER.search(line)
    return (m.group(1), m.group(2)) if m else None


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


def dds_config(ip: str, peers: list[str]) -> str:
    peer_xml = "".join(f'<Peer address="{p}"/>' for p in [ip, *peers])
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- Written by guide_core.ros.container: DDS on the guide-net interface only, unicast
     discovery (overlay networks carry no multicast) of this container and the master. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface address="{ip}"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
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
    zones, want = scene["zones"], scene["counts"]
    if not zones:
        return sum(counts.values()) == want[0]
    if zones == [-1]:
        # ponytail: checks the zones it saw, not that it saw every zone (the grid is the task's)
        return bool(counts) and all(n == want[0] for n in counts.values())
    return all(counts[z] == n for z, n in zip(zones, want))


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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_container.py`
Expected: all pass (19 with the parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add guide_core/guide_core/ros/container.py guide_core/test/test_container.py
git commit -m "guide_core: container runner parts: plan, overlay DDS config, markers, delivery"
```

---

### Task 8: Container runner — GUIDE child, plan and slave modes, entrypoint

**Files:**
- Modify: `guide_core/guide_core/ros/container.py` (append the runtime part)
- Create: `docker/entrypoint.sh`
- Test: `guide_core/test/test_container.py`

**Interfaces:**
- Consumes: Task 7 functions; GUIDE flags `--id`, `--bringup`, `--tasks-dir`, `--set` (Tasks 2, 6); markers (Task 4).
- Produces:
  - `main(argv: list[str] | None = None) -> int`, run as `python -m guide_core.ros.container`.
  - Flags: `--sim-id --plan --output --scratch --tasks-dir --master --dds-subnet --set`.
  - Environment: `GUIDE_CMD` replaces the simulator command (mock image, tests); `ISAACSIM_PYTHON`.
  - Topic: `/Sim_<id>/dataset_delivered` (`std_msgs/String`, JSON `{"dataset", "target", "complete"}`, transient local, depth 100).

- [ ] **Step 1: Write the failing test** (append to `test_container.py`; `import sys` joins the top import block)

```python
import sys


def test_slave_mode_delivers_what_the_simulator_finalizes(tmp_path, monkeypatch):
    made = tmp_path / "scratch" / "dataset_7_0_x"
    fake = tmp_path / "fake_guide.py"
    fake.write_text(
        "import pathlib\n"
        f"d = pathlib.Path({str(made)!r}); (d / 'meta').mkdir(parents=True)\n"
        "(d / 'meta' / 'info.json').write_text('{}')\n"
        "print('GUIDE_DATASET_READY', d, flush=True)\n"
    )
    monkeypatch.setenv("GUIDE_CMD", f"{sys.executable} {fake}")
    monkeypatch.setenv("CYCLONEDDS_URI", "")  # main() sets it; monkeypatch restores it
    out = tmp_path / "out"
    out.mkdir()

    code = c.main(["--sim-id", "7", "--scratch", str(tmp_path / "scratch"), "--output", str(out)])

    assert code == 1  # the simulator exited on its own: a slave reports that as a failure
    assert (out / "Sim_7" / "dataset_7_0_x" / "meta" / "info.json").is_file()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_container.py -k slave`
Expected: FAIL, `AttributeError: module ... has no attribute 'main'`.

- [ ] **Step 3: Implement** (append to `container.py`; extend its imports with `argparse, contextlib, queue, shlex, signal, sys, tempfile, threading, time`)

```python
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="guide-container", description=__doc__.split("\n\n")[0])
    p.add_argument("--sim-id", type=int, default=0, help="This simulator's Sim_<id> namespace.")
    p.add_argument("--plan", help="Plan file or s3:// URL; without one, wait for a master.")
    p.add_argument("--output", help="Directory or s3://bucket/prefix for finished datasets "
                   "(overrides the plan's). Unset: they stay in --scratch.")
    p.add_argument("--scratch", default="/scratch", help="Local recording directory.")
    p.add_argument("--tasks-dir", default="/root/.guide/tasks", help="Where Register builds tasks.")
    p.add_argument("--master", help="The master's address on guide-net: the only peer besides us.")
    p.add_argument("--dds-subnet", help="guide-net's subnet, e.g. 10.42.0.0/24: DDS stays on it.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Passed to GUIDE: an init.yaml override.")
    return p.parse_args(argv)


def guide_cmd(a: argparse.Namespace) -> list[str]:
    if "GUIDE_CMD" in os.environ:  # the mock image's stand-in simulator; tests
        cmd = shlex.split(os.environ["GUIDE_CMD"])
    else:
        from ament_index_python.packages import get_package_prefix

        exe = Path(get_package_prefix("guide_core")) / "lib" / "guide_core" / "GUIDE"
        cmd = [os.environ.get("ISAACSIM_PYTHON", sys.executable), str(exe)]
    cmd += ["--id", str(a.sim_id), "--bringup", "true", "--tasks-dir", a.tasks_dir]
    for pair in a.set:
        cmd += ["--set", pair]
    return cmd


def dds_uri(a: argparse.Namespace) -> str:
    if not a.dds_subnet:  # sealed: the shipped localhost config
        from ament_index_python.packages import get_package_share_directory

        return f"file://{get_package_share_directory('guide_core')}/config/cyclonedds_localhost.xml"
    path = Path(tempfile.gettempdir()) / "cyclonedds.xml"
    path.write_text(dds_config(overlay_ip(a.dds_subnet), [a.master] if a.master else []))
    return f"file://{path}"


def pump(stream, events: queue.Queue) -> None:
    """Echo the simulator's output (docker logs) and queue its dataset markers."""
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()
        marker = parse_marker(line)
        if marker:
            events.put(marker)
    events.put(("EXIT", ""))


def next_event(events: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return events.get(timeout=1.0)
        except queue.Empty:
            continue
    return None


def call(node, client, request, timeout: float):
    import rclpy

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        raise TimeoutError(f"{client.srv_name} did not answer in {timeout:.0f} s")
    return future.result()


def wait(client, timeout: float = 1800.0) -> None:
    # A cold start compiles shaders (minutes); a solver waits for MoveIt and joint states.
    if not client.wait_for_service(timeout_sec=timeout):
        raise TimeoutError(f"{client.srv_name} did not come up in {timeout:.0f} s")


def run_plan(node, scenes: list, scratch: Path, events, handle, stop) -> bool:
    from guide_msgs.srv import Demonstration, RegisterScene

    register = node.create_client(RegisterScene, "Register")
    wait(register)
    for i, scene in enumerate(scenes):
        # Register fetches and builds an unknown task first: allow for a long build.
        reply = call(node, register, RegisterScene.Request(path=scene["task"]), timeout=3600)
        if not reply.success or reply.id != i:
            raise RuntimeError(f"Register {scene['task']!r} failed: {reply.message or reply.id}")
    for i, scene in enumerate(scenes):
        client = node.create_client(Demonstration, f"Scene_{i}/generate_demonstration")
        wait(client)
        request = Demonstration.Request(
            path=str(scratch / f"scene_{i}"), zones=scene["zones"], counts=scene["counts"])
        reply = call(node, client, request, timeout=60)
        if not reply.success:
            raise RuntimeError(f"scene {i}: {reply.message}")
    results = []
    while len(results) < len(scenes):
        event = next_event(events, stop)
        if event is None:
            return False  # stopped
        kind, value = event
        if kind == "EXIT":
            raise RuntimeError("the simulator exited before the plan finished")
        scene = None
        if kind == "READY":
            scene = scenes[int(Path(value).relative_to(scratch).parts[0].removeprefix("scene_"))]
        results.append(handle(kind, value, scene))
    return all(results)


def serve(events, handle, stop) -> bool:
    """Slave mode: deliver every dataset the master has recorded, until stopped."""
    while (event := next_event(events, stop)) is not None:
        kind, value = event
        if kind == "EXIT":
            return False  # the simulator died
        handle(kind, value)
    return True


def main(argv: list[str] | None = None) -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import String

    a = parse_args(argv)
    ns = f"Sim_{a.sim_id}"
    try:
        plan = load_plan(read_text(a.plan)) if a.plan else None
    except (OSError, ValueError) as e:
        print(f"[container] {e}", flush=True)
        return 1  # before Isaac starts
    output = a.output or (plan or {}).get("output")
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["CYCLONEDDS_URI"] = dds_uri(a)  # for GUIDE, its launches and our own node

    guide = subprocess.Popen(guide_cmd(a), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True)
    events: queue.Queue = queue.Queue()
    threading.Thread(target=pump, args=(guide.stdout, events), daemon=True).start()
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}

    rclpy.init()
    node = rclpy.create_node("guide_container", namespace=ns)
    announce = node.create_publisher(String, "dataset_delivered", QoSProfile(
        depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def handle(kind: str, value: str, scene: dict | None = None) -> bool:
        if kind != "READY":
            print(f"[container] {value} recorded nothing", flush=True)
            return False
        dataset = Path(value)
        ok = complete(scene, zone_counts(dataset)) if scene else True
        try:
            target = deliver(dataset, output, ns) if output else str(dataset)
        except Exception as e:  # keep the local copy; announce the failure; keep serving
            print(f"[container] delivering {dataset} failed, kept in scratch: {e}", flush=True)
            target, ok = None, False
        announce.publish(String(data=json.dumps(
            {"dataset": dataset.name, "target": target, "complete": ok})))
        return ok

    try:
        ok = (serve(events, handle, stop) if plan is None
              else run_plan(node, plan["scenes"], scratch, events, handle, stop))
    except Exception as e:
        print(f"[container] {e}", flush=True)
        ok = False
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(guide.pid, signal.SIGINT)  # GUIDE finalizes every scene on SIGINT
        # Deliver what the shutdown finalized: 150 s (finalize 15 s/scene + uploads) fits
        # inside the 180 s stop timeout the docs give `docker run`.
        deadline = time.monotonic() + 150.0
        while time.monotonic() < deadline:
            try:
                kind, value = events.get(timeout=1.0)
            except queue.Empty:
                if guide.poll() is not None:
                    break
                continue
            if kind == "EXIT":
                break
            handle(kind, value)
        if guide.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(guide.pid, signal.SIGKILL)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `docker/entrypoint.sh`**

```bash
#!/usr/bin/env bash
# GUIDE container entry point: ROS 2 + the workspace, then the runner (guide_core.ros.container).
source /opt/ros/jazzy/setup.bash
source "${GUIDE_WS:-/root/ros2_ws/install}/setup.bash"
exec "${GUIDE_PYTHON:-/root/ros2_ws/.venv/bin/python}" -m guide_core.ros.container "$@"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `chmod +x docker/entrypoint.sh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/ros2_ws/.venv/bin/python -m pytest -q guide_core/test/test_container.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add guide_core/guide_core/ros/container.py guide_core/test/test_container.py docker/entrypoint.sh
git commit -m "guide_core: container runner: GUIDE child, plan and slave modes, delivery announcements"
```

---

### Task 9: Mock image and the overlay communication test

**Files:**
- Create: `docker/mock/mock_sim.py`, `docker/test_comms.sh`
- Modify: `docker/Dockerfile` (stages `mock-build`, `mock`)

**Interfaces:**
- Consumes: the runner (Task 8) with `GUIDE_CMD` set to the mock, plus `guide_msgs`.
- Produces: image `guide:mock`; `docker/test_comms.sh` exits 0 when every check passes.

- [ ] **Step 1: Write `docker/mock/mock_sim.py`**

```python
#!/usr/bin/env python3
"""A stand-in GUIDE for communication tests: same namespace, services and markers, no Isaac.

Register adds a scene and serves its Scene_<i>/generate_demonstration (in the real system the
task's solver serves it). A request writes a LeRobot-shaped dataset, one guide_episodes line
per episode, and prints GUIDE_DATASET_READY like the real recorder. /Sim_<id>/clock runs at
10x real time, so the clocks of two mocks differ.
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from guide_msgs.srv import Demonstration, RegisterScene
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def episodes(zones: list, counts: list) -> list:
    if not zones:
        return [None] * counts[0]
    if zones == [-1]:
        return [z for z in (0, 1) for _ in range(counts[0])]  # the mock's grid: two zones
    return [z for z, n in zip(zones, counts) for _ in range(n)]


class MockSim(Node):
    def __init__(self, sim_id: int):
        super().__init__("GUIDE", namespace=f"Sim_{sim_id}")
        self.sim_id, self.scenes, self.t0 = sim_id, [], time.monotonic()
        self.create_service(RegisterScene, "Register", self.register)
        self.clock = self.create_publisher(Clock, "clock", 10)
        self.create_timer(0.01, self.tick)

    def tick(self):
        t = (time.monotonic() - self.t0) * 10.0
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = int(t), int(t % 1 * 1e9)
        self.clock.publish(msg)

    def register(self, request, response):
        i = len(self.scenes)
        self.scenes.append(request.path)
        self.create_service(Demonstration, f"Scene_{i}/generate_demonstration",
                            lambda req, res: self.generate(i, req, res))
        response.id, response.offset, response.success = i, [0.0, 2.0 * i, 0.0], True
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
        print(f"GUIDE_DATASET_READY {root}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    args, _ = parser.parse_known_args()  # the runner's GUIDE flags are ignored here
    rclpy.init()
    executor = MultiThreadedExecutor()
    executor.add_node(MockSim(args.id))
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the mock stages to `docker/Dockerfile`**

```dockerfile
FROM ros:jazzy-ros-base AS mock-build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY guide_msgs /ws/src/guide_msgs
COPY guide_core /ws/src/guide_core
RUN source /opt/ros/jazzy/setup.bash && cd /ws \
 && colcon build --merge-install --install-base /opt/guide --packages-select guide_msgs guide_core

FROM ros:jazzy-ros-core AS mock
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rmw-cyclonedds-cpp python3-boto3 python3-yaml iproute2 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=mock-build /opt/guide /opt/guide
COPY docker/mock/mock_sim.py /opt/guide/mock_sim.py
COPY docker/entrypoint.sh /usr/local/bin/guide-entrypoint
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp GUIDE_WS=/opt/guide GUIDE_PYTHON=python3 \
    GUIDE_CMD="python3 /opt/guide/mock_sim.py"
ENTRYPOINT ["guide-entrypoint"]
```

Run: `docker build -f docker/Dockerfile --target mock -t guide:mock . && docker image ls guide:mock`
Expected: built, under 1 GB.

- [ ] **Step 3: Write `docker/test_comms.sh`**

```bash
#!/usr/bin/env bash
# Communication test on guide-net: two mock slaves, a stand-in master (fixed IP), MinIO.
# Needs Task 0 (swarm + guide-net 10.42.0.0/24) and guide:mock. Exit 0 = every check passed.
set -euo pipefail
NET=guide-net SUBNET=10.42.0.0/24 MASTER=10.42.0.2
S3=(-e AWS_ENDPOINT_URL=http://guide-minio:9000 -e AWS_ACCESS_KEY_ID=guide
    -e AWS_SECRET_ACCESS_KEY=guidesecret -e AWS_DEFAULT_REGION=us-east-1)
names=(guide-master guide-sim-1 guide-sim-2 guide-minio)
cleanup() { docker rm -f "${names[@]}" >/dev/null 2>&1 || true; }
trap cleanup EXIT; cleanup
check() { echo "CHECK: $1"; }
fail() { echo "FAILED: $1"; exit 1; }   # explicit: set -e ignores `! cmd` and && lists

docker run -d --name guide-minio --network $NET -e MINIO_ROOT_USER=guide \
  -e MINIO_ROOT_PASSWORD=guidesecret minio/minio:RELEASE.2025-04-22T22-12-26Z server /data >/dev/null
docker run -d --name guide-master --network $NET --ip $MASTER "${S3[@]}" --entrypoint bash guide:mock -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && python3 -c '
from guide_core.ros.container import dds_config, overlay_ip
open(\"/tmp/cdds.xml\", \"w\").write(dds_config(overlay_ip(\"$SUBNET\"), []))' && sleep infinity" >/dev/null
m() { docker exec -e CYCLONEDDS_URI=file:///tmp/cdds.xml guide-master bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && $1"; }
sleep 3
m "python3 -c 'import boto3; boto3.client(\"s3\").create_bucket(Bucket=\"guide\")'"
for id in 1 2; do
  docker run -d --name guide-sim-$id --network $NET "${S3[@]}" guide:mock \
    --sim-id $id --master $MASTER --dds-subnet $SUBNET --output s3://guide/out >/dev/null
done

check "the master discovers both slaves"
for _ in $(seq 60); do
  m "ros2 service list" | grep -q /Sim_1/Register && m "ros2 service list" | grep -q /Sim_2/Register && break
  sleep 2
done
services=$(m "ros2 service list")
grep -q /Sim_1/Register <<<"$services" && grep -q /Sim_2/Register <<<"$services" \
  || fail "the master does not see both slaves"

check "a slave registers, generates, delivers to S3, announces it"
m "ros2 service call /Sim_1/Register guide_msgs/srv/RegisterScene \"{path: block_bin}\"" \
  | grep -q "success=True" || fail "Register"
m "ros2 service call /Sim_1/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \"{path: /scratch/s0, zones: [2], counts: [3]}\"" \
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

check "Sim_1's clock comes only from Sim_1"
m "ros2 topic info /Sim_1/clock" | grep -q "Publisher count: 1" || fail "/Sim_1/clock has other publishers"

check "slaves do not see each other"
docker exec -e CYCLONEDDS_URI=file:///tmp/cyclonedds.xml guide-sim-1 bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && ros2 node list" \
  | grep -q /Sim_2/ && fail "Sim_1 sees Sim_2"

check "the host sees no simulator"
(source /opt/ros/jazzy/setup.bash && timeout 10 ros2 node list 2>/dev/null) | grep -q /Sim_ \
  && fail "the host sees a simulator"

echo "ALL CHECKS PASSED"
```

- [ ] **Step 4: Run it**

Run: `chmod +x docker/test_comms.sh && docker/test_comms.sh`
Expected: `ALL CHECKS PASSED`.

If "the master discovers both slaves" fails, Cyclone is not answering unicast discovery from peers outside the master's list. Fall back to the master listing the slaves: give `dds_config` the slaves' addresses on the master side, and note it in the spec's DDS section. Re-run.

- [ ] **Step 5: Two simulators, no cross-talk** (Review Focus 5). The last four checks of Step 3 cover it: per-simulator clocks, no global `/clock`, one publisher per clock, slaves blind to each other. Make one check fail on purpose to prove it can: comment out `--dds-subnet` for `guide-sim-2`. Expected: the master no longer sees Sim_2, and the script stops at `FAILED: the master does not see both slaves`. Restore the flag.

- [ ] **Step 6: Plan mode, sealed, and a bad plan** (Review Focus 4)

```bash
tmp=$(mktemp -d)
printf 'scenes:\n  - {task: block_bin, zones: [1, 2], counts: [2, 1]}\n  - {task: cube_stack, counts: [2]}\n' > $tmp/plan.yaml
docker run --rm --network none -v $tmp:/io guide:mock --plan /io/plan.yaml --output /io/out; echo "exit $?"
ls $tmp/out/Sim_0
printf 'scenes:\n  - {task: block_bin, zones: [1, 1], counts: [2, 2]}\n' > $tmp/bad.yaml
docker run --rm --network none -v $tmp:/io guide:mock --plan /io/bad.yaml; echo "exit $?"
```
Expected: `exit 0` with two `dataset_0_*` folders, then `plan: scene 0: one count per distinct zone >= 0` and `exit 1` within seconds.

- [ ] **Step 7: Commit**

```bash
git add docker/Dockerfile docker/mock/mock_sim.py docker/test_comms.sh
git commit -m "docker: mock image + guide-net communication test (discovery, S3, clocks, isolation)"
```

---

### Task 10: Deploy image and the GPU end-to-end run

**Files:**
- Modify: `docker/Dockerfile` (stages `trim`, `deploy`)
- Create: `docker/trim.sh` (first version: build leftovers only), `docker/e2e_plan.yaml`

**Interfaces:**
- Consumes: `guide:build` (Task 1), the runner (Task 8), `--bringup` (Task 6).
- Produces: image `guide:deploy` with `ENTRYPOINT ["guide-entrypoint"]`.

- [ ] **Step 1: Write `docker/trim.sh` (version 1)**

```bash
#!/usr/bin/env bash
# Delete what headless GUIDE generation never loads from the venv the deploy image copies.
set -euo pipefail
ISAAC=/root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim
rm -rf "$ISAAC"/kit/cache "$ISAAC"/kit/logs "$ISAAC"/kit/data   # build-time Kit state
find /root/ros2_ws/.venv -name __pycache__ -prune -o -name '*.pyc' -print -delete >/dev/null
```

- [ ] **Step 2: Add the stages**

```dockerfile
FROM build AS trim
RUN /root/ros2_ws/src/guide/docker/trim.sh

FROM base AS deploy
COPY --from=trim /root/ros2_ws/.venv /root/ros2_ws/.venv
COPY --from=trim /root/ros2_ws/install /root/ros2_ws/install
COPY --from=build /root/ros2_ws/src/guide/modules/isaac6-safe-pins.txt /root/ros2_ws/pins.txt
COPY docker/entrypoint.sh /usr/local/bin/guide-entrypoint
ENV ISAACSIM_PYTHON=/root/ros2_ws/.venv/bin/python GUIDE_PINS=/root/ros2_ws/pins.txt
RUN mkdir -p /scratch
ENTRYPOINT ["guide-entrypoint"]
```

Run: `docker build -f docker/Dockerfile --target deploy -t guide:deploy . && docker image ls guide`
Expected: built; record the size, which is the baseline for Task 11.

- [ ] **Step 3: Write `docker/e2e_plan.yaml`**

```yaml
# Two tasks in one simulator (scenes 0 and 1), small counts.
scenes:
  - {task: block_bin, zones: [2], counts: [1]}
  - {task: cube_stack, zones: [], counts: [1]}
```

- [ ] **Step 4: Plan mode on one GPU**

```bash
mkdir -p ~/dataset/docker_e2e
docker run --rm --gpus device=1 --stop-timeout 180 \
  -v $PWD/docker/e2e_plan.yaml:/plan.yaml:ro -v ~/dataset/docker_e2e:/output \
  guide:deploy --plan /plan.yaml --output /output --set render_device=cuda:0; echo "exit $?"
```
Expected:
- `Renderer on cuda:0.` and `exit 0`;
- two folders in `~/dataset/docker_e2e/Sim_0/`, owned by you, each with one `guide_episodes.jsonl` line.

Then load one with `LeRobotDataset(..., root=...)` in `~/ros2_ws/.venv` and check its frame count equals its video frame count.

- [ ] **Step 5: Slave mode with an S3 task that is not installed**

```bash
S3=(-e AWS_ENDPOINT_URL=http://guide-minio:9000 -e AWS_ACCESS_KEY_ID=guide
    -e AWS_SECRET_ACCESS_KEY=guidesecret -e AWS_DEFAULT_REGION=us-east-1)
docker run -d --name guide-minio --network guide-net -e MINIO_ROOT_USER=guide \
  -e MINIO_ROOT_PASSWORD=guidesecret minio/minio:RELEASE.2025-04-22T22-12-26Z server /data
docker run -d --name guide-master --network guide-net --ip 10.42.0.2 "${S3[@]}" --entrypoint bash guide:mock -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && python3 -c '
from guide_core.ros.container import dds_config, overlay_ip
open(\"/tmp/cdds.xml\", \"w\").write(dds_config(overlay_ip(\"10.42.0.0/24\"), []))' && sleep infinity"
m() { docker exec -e CYCLONEDDS_URI=file:///tmp/cdds.xml guide-master bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/guide/setup.bash && $1"; }
m "python3 -c 'import boto3; boto3.client(\"s3\").create_bucket(Bucket=\"guide\")'"
tar -czf $PRIV/cube_stack.tar.gz -C guide_tasks cube_stack
docker run --rm -v $PRIV:/b --network guide-net "${S3[@]}" --entrypoint python3 guide:mock -c \
  "import boto3; s=boto3.client('s3'); s.create_bucket(Bucket='tasks'); s.upload_file('/b/cube_stack.tar.gz','tasks','cube_stack.tar.gz')"
docker run -d --name guide-e2e --gpus device=1 --stop-timeout 180 --network guide-net "${S3[@]}" \
  --entrypoint bash guide:deploy -c 'rm -rf /root/ros2_ws/install/cube_stack && exec guide-entrypoint "$@"' _ \
  --sim-id 3 --master 10.42.0.2 --dds-subnet 10.42.0.0/24 --output s3://guide/out --set render_device=cuda:0
m "ros2 service call /Sim_3/Register guide_msgs/srv/RegisterScene \"{path: 's3://tasks/cube_stack.tar.gz'}\""
m "ros2 service call /Sim_3/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \"{path: /scratch/s0, zones: [], counts: [1]}\""
m "timeout 1200 ros2 topic echo --once --qos-durability transient_local /Sim_3/dataset_delivered"
```
Expected:
- `docker logs guide-e2e` shows rosdep, colcon `Finished <<< cube_stack` and the task launch;
- the delivered JSON says `"complete": true`;
- `out/Sim_3/dataset_3_0_*` is in MinIO.

Clean up with `docker rm -f guide-e2e guide-master guide-minio`.

- [ ] **Step 6: `docker stop` in the middle** (Review Focus 1)

Run the Step 4 command with `counts: [20]` in a copy of the plan, detached (`-d --name guide-stop`). After the third `Finished episode` in `docker logs`, run `docker stop -t 180 guide-stop`.
Expected:
- the container exits within the timeout;
- `~/dataset/docker_e2e/Sim_0/` holds the finalized datasets, which load with `LeRobotDataset`;
- `docker logs` shows no "delivering ... failed".

- [ ] **Step 7: Commit**

```bash
git add docker/Dockerfile docker/trim.sh docker/e2e_plan.yaml
git commit -m "docker: deploy image; GPU end-to-end plan/slave/stop runs"
```

---

### Task 11: Trim the deploy image, measuring each cut

**Files:**
- Modify: `docker/trim.sh`, `README.md`/`INSTALLATION.md` (the isaacsim subset is a docs change and goes through the Task 1 loop)
- Create: `docker/kit_exts_keep.txt`

**Interfaces:**
- Consumes: Task 10's end-to-end plan run as the gate after every cut.

- [ ] **Step 1: Generate the extension keep list** from an untrimmed run's Kit log

```bash
docker run --name guide-keep --gpus device=1 -v $PWD/docker/e2e_plan.yaml:/plan.yaml:ro guide:deploy \
  --plan /plan.yaml --set render_device=cuda:0
docker cp guide-keep:/root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim/kit/logs $PRIV/kitlogs
docker rm guide-keep
grep -rhoP '\[ext: \K[^\] ]+(?=\] startup)' $PRIV/kitlogs | sed 's/-[0-9].*//' | sort -u > docker/kit_exts_keep.txt
wc -l docker/kit_exts_keep.txt
```
Expected: about 308 names, matching the count measured on the host.

- [ ] **Step 2: Cut never-loaded extensions and their test/doc folders.** Add to `docker/trim.sh`:

```bash
KEEP=/root/ros2_ws/src/guide/docker/kit_exts_keep.txt   # regenerate (Task 11 Step 1) when a task needs more
for dir in "$ISAAC"/extscache/* "$ISAAC"/exts/* "$ISAAC"/extsDeprecated/*; do
  name=$(basename "$dir"); name=${name%%-[0-9]*}       # extscache dirs are <name>-<version>
  grep -qxF "$name" "$KEEP" || rm -rf "$dir"
done
find "$ISAAC" -depth -type d \( -name tests -o -name docs \) -exec rm -rf {} +
```
Rebuild `deploy`, run Task 10 Step 4. Expected: `exit 0`, and the image is about 10 GB smaller. If Kit reports a missing extension, add its name to the keep list.

- [ ] **Step 3: isaacsim subset instead of `[all]`.** This is a docs change, so it goes through the Task 1 loop. README step 3's Isaac line becomes the subset from `modules/isaac6-install.md` (untracked; it says `[all]` is broken in 6.0.1.0 GA, while this host's venv did install `[all,extscache]`, so the build decides):

```bash
uv pip install --python .venv/bin/python \
  "isaacsim[extscache]==6.0.1.0" isaacsim-core==6.0.1.0 isaacsim-ros2==6.0.1.0 \
  isaacsim-sensor==6.0.1.0 isaacsim-robot==6.0.1.0 isaacsim-storage==6.0.1.0 \
  isaacsim-asset==6.0.1.0 \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
```
Rebuild everything, then run Task 10 Step 4.
Expected: `exit 0`; record the size. If Kit is now missing an extension that the keep list names (for example `isaacsim.util.clash_detection` or the Replicator YAML extension), add the `isaacsim-*` package that ships it and rebuild.

- [ ] **Step 4: Strip shared libraries.** Add `find /root/ros2_ws/.venv -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true` to `trim.sh`. Rebuild, then run Task 10 Step 4. Expected: `exit 0`. If anything fails to load, revert this step: the saving is not worth a broken library.

- [ ] **Step 5: Record the sizes** (baseline → each step) in INSTALLATION.md's Docker section (Task 13), then commit:

```bash
git add docker/trim.sh docker/kit_exts_keep.txt README.md INSTALLATION.md
git commit -m "docker: trim the deploy image (unused Kit extensions, isaacsim subset, stripped libraries)"
```

---

### Task 12: Warm the shader and asset caches, commit `-warm`

**Files:**
- Create: `docker/warm.sh`, `docker/warm_plan.yaml`

- [ ] **Step 1: Write `docker/warm_plan.yaml`**

```yaml
# One episode of each shipped task: compiles every shader and fetches every asset they use.
scenes:
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
  --entrypoint bash "$IMAGE" -c 'guide-entrypoint --plan /warm_plan.yaml --set render_device=cuda:0 \
  && rm -rf /scratch/* /root/.ros/log /root/.nvidia-omniverse/logs \
            /root/ros2_ws/.venv/lib/python3.12/site-packages/isaacsim/kit/logs'
docker commit --change 'ENTRYPOINT ["guide-entrypoint"]' --change 'CMD []' guide-warm "${IMAGE}-warm"
docker rm guide-warm
```

- [ ] **Step 3: Run it, then measure cold vs warm starts and portability**

```bash
chmod +x docker/warm.sh && docker/warm.sh guide:deploy 1
for img in guide:deploy guide:deploy-warm; do for gpu in 1 0; do
  /usr/bin/time -f "$img gpu$gpu %e s" docker run --rm --gpus device=$gpu \
    -v $PWD/docker/warm_plan.yaml:/p.yaml:ro $img --plan /p.yaml --set render_device=cuda:0 >/dev/null
done; done
```
Expected: the warm image is faster on GPU 1 (A4000, the card it was warmed on). Record whether GPU 0 (A2000) also profits, because that decides between one warm image per GPU model and one for all. Also record the warm layer's size: `docker history guide:deploy-warm | head -2`.

- [ ] **Step 4: Commit**

```bash
git add docker/warm.sh docker/warm_plan.yaml
git commit -m "docker: warm.sh bakes shader/asset caches into <image>-warm"
```

---

### Task 13: Documentation

**Files:**
- Modify: `README.md` (a "Docker" section after "Usage"), `INSTALLATION.md` (Docker details: sizes, measurements, DDS)

- [ ] **Step 1: README "Docker" section.** Cover:
  - host prerequisites (Task 0, short);
  - the three build commands;
  - plan mode `docker run` (Task 10 Step 4) and slave mode on `guide-net` (`--sim-id --master --dds-subnet --output`);
  - the plan file format (from the spec);
  - the container flags table (Task 8 `parse_args`);
  - the GUIDE flags `--set/--bringup/--tasks-dir`;
  - the task bundle layout (`deps.repos`, `requirements.txt`);
  - `/Sim_<id>/clock` and `/Sim_<id>/dataset_delivered`;
  - `--stop-timeout 180`;
  - `docker/warm.sh`;
  - `docker/test_comms.sh`.

  Every command in the section must be one that was actually run in Tasks 9–12.

- [ ] **Step 2: INSTALLATION.md:**
  - the image sizes from Task 11 and the warm-start measurements from Task 12;
  - why DDS is unicast on the overlay;
  - why the image is built from the checkout and not cloned (`origin/dev` lags local `dev`);
  - the `block_bin_eval` `/clock` remap note.

- [ ] **Step 3: Full check**

Run: `docker build -f docker/Dockerfile --target test -t guide:test . && docker run --rm guide:test && docker/test_comms.sh`
Expected: all tests pass (519 + the new ones) and `ALL CHECKS PASSED`.

- [ ] **Step 4: Commit and push the branch** (no PR)

```bash
git add README.md INSTALLATION.md
git commit -m "docs: Docker images, container flags, plan format, guide-net"
git push -u origin feat/docker
```
