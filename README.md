# GUIDE Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![repo size](https://img.shields.io/github/repo-size/ABC-iRobotics/guide)
![GitHub Repo stars](https://img.shields.io/github/stars/ABC-iRobotics/guide)
![GitHub forks](https://img.shields.io/github/forks/ABC-iRobotics/guide)

## Introduction
The **GUIDE** framework is a modular, scalable, and task-agnostic imitation learning framework for robotics. It interfaces **Isaac Sim** with *ROS 2 Jazzy* and **MoveIt 2**, allowing users to specify complex manipulation tasks, orchestrate simulation environments, and seamlessly record expert demonstrations.

The resulting demonstrations are saved natively in the [LeRobot dataset format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format).

## Repository Structure
This repository contains the full GUIDE framework and serves as the main entry point:
- `guide_core`: Core simulation orchestration and ROS 2 bridging.
- `guide_ex`: Task execution and composite node structure for logic flow.
- `guide_msgs`: Standardized message interfaces.
- `guide_tasks`: Contains specific tasks (e.g., `block_bin`).
- `modules`: Vendored dependencies (e.g., `pymoveit2`, `irob_lerobot_ros`).

## Prerequisites

- [Ubuntu 24.04](https://ubuntu.com/)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- [MoveIt 2 (Jazzy)](https://moveit.picknik.ai/main/index.html)
- NVIDIA GPU with a recent driver (Isaac Sim 6.0 requirement)
- [`uv`](https://docs.astral.sh/uv/) — used to create the Python 3.12 environment

> **Isaac Sim 6.0.1** is installed with `pip` into a project virtual environment during
> [Installation](#installation) — no standalone install is needed. Because 6.0 runs on **Python 3.12,
> the same interpreter as ROS 2 Jazzy**, ROS 2 works *natively* (no bundled rclpy, no message overlay).

## Installation

**1. Create the workspace and clone the packages** into `src/`:
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/ABC-iRobotics/guide.git
git clone https://github.com/ABC-iRobotics/irob_franka_ros2.git
git clone https://github.com/ABC-iRobotics/irob_franka_description.git
git clone https://github.com/PickNikRobotics/topic_based_ros2_control.git
```

**2. Clone GUIDE's vendored modules.** The `.gitmodules` gitlinks are not committed, so
`--recurse-submodules` pulls nothing; clone them explicitly:
```bash
cd ~/ros2_ws/src/guide/modules
git clone https://github.com/ABC-iRobotics/irob_pymoveit2.git pymoveit2
git clone https://github.com/ABC-iRobotics/irob_lerobot_ros.git
```

**3. Create the Python 3.12 environment and install Isaac Sim 6.0.1 + dependencies:**
```bash
cd ~/ros2_ws
uv venv --python /usr/bin/python3.12 .venv
PINS=src/guide/modules/isaac6-safe-pins.txt

# PyTorch first — match the wheel index to your CUDA version (cu130 shown):
uv pip install --python .venv/bin/python torch==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu130

# Isaac Sim 6.0.1 (--prerelease=allow is required by a pre-release build dependency):
uv pip install --python .venv/bin/python "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

# GUIDE runtime deps (-c protects Isaac's torch/numpy pins from being upgraded):
uv pip install --python .venv/bin/python python-fcl "lerobot==0.6.0" "transformers>=5.4,<5.6" -c $PINS
```
> Always pass `-c $PINS` when installing torch-dependent packages — without it the resolver
> re-resolves torch/numpy and breaks the CUDA/Isaac stack. lerobot 0.6.0 caps `numpy<2.3.0`,
> which conflicts with Isaac's exact `numpy==2.3.1`; `-c $PINS` forces Isaac's version (works
> for lerobot at runtime). lerobot 0.6.0 also needs `transformers 5.4-5.6` + `huggingface-hub 1.x`.

**4. Build the workspace** (`.venv` is hidden, so colcon skips it automatically):
```bash
source /opt/ros/jazzy/setup.bash          # or setup.zsh
cd ~/ros2_ws
colcon build
source install/setup.bash                 # or setup.zsh
```

## Usage

Launch GUIDE from the `guide_core` package. This will start a singleton Isaac Sim instance, 
and a ROS 2 node that can register new scenes as needed.

```bash
ros2 launch guide_core bringup.launch.py
```

Launch a task from the `guide_tasks` package. For example, the `block_bin` pick-and-place
demonstration task (the launch file starts Isaac Sim through the `.venv` interpreter and
accepts the Isaac EULA automatically):

```bash
ros2 launch block_bin bringup.launch.py
```
> The first launch spends ~2 minutes compiling RTX shaders before the viewport appears.

In a separate terminal, trigger demonstration generation via a ROS 2 service. `path` is the
directory the dataset is written under (leave empty for the default `~/dataset`); `zones` and
`counts` are parallel arrays saying how many successful episodes to record per zone:

```bash
# 5 free (unstratified) demonstrations — zone -1 means "draw anywhere in the region"
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '', zones: [-1], counts: [5]}"

# 4 demonstrations with the target cube in zone 2, and 10 in zone 16
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '', zones: [2, 16], counts: [4, 10]}"

# 5 demonstrations in EVERY zone — an empty `zones` sweeps the whole grid
# (block_bin has 20 zones, so this records 100 episodes)
ros2 service call /Sim_0/Scene_0/generate_demonstration guide_msgs/srv/Demonstration \
  "{path: '', zones: [], counts: [5]}"
```
Counts are *successful* episodes: a failed attempt is discarded and retried, so the episode
count is exact regardless of the task's success rate. The dataset is saved to
`<path>/<task>_<timestamp>/` in the LeRobot format.
*(Exact launch commands and service calls depend on the instantiated task configuration.)*

### Zoned randomization

A task can partition its position-randomization region into a grid of square **zones**, so a
dataset can be stratified over the workspace instead of sampled uniformly — useful for
measuring where a policy fails, or for deliberately balancing coverage. Enable it on the
position spec in the task's `config/randomize.yaml`:

```yaml
position:
  value: [0.0, 0.0, 0.025]
  random:
    low: [-0.25, 0.0, 0.0]
    high: [0.25, 0.4, 0.0]
  grid:
    enabled: true
    resolution: 0.1     # 0.1 m cells -> 5 columns x 4 rows = 20 zones
```

Zones are numbered row-major, 0-indexed from the `(min-x, min-y)` corner. The scene chooses
*which* prim gets placed in the requested zone by overriding `zone_target()` (in `block_bin`,
the color-selected block; the other blocks stay free as disturbances). At most one
grid-enabled instruction per scene.

The grid is inert unless a request asks for a zone, so a gridded task still generates ordinary
free demonstrations exactly as before. See [`docs/design/zoned-randomization.md`](docs/design/zoned-randomization.md)
for the full design.

### Dataset metadata

Alongside the LeRobot files, GUIDE writes a reproducibility sidecar into `<dataset>/meta/`:

- `guide_info.json` — run-level constants: master seed, grid layout, curated scene config
  (robots, cameras, USD asset) and provenance (ROS distro, Python, Isaac Sim and GUIDE versions).
- `guide_episodes.jsonl` — one line per *saved* episode: its seed, every drawn randomization
  value, the task string, target/goal prims, the zone and its cell bounds, the robot's starting
  joint configuration, and the main object's pose.

Together these let any episode be replayed exactly: feeding a stored record back as the
randomization injection reproduces the scene verbatim.

## Troubleshooting

- **Nodes are not discovered on the same host** (`ros2 node list` / `rqt_graph` hang): if your
  loopback interface has no multicast, point every terminal at the bundled localhost Cyclone DDS config:
  ```bash
  export CYCLONEDDS_URI=file://$HOME/ros2_ws/install/guide_core/share/guide_core/config/cyclonedds_localhost.xml
  ```
- **`colcon build` fails in message generation** (`No module named 'em'`): another Python is ahead
  of 3.12 in your `PATH`. Force the interpreter for CMake/rosidl:
  ```bash
  mkdir -p ~/.colcon
  printf 'build:\n  cmake-args:\n    - -DPython3_EXECUTABLE=/usr/bin/python3\n' > ~/.colcon/defaults.yaml
  ```
- Official docs: [Isaac Sim 6.0](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) · [MoveIt 2](https://moveit.picknik.ai/main/index.html) · [LeRobot](https://github.com/huggingface/lerobot)

## Author

[András Makány](https://github.com/andras-makany) - PhD student at Obuda University

## Citation (BibTeX)
```
@INPROCEEDINGS{MakanyGalambos2025a,
  author={Makány, András and Galambos, Péter},
  booktitle={2025 IEEE 23rd Jubilee International Symposium on Intelligent Systems and Informatics (SISY)}, 
  title={A Framework for Generating Synthetic Expert Demonstrations in Digital Twin-based Robot Learning}, 
  year={2025},
  month={sep},
  pages={51--56},
  address={Subotica, Serbia},
  doi={10.1109/SISY67000.2025.11205394}
}
```

## Acknowledgements

Program of the Ministry for Culture and Innovation from the source of the National Research, Development and Innovation Fund.Project 2024-1.2.3-HU-RIZONT-00069 has been implemented with support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development, and Innovation Fund, financed under the 2024-1.2.3-HU-RIZONT funding scheme.

## License

This software is released under the GNU General Public License v3.0, see [LICENSE](./LICENSE).
