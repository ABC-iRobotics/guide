# GUIDE Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![repo size](https://img.shields.io/github/repo-size/ABC-iRobotics/guide)
![GitHub Repo stars](https://img.shields.io/github/stars/ABC-iRobotics/guide)
![GitHub forks](https://img.shields.io/github/forks/ABC-iRobotics/guide)

## Introduction
The **GUIDE** framework is a modular, scalable, and task-agnostic imitation learning framework for robotics. It interfaces **Isaac Sim** with *ROS 2 Humble* and **MoveIt 2**, allowing users to specify complex manipulation tasks, orchestrate simulation environments, and seamlessly record expert demonstrations.

The resulting demonstrations are saved natively in the [LeRobot dataset format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format).

## Repository Structure
This repository contains the full GUIDE framework and serves as the main entry point:
- `guide_core`: Core simulation orchestration and ROS 2 bridging.
- `guide_exo`: Task execution and composite node structure for logic flow.
- `guide_msgs`: Standardized message interfaces.
- `guide_tasks`: Contains specific tasks (e.g., `block_bin`).
- `modules`: Git submodules for external robot configurations (e.g., `irob_franka_ros2`).

## Prerequisites

- [Ubuntu 22.04 PC](https://ubuntu.com/certified)
- [ROS 2 Humble (Python3)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [MoveIt 2](https://moveit.picknik.ai/main/index.html)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)

## Setup guide

1. Navigate into your ROS 2 workspace source folder:
```bash
cd ~/ros2_ws/src
```

2. Clone the repository **with its submodules**:
```bash
git clone --recurse-submodules https://github.com/ABC-iRobotics/guide.git
```
*(Note: The `--recurse-submodules` flag is required to pull the external robot configurations located in the `modules/` folder.)*

3. Ignore submodules from being parsed as individual ROS packages to avoid workspace collisions (handled by internal build scripts, but ensure `COLCON_IGNORE` is present in `modules/`).

4. Build the workspace:
```bash
cd ~/ros2_ws
colcon build --packages-up-to guide
source install/setup.bash
```

## Usage

You can launch specific tasks from the `guide_tasks` package. For example, to run the `block_bin` pick-and-place demonstration task:

```bash
ros2 launch block_bin bringup.launch.py
```

Using a separate terminal, the demonstration generation can be triggered via a ROS 2 service:

```bash
ros2 run block_bin solve_task --ros-args -p namespace:=/Sim_0/Scene_0
```
*(Exact launch commands and service calls depend on the instantiated task configuration.)*

## Troubleshooting

In case of any issues, check the official resources:
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)

## Author

[András Makány](https://github.com/andras-makany) - PhD student at Obuda University

## License

This software is released under the GNU General Public License v3.0, see [LICENSE](./LICENSE).
