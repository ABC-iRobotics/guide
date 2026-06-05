<<<<<<< HEAD
# Isaac Sim Imitation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![repo size](https://img.shields.io/github/repo-size/ABC-iRobotics/isaac_sim_imitation_learning)
![GitHub Repo stars](https://img.shields.io/github/stars/ABC-iRobotics/isaac_sim_imitation_learning)
![GitHub forks](https://img.shields.io/github/forks/ABC-iRobotics/isaac_sim_imitation_learning)

## Introduction
Imitation learning framework for the **TM5-900** collaborative robot arm equipped with the **OnRobot RG6** gripper. The package interfaces **Isaac Sim 4.5** with *ROS 2 Humble* and can be used as an expert demonstration generator.

The resulting demonstrations are saved in the [LeRobot dataset format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format).

## Features

- [Ubuntu 22.04 PC](https://ubuntu.com/certified/laptops?q=&limit=20&category=Laptop&vendor=Dell&vendor=HP&vendor=Lenovo&release=22.04+LTS)
- [ROS 2 Humble (Python3)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format)
- SceneHandler node using [Omniverse](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/Modules.html) libraries
- AnalyticSolver node using ground truth information to solve the randomized task
- TrajectoryRecorder node records robot state - action pairs

## Prerequisites

- [MoveIt 2](https://moveit.picknik.ai/main/index.html)
- [TMFlow 2.2](https://www.tm-robot.com/en/tmflow)
- [tm_drive](https://github.com/TechmanRobotInc/tmr_ros2)
- [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [onrobot-ros2](https://github.com/ABC-iRobotics/onrobot-ros2)
- [tm5-900_rg6_moveit_config](https://github.com/ABC-iRobotics/tm5-900_rg6_moveit_config)

## Setup guide

Using the [previous](#prerequisites) section's links, install every prerequisites.

Navigate into your ROS 2 workspace and copy this repo to your source folder
```
cd src && git clone https://github.com/ABC-iRobotics/isaac_sim_imitation_learning.git && cd ..
```

Build the package (and installing python prerequisites)
```
colcon build --packages-select isaac_sim_msgs isaac_sim_scene_handler analytic_solver
=======
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
>>>>>>> 1a59891 (feat: GUIDE 1.0)
```

## Usage

<<<<<<< HEAD
After installation use the TMFlow to make a simulated robot. Set up the [Ethernet Slave](https://github.com/TechmanRobotInc/TM_Export) settings.

The expert demonstration generation demo can be initialized with the included launch file.

```
ros2 launch isaac_sim_imitation_learning bringup_robot_scene.launch.py robot_ip:=<robot_ip>
```
where *robot_ip* is the simulated robot's ip address.

Using a separate terminal the demonstration generation can be called via a ROS 2 service.

```
ros2 service call /TrajectoryRecorder/GetTrajectory isaac_sim_msgs/srv/Demonstration "{amount:<num_of_demos>, path: '<save_path>'}"
```

where *num_of_demos* is the number of requested successful demonstrations and *save_path* is the absolute path ("~" can be used) where the completed demonstrations are saved.
=======
You can launch specific tasks from the `guide_tasks` package. For example, to run the `block_bin` pick-and-place demonstration task:

```bash
ros2 launch block_bin bringup.launch.py
```

Using a separate terminal, the demonstration generation can be triggered via a ROS 2 service:

```bash
ros2 run block_bin solve_task --ros-args -p namespace:=/Sim_0/Scene_0
```
*(Exact launch commands and service calls depend on the instantiated task configuration.)*
>>>>>>> 1a59891 (feat: GUIDE 1.0)

## Troubleshooting

In case of any issues, check the official resources:
<<<<<<< HEAD
- [OnRobot RG6](https://onrobot.com/en/products/rg6-finger-gripper)
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [TMFlow](https://www.tm-robot.com/en/tmflow)

## Author

[Makány András](https://github.com/andras-makany)  - Graduate student at Obuda University

## License

This software is released under the MIT License, see [LICENSE](./LICENSE).
=======
- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)

## Author

[András Makány](https://github.com/andras-makany) - PhD student at Obuda University

## License

This software is released under the GNU General Public License v3.0, see [LICENSE](./LICENSE).
>>>>>>> 1a59891 (feat: GUIDE 1.0)
