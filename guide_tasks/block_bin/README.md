# Block Bin Task

This package is a demonstration task designed for the GUIDE framework. It implements a basic block pick-and-place operation using the `guide_exo` composite node architecture.

## Overview
The `block_bin` task provides an example of how to build and execute complex manipulation sequences using the GUIDE task framework, controlling an Isaac Sim simulated robot arm via ROS 2 and MoveIt 2.

## Dependencies
- `guide_core`: Core simulation orchestration and ROS 2 bridging.
- `guide_exo`: Task execution and composite node structure.
- `guide_msgs`: Standardized message types for the GUIDE framework.
- `irob_lerobot_ros`: Robot hardware abstraction and MoveIt 2 integration.

## Usage
The task can be launched via the provided ROS 2 launch files. Ensure that the GUIDE framework and Isaac Sim environment are properly configured before execution.

```bash
ros2 launch block_bin bringup.launch.py
```

## Maintainer
András Makány (makany.andras@uni-obuda.hu)
