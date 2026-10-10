#!/usr/bin/env bash
# GUIDE container entry point: ROS 2 + the workspace, then the runner (docker/runner).
source /opt/ros/jazzy/setup.bash
source "${GUIDE_WS:-/root/ros2_ws/install}/setup.bash"
exec "${GUIDE_PYTHON:-/root/ros2_ws/.venv/bin/python}" /opt/guide/guide_container.py "$@"
```
