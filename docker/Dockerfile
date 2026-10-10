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
RUN uv pip install --python /root/ros2_ws/.venv/bin/python pytest lark \
      -c /root/ros2_ws/src/guide/modules/isaac6-safe-pins.txt
WORKDIR /root/ros2_ws/src/guide
# INSTALLATION.md §7 plus the runner's tests. No GPU: the tests stub Isaac.
CMD ["bash", "-c", "source /root/ros2_ws/install/setup.bash && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /root/ros2_ws/.venv/bin/python -m pytest -q guide_core/test guide_ex/test guide_tasks/cube_stack/test docker/runner --ignore-glob='*test_flake8.py' --ignore-glob='*test_pep257.py' --ignore-glob='*test_copyright.py'"]
