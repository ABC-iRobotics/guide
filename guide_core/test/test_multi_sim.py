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
    entities = group.get_sub_entities()

    (remap,) = [e for e in entities if isinstance(e, SetRemap)]
    assert perform_substitutions(context, remap.src) == "/clock"
    assert perform_substitutions(context, remap.dst) == "/Sim_3/clock"
    solvers = [e for e in entities if isinstance(e, Node)]
    assert [s._Node__arguments for s in solvers] == [["--namespace", "/Sim_3/Scene_2"]]
