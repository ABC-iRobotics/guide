import os

import ament_index_python.packages
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Monkey-patch get_package_share_directory to isolate the submodule from workspace collisions
_original_get_pkg_share_dir = ament_index_python.packages.get_package_share_directory


def _custom_get_pkg_share_dir(package_name):
    if package_name in ["franka_fr3_moveit_config", "franka_description"]:
        # Redirect to the submodule installed inside block_bin
        block_bin_share = _original_get_pkg_share_dir("block_bin")
        return os.path.join(block_bin_share, "modules", "irob_franka_ros2", package_name)
    return _original_get_pkg_share_dir(package_name)


ament_index_python.packages.get_package_share_directory = _custom_get_pkg_share_dir


def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return file.read()
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():

    num_env = 1

    move_groups = []

    for i in range(num_env):
        move_group = os.path.join(
            get_package_share_directory("franka_fr3_moveit_config"),
            "launch",
            "moveit.launch.py",
        )
        move_groups.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(move_group),
                launch_arguments={
                    "namespace": f"/Sim_0/Scene_{i}/franka",
                    "robot_ip": "dont-care",
                    "use_fake_hardware": "true",
                    "connected_to": f"Scene_{i}",
                    "xyz": '"-0.3 0 0"',
                }.items(),
            )
        )

    testers = []

    for i in range(num_env):
        testers.append(
            Node(
                package="block_bin",
                executable="solve_task",
                name=f"block_bin_solver_node_{i}",
                parameters=[{"use_sim_time": True}],
                arguments=["--namespace", f"/Sim_0/Scene_{i}"],
                remappings=[
                    ("/trajectory_execution_event", "trajectory_execution_event"),
                    ("/attached_collision_object", "attached_collision_object"),
                    ("/collision_object", "collision_object"),
                ],
            )
        )

    return LaunchDescription(move_groups + testers)
