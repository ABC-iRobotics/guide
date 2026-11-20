"""The peg_transfer solver for each scene. OSC only (controller: oscbf): no move_group."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_nodes(context, *args, **kwargs):
    num_env = int(LaunchConfiguration("num_env").perform(context))

    # Same-host DDS discovery needs guide_core's localhost cyclonedds config.
    if "CYCLONEDDS_URI" not in os.environ:
        cdds_cfg = os.path.join(
            get_package_share_directory("guide_core"), "config", "cyclonedds_localhost.xml"
        )
        if os.path.isfile(cdds_cfg):
            os.environ["CYCLONEDDS_URI"] = f"file://{cdds_cfg}"

    # The solver imports lerobot, which lives in the '.venv' (override: ISAACSIM_PYTHON).
    venv_python = os.environ.get(
        "ISAACSIM_PYTHON",
        os.path.join(os.path.expanduser("~"), "ros2_ws", ".venv", "bin", "python"),
    )
    return [
        Node(
            package="peg_transfer",
            executable="solve_task",
            name=f"peg_transfer_solver_node_{i}",
            prefix=venv_python,
            parameters=[{"use_sim_time": True}],
            arguments=["--namespace", f"/Sim_0/Scene_{i}", "--controller", "oscbf"],
        )
        for i in range(num_env)
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "num_env", default_value="1", description="Number of environments to launch"
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
