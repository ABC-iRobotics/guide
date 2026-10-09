"""MoveIt and the cube_stack solver for each scene (same bring-up as block_bin's)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_nodes(context, *args, **kwargs):
    num_env = int(LaunchConfiguration("num_env").perform(context))
    first = int(LaunchConfiguration("first_scene").perform(context))
    sim = f"/Sim_{LaunchConfiguration('sim_id').perform(context)}"

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
    guide_moveit = os.path.join(
        get_package_share_directory("franka_fr3_moveit_config"), "launch", "guide_moveit.launch.py"
    )

    nodes = []
    for i in range(first, first + num_env):
        ns = f"{sim}/Scene_{i}/franka"
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(guide_moveit),
                launch_arguments={
                    "namespace": ns,
                    "connected_to": f"Scene_{i}",
                    "base_frame": f"Scene_{i}",
                    "xyz": "-0.3 0 0",
                    "rpy": "0 0 0",
                    "joint_states_topic": f"{ns}/joint_states",
                    "joint_commands_topic": f"{ns}/joint_command",
                }.items(),
            )
        )
        nodes.append(
            Node(
                package="cube_stack",
                executable="solve_task",
                name=f"cube_stack_solver_node_{i}",
                prefix=venv_python,
                parameters=[{"use_sim_time": True}],
                arguments=["--namespace", f"{sim}/Scene_{i}"],
                remappings=[
                    ("/trajectory_execution_event", "trajectory_execution_event"),
                    ("/attached_collision_object", "attached_collision_object"),
                    ("/collision_object", "collision_object"),
                ],
            )
        )
    # Each simulator publishes its own clock (/Sim_N/clock); use_sim_time nodes listen on
    # /clock, so point every node of this launch, the included MoveIt ones too, at it.
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *nodes])]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "num_env", default_value="1", description="Number of scenes to launch"
            ),
            DeclareLaunchArgument(
                "first_scene", default_value="0", description="Id of the first scene to launch"
            ),
            DeclareLaunchArgument(
                "sim_id", default_value="0", description="Simulator id: the Sim_<id> namespace"
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
