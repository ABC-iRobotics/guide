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

    # Same-host DDS discovery on this machine needs the localhost cyclonedds
    # config (loopback has no MULTICAST flag + multiple NICs), otherwise these
    # nodes are invisible to rqt / ros2 CLI. Point every node here at the same
    # config guide_core ships; export it in the shell you run rqt from too.
    if "CYCLONEDDS_URI" not in os.environ:
        cdds_cfg = os.path.join(
            get_package_share_directory("guide_core"),
            "config",
            "cyclonedds_localhost.xml",
        )
        if os.path.isfile(cdds_cfg):
            os.environ["CYCLONEDDS_URI"] = f"file://{cdds_cfg}"

    # solve_task imports irob_lerobot_ros -> lerobot (torch), which lives in the
    # Isaac '.venv' (Python 3.12, same as Jazzy). The block_bin console script's
    # shebang points at the system python3 that lacks lerobot, so run it under the
    # venv interpreter via a launch prefix. pymoveit2 + irob_lerobot_ros come from
    # the sourced workspace; lerobot from the venv. Override via ISAACSIM_PYTHON.
    venv_python = os.environ.get(
        "ISAACSIM_PYTHON",
        os.path.join(os.path.expanduser("~"), "ros2_ws", ".venv", "bin", "python"),
    )

    # GUIDE FR3 MoveIt bring-up. This is the Jazzy topic_based variant
    # (franka_fr3_moveit_config/launch/guide_moveit.launch.py): MoveIt plans and
    # executes through topic_based_ros2_control straight into the Isaac-simulated
    # robot instead of the old use_fake_hardware / libfranka paths.
    guide_moveit = os.path.join(
        get_package_share_directory("franka_fr3_moveit_config"),
        "launch",
        "guide_moveit.launch.py",
    )

    move_groups = []
    for i in range(first, first + num_env):
        # The robot for scene i lives under this namespace; Isaac publishes its
        # joint states and subscribes its joint commands here.
        ns = f"{sim}/Scene_{i}/franka"
        move_groups.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(guide_moveit),
                launch_arguments={
                    "namespace": ns,
                    # Customizable base: mount the arm at scene i's frame with the
                    # scene's robot offset (was connected_to/xyz on the old launch).
                    "connected_to": f"Scene_{i}",
                    "base_frame": f"Scene_{i}",
                    "xyz": "-0.3 0 0",
                    "rpy": "0 0 0",
                    # topic_based hardware <-> Isaac Sim, per scene.
                    "joint_states_topic": f"{ns}/joint_states",
                    "joint_commands_topic": f"{ns}/joint_command",
                }.items(),
            )
        )

    testers = []
    for i in range(first, first + num_env):
        testers.append(
            Node(
                package="block_bin",
                executable="solve_task",
                name=f"block_bin_solver_node_{i}",
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
    return [GroupAction([SetRemap(src="/clock", dst=f"{sim}/clock"), *move_groups, *testers])]


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
