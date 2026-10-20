import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_nodes(context, *args, **kwargs):

    num_env = int(LaunchConfiguration("num_env").perform(context))

    # The motion layer: robots.franka.controller in this task's config/init.yaml, the switch GUIDE
    # reads when it registers the scene (S2b). oscbf drives the arm from Isaac's physics step, so
    # no move_group or ros2_control is started for it.
    with open(os.path.join(get_package_share_directory("block_bin"), "config", "init.yaml")) as f:
        config = yaml.safe_load(f)
    controller = config["robots"]["franka"].get("controller", "oscbf")  # OSC only
    # dataset.camera_transition: GUIDE moves the cameras onto their mounts for `duration` s at the
    # start of every episode, and the solver holds the arm still for as long plus one frame period
    # at dataset.fps (GUIDE validates both).
    transition = (config.get("dataset") or {}).get("camera_transition") or {}
    startup_padding = transition.get("duration", 0.0) if transition.get("enabled") is True else 0.0
    fps = (config.get("dataset") or {}).get("fps", 10)

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
    for i in range(num_env if controller == "moveit" else 0):
        # The robot for scene i lives under this namespace; Isaac publishes its
        # joint states and subscribes its joint commands here.
        ns = f"/Sim_0/Scene_{i}/franka"
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
    for i in range(num_env):
        testers.append(
            Node(
                package="block_bin",
                executable="solve_task",
                name=f"block_bin_solver_node_{i}",
                prefix=venv_python,
                parameters=[{"use_sim_time": True}],
                arguments=["--namespace", f"/Sim_0/Scene_{i}", "--controller", controller,
                           "--startup-padding", str(startup_padding), "--fps", str(fps)],
                remappings=[
                    ("/trajectory_execution_event", "trajectory_execution_event"),
                    ("/attached_collision_object", "attached_collision_object"),
                    ("/collision_object", "collision_object"),
                ],
            )
        )
    return move_groups + testers


def generate_launch_description():

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "num_env", default_value="1", description="Number of environments to launch"
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
