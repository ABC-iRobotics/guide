import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    package_name = "guide_core"

    package_share_dir = get_package_share_directory(package_name)
    node_path = os.path.join(
        os.path.dirname(os.path.dirname(package_share_dir)),
        "lib",
        package_name,
        "GUIDE",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "python_executable",
                default_value=os.path.join(os.path.expanduser("~"), "isaacsim/python.sh"),
                description="Path to the Isaac Sim Python interpreter (python.sh file)",
            ),
            ExecuteProcess(
                cmd=[LaunchConfiguration("python_executable"), node_path, "--debug", "False"],
                name="GUIDE",
                output="both",
            ),
        ]
    )
