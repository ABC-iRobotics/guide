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

    # Isaac Sim 6.0 is pip-installed into the '.venv' virtualenv on Python 3.12 —
    # the SAME interpreter as ROS 2 Jazzy — so ROS 2 works natively: no bundled
    # rclpy, no PYTHONPATH/LD_LIBRARY_PATH scrubbing, no message-package overlay.
    # Source Jazzy + the workspace before launching; the spawned venv Python then
    # picks up rclpy/guide_msgs from the inherited environment. Override the
    # interpreter with ISAACSIM_PYTHON.
    default_python = os.environ.get(
        "ISAACSIM_PYTHON",
        os.path.join(os.path.expanduser("~"), "ros2_ws", ".venv", "bin", "python"),
    )

    # Isaac Sim 6.0 requires the EULA to be accepted non-interactively.
    env = {"OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", "YES")}

    # Same-host DDS discovery on this machine needs a localhost cyclonedds config
    # (loopback has no MULTICAST flag + multiple NICs). Point GUIDE at it; export
    # the same file in the shell you run ros2/rqt from. Respect a user override.
    if "CYCLONEDDS_URI" not in os.environ:
        cdds_cfg = os.path.join(package_share_dir, "config", "cyclonedds_localhost.xml")
        if os.path.isfile(cdds_cfg):
            env["CYCLONEDDS_URI"] = f"file://{cdds_cfg}"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_topics",
                default_value=os.environ.get("GUIDE_CAMERA_TOPICS", "false"),
                description="Publish the cameras as ROS 2 image topics (/cam_*). Needed "
                "for policy evaluation; during demonstration generation it is pure cost "
                "-- a second render product per camera plus ~166 MB/s of raw images over "
                "localhost DDS that nothing reads. Read by create_camera_graphs, where it "
                "turns topics on even when the task's publish_camera_topics is false.",
            ),
            DeclareLaunchArgument(
                "python_executable",
                default_value=default_python,
                description="Path to the Isaac Sim Python interpreter (the '.venv' Python 3.12 "
                "for pip-installed Isaac Sim 6.0; override via ISAACSIM_PYTHON).",
            ),
            ExecuteProcess(
                cmd=[LaunchConfiguration("python_executable"), node_path, "--debug", "False"],
                name="GUIDE",
                output="both",
                sigterm_timeout="180",
                sigkill_timeout="10",
                additional_env={**env, "GUIDE_CAMERA_TOPICS": LaunchConfiguration("camera_topics")},
            ),
        ]
    )
