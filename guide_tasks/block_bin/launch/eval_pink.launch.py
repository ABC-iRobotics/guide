# robot_description publisher for policy evaluation through Pink IK
# (block_bin/eval_policy_pink.py).
#
# Deliberately the smallest possible bring-up: robot_state_publisher and nothing else.
# No move_group, no servo_node, no controller_manager, no topic_based_ros2_control.
# That is the whole point -- eval_policy_pink solves the IK in its own process and
# writes joint targets straight onto `joint_command`, and topic_based_ros2_control
# re-publishes its own (now stale) command as soon as the sim drifts away from it,
# which fights the policy. The same reason eval_policy.py says to evaluate without
# bringup.launch.py applies here; the difference is that Pink needs the URDF, so the
# one node that publishes it comes up on its own.
#
# Topic wiring per scene, namespace /Sim_0/Scene_i/franka:
#   robot_state_publisher --/robot_description (latched)--> eval_policy_pink
#   eval_policy_pink      --/joint_command--------------->  Isaac
#
# The xacro invocation mirrors guide_moveit.launch.py's exactly, including the
# ros2_control tags this node has no use for: an identical description means FK here
# and FK under the MoveIt bring-up cannot disagree. Keep the two in step if the arm
# mounting changes.
#
#   ros2 launch block_bin eval_pink.launch.py
#   ros2 launch block_bin eval_pink.launch.py num_env:=4

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def describe_robot(scene: str, namespace: str):
    """The same robot_description guide_moveit.launch.py builds for this scene."""
    urdf = os.path.join(
        get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.urdf.xacro'
    )
    return {
        'robot_description': ParameterValue(
            Command(
                [FindExecutable(name='xacro'), ' ', urdf,
                 ' hand:=true ee_id:=franka_hand ros2_control:=true',
                 ' use_topic_based:=true',
                 f' connected_to:={scene} base_frame:={scene}',
                 ' xyz:="-0.3 0 0" rpy:="0 0 0"',
                 f' joint_states_topic:={namespace}/joint_states',
                 f' joint_commands_topic:={namespace}/joint_command']
            ),
            value_type=str,
        )
    }


def generate_nodes(context, *args, **kwargs):
    num_env = int(LaunchConfiguration('num_env').perform(context))

    # Same-host DDS discovery needs the localhost cyclonedds config, as in bringup.
    if 'CYCLONEDDS_URI' not in os.environ:
        cdds_cfg = os.path.join(
            get_package_share_directory('guide_core'), 'config', 'cyclonedds_localhost.xml'
        )
        if os.path.isfile(cdds_cfg):
            os.environ['CYCLONEDDS_URI'] = f'file://{cdds_cfg}'

    nodes = []
    for i in range(num_env):
        scene = f'Scene_{i}'
        namespace = f'/Sim_0/{scene}/franka'
        nodes.append(
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                namespace=namespace,
                output='screen',
                parameters=[describe_robot(scene, namespace), {'use_sim_time': True}],
                # Isaac's TF graphs are namespaced; publishing absolutely would put this
                # arm's frames on a topic nothing in the scene reads.
                remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'num_env', default_value='1', description='Number of environments to launch'
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
