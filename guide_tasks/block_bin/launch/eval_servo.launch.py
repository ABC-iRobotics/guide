# MoveIt + MoveIt Servo bring-up for policy evaluation (block_bin/eval_policy_servo.py).
#
# Same MoveIt stack bringup.launch.py starts, minus the solve_task demonstration solver
# (a policy is driving the arm now, not the node tree) and plus a servo_node per scene.
# The eval script is NOT launched here: it wants a terminal, a --policy path and the
# venv interpreter, so run it yourself once this is up.
#
# Topic wiring per scene, namespace /Sim_0/Scene_i/franka:
#   eval_policy_servo --/servo_node/delta_twist_cmds--> servo_node
#   servo_node --/fr3_arm_controller/joint_trajectory--> JTC --> topic_based --> Isaac
#
# Servo needs the same robot_description as move_group, so the xacro invocation below
# mirrors guide_moveit.launch.py's. Keep the two in step if the arm mounting changes.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml


def load_yaml(package_name, file_path):
    absolute_file_path = os.path.join(get_package_share_directory(package_name), file_path)
    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def describe_robot(scene: str):
    """robot_description / semantic / kinematics / joint limits for one scene's arm."""
    xacro = FindExecutable(name='xacro')
    urdf = os.path.join(
        get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.urdf.xacro'
    )
    srdf = os.path.join(
        get_package_share_directory('franka_description'), 'robots', 'fr3', 'fr3.srdf.xacro'
    )
    return [
        {
            'robot_description': ParameterValue(
                Command(
                    [xacro, ' ', urdf, ' hand:=true ee_id:=franka_hand ros2_control:=true',
                     ' use_topic_based:=true',
                     f' connected_to:={scene} base_frame:={scene}',
                     ' xyz:="-0.3 0 0" rpy:="0 0 0"',
                     f' joint_states_topic:=/Sim_0/{scene}/franka/joint_states',
                     f' joint_commands_topic:=/Sim_0/{scene}/franka/joint_command']
                ),
                value_type=str,
            )
        },
        {
            'robot_description_semantic': ParameterValue(
                Command([xacro, ' ', srdf, f' hand:=true ee_id:=franka_hand connected_to:={scene}']),
                value_type=str,
            )
        },
        # block_bin's own kinematics, not franka_fr3_moveit_config's: these parameters
        # reach servo_node only (move_group is brought up by guide_moveit.launch.py with
        # its own copy), so the servo loop gets pick_ik's redundancy resolution while
        # planning and demonstration recording keep the stock KDL solver. See the file.
        {'robot_description_kinematics': load_yaml('block_bin', 'config/kinematics.yaml')},
        {'robot_description_planning': load_yaml(
            'franka_fr3_moveit_config', 'config/fr3_joint_limits.yaml')},
    ]


def generate_nodes(context, *args, **kwargs):
    num_env = int(LaunchConfiguration('num_env').perform(context))

    # Same-host DDS discovery needs the localhost cyclonedds config, as in bringup.
    if 'CYCLONEDDS_URI' not in os.environ:
        cdds_cfg = os.path.join(
            get_package_share_directory('guide_core'), 'config', 'cyclonedds_localhost.xml'
        )
        if os.path.isfile(cdds_cfg):
            os.environ['CYCLONEDDS_URI'] = f'file://{cdds_cfg}'

    guide_moveit = os.path.join(
        get_package_share_directory('franka_fr3_moveit_config'), 'launch', 'guide_moveit.launch.py'
    )
    # ServoNode reads its parameters under this prefix (generate_parameter_library).
    servo_params = {
        'moveit_servo': load_yaml('block_bin', 'config/servo.yaml'),
        'use_sim_time': True,
    }

    nodes = []
    for i in range(num_env):
        scene = f'Scene_{i}'
        ns = f'/Sim_0/{scene}/franka'
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(guide_moveit),
                launch_arguments={
                    'namespace': ns,
                    'connected_to': scene,
                    'base_frame': scene,
                    'xyz': '-0.3 0 0',
                    'rpy': '0 0 0',
                    'joint_states_topic': f'{ns}/joint_states',
                    'joint_commands_topic': f'{ns}/joint_command',
                }.items(),
            )
        )
        nodes.append(
            Node(
                package='moveit_servo',
                executable='servo_node',
                name='servo_node',
                namespace=ns,
                output='screen',
                parameters=describe_robot(scene) + [servo_params],
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
