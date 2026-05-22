#!/usr/bin/env python3
"""
Minimal launcher — starts only the kinematic MPC node.

Use this when AMCL + EKF are already running (e.g. started separately
via mpc_bringup_launch.py or ebot_bringup_launch.py).

Usage:
    ros2 launch kinematic_mpc mpc_launch.py
    ros2 launch kinematic_mpc mpc_launch.py debug:=True   # steer-only mode
    ros2 launch kinematic_mpc mpc_launch.py use_sim_time:=True

For the full stack (AMCL + EKF + MPC) use:
    ros2 launch kinematic_mpc mpc_bringup_launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation clock — True only in Gazebo',
    )
    arg_debug = DeclareLaunchArgument(
        'debug', default_value='False',
        description='True = mpc_debug_node (STEER_ONLY_MODE); False = mpc_node (full)',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_debug    = LaunchConfiguration('debug')

    mpc = Node(
        package='kinematic_mpc',
        executable='mpc_node',
        name='kinematic_mpc_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(use_debug),
    )

    mpc_debug = Node(
        package='kinematic_mpc',
        executable='mpc_debug_node',
        name='kinematic_mpc_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_debug),
    )

    return LaunchDescription([
        arg_sim_time,
        arg_debug,
        mpc,
        mpc_debug,
    ])
