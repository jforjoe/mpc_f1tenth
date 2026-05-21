#!/usr/bin/env python3
"""
Minimal MPC bringup — starts only what kinematic_mpc needs:
    1. map_server   — serves the SLAM map to AMCL
    2. AMCL         — publishes map→odom TF (localisation)
    3. EKF          — publishes /odometry/filtered + odom→base_link TF
    4. mpc_node     — the kinematic MPC controller

Does NOT start Nav2's controller_server, planner_server, bt_navigator,
costmaps, or waypoint_follower — those are unused and waste CPU.

Usage (after colcon build):
    ros2 launch kinematic_mpc mpc_launch.py

Override arguments:
    ros2 launch ... map:=/path/to/map.yaml use_sim_time:=False
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ---------------------------------------------------------------------------
# Paths resolved at parse time
# ---------------------------------------------------------------------------
_EBOT_DIR   = get_package_share_directory('ebot_nav2')
_NAV2_DIR   = get_package_share_directory('nav2_bringup')

_MAP_YAML       = os.path.join(_EBOT_DIR, 'maps',   'map.yaml')
_NAV2_PARAMS    = os.path.join(_EBOT_DIR, 'params', 'nav2_params.yaml')
_EKF_YAML       = os.path.join(_EBOT_DIR, 'config', 'ekf.yaml')
_RVIZ_CONFIG    = os.path.join(_EBOT_DIR, 'rviz',   'nav2_default_view.rviz')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation clock (set True only in Gazebo)',
    )
    arg_map = DeclareLaunchArgument(
        'map',
        default_value=_MAP_YAML,
        description='Full path to map.yaml produced by SLAM',
    )
    arg_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Launch RViz2 for visualization — ros2 launch ... rviz:=True',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml     = LaunchConfiguration('map')
    use_rviz     = LaunchConfiguration('rviz')

   
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_NAV2_DIR, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map':          map_yaml,
            'use_sim_time': use_sim_time,
            'params_file':  _NAV2_PARAMS,
            'autostart':    'true',
        }.items(),
    )

    # -----------------------------------------------------------------------
    # 3.  EKF — fuses /odom + IMU → /odometry/filtered + odom→base_link TF
    # -----------------------------------------------------------------------
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            _EKF_YAML,
            {'use_sim_time': use_sim_time},
        ],
    )

    # -----------------------------------------------------------------------
    # 4.  Kinematic MPC node
    # -----------------------------------------------------------------------
    mpc = Node(
        package='kinematic_mpc',
        executable='mpc_node',
        name='kinematic_mpc_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # -----------------------------------------------------------------------
    # 5.  RViz2  (optional — pass rviz:=True to enable)
    # -----------------------------------------------------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', _RVIZ_CONFIG],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        LogInfo(msg='[mpc_launch] Starting: map_server + AMCL + EKF + kinematic_mpc'),
        arg_sim_time,
        arg_map,
        arg_rviz,
        localization,   # map_server + AMCL + lifecycle_manager
        ekf,            # robot_localization EKF
        mpc,            # kinematic MPC controller
        rviz,           # RViz2 (only when rviz:=True)
    ])
