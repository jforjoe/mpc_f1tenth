#!/usr/bin/env python3
"""
Unified MPC bringup for kinematic_mpc — run this on the COMPUTE (Jetson / onboard PC).

Usage
-----
Localization mode  (map already built — use for racing):
    ros2 launch kinematic_mpc mpc_bringup_launch.py

SLAM / mapping mode  (building a new map):
    ros2 launch kinematic_mpc mpc_bringup_launch.py slam:=True

Steer-only debug mode  (MPC steers, you throttle manually via RC):
    ros2 launch kinematic_mpc mpc_bringup_launch.py debug:=True

On the REMOTE LAPTOP for visualization (separate terminal):
    ros2 launch kinematic_mpc mpc_viz_launch.py

Components launched
-------------------
    slam=False  →  map_server + AMCL + EKF + mpc_node      (race / deployment)
    slam=True   →  slam_toolbox       + EKF + mpc_node      (mapping)
    debug=True  →  replaces mpc_node with mpc_debug_node    (STEER_ONLY_MODE)

Override arguments
------------------
    map:=/path/to/map.yaml      full path to map.yaml  (localization mode only)
    use_sim_time:=True          Gazebo only
    slam:=True/False            mapping vs localization
    debug:=True/False           steer-only debug vs full MPC
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ---------------------------------------------------------------------------
# Paths resolved at parse time
# ---------------------------------------------------------------------------
_MPC_DIR  = get_package_share_directory('kinematic_mpc')
_EBOT_DIR = get_package_share_directory('ebot_nav2')
_NAV2_DIR = get_package_share_directory('nav2_bringup')
_SLAM_DIR = get_package_share_directory('slam_toolbox')

_MAP_YAML    = os.path.join(_EBOT_DIR, 'maps',   'map.yaml')
_NAV2_PARAMS = os.path.join(_MPC_DIR,  'params', 'nav2_params.yaml')
_EKF_YAML    = os.path.join(_MPC_DIR,  'config', 'ekf.yaml')
_SLAM_PARAMS = os.path.join(_EBOT_DIR, 'config', 'mapper_params_online_async.yaml')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    arg_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation clock — True only in Gazebo',
    )
    arg_map = DeclareLaunchArgument(
        'map', default_value=_MAP_YAML,
        description='Full path to map.yaml used by AMCL (ignored when slam:=True)',
    )
    arg_slam = DeclareLaunchArgument(
        'slam', default_value='False',
        description='True = slam_toolbox mapping; False = AMCL localization against saved map',
    )
    arg_debug = DeclareLaunchArgument(
        'debug', default_value='False',
        description='True = mpc_debug_node (STEER_ONLY_MODE, speed from RC); False = mpc_node (full)',
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml     = LaunchConfiguration('map')
    use_slam     = LaunchConfiguration('slam')
    use_debug    = LaunchConfiguration('debug')

    # -----------------------------------------------------------------------
    # 1a. SLAM toolbox — online async mapping  (slam:=True)
    #     Config: ebot_nav2/config/mapper_params_online_async.yaml (unchanged)
    #     Publishes: /map  +  map→odom TF
    # -----------------------------------------------------------------------
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_SLAM_DIR, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': _SLAM_PARAMS,
            'use_sim_time':     use_sim_time,
        }.items(),
        condition=IfCondition(use_slam),
    )

    # -----------------------------------------------------------------------
    # 1b. Localization — map_server + AMCL  (slam:=False, default)
    #     Config: kinematic_mpc/params/nav2_params.yaml
    #     Publishes: /map  +  map→odom TF via AMCL particle filter
    # -----------------------------------------------------------------------
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
        condition=UnlessCondition(use_slam),
    )

    # -----------------------------------------------------------------------
    # 2. EKF — fuses /odom + IMU → /odometry/filtered + odom→base_link TF
    #    Config: kinematic_mpc/config/ekf.yaml
    # -----------------------------------------------------------------------
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[_EKF_YAML, {'use_sim_time': use_sim_time}],
    )

    # -----------------------------------------------------------------------
    # 3a. Full MPC node — MPC controls both steering and speed  (debug:=False)
    # -----------------------------------------------------------------------
    mpc = Node(
        package='kinematic_mpc',
        executable='mpc_node',
        name='kinematic_mpc_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(use_debug),
    )

    # -----------------------------------------------------------------------
    # 3b. Debug MPC node — MPC steers only, RC controls speed  (debug:=True)
    # -----------------------------------------------------------------------
    mpc_debug = Node(
        package='kinematic_mpc',
        executable='mpc_debug_node',
        name='kinematic_mpc_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_debug),
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        LogInfo(msg='[mpc_bringup] Launching kinematic_mpc stack on compute'),
        arg_sim_time,
        arg_map,
        arg_slam,
        arg_debug,
        slam,           # slam_toolbox     — active only when slam:=True
        localization,   # AMCL + map_server — active only when slam:=False
        ekf,            # EKF — always active
        # mpc,            # full MPC — active when debug:=False
        mpc_debug,      # debug MPC — active when debug:=True
    ])
