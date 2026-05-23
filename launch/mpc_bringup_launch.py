#!/usr/bin/env python3
"""
Unified MPC bringup — mirrors ebot_bringup_launch with the full nav2 stack.

COMPUTE (headless Jetson / onboard PC) — default, no display needed:
    ros2 launch kinematic_mpc mpc_bringup_launch.py

COMPUTE with display attached (VNC / monitor):
    ros2 launch kinematic_mpc mpc_bringup_launch.py viz:=True

SLAM / mapping mode (build a new map):
    ros2 launch kinematic_mpc mpc_bringup_launch.py slam:=True viz:=False

REMOTE LAPTOP (visualization only — DDS peer, no nav2 running locally):
    ros2 launch kinematic_mpc mpc_viz_launch.py

MPC node — run SEPARATELY after bringup is up:
    ros2 launch kinematic_mpc mpc_launch.py
    ros2 launch kinematic_mpc mpc_launch.py debug:=True   # steer-only

Override arguments
------------------
    viz:=True/False         Start RViz2 locally (default False — headless compute; True on display)
    slam:=True/False        SLAM mapping vs AMCL localization against saved map (default False)
    map:=/path/to/map.yaml  Map used by AMCL (ignored when slam:=True)
    use_sim_time:=True      Gazebo only
    params_file:=...        Override kinematic_mpc/params/nav2_params.yaml
    autostart:=true         Auto-activate nav2 lifecycle nodes
    use_composition:=True   Use component_container_isolated for faster IPC
    use_respawn:=False      Respawn nodes on crash (non-composition mode only)
    log_level:=info         Log verbosity

Components launched
-------------------
    Always:     EKF (robot_localization)
                nav2 navigation stack — controller_server, planner_server, bt_navigator,
                  behaviors_server, global_costmap, local_costmap (navigation_launch.py)
    slam=False: map_server + AMCL (localization_launch.py)
    slam=True:  slam_toolbox online-async + map_saver_server + lifecycle_manager_slam
    viz=True:   RViz2 with mpc.rviz — map, lidar, AMCL particles, global/local costmaps,
                  car pose, EKF velocity, MPC raceline/horizon/ref_horizon
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from nav2_common.launch import RewrittenYaml

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
_RVIZ_CFG    = os.path.join(_MPC_DIR,  'rviz',   'mpc.rviz')

# ---------------------------------------------------------------------------
# Parse-time sanity check — fail loudly if the package needs a rebuild rather
# than letting nodes crash silently with missing parameters.
# Run:  colcon build --packages-select kinematic_mpc   then re-source.
# ---------------------------------------------------------------------------
_REQUIRED = {
    'EKF config':      _EKF_YAML,
    'nav2 params':     _NAV2_PARAMS,
    'map yaml':        _MAP_YAML,
    'slam params':     _SLAM_PARAMS,
}
_missing = [f'  {label}: {path}' for label, path in _REQUIRED.items() if not os.path.isfile(path)]
if _missing:
    raise FileNotFoundError(
        '\n[mpc_bringup] Required files not found in install space '
        '— did you forget to rebuild?\n'
        '  Run:  colcon build --packages-select kinematic_mpc  &&  source install/setup.bash\n'
        'Missing:\n' + '\n'.join(_missing)
    )


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    namespace       = LaunchConfiguration('namespace')
    use_namespace   = LaunchConfiguration('use_namespace')
    slam            = LaunchConfiguration('slam')
    map_yaml        = LaunchConfiguration('map')
    use_sim_time    = LaunchConfiguration('use_sim_time')
    params_file     = LaunchConfiguration('params_file')
    autostart       = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn     = LaunchConfiguration('use_respawn')
    log_level       = LaunchConfiguration('log_level')
    rviz_config     = LaunchConfiguration('rviz_config')
    viz             = LaunchConfiguration('viz')

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={'use_sim_time': use_sim_time, 'yaml_filename': map_yaml},
        convert_types=True,
    )

    args = [
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),
        DeclareLaunchArgument(
            'use_namespace', default_value='False',
            description='Whether to apply a namespace to the navigation stack'),
        DeclareLaunchArgument(
            'slam', default_value='False',
            description='True=slam_toolbox mapping; False=AMCL localization against saved map'),
        DeclareLaunchArgument(
            'map', default_value=_MAP_YAML,
            description='Full path to map.yaml used by AMCL (ignored when slam:=True)'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='False',
            description='Use simulation clock — True only in Gazebo'),
        DeclareLaunchArgument(
            'params_file', default_value=_NAV2_PARAMS,
            description='Full path to the nav2 parameters yaml'),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Automatically startup the nav2 stack'),
        DeclareLaunchArgument(
            'use_composition', default_value='True',
            description='Use composed bringup (component_container_isolated)'),
        DeclareLaunchArgument(
            'use_respawn', default_value='False',
            description='Respawn nodes on crash (non-composition mode only)'),
        DeclareLaunchArgument(
            'log_level', default_value='info',
            description='Log verbosity'),
        DeclareLaunchArgument(
            'async_param', default_value=_SLAM_PARAMS,
            description='slam_toolbox online-async params file'),
        DeclareLaunchArgument(
            'rviz_config', default_value=_RVIZ_CFG,
            description='Full path to the RViz config file'),
        DeclareLaunchArgument(
            'viz', default_value='False',
            description='Start RViz2 locally — False by default (headless compute); pass True with display'),
    ]

    # -----------------------------------------------------------------------
    # 1. SLAM toolbox — online async mapping  (slam:=True)
    #    Publishes: /map  +  map→odom TF
    # -----------------------------------------------------------------------
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_SLAM_DIR, 'launch', 'online_async_launch.py')
        ),
        condition=IfCondition(slam),
        launch_arguments=[
            ('slam_params_file', LaunchConfiguration('async_param')),
            ('use_sim_time',     use_sim_time),
        ],
    )

    # -----------------------------------------------------------------------
    # 2. EKF — fuses /odom + IMU → /odometry/filtered + odom→base_link TF
    # -----------------------------------------------------------------------
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[_EKF_YAML, {'use_sim_time': use_sim_time}],
    )

    # -----------------------------------------------------------------------
    # 3. RViz — optional, disable on headless compute with viz:=False
    #    Shows: map, lidar, AMCL particles, global/local costmap, MPC markers
    # -----------------------------------------------------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(viz),
    )

    # -----------------------------------------------------------------------
    # 4. Nav2 group: container + localization/slam pieces + full nav stack
    # -----------------------------------------------------------------------
    nav2_group = GroupAction([
        PushRosNamespace(condition=IfCondition(use_namespace), namespace=namespace),

        # Shared component container (localization + navigation sub-launches attach here)
        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': autostart}],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings,
            output='screen',
        ),

        # SLAM mode: map_saver_server + its lifecycle manager
        Node(
            condition=IfCondition(slam),
            package='nav2_map_server',
            executable='map_saver_server',
            output='screen',
            respawn=use_respawn,
            respawn_delay=2.0,
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[configured_params],
        ),
        Node(
            condition=IfCondition(slam),
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_slam',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart':    autostart},
                {'node_names':   ['map_saver']},
            ],
        ),

        # Localization mode: map_server + AMCL  (slam:=False, default)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(_NAV2_DIR, 'launch', 'localization_launch.py')
            ),
            condition=UnlessCondition(slam),
            launch_arguments={
                'namespace':       namespace,
                'map':             map_yaml,
                'use_sim_time':    use_sim_time,
                'autostart':       autostart,
                'params_file':     params_file,
                'use_composition': use_composition,
                'use_respawn':     use_respawn,
                'container_name':  'nav2_container',
            }.items(),
        ),

        # Full navigation stack — controller_server, planner_server, bt_navigator,
        #   behaviors_server, global_costmap, local_costmap  (always active)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(_NAV2_DIR, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'namespace':       namespace,
                'use_sim_time':    use_sim_time,
                'autostart':       autostart,
                'params_file':     params_file,
                'use_composition': use_composition,
                'use_respawn':     use_respawn,
                'container_name':  'nav2_container',
            }.items(),
        ),
    ])

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))
    for a in args:
        ld.add_action(a)
    ld.add_action(slam_toolbox)
    ld.add_action(ekf)
    ld.add_action(rviz)
    ld.add_action(nav2_group)
    return ld
