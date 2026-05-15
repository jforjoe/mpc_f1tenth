import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg       = get_package_share_directory('kinematic_mpc')
    pf_params = os.path.join(pkg, 'config', 'particle_filter_sim.yaml')

    # NOTE: map_server and lifecycle_manager are already launched by
    # gym_bridge_launch.py — do NOT duplicate them here.

    return LaunchDescription([

        # Particle filter
        # /scan is already at /ego_racecar/scan in sim; remap so PF gets it.
        # /odom remapped from sim's ground-truth odom (good enough for motion model).
        Node(
            package='particle_filter',
            executable='particle_filter',
            name='particle_filter_node',
            output='screen',
            parameters=[pf_params],
            remappings=[
                ('/scan', '/scan'),
                ('/odom', '/ego_racecar/odom'),
            ],
        ),

        # MPC — reads /pf/pose/odom
        Node(
            package='kinematic_mpc',
            executable='kinematic_mpc_node',
            name='kinematic_mpc_node',
            output='screen',
        ),
    ])
