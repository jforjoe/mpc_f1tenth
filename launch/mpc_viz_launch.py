#!/usr/bin/env python3
"""
Remote visualization for kinematic_mpc — run this on the LAPTOP.

The car runs:   ros2 launch kinematic_mpc mpc_launch.py
The laptop runs: ros2 launch kinematic_mpc mpc_viz_launch.py

Requirements on the laptop (one-time setup):
    export ROS_DOMAIN_ID=3                    # must match the car
    export ROS_LOCALHOST_ONLY=0               # allow cross-machine DDS traffic
    source /opt/ros/humble/setup.bash
    source <your_ws>/install/setup.bash

RViz shows:
    - Map + LiDAR scan (car sees the same map it localises against)
    - AMCL particle cloud (localisation confidence)
    - Car pose arrow  (/amcl_pose)
    - TF frames  (map → odom → base_link — orientation of the car)
    - EKF velocity arrow  (/odometry/filtered)
    - MPC raceline  /mpc/raceline   (green  — full raceline, published once)
    - MPC predicted horizon  /mpc/horizon   (blue  — what the solver plans)
    - MPC reference horizon  /mpc/ref_horizon  (yellow — what the solver chases)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('kinematic_mpc'), 'rviz', 'mpc.rviz'
    )

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
