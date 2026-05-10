from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kinematic_mpc',
            executable='kinematic_mpc_node',
            name='kinematic_mpc_node',
            output='screen',
        ),
    ])
