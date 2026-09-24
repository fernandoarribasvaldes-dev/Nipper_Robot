import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    pkg_project_bringup = get_package_share_directory('ros_gz_example_bringup')
    pkg_project_gazebo = get_package_share_directory('ros_gz_example_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ---------------------------------------------------------
    # 1. Launch Gazebo with the new multi-wheelset world
    # ---------------------------------------------------------
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': PathJoinSubstitution([
            pkg_project_gazebo,
            'worlds',
            'nipper_cobots.sdf'
        ])}.items(),
    )

    # ---------------------------------------------------------
    # 2. ros_gz_bridge for all wheelsets
    # ---------------------------------------------------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(
                pkg_project_bringup,
                'config',
                'ros_gz_example_bridge_cobots.yaml'
            ),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    # ---------------------------------------------------------
    # 3. Optional RViz (but note: you have no unified robot)
    # ---------------------------------------------------------
    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', os.path.join(pkg_project_bringup, 'config', 'diff_drive.rviz')],
       condition=IfCondition(LaunchConfiguration('rviz'))
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false',  # default false because RViz is not set up
                              description='Open RViz.'),

        gz_sim,
        bridge,
        rviz,
    ])
