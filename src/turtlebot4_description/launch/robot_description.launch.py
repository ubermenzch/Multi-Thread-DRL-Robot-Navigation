# Copyright 2021 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)


import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess


ARGUMENTS = [
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('use_sim_time', default_value='false',
                          choices=['true', 'false'],
                          description='use_sim_time'),
]


def generate_launch_description():
    # pkg_turtlebot4_description = get_package_share_directory('turtlebot4_description')

    # xacro_file = PathJoinSubstitution([pkg_turtlebot4_description,
    #                                    'urdf',
    #                                    LaunchConfiguration('model'),
    #                                    'turtlebot4.urdf.xacro'])
    # xacro_file = "/home/reinis/DRL_Navigation_ROS2/src/turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro"
    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[
    #         {'use_sim_time': LaunchConfiguration('use_sim_time')},
    #         {'robot_description': Command(['xacro', ' ', xacro_file, ' ', 'gazebo:=ignition'])},
    #     ],
    # )
    #
    # world_path = "/home/reinis/DRL_Navigation_ROS2/src/drl_navigation_ros2/assets/TD3.world"
    #
    # declare_simulator_cmd = DeclareLaunchArgument(
    #     name='headless',
    #     default_value='False',
    #     description='Whether to execute gzclient')
    #
    # declare_use_sim_time_cmd = DeclareLaunchArgument(
    #     name='use_sim_time',
    #     default_value='true',
    #     description='Use simulation (Gazebo) clock if true')
    #
    # declare_use_simulator_cmd = DeclareLaunchArgument(
    #     name='use_simulator',
    #     default_value='True',
    #     description='Whether to start the simulator')
    #
    # declare_world_cmd = DeclareLaunchArgument(
    #     name='world',
    #     default_value=world_path,
    #     description='Full path to the world model file to load')
    #
    #
    # # Create the launch description and populate
    # ld = LaunchDescription()
    #
    # # Declare the launch options
    # ld.add_action(declare_simulator_cmd)
    # ld.add_action(declare_use_sim_time_cmd)
    # ld.add_action(declare_use_simulator_cmd)
    # ld.add_action(declare_world_cmd)
    #
    # # Define LaunchDescription variable
    # ld = LaunchDescription(ARGUMENTS)
    # # Add nodes to LaunchDescription
    # ld.add_action(robot_state_publisher)
    # return ld
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_name = 'rrbot_description'
    world_file_name = 'empty.world'

    world = "/home/reinis/DRL_Navigation_ROS2/src/drl_navigation_ros2/assets/TD3.world"

    urdf = "/home/reinis/DRL_Navigation_ROS2/src/turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro"

    # xml = open(urdf, 'r').read()
    #
    # xml = xml.replace('"', '\\"')

    # swpan_args = '{name: \"my_robot\", xml: \"' + xml + '\" }'

    xacro_file = "/home/reinis/DRL_Navigation_ROS2/src/turtlebot4_description/urdf/lite/turtlebot4.urdf.xacro"
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': Command(['xacro', ' ', xacro_file, ' ', 'gazebo:=ignition'])},
        ],
    )

    ld = LaunchDescription([
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world],
            output='screen'),

        ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/gazebo',
                 'use_sim_time', use_sim_time],
            output='screen'),

        # ExecuteProcess(
        #     cmd=['ros2', 'service', 'call', '/spawn_entity',
        #          'gazebo_msgs/SpawnEntity', swpan_args],
        #     output='screen'),
    ])

    ld.add_action(robot_state_publisher)
    return ld
