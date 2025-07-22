import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros.actions

# 核心修改：添加可配置的参数化启动
def generate_multi_agent_launch(agent_id):
    # 为每个智能体生成唯一的命名空间
    namespace = f"agent_{agent_id}"
    
    # 计算唯一端口（避免冲突）
    base_port = 11311  # 默认ROS Master端口
    ros_master_port = base_port + agent_id
    gzserver_port = 11345 + agent_id * 2  # Gazebo服务器端口
    gzclient_port = gzserver_port + 1    # Gazebo客户端端口
    
    # 设置环境变量（关键修改）
    os.environ["ROS_MASTER_URI"] = f"http://localhost:{ros_master_port}"
    os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{gzserver_port}"
    
    # 动态配置启动参数
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    pause = LaunchConfiguration("pause", default="false")
    
    # 命名空间感知的世界文件
    world_file_name = f"turtlebot3_drl_{namespace}/{os.environ['TURTLEBOT3_MODEL']}.model"
    world = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), 
        "worlds", 
        world_file_name
    )
    
    # 确保世界文件存在（不存在则创建）
    if not os.path.exists(world):
        create_world_copy(namespace)
    
    launch_file_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    
    return LaunchDescription([
        # 声明命名空间参数
        DeclareLaunchArgument(
            'namespace', default_value=namespace,
            description='Top-level namespace'
        ),
        
        # Gazebo服务器（headless模式）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
            ),
            launch_arguments={
                "world": world,
                "pause": pause,
                "verbose": "false",  # 减少日志输出
                "physics": "ode",    # 优化性能
                "extra_gazebo_args": f"-s {gzserver_port}"  # 指定Gazebo端口
            }.items()
        ),
        
        # Gazebo客户端（仅主智能体显示）
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
            ),
            # 仅第一个智能体启动GUI
            condition=launch.conditions.IfCondition(
                str(agent_id == 0)
            ),
            launch_arguments={
                "gui_args": f"--gazebo-server-port={gzserver_port}"
            }.items()
        ),
        
        # 命名空间感知的机器人状态发布器
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                launch_file_dir, "/robot_state_publisher.launch.py"
            ]),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "namespace": namespace  # 关键：传递命名空间
            }.items()
        ),
        
        # 机器人控制器（使用命名空间）
        launch_ros.actions.Node(
            package='robot_controller',
            namespace=namespace,
            executable='controller_node',
            name='controller'
        )
    ])

# 创建世界文件的副本（确保唯一性）
def create_world_copy(namespace):
    src_world = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), 
        "worlds", 
        f"turtlebot3_drl/{os.environ['TURTLEBOT3_MODEL']}.model"
    )
    
    dst_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), 
        "worlds", 
        f"turtlebot3_drl_{namespace}"
    )
    
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_world, dst_dir)
    
    # 添加微小差异（可选）
    with open(os.path.join(dst_dir, f"{os.environ['TURTLEBOT3_MODEL']}.model"), "a") as f:
        f.write(f"\n<!-- Unique world for agent {namespace} -->")