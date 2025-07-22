import time
import rclpy
from ros_nodes import (
    ScanSubscriber,
    OdomSubscriber,
    ResetWorldClient,
    SetModelStateClient,
    CmdVelPublisher,
    MarkerPublisher,
    PhysicsClient,
    SensorSubscriber,
)
import numpy as np
from geometry_msgs.msg import Pose, Twist
from squaternion import Quaternion


class ROS_env:
    def __init__(
        self,
        agent_id=0,
        init_target_distance=2.0,
        target_dist_increase=0.001,
        max_target_dist=8.0,
        target_reached_delta=0.5,
        collision_delta=0.4,
        args=None,
    ):
    
        self.BASE_PORT = 10000
        # 为每个智能体设置唯一命名空间
        self.namespace = f"agent_{agent_id}"
        self.agent_id = agent_id
        
        # 设置独立ROS Master和Gazebo端口
        self.ros_port=-1
        self.gzserver_port=-1
        self.set_isolated_ports()
        
        # 启动独立的Gazebo实例
        self.launch_gazebo_instance()
        
        # 初始化ROS节点（带命名空间）
        rclpy.init(args=args, namespace=self.namespace)

        """创建命名空间感知的ROS组件"""
        # 所有话题和服务添加命名空间前缀
        self.cmd_vel_publisher = CmdVelPublisher(namespace=self.namespace)
        self.scan_subscriber = ScanSubscriber(namespace=self.namespace)
        self.odom_subscriber = OdomSubscriber(namespace=self.namespace)
        self.robot_state_publisher = SetModelStateClient(namespace=self.namespace)
        self.world_reset = ResetWorldClient(namespace=self.namespace)
        self.physics_client = PhysicsClient(namespace=self.namespace)
        self.publish_target = MarkerPublisher(namespace=self.namespace)
        self.sensor_subscriber = SensorSubscriber(namespace=self.namespace)
        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        self.target = self.set_target_position([0.0, 0.0])



    def set_isolated_ports(self):
        """为每个智能体设置独立端口"""
        self.ros_port = self.BASE_PORT + self.agent_id
        self.gzserver_port = self.BASE_PORT + 1000 + self.agent_id * 2

        # ROS Master端口
        os.environ["ROS_MASTER_URI"] = f"http://localhost:{self.ros_port}"
        
        # Gazebo服务器端口
        os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{self.gzserver_port}"
        
        # Gazebo客户端端口
        self.gzclient_port = self.gzserver_port + 1
    
    def launch_gazebo_instance(self):
        """启动独立的Gazebo实例"""
        # 生成唯一的世界文件副本
        world_file = self.create_unique_world()
        
        # 启动命令
        cmd = [
            "ros2", "launch", "turtlebot3_gazebo", "turtlebot3_world.launch.py",
            f"world:={world_file}",
            f"gzserver_port:={self.gzserver_port}",
            f"gzclient_port:={self.gzclient_port}",
            f"namespace:={self.namespace}",
            "gui:=false" if self.agent_id > 0 else "gui:=true"
        ]
        
        # 启动Gazebo进程
        self.gazebo_process = subprocess.Popen(cmd)
        time.sleep(5)  # 等待初始化
    
    def create_unique_world(self):
        """创建轻微差异化的世界文件副本"""
        # 原始世界文件路径
        src_world = os.path.join(
            get_package_share_directory("turtlebot3_gazebo"),
            "worlds",
            "turtlebot3_world.world"
        )
        
        # 目标世界文件路径
        dst_dir = os.path.join("/tmp", "gazebo_worlds", self.namespace)
        os.makedirs(dst_dir, exist_ok=True)
        dst_world = os.path.join(dst_dir, "turtlebot3_world.world")
        
        # 复制并修改世界文件
        shutil.copy(src_world, dst_world)
        
        # 添加微小差异（可选）
        with open(dst_world, "a") as f:
            f.write(f"\n<!-- Unique world for agent {self.agent_id} -->")
        
        return dst_world


    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        self.physics_client.unpause_physics()
        time.sleep(0.1)
        rclpy.spin_once(self.sensor_subscriber)
        self.physics_client.pause_physics()

        (
            latest_scan,
            latest_position,
            latest_orientation,
        ) = self.sensor_subscriber.get_latest_sensor()

        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(self):
        self.world_reset.reset_world()
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )

        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.set_positions()

        self.publish_target.publish(self.target[0], self.target[1])

        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    def eval(self, scenario):
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)

        self.target = [scenario[-1].x, scenario[-1].y]
        self.publish_target.publish(self.target[0], self.target[1])

        for element in scenario[:-1]:
            self.set_position(element.name, element.x, element.y, element.angle)

        self.physics_client.unpause_physics()
        time.sleep(1)
        latest_scan, distance, cos, sin, _, _, a, reward = self.step(
            lin_velocity=0.0, ang_velocity=0.0
        )
        return latest_scan, distance, cos, sin, False, False, a, reward

    def set_target_position(self, robot_position):
        pos = False
        while not pos:
            x = np.clip(
                robot_position[0]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -4.0,
                4.0,
            )
            y = np.clip(
                robot_position[1]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -4.0,
                4.0,
            )
            pos = self.check_position(x, y, 1.2)
        self.element_positions.append([x, y])
        return [x, y]

    def set_random_position(self, name):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.element_positions.append([x, y])
        self.set_position(name, x, y, angle)

    def set_robot_position(self):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.set_position("turtlebot3_waffle", x, y, angle)
        return x, y

    def set_position(self, name, x, y, angle):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        self.robot_state_publisher.set_state(name, pose)
        rclpy.spin_once(self.robot_state_publisher)

    def set_positions(self):
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            self.set_random_position(name)

        robot_position = self.set_robot_position()
        self.target = self.set_target_position(robot_position)

    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

    def check_collision(self, laser_scan):
        if min(laser_scan) < self.collision_delta:
            return True
        return False

    def check_target(self, distance, collision):
        if distance < self.target_reached_delta and not collision:
            self.target_dist += self.target_dist_increase
            if self.target_dist > self.max_target_dist:
                self.target_dist = self.max_target_dist
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        # Calculate robot heading from odometry data
        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        return distance, cos, sin, angle

    @staticmethod
    def get_reward(goal, collision, action, laser_scan):
        if goal:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
