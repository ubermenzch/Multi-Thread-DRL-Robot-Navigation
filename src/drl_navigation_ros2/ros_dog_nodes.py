import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import Point
from std_msgs.msg import String
import json
from collections import deque
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import OccupancyGrid
import utils
from tf_transformations import euler_from_quaternion

SEVERITY = LoggingSeverity.ERROR

class DogCmdVelPublisher(Node):
    def __init__(self,robot_type):
        super().__init__("cmd_vel_publisher")
        self.robot_type = robot_type
        topic_name = "error"
        match self.robot_type:
            case "unitree":
                topic_name = "utlidar/robot_odom"
                break 
            case "ysc":
                topic_name = "/catch_turtle/ctrl_instruct"
                break
        self.publisher_ = self.create_publisher(Twist, topic_name, 1)

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        twist_msg = Twist()
        # Set linear and angular velocities
        twist_msg.linear.x = float(linear_velocity)  # Example linear velocity (m/s)
        if self.robot_type == "ysc": # 如果机器狗是云深处则角速度取反
            angular_velocity = - angular_velocity
        twist_msg.angular.z = float(angular_velocity)  # Example angular velocity (rad/s)
        self.publisher_.publish(twist_msg)


class TargetPathSubscriber(Node):
    def __init__(self):
        super().__init__("target_path_subscriber")
        # 创建路径信息（字符串类型）的订阅器
        self.subscriber_ = self.create_subscription(
            String,  # 消息类型为 std_msgs/String
            '/navigation_control',  # 话题名称
            self.navigation_control_callback,  # 回调函数
            1  # 队列大小
        )
        self.path_points = deque()

    def get_next_target_gnss(self):
        #path_points不为空
        if self.has_target_path():
            return self.path_points.popleft()
        else:
            return None

    def navigation_control_callback(self, msg):
        json_str = msg.data
        # 解析JSON字符串为Python字典
        path_data = json.loads(json_str)
        # 提取基本字段
        action = path_data['action'] # 1-开始执行室外自主出行 13-暂停室外自主出行 0-停止室外自主出行
        if not self.has_target_path() and action == 1: # 在当前路径规划未完成前（path_points不为空），不会接受新的路径规划
            self.path_points = deque(path_data['points'])
    
    def has_target_path(self):
        return self.path_points

class GnssSubscriber(Node):
    def __init__(self,robot_type):
        super().__init__('gnss_subscriber')
        # 创建订阅者
        self.robot_type = robot_type
        self.subscriber_ = None
        match self.robot_type:
            case "ysc":
                self.subscriber_ = self.create_subscription(NavSatFix,
                '/fix',
                self.fix_callback,
                1
                )
            case "unitree":
                self.subscriber_ = self.create_subscription(String,
                '/gnss',
                self.gnss_callback,
                1
                )
        self.is_gnss_valid=False
        self.gnss = {'latitude':-1.0,'longitude':-1.0}
    
    def fix_callback(self, msg):
        # STATUS_NO_FIX	-1	设备无法定位（无任何定位信息）
        # STATUS_FIX	0	设备能够定位（标准定位精度）
        # STATUS_SBAS_FIX	1	设备使用 SBAS（卫星增强系统）定位
        # STATUS_GBAS_FIX	2	设备使用 GBAS（地面增强系统）定位
        if msg.status.status != NavSatStatus.STATUS_NO_FIX:# 确认GNSS数据有效
            self.is_gnss_valid=True
            self.gnss['latitude'] = msg.latitude # 维度
            self.gnss['longitude'] = msg.longitude # 经度
        else:
            self.is_gnss_valid=False
    
    def gnss_callback(self, msg):
        # 解析JSON数据
        data = json.loads(msg.data)
            
        # 提取字段值
        fixed = data['fixed']
        if fixed != 0:
            self.gnss['latitude'], self.gnss['longitude'] = wgs2gcj(latitude = data['latitude'],
            longitude = data['longitude']
            )
            self.is_gnss_valid = True
        else:
            self.is_gnss_valid = False

class OdomSubscriber(Node):
    def __init__(self,robot_type):
        super().__init__("odom_subscriber")
        self.robot_type = robot_type
        match self.robot_type:
            case "ysc":
                self.subscriber_ = self.create_subscription(Odometry,
                "/leg_odom2",
                self.odom_callback,
                1
                )
            case "unitree":
                self.subscriber_ = self.create_subscription(Odometry,
                "/utlidar/robot_odom",
                self.odom_callback,
                1
                )
        self.is_odom_valid = False
        self.odom = None
        # 智能体朝向
        self.heading = None #弧度表示
        self.orientation = None #度数表示
        
    def odom_callback(self, msg):
        # 更新里程计数据
        self.odom = msg
        # 获取姿态四元数
        orientation = msg.pose.pose.orientation
        # 将四元数转换为欧拉角 (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]) 
        # 规范化角度到 [-π, π] 范围
        self.heading = math.atan2(math.sin(yaw), math.cos(yaw))
        # 转换为角度值 (-180° to 180°)，朝东为0度
        self.orientation = self.heading * 180 / math.pi

        self.is_odom_valid = True

class ReusedCostMapSubcriber(Node):
    # 角度采样单位angle_resolution为1.5度（转换为弧度为0.02617993878）
    def __init__(self,angle_resolution=0.02617993878,scan_range=5.0,output_rate=10.0):
        super.__init__('reused_costmap_subscriber')
        self.subscriber_ = self.create_subscription(
            OccupancyGrid,
            "/planner/reused_costmap",
            self.reused_cost_map_callback,
            1
        )
        self.OBS_VALUE = 100 # 栅格图中代表障碍物的值
        self.UNKNOWN_VALUE = -1 # 栅格图中代表未知区域的值
        # 一个分区bin内的角度采样单位
        self.angle_resolution = angle_resolution 
        # 多少米内的障碍物为需要被考虑的障碍物，也就是障碍物探测的有效距离
        self.scan_range = scan_range
        self.reused_costmap = None
        self.is_reused_costmap_valid = False
        self.output_rate = output_rate # 以hz为单位，指复用地图下发的频率，这个output_rate的值要和map_reuse.py中的output_rate值相等
        self.timer = None
    
    def reused_cost_map_callback(self, reused_costmap: OccupancyGrid):
        # 复用地图的有效持续时间
        duration = 1.0/self.output_rate
        # 获取ROS系统当前时间戳
        now = self.get_clock().now()
        current_stamp = now.to_msg()
        # 如果复用地图有效持续时间已过则不采用
        if get_time_diff_ns(current_stamp,reused_costmap.header.stamp) < duration*1e9:
            self.reused_costmap = reused_costmap
            self.is_reused_costmap_valid = True
            # 接收到新下发的reused_costmap后
            if self.timer == None: # 若定时器未创建则创建定时器，用以定时失效地图
                self.timer = self.create_timer(duration,timer_callback)
            else:
                # 若定时器已创建则重置定时器
                self.timer.cancel()
                self.timer.reset()
        else:
            self.reused_costmap = None
            self.is_reused_costmap_valid = False
    
    def timer_callback(self):
        # 到时间则自动失效reused_costmap
        self.reused_costmap = None
        self.is_reused_costmap_valid = False

    # 从地图中扫描得到前方距离最近的obs障碍物
    def get_obs_min_distance(self):
        # 极坐标扫描：角度从0到π，径向按地图分辨率步进
        angle = 0
        bin_angle = math.pi/self.bin_num
        obs_min_distance = []
        while angle <= math.pi:
            scan_angle = 0
            dist = 0
            bin_obs = [self.scan_range]
            while scan_angle <= bin_angle:
                cos_value = math.cos(angle+scan_angle)
                sin_value = math.sin(angle+scan_angle)
                while dist <= self.scan_range:
                    # 计算当前扫描点的世界坐标
                    x = dist * cos_value
                    y = dist * sin_value
                    # 转换到网格索引
                    index_x = int(math.floor((x - self.reused_costmap.info.origin.position.x) / self.reused_costmap.info.resolution))
                    index_y = int(math.floor((y - self.reused_costmap.info.origin.position.y) / self.reused_costmap.info.resolution))
                    # 检查索引是否有效
                    if (0 <= index_x < self.reused_costmap.info.width) and (0 <= index_y < self.reused_costmap.info.height):
                        # 检查是否为障碍物 (值 100)
                        if self.reused_costmap.data[index_y * self.reused_costmap.info.width + index_x] in [self.OBS_VALUE,self.UNKNOWN_VALUE]:
                            bin_obs.append(dist)
                            break # 标记并跳出当前径向扫描
                    dist += self.reused_costmap.info.resolution  # 移动到下一个径向点
                scan_angle += self.angle_resolution
                obs_min_distance.append(min(bin_obs))

            angle += self.bin_angle  # 移动到下一个角度

        return obs_min_distance

class ScanSubscriber(Node):
    def __init__(self):
        super().__init__("scan_subscriber")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.listener_callback, 1
        )
        self.latest_scan = None

    def listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]

    def get_latest_scan(self):
        return self.latest_scan

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__("sensor_subscriber")
        self.get_logger().set_level(SEVERITY)
        self.subscriber_ = self.create_subscription(
            LaserScan, "scan", self.scan_listener_callback, 1
        )
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.odom_listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None
        self.latest_scan = None

    def scan_listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]

    def odom_listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation

    def get_latest_sensor(self):
        # print(self.latest_scan, self.latest_position, self.latest_heading)
        return self.latest_scan, self.latest_position, self.latest_heading


class ResetWorldClient(Node):
    def __init__(self):
        super().__init__("reset_world_client")
        self.get_logger().set_level(SEVERITY)
        self.reset_client = self.create_client(Empty, "/reset_world")
        self.wait_for_service(self.reset_client, "reset_world")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def reset_world(self):
        self.get_logger().info("Calling /gazebo/reset_world service...")
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("World reset successfully.")
        else:
            self.get_logger().error(f"Failed to reset world: {future.exception()}")


class PhysicsClient(Node):
    def __init__(self):
        super().__init__("physics_client")
        self.get_logger().set_level(SEVERITY)
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.pause_client = self.create_client(Empty, "/pause_physics")

        self.wait_for_service(self.unpause_client, "unpause_physics")
        self.wait_for_service(self.pause_client, "pause_physics")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def pause_physics(self):
        self.get_logger().info("Calling /gazebo/pause_physics service...")
        request = Empty.Request()
        future = self.pause_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics paused successfully.")
        else:
            self.get_logger().error(f"Failed to pause physics: {future.exception()}")

    def unpause_physics(self):
        self.get_logger().info("Calling /gazebo/unpause_physics service...")
        request = Empty.Request()
        future = self.unpause_client.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics unpaused successfully.")
        else:
            self.get_logger().error(f"Failed to unpause physics: {future.exception()}")


class SetModelStateClient(Node):
    def __init__(self):
        super().__init__("set_entity_state_client")
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        #print("SetModelStateClient::create_client Done")
        while not self.client.wait_for_service(timeout_sec=1.0):
            print("SetModelStateClient::wait_for_service")
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose):
        self.request.state.name = name
        self.request.state.pose = new_pose
        self.future = self.client.call_async(self.request)

class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_publisher")
        self.get_logger().set_level(SEVERITY)
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 1)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        twist_msg = Twist()
        # Set linear and angular velocities
        twist_msg.linear.x = float(linear_velocity)  # Example linear velocity (m/s)
        twist_msg.angular.z = float(
            angular_velocity
        )  # Example angular velocity (rad/s)
        self.publisher_.publish(twist_msg)



class MarkerPublisher(Node):
    def __init__(self):
        super().__init__("marker_publisher")
        self.get_logger().set_level(SEVERITY)
        self.publisher = self.create_publisher(Marker, "visualization_marker", 1)

    def publish(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.publisher.publish(marker)
        self.get_logger().info("Publishing Marker")


def run_scan(args=None):
    rclpy.init()
    reading_laser = ScanSubscriber()
    reading_laser.get_logger().info("Hello friend!")
    rclpy.spin(reading_laser)

    reading_laser.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    run_scan()
