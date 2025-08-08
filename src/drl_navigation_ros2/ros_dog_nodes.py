import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import json
from collections import deque
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import OccupancyGrid
import utils
from tf_transformations import euler_from_quaternion
import math


class DogCmdVelPublisher(Node):
    def __init__(self,robot_type):
        super().__init__("cmd_vel_publisher")
        self.robot_type = robot_type
        topic_name = "error"
        if self.robot_type == "unitree":
            topic_name = "/catch_turtle/ctrl_instruct"
        elif self.robot_type == "ysc":
            topic_name = "/catch_turtle/ctrl_instruct"
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
            '/navigation_control2',  # 话题名称
            self.navigation_control_callback,  # 回调函数
            1  # 队列大小
        )
        self.path_points = deque()
        self.total_point_count = 0  # 总路径点数量

    def get_next_target_gnss(self):
        #path_points不为空
        if self.has_target_path():
            return self.path_points.popleft()
        else:
            return None

    def navigation_control_callback(self, msg):
        if self.has_target_path(): # 如果当前路径规划未完成（path_points不为空），则不接受新的路径规划
            return

        # 解析JSON字符串为Python字典
        path_data = json.loads(msg.data)
        # 提取基本字段
        action = path_data['action'] # 1-开始执行室外自主出行 13-暂停室外自主出行 0-停止室外自主出行
        if action == 1:
            self.path_points = deque(path_data['points'])
            self.total_point_count = len(self.path_points)
    
    def has_target_path(self):
        return bool(self.path_points)

class GnssSubscriber(Node):
    def __init__(self,robot_type):
        super().__init__('gnss_subscriber')
        # 创建订阅者
        self.robot_type = robot_type
        if self.robot_type == "ysc":
            self.subscriber_ = self.create_subscription(NavSatFix,
            '/fix',
            self.fix_callback,
            1
            )
        elif self.robot_type == "unitree":
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
        if msg.status.status == NavSatStatus.STATUS_GBAS_FIX:# 确认GNSS数据有效
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
        if self.robot_type == "ysc":
            self.subscriber_ = self.create_subscription(Odometry,
            "/leg_odom2",
            self.odom_callback,
            1
            )
        elif self.robot_type == "unitree":
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
        # 转换为[-π, π]范围的弧度
        self.heading = math.atan2(math.sin(yaw), math.cos(yaw))
        self.orientation = yaw # 度数，朝正东为0度，逆时针为正
        self.is_odom_valid = True

class ReusedCostMapSubscriber(Node):
    # 角度采样单位angle_resolution为1.5度（转换为弧度为0.02617993878）
    def __init__(self,angle_resolution=(1/180)*math.pi,
                 scan_range=5.0,
                 output_rate=10.0,
                 bin_num=20,
                 neglect_angle=49.0):
        super().__init__('reused_costmap_subscriber')
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
        # 前方180度范围内分成多少个分区
        self.bin_num = bin_num 
        # 以hz为单位，指复用地图下发的频率，这个output_rate的值要和map_reuse.py中的output_rate值相等
        self.output_rate = output_rate
        self.reused_costmap = None
        self.is_reused_costmap_valid = False
        self.timer = None
        self.neglect_angle = neglect_angle

    def reused_cost_map_callback(self, reused_costmap: OccupancyGrid):
        # 复用地图的有效持续时间
        duration = (1.0/self.output_rate)*2
        # 获取ROS系统当前时间戳
        now = self.get_clock().now()
        current_stamp = now.to_msg()
        # 如果复用地图有效持续时间已过则不采用
        if utils.get_time_diff_ns(current_stamp,reused_costmap.header.stamp) < duration*1e9:
            self.reused_costmap = reused_costmap
            self.is_reused_costmap_valid = True
            # 接收到新下发的reused_costmap后
            if self.timer == None: # 若定时器未创建则创建定时器，用以定时失效地图
                self.timer = self.create_timer(duration,self.timer_callback)
            else:
                # 若定时器已创建则重置定时器
                self.timer.cancel()
                self.timer.reset()
        else:
            self.is_reused_costmap_valid = False
    
    def timer_callback(self):
        # 到时间则自动失效reused_costmap
        self.is_reused_costmap_valid = False

    # 将坐标转换为栅格图data索引
    # data数据从栅格图的左下角开始，向右为x轴正方向，向上为y轴正方向，x对应列从左向右递增，y对应行从下向上递增，
    # 注意：栅格图的左下角坐标为(0,0)，存储在data中的顺序为(0,0) -> (1,0) -> (2,0) -> ... -> (0,1) -> (1,1) -> ...
    def x_y_to_index(self, x, y,edge_length):
        index_x = int(math.floor(x))
        index_y = int(math.floor(y))
        
        return index_x+index_y*edge_length

    # 从地图中扫描得到前方距离最近的obs障碍物
    def get_obs_min_distance(self):
        if not self.is_reused_costmap_valid:
            return [0] * self.bin_num # 如果复用地图无效，则返回一个全0的列表（全是障碍物）
        # 极坐标扫描：角度从0到π，径向按地图分辨率步进
        neglect_angle_radian = (self.neglect_angle * math.pi) / 180.0 # 将角度转换为弧度
        angle = -math.pi/2 + neglect_angle_radian
        bin_angle = (math.pi-(neglect_angle_radian*2))/self.bin_num
        resolution = self.reused_costmap.info.resolution
        edge_length = self.reused_costmap.info.height
        obs_min_distance = []
        max_angle = math.pi/2 - neglect_angle_radian
        while max_angle - angle > 1e-6: # 角度从neglect_angle_radian到π-neglect_angle_radian:
            # print(f"Scanning at angle {angle} rad. Max Angle: {max_angle}.")
            scan_angle = 0
            bin_obs = [self.scan_range]
            while scan_angle <= bin_angle:
                dist = 0
                cos_value = math.cos(angle+scan_angle)
                sin_value = math.sin(angle+scan_angle)
                while dist <= self.scan_range:
                    # 计算当前扫描点的世界坐标
                    x = dist * cos_value
                    y = dist * sin_value
                    # print(f"Scanning at angle {angle+scan_angle} rad, distance {dist} m, position ({x}, {y})")
                    # 转换到网格索引
                    absolute_x = x - self.reused_costmap.info.origin.position.x
                    absolute_y = y - self.reused_costmap.info.origin.position.y
                    index_x = int(math.floor(absolute_x / resolution))
                    index_y = int(math.floor(absolute_y / resolution))
                    # 检查索引是否有效
                    if (0 <= index_x < edge_length) and (0 <= index_y < edge_length):
                        # 检查是否为障碍物 (值 100)
                        if not self.reused_costmap.data[self.x_y_to_index(index_x,index_y,edge_length)] == 0:
                            # print("index_x = ",index_x,"index_y = ",index_y)
                            bin_obs.append(dist)
                            break # 标记并跳出当前径向扫描
                    dist += resolution  # 移动到下一个径向点
                scan_angle += self.angle_resolution
            obs_min_distance.append(min(bin_obs))
            angle += bin_angle  # 移动到下一个角度

        return obs_min_distance
