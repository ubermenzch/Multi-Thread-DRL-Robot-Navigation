#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import math
from bisect import bisect_left
import os
import colorama
from colorama import Fore, Back, Style
from geometry_msgs.msg import Pose2D

def euler_from_quaternion(quaternion):
    """四元数转欧拉角"""
    x, y, z, w = quaternion
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = min(max(t2, -1.0), 1.0)
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

colorama.init()

class VisualCostmapSyncNode(Node):
    def __init__(self):
        super().__init__('visual_costmap_sync')
        
        # 参数配置
        self.odom_history_length = 2000
        self.odom_tolerance = 0.1
        self.output_rate = 10.0
        self.keep_dog_centered = True
        self.console_width = 80
        self.console_height = 40
        
        # 数据缓存
        self.odom_timestamps = []
        self.odom_data = []
        self.latest_costmap = None
        self.latest_costmap_time = None
        
        # 订阅和发布
        # /leg_odom2 /odometry/filtered
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/planner/costmap', self.costmap_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/leg_odom2', self.odom_callback, 10)
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, '/planner/reused_costmap', 10)
        
        # 调试可视化
        self.debug_visualization = True
        
        # 定时器
        self.timer = self.create_timer(1.0/self.output_rate, self.publish_latest_costmap)
        
        self.get_logger().info("Visual Costmap Sync Node initialized")

    def odom_callback(self, msg):
        current_time = self.ros_time_to_float(msg.header.stamp)
        
        if self.odom_timestamps and current_time <= self.odom_timestamps[-1]:
            self.get_logger().warn(f"Non-increasing odometry timestamp: {current_time} <= {self.odom_timestamps[-1]}")
            return
        
        insert_pos = bisect_left(self.odom_timestamps, current_time)
        if insert_pos != len(self.odom_timestamps):
            self.get_logger().error("Unexpected insert position")
            return
        
        self.odom_timestamps.append(current_time)
        self.odom_data.append(msg)
        
        if len(self.odom_timestamps) > self.odom_history_length:
            self.odom_timestamps.pop(0)
            self.odom_data.pop(0)
    
    def costmap_callback(self, msg):
        self.latest_costmap = msg
        self.latest_costmap_time = self.ros_time_to_float(msg.header.stamp)
        self.get_logger().debug(f"Received costmap at time: {self.latest_costmap_time}")
    
    def find_closest_odom_index(self, target_time):
        """更精确的时间戳匹配方法"""
        if not self.odom_timestamps:
            return None, float('inf')
            
        # 使用二分查找找到最近的时间戳
        pos = bisect_left(self.odom_timestamps, target_time)
        
        # 处理边界情况
        if pos == 0:
            return 0, abs(self.odom_timestamps[0] - target_time)
        if pos == len(self.odom_timestamps):
            return len(self.odom_timestamps)-1, abs(self.odom_timestamps[-1] - target_time)
        
        # 比较前后两个时间点
        before_diff = abs(self.odom_timestamps[pos-1] - target_time)
        after_diff = abs(self.odom_timestamps[pos] - target_time)
        
        return (pos-1, before_diff) if before_diff < after_diff else (pos, after_diff)

    def publish_latest_costmap(self):
        if self.latest_costmap is None or not self.odom_timestamps:
            return
        
        current_odom = self.odom_data[-1]
        current_time = self.odom_timestamps[-1]
        
        closest_idx, time_diff = self.find_closest_odom_index(self.latest_costmap_time)
        
        if time_diff > self.odom_tolerance:
            self.get_logger().warn(
                f"Time mismatch: costmap={self.latest_costmap_time:.3f}, "
                f"closest odom={self.odom_timestamps[closest_idx]:.3f}, "
                f"diff={time_diff:.3f}s")
            return
        
        closest_odom = self.odom_data[closest_idx]
        delta_pose = self.calculate_pose_difference(closest_odom, current_odom)
        
        # 调试输出位姿变化
        self.get_logger().debug(
            f"Delta pose - dx: {delta_pose.x:.2f}m, dy: {delta_pose.y:.2f}m, "
            f"dtheta: {math.degrees(delta_pose.theta):.2f}°")
        
        transformed_costmap = self.apply_transform_to_costmap(
            self.latest_costmap, delta_pose, self.keep_dog_centered)
        
        transformed_costmap.header.stamp = self.get_clock().now().to_msg()
        transformed_costmap.header.frame_id = "map"
        
        if self.debug_visualization:
            self.visualize_costmap(transformed_costmap)
        
        self.costmap_pub.publish(transformed_costmap)

    def calculate_pose_difference(self, odom1, odom2):
        """计算两个odom之间的相对位姿变化（修正坐标系）"""
        x1, y1 = odom1.pose.pose.position.x, odom1.pose.pose.position.y
        x2, y2 = odom2.pose.pose.position.x, odom2.pose.pose.position.y
        
        _, _, yaw1 = euler_from_quaternion([
            odom1.pose.pose.orientation.x,
            odom1.pose.pose.orientation.y,
            odom1.pose.pose.orientation.z,
            odom1.pose.pose.orientation.w])
        
        _, _, yaw2 = euler_from_quaternion([
            odom2.pose.pose.orientation.x,
            odom2.pose.pose.orientation.y,
            odom2.pose.pose.orientation.z,
            odom2.pose.pose.orientation.w])
        
        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_yaw = yaw2 - yaw1
        
        # 将位移转换到odom1的坐标系下
        rotated_dx = delta_x * math.cos(yaw1) + delta_y * math.sin(yaw1)
        rotated_dy = -delta_x * math.sin(yaw1) + delta_y * math.cos(yaw1)
        
        return Pose2D(x=rotated_dx, y=rotated_dy, theta=delta_yaw)

    def apply_transform_to_costmap(self, costmap, delta_pose, keep_centered=True):
        """应用位姿变换到costmap"""
        new_costmap = OccupancyGrid()
        new_costmap.header = costmap.header
        new_costmap.info = costmap.info
        
        resolution = costmap.info.resolution
        width = costmap.info.width
        height = costmap.info.height
        
        # 计算单元格位移量
        dx_cells = int(round(-delta_pose.x / resolution))
        dy_cells = int(round(-delta_pose.y / resolution)) #需要加负号吗
        dtheta = delta_pose.theta
        
        original_data = np.array(costmap.data, dtype=np.int8).reshape(height, width)
        new_data = np.full_like(original_data, -1)  # 初始化为未知
        
        center_x = width // 2
        center_y = height // 2
        
        # 创建变换矩阵
        cos_theta = math.cos(dtheta)
        sin_theta = math.sin(dtheta)
        
        # 对每个目标单元格计算原始位置
        for y_new in range(height):
            for x_new in range(width):
                # 相对于中心点的坐标
                rel_x = x_new - center_x
                rel_y = y_new - center_y
                
                # 应用旋转（逆变换）
                x_rot = rel_x * cos_theta - rel_y * sin_theta
                y_rot = rel_x * sin_theta + rel_y * cos_theta
                
                # 应用平移（逆变换）
                x_orig = round(x_rot + center_x - dx_cells)
                y_orig = round(y_rot + center_y - dy_cells)
                
                # 检查边界
                if 0 <= x_orig < width and 0 <= y_orig < height:
                    new_data[y_new, x_new] = original_data[y_orig, x_orig]
        
        new_costmap.data = new_data.flatten().tolist()
        return new_costmap

    def visualize_costmap(self, costmap):
        """可视化costmap（修正坐标系方向）"""
        os.system('cls' if os.name == 'nt' else 'clear')
        data = np.array(costmap.data).reshape(costmap.info.height, costmap.info.width)
        
        center_x = costmap.info.width // 2
        center_y = costmap.info.height // 2
        half_width = self.console_width // 2
        half_height = self.console_height // 2
        
        print(f"\n{Fore.CYAN}=== Costmap Visualization ==={Style.RESET_ALL}")
        print(f"Size: {costmap.info.width}x{costmap.info.height} | "
              f"Res: {costmap.info.resolution:.3f}m/cell | "
              f"Center: ({center_x}, {center_y})")
        
        # 注意Y轴方向（从顶部开始）
        for y in range(costmap.info.height-1,-1,-1):
            row_str = ""
            for x in range(max(0, center_x-half_width), min(costmap.info.width, center_x+half_width)):
                val = data[y, x]
                
                if x == center_x and y == center_y:
                    row_str += f"{Back.YELLOW}{Fore.BLACK}@{Style.RESET_ALL}"
                elif val < 0:
                    row_str += f"{Fore.WHITE}?"
                elif val == 0:
                    row_str += f"{Fore.GREEN}."
                elif val < 65:
                    row_str += f"{Fore.YELLOW}%"
                else:
                    row_str += f"{Fore.RED}#"
            
            print(row_str)
        
        print(f"\n{Fore.CYAN}Legend:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}.{Style.RESET_ALL} Free  {Fore.YELLOW}%{Style.RESET_ALL} Low obstacle  "
              f"{Fore.RED}#{Style.RESET_ALL} High obstacle  {Fore.WHITE}?{Style.RESET_ALL} Unknown  "
              f"{Back.YELLOW}{Fore.BLACK}@{Style.RESET_ALL} Robot")
        print(f"{Fore.CYAN}=== (Showing center {self.console_width}x{self.console_height} area) ==={Style.RESET_ALL}")

    def ros_time_to_float(self, time_msg):
        return float(time_msg.sec) + float(time_msg.nanosec) / 1e9

def main(args=None):
    rclpy.init(args=args)
    node = VisualCostmapSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exiting...{Style.RESET_ALL}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()