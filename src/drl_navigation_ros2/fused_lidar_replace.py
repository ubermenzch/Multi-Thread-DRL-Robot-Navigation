#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose2D
import numpy as np
import math
from bisect import bisect_left
import os
import colorama
from colorama import Fore, Back, Style
import struct # 用于解析点云

# --- 辅助函数 ---

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

def read_points(cloud_msg: PointCloud2):
    """手动解析PointCloud2消息，生成(x, y, z)点。"""
    point_step = cloud_msg.point_step
    data = cloud_msg.data
    x_offset, y_offset, z_offset = -1, -1, -1
    for field in cloud_msg.fields:
        if field.name == 'x': x_offset = field.offset
        elif field.name == 'y': y_offset = field.offset
        elif field.name == 'z': z_offset = field.offset
    
    if x_offset == -1 or y_offset == -1 or z_offset == -1:
        return

    byte_order = '<' if not cloud_msg.is_bigendian else '>'
    num_points = cloud_msg.width * cloud_msg.height
    for i in range(num_points):
        point_start_index = i * point_step
        x, = struct.unpack_from(f'{byte_order}f', data, point_start_index + x_offset)
        y, = struct.unpack_from(f'{byte_order}f', data, point_start_index + y_offset)
        z, = struct.unpack_from(f'{byte_order}f', data, point_start_index + z_offset)
        yield (x, y, z)

colorama.init()

class FusedCostmapNode(Node):
    def __init__(self):
        super().__init__('fused_costmap_node')
        self.max_secs = 0.0
        # --- 参数配置 ---
        # 复用地图相关参数
        self.odom_history_length = 2000
        self.odom_tolerance = 0.1
        self.output_rate = 10.0
        # 点云融合相关参数
        self.fusion_radius = 2.0  # 只融合2米内的点云
        self.point_min_height = -0.3 # 点云高度过滤下限
        self.point_max_height = 0.8  # 点云高度过滤上限
        
        # 可视化参数
        self.debug_visualization = False
        self.console_width = 80
        self.console_height = 40
        
        # --- 数据缓存 ---
        self.odom_timestamps = []
        self.odom_data = []
        self.latest_costmap = None
        self.latest_costmap_time = None
        self.latest_pointcloud = None # 新增：缓存最新的点云数据
        
        # --- 订阅和发布 ---
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/planner/costmap', self.costmap_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/leg_odom2', self.odom_callback, 10)
        # 新增：订阅点云话题
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar_points', self.pointcloud_callback, 10)
        
        # 发布融合后的地图
        self.fused_costmap_pub = self.create_publisher(
            OccupancyGrid, '/planner/reused_costmap', 10)
        
        # --- 定时器 ---
        self.timer = self.create_timer(1.0/self.output_rate, self.publish_fused_costmap)
        
        self.get_logger().info("Fused Costmap Node initialized")
        self.get_logger().info(f"Replacing original costmap with radar data within {self.fusion_radius}m radius.")

    def odom_callback(self, msg):
        current_time = self.ros_time_to_float(msg.header.stamp)
        if self.odom_timestamps and current_time <= self.odom_timestamps[-1]: return
        self.odom_timestamps.append(current_time)
        self.odom_data.append(msg)
        if len(self.odom_timestamps) > self.odom_history_length:
            self.odom_timestamps.pop(0)
            self.odom_data.pop(0)
    
    def costmap_callback(self, msg):
        self.latest_costmap = msg
        self.latest_costmap_time = self.ros_time_to_float(msg.header.stamp)

    def pointcloud_callback(self, msg):
        """新增：点云回调函数，仅缓存最新消息"""
        self.latest_pointcloud = msg

    def find_closest_odom_index(self, target_time):
        """更精确的时间戳匹配方法"""
        if not self.odom_timestamps: return None, float('inf')
        pos = bisect_left(self.odom_timestamps, target_time)
        if pos == 0: return 0, abs(self.odom_timestamps[0] - target_time)
        if pos == len(self.odom_timestamps): return len(self.odom_timestamps)-1, abs(self.odom_timestamps[-1] - target_time)
        before_diff = abs(self.odom_timestamps[pos-1] - target_time)
        after_diff = abs(self.odom_timestamps[pos] - target_time)
        return (pos-1, before_diff) if before_diff < after_diff else (pos, after_diff)

    def publish_fused_costmap(self):
        """
        核心函数：先生成复用地图，然后融合点云，最后发布。
        """
        if self.latest_costmap is None or not self.odom_timestamps:
            return
        
        current_odom = self.odom_data[-1]
        closest_idx, time_diff = self.find_closest_odom_index(self.latest_costmap_time)
        
        if time_diff > self.odom_tolerance:
            self.get_logger().warn(
                f"Time mismatch for costmap reuse: diff={time_diff:.3f}s > {self.odom_tolerance}s")
            return
        import time
        start_time = time.perf_counter()
        # 1. 计算位姿差并生成"复用"的代价地图
        closest_odom = self.odom_data[closest_idx]
        delta_pose = self.calculate_pose_difference(closest_odom, current_odom)
        transformed_costmap = self.apply_transform_to_costmap(self.latest_costmap, delta_pose)
        
        # 2. 完全替换2m范围内的雷达栅格图
        if self.latest_pointcloud:
            final_costmap = self.replace_with_radar_costmap(transformed_costmap, self.latest_pointcloud)
        else:
            final_costmap = transformed_costmap

        # 3. 发布最终地图
        final_costmap.header.stamp = self.get_clock().now().to_msg()
        final_costmap.header.frame_id = "map" # 最终地图还是在map坐标系下
        
        if self.debug_visualization:
            self.visualize_costmap(final_costmap)
        end_time = time.perf_counter()
        duration = end_time - start_time
        self.max_secs=max(self.max_secs, duration)
        print(f"函数的运行时间为: {duration:.6f} 秒")
        print(f"最大运行时间为: {self.max_secs:.6f} 秒")
        self.fused_costmap_pub.publish(final_costmap)

    def replace_with_radar_costmap(self, base_costmap, cloud_msg):
        """
        核心修改：在2m范围内直接用雷达栅格图替换原始地图
        """
        self.get_logger().debug("Replacing costmap with radar data within 2m radius...")
        info = base_costmap.info
        resolution = round(info.resolution, 2)
        width = info.width
        height = info.height
        
        # 创建雷达栅格图，初始值为0 (自由)
        # print(f"基础地图大小: {width}x{height}, 分辨率: {resolution}m/栅格")
        radar_edge = int((self.fusion_radius/resolution)*2)
        radar_grid = np.zeros((radar_edge, radar_edge), dtype=np.int8)
        radar_bias = info.height/2 - radar_edge/2
        # print(f"雷达栅格图大小: {radar_edge}x{radar_edge}, 偏移: {radar_bias}") 

        origin_x = - (radar_edge / 2.0) * resolution
        origin_y = - (radar_edge / 2.0) * resolution
        
        # 收集所有在2m范围内的点
        for point in read_points(cloud_msg):
            px, py, pz = point
            
            # 计算点到机器人中心的距离
            # distance = math.sqrt(px**2 + py**2)
            # if distance > self.fusion_radius:
            #     continue  # 跳过2m范围外的点

            if abs(px) > self.fusion_radius or abs(py) > self.fusion_radius:
                continue  # 跳过2m范围外的点
            
            
            if self.point_min_height <= pz <= self.point_max_height:
                # 转换为栅格坐标
                grid_x = int((px - origin_x) / resolution)
                grid_y = int((py - origin_y) / resolution)
                if 0 <= grid_x < radar_edge and 0 <= grid_y < radar_edge:
                    # 标记为障碍物
                    # print(f"Point ({px:.2f}, {py:.2f}, {pz:.2f}) mapped to grid ({grid_x}, {grid_y})")
                    radar_grid[grid_y, grid_x] = 100  # 标记为障碍物
                # for dy in [-1, 0, 1]:
                #     for dx in [-1, 0, 1]:
                #         nx, ny = grid_x + dx, grid_y + dy
                #         if 0 <= nx < radar_edge and 0 <= ny < radar_edge:
                #             radar_grid[ny, nx] = 100

        # 标记雷达感知范围内的区域
        for y in range(radar_edge):
            for x in range(radar_edge):
                base_costmap.data[int((y+radar_bias)*width+x+radar_bias)] = radar_grid[y, x]

        #特殊处理：确保机器人当前位置始终被标记为自由
        center_x = width // 2
        center_y = height // 2
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = center_x + dx, center_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    base_costmap.data[ny*width+nx] = 0

        # 可视化雷达感知范围的边界（调试用）
        if self.debug_visualization:
            # 创建基础地图的副本用于可视化
            grid_data = np.array(base_costmap.data, dtype=np.int8).reshape(height, width)
            self.visualize_radar_boundary(grid_data, width, height, origin_x, origin_y, resolution)

        return base_costmap

    def visualize_radar_boundary(self, grid_data, width, height, origin_x, origin_y, resolution):
        """在终端可视化雷达覆盖范围边界"""
        # 计算覆盖范围半径（栅格数）
        radius_in_cells = int(self.fusion_radius / resolution)
        
        # 计算地图中心
        center_x = width // 2
        center_y = height // 2
        
        # 标记覆盖范围边界
        for y in range(height):
            for x in range(width):
                # 计算到中心的距离（栅格单位）
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if abs(distance - radius_in_cells) < 2:  # 边界约2个栅格宽
                    if grid_data[y, x] == 0:  # 只在自由空间标记
                        grid_data[y, x] = 50  # 特殊值表示边界

    # --- 以下函数保持不变 ---

    def calculate_pose_difference(self, odom1, odom2):
        """计算两个odom之间的相对位姿变化（修正坐标系）"""
        x1, y1 = odom1.pose.pose.position.x, odom1.pose.pose.position.y
        x2, y2 = odom2.pose.pose.position.x, odom2.pose.pose.position.y
        
        # --- 已修正的部分 ---
        # 直接从 orientation 对象创建列表
        q1 = odom1.pose.pose.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
        
        q2 = odom2.pose.pose.orientation
        _, _, yaw2 = euler_from_quaternion([q2.x, q2.y, q2.z, q2.w])
        # --- 修正结束 ---
        
        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_yaw = yaw2 - yaw1
        
        # 将位移转换到odom1的坐标系下
        rotated_dx = delta_x * math.cos(yaw1) + delta_y * math.sin(yaw1)
        rotated_dy = -delta_x * math.sin(yaw1) + delta_y * math.cos(yaw1)
        
        return Pose2D(x=rotated_dx, y=rotated_dy, theta=delta_yaw)

    def apply_transform_to_costmap(self, costmap, delta_pose):
        new_costmap = OccupancyGrid()
        new_costmap.header, new_costmap.info = costmap.header, costmap.info
        res, width, height = costmap.info.resolution, costmap.info.width, costmap.info.height
        dx_cells = int(round(-delta_pose.x / res))
        dy_cells = int(round(-delta_pose.y / res))
        dtheta = delta_pose.theta
        original_data = np.array(costmap.data, dtype=np.int8).reshape(height, width)
        new_data = np.full_like(original_data, -1)
        center_x, center_y = width // 2, height // 2
        cos_theta, sin_theta = math.cos(dtheta), math.sin(dtheta)
        for y_new in range(height):
            for x_new in range(width):
                rel_x, rel_y = x_new - center_x, y_new - center_y
                x_rot, y_rot = rel_x * cos_theta - rel_y * sin_theta, rel_x * sin_theta + rel_y * cos_theta
                x_orig, y_orig = round(x_rot + center_x - dx_cells), round(y_rot + center_y - dy_cells)
                if 0 <= x_orig < width and 0 <= y_orig < height:
                    new_data[y_new, x_new] = original_data[y_orig, x_orig]
        new_costmap.data = new_data.flatten().tolist()
        return new_costmap

    def visualize_costmap(self, costmap):
        """可视化costmap（无闪烁版本）"""
        data = np.array(costmap.data).reshape(costmap.info.height, costmap.info.width)

        center_x = costmap.info.width // 2
        center_y = costmap.info.height // 2
        half_width = self.console_width // 2
        half_height = self.console_height // 2

        # 移动光标到终端左上角
        print("\033[H", end="")

        # 打印标题（只打印一次）
        print(f"{Fore.CYAN}=== Costmap Visualization ==={Style.RESET_ALL}")
        print(f"Size: {costmap.info.width}x{costmap.info.height} | "
            f"Res: {costmap.info.resolution:.3f}m/cell | "
            f"Center: ({center_x}, {center_y})")

        # 打印地图内容
        for y in range(costmap.info.height - 1, -1, -1):
            row_str = ""
            for x in range(max(0, center_x - half_width), min(costmap.info.width, center_x + half_width)):
                val = data[y, x]

                if x == center_x and y == center_y:
                    row_str += f"{Back.YELLOW}{Fore.BLACK}@{Style.RESET_ALL}"
                elif val < 0:
                    row_str += f"{Fore.WHITE}?"
                elif val == 0:
                    row_str += f"{Fore.GREEN}."
                elif val == 50:  # 雷达边界特殊显示
                    row_str += f"{Fore.BLUE}o{Style.RESET_ALL}"
                elif val < 65:
                    row_str += f"{Fore.YELLOW}%"
                else:
                    row_str += f"{Fore.RED}#"
            print(row_str)

        # 打印图例
        print(f"\n{Fore.CYAN}Legend:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}.{Style.RESET_ALL} Free  {Fore.YELLOW}%{Style.RESET_ALL} Low obstacle  "
            f"{Fore.RED}#{Style.RESET_ALL} High obstacle  {Fore.WHITE}?{Style.RESET_ALL} Unknown  "
            f"{Fore.BLUE}o{Style.RESET_ALL} Radar border  "
            f"{Back.YELLOW}{Fore.BLACK}@{Style.RESET_ALL} Robot")
        print(f"{Fore.CYAN}=== (Showing center {self.console_width}x{self.console_height} area) ==={Style.RESET_ALL}")

        # 清除多余行（防止上一次输出更长）
        print("\033[J", end="")

    def ros_time_to_float(self, time_msg):
        return float(time_msg.sec) + float(time_msg.nanosec) / 1e9

def main(args=None):
    rclpy.init(args=args)
    node = FusedCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exiting...{Style.RESET_ALL}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
