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
    def __init__(self, args=None):
        rclpy.init(args=args)
        self.cmd_vel_publisher = CmdVelPublisher()
        self.scan_subscriber = ScanSubscriber()
        self.odom_subscriber = OdomSubscriber()
        self.robot_state_publisher = SetModelStateClient()
        self.world_reset = ResetWorldClient()
        self.physics_client = PhysicsClient()
        self.publish_target = MarkerPublisher()
        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        self.sensor_subscriber = SensorSubscriber()
        self.target_dist = 2.0
        self.target = self.set_target_position([0.0, 0.0])

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
        if min(laser_scan) < 0.4:
            return True
        return False

    def check_target(self, distance, collision):
        if distance < 0.5 and not collision:
            self.target_dist += 0.001
            if self.target_dist > 8:
                self.target_dist = 8
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
