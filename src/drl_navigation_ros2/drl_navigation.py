from ros_nodes import (
    TargetPathSubscriber,
    GnssSubscriber,
    DogCmdVelPublisher,
    ReusedCostMapSubcriber,
)
import math
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from SAC.SAC_dog import SAC
import rclpy
from geometry_msgs.msg import PointStamped
import utils
import numpy as np


class Agent:
    def __init__(self,robot_type,is_indoor_test=False,scan_range=5.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("GPU CUDA Ready")
        else:
            print("CPU CUDA Ready")
        self.robot_type = robot_type # robot_type = {ysc,unitree}
        self.is_indoor_test=is_indoor_test
        self.action_dim = 2  # number of actions produced by the model
        self.max_action = 1  # maximum absolute value of output actions
        self.state_dim = 25  # number of input values in the neural network (vector length of state input)
        self.bin_num=self.state_dim-5
        self.model = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            load_model=True,
            robot_type = self.robot_type,
        )  # instantiate a model
        self.dog_cmd_vel_publisher=DogCmdVelPublisher(robot_type = self.robot_type)
        self.target_path_subscriber=TargetPathSubscriber()
        self.fix_subscriber = FixSubscriber()
        self.reused_costmap_subscriber = ReusedCostMapSubcriber(scan_range=scan_range)
        self.odom_subscriber = OdomSubscriber(robot_type)

    def run(self):
        while True:
            last_action = [0.0,0.0]
            has_target = self.target_path_subscriber.has_target_path()
            has_gnss = self.fix_subscriber.is_gnss_valid
            has_map = self.reused_costmap_subscriber.is_reused_costmap_valid
            while True:
                # 若数据未全部有效，则发送停止指令（线速度、角速度均为0的控制指令）
                if has_target and (self.is_indoor_test or has_gnss) and has_map:
                    # 计算智能体前方180度范围内20个分区中距离自身最近的障碍物距离
                    obs_min_distance = self.reused_costmap_subscriber.get_obs_min_distance()

                    # 计算目标距离智能体的距离
                    target_gps = self.target_path_subscriber.get_next_target_gnss()
                    distance = -1.0
                    match self.robot_type:
                        case "ysc":
                            distance = utils.distance(self.fix_subscriber.latitude,
                            self.fix_subscriber.longitude,
                            target_gps['latitude'],
                            target_gps['longitude']
                            )
                        case "unitree":
                            gcjLat, gcjLng = utils.wgs2gcj(self.fix_subscriber.latitude,self.fix_subscriber.longitude)
                            distance = utils.distance(gcjLat,
                            gcjLng,
                            target_gps['latitude'],
                            target_gps['longitude']
                            )
                    # 计算目标相对于智能体的角度（以智能体坐标系为基准）
                    agent_vector = np.array([np.cos(self.odom_subscriber.heading), np.sin(self.odom_subscriber.heading)])
                    target_vector = np.array([target_x - agent_x, target_y - agent_y])
                    cos = 1.0
                    sin = 0.0
                    target_vector_norm = np.linalg.norm(target_vector)
                    if target_vector_norm > 1e-6:
                        #   点积 = |a|·|b|·cosθ → cosθ = (a·b) / (|a||b|)
                        #   叉积 = |a|·|b|·sinθ → sinθ = (a×b) / (|a||b|)
                        dot_product = np.dot(pose_vector, goal_vector)
                        cross_product = np.cross(pose_vector, goal_vector)  # 2D叉积返回标量
                        cos = dot_product /  target_vector_norm
                        sin = cross_product / target_vector_norm
                    state = model.prepare_state(obs_min_distance, distance, cos, sin, last_action)
                    action = model.get_action(state, add_noise=False)
                    action = [(action[0] + 1) / 2, action[1]] # 把线速度从[-1,1]规范到[0,1]
                    dog_cmd_vel_publisher.publish_cmd_vel(linear_velocity=last_action[0],angular_velocity=action[1])
                    last_action=action
                else:
                    dog_cmd_vel_publisher.publish_cmd_vel(linear_velocity=0.0,angular_velocity=0.0)
                    last_action = [0.0,0.0]

def main(args=None):
    agent = Agent(robot_type="ysc")
    agent.run()

if __name__ == "__main__":
    main()