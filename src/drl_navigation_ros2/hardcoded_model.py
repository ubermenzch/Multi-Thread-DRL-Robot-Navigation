from math import atan2

import numpy as np
from numpy import clip
from torch.utils.tensorboard import SummaryWriter
import yaml


class HCM(object):
    def __init__(
        self,
        state_dim,
        max_action,
        save_samples,
        file_location="src/drl_navigation_ros2/assets/data.yml",
    ):
        self.max_action = max_action
        self.state_dim = state_dim
        self.writer = SummaryWriter()
        self.iterator = 0
        self.save_samples = save_samples
        self.file_location = file_location

    def get_action(self, state):
        sin = state[-3]
        cos = state[-4]
        angle = atan2(sin, cos)
        laser_nr = self.state_dim - 5
        limit = 1.5

        if min(state[4 : self.state_dim - 9]) < limit:
            idx = state[:laser_nr].index(min(state[:laser_nr]))
            if idx > laser_nr / 2:
                sign = -1
            else:
                sign = 1

            idx = clip(idx + sign * 5 * (limit / min(state[:laser_nr])), 0, laser_nr)

            angle = ((3.14 / (laser_nr)) * idx) - 1.57

        rot_vel = clip(angle, -1.0, 1.0)
        lin_vel = -abs(rot_vel / 2)
        return [lin_vel, rot_vel]

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99999,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        pass

    def save(self, filename, directory):
        pass

    def load(self, filename, directory):
        pass

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):

        latest_scan = np.array(latest_scan)

        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0

        max_bins = self.state_dim - 5
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
        state = min_values + [distance, cos, sin] + [action[0], action[1]]

        assert len(state) == self.state_dim
        terminal = 1 if collision or goal else 0

        self.iterator += 1
        if self.save_samples and self.iterator < 10_000:
            action = action if type(action) is list else action
            action = [float(a) for a in action]
            sample = {
                self.iterator: {
                    "latest_scan": latest_scan.tolist(),
                    "distance": distance.tolist(),
                    "cos": cos.tolist(),
                    "sin": sin,
                    "collision": collision,
                    "goal": goal,
                    "action": action,
                }
            }
            with open(self.file_location, "a") as outfile:
                yaml.dump(sample, outfile, default_flow_style=False)

        return state, terminal
