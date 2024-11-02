from pathlib import Path

from TD3.TD3 import TD3
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining


def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 10  # how many episodes to use to run evaluation
    max_epochs = 100  # max number of epochs
    epoch = 0  # starting epoch number
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 2  # train and update network parameters every n episodes
    training_iterations = 500  # how many batches to use for single training cycle
    batch_size = 40  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        50  # number of training iterations to run during pre-training
    )
    save_every = 100  # save the model every n training cycles

    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
    )  # instantiate a model

    ros = ROS_env()  # instantiate ROS environment
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=5e3, random_seed=42),
            reward_function=ros.get_reward,
        )  # instantiate pre-trainind
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training
    else:
        replay_buffer = ReplayBuffer(
            buffer_size=5e3, random_seed=42
        )  # if not experiences are loaded, instantiate an empty buffer

    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state

    while epoch < max_epochs:  # train until max_epochs is reached
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get state a state representation from returned data from the environment
        action = model.get_action(state, True)  # get an action from the model
        action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(
            -max_action, max_action
        )  # add random noise to the model
        a_in = [
            (action[0] + 1) / 2,
            action[1],
        ]  # clip linear velocity to [0, 0.5] m/s range

        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )  # get data from the environment
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )  # get a next state representation
        replay_buffer.add(
            state, action, reward, terminal, next_state
        )  # add experience to the replay buffer

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            eval(
                model=model,
                env=ros,
                scenarios=eval_scenarios,
                epoch=epoch,
                max_steps=max_steps,
            )  # run evaluation


def eval(model, env, scenarios, epoch, max_steps):
    """Function to run evaluation"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
    avg_reward = 0.0
    col = 0
    gl = 0
    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            action = model.get_action(state, False)
            a_in = [(action[0] + 1) / 2, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            col += collision
            gl += goal
    avg_reward /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
