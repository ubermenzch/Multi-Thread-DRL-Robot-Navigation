from hardcoded_model import HCM
from ros_python import ROS_env
import numpy as np
from utils import record_eval_positions


def main(args=None):
    action_dim = 2
    max_action = 1
    state_dim = 25
    nr_eval_episodes = 10_000
    scenarios = record_eval_positions(n_eval_scenarios=nr_eval_episodes)
    max_epochs = 100
    epoch = 0
    episodes_per_epoch = 1
    episode = 0
    max_steps = 300
    steps = 0

    model = HCM(state_dim=state_dim, max_action=max_action, save_samples=True)
    ros = ROS_env()
    latest_scan, distance, cos, sin, collision, goal, action, reward = ros.step(lin_velocity=0.0, ang_velocity=0.0)

    while epoch < max_epochs:
        state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, action)
        action = model.get_action(state)
        action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(-max_action, max_action)
        action[0] = (action[0] + 1) / 2

        latest_scan, distance, cos, sin, collision, goal, action, reward = ros.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )

        if terminal or steps == max_steps:
            (
                latest_scan,
                distance,
                cos,
                sin,
                collision,
                goal,
                action,
                reward,
            ) = ros.reset()
            episode += 1
            steps = 0
        else:
            steps += 1

        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            eval(
                model=model,
                env=ros,
                scenarios=scenarios,
                epoch=epoch,
                max_steps=max_steps,
            )


def eval(model, env, scenarios, epoch, max_steps):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
    avg_reward = 0.0
    col = 0
    gl = 0
    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, action, reward = env.eval(scenario=scenario)
        while count < max_steps:
            state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, action)
            if terminal:
                break
            action = model.get_action(state)
            action[0] = (action[0] + 1) / 2
            latest_scan, distance, cos, sin, collision, goal, action, reward = env.step(
                lin_velocity=action[0], ang_velocity=action[1]
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
    model.writer.add_scalar("avg_reward", avg_reward, epoch)
    model.writer.add_scalar("avg_col", avg_col, epoch)
    model.writer.add_scalar("avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
