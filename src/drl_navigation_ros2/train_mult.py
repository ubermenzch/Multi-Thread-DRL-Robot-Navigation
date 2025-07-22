from pathlib import Path
#from TD3.TD3 import TD3
from SAC.SAC_mult import SAC
from ros_mult_python import ROS_env
from replay_buffer_mult import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining
from threading import Thread
import torch.multiprocessing as mp

action_dim = 2  # number of actions produced by the model
max_action = 1  # maximum absolute value of output actions
state_dim = 25  # number of input values in the neural network (vector length of state input)
# how many episodes to use to run evaluation 
nr_eval_episodes = 20  # 用训练好的模型运行nr_eval_episodes个完整的回合（episodes）以评估模型性能。默认值为10
# 训练整体思路是：一个轮次中，让智能体跑多个回合，这多个回合的数据在训练中被使用，训练时通过抽取回合数据来更新模型
# max number of epochs. 
max_epochs = 100  # 设置整个训练过程的最大迭代次数。默认值为100。

# how many episodes to run in single epoch
episodes_per_epoch = 70  # 在1轮（epoch）内收集episodes_per_epoch个回合的经验数据用于训练。默认值为70.
# train and update network parameters every n episodes
train_every_n = 2  # 每train_every_n个回合后执行一次训练。默认值为2.
# how many batches to use for single training cycle
training_iterations = 500  # 每次执行训练时，用随机抽取的批次数据更新模型training_iterations次。默认值为500.
# batch size for each training iteration
batch_size = 40  # 从经验池中随机抽取batch_size条经验作为一个训练批次来对模型参数进行更新。默认值为40.
# maximum number of steps in single episode
max_steps = 300  # 限制每个回合最多执行max_steps步操作，超过会强制结束回合。默认值为300.
# starting step number
steps = 0  # 当前回合的步数。默认值为0.
# whether to load experiences from assets/data.yml
load_saved_buffer = True  # 是否加载预存经验池。默认值为True。
# whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
pretrain = True  # 是否执行预训练。默认值为True，同时要求load_saved_buffer也必须为True。
# number of training iterations to run during pre-training
pretraining_iterations = (
    50  # 预训练迭代次数，默认值50
)
# save the model every n training cycles
save_every = 100  # 每save_every次训练后保存一次模型，默认值100
num_agents = 4  # 智能体数量
model = None

# 多线程缓冲池
replay_buffer=ReplayBuffer(buffer_size=5e5, random_seed=42)

def main(args=None):
    """Main training function"""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    if torch.cuda.is_available():
        print("GPU CUDA Ready")
    else:
        print("CPU CUDA Ready")

    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=True,
    )  # instantiate a model
    
    manager = mp.Manager()

    # 创建多智能体训练队列
    agents = []

    #生成多个评估环境，用于评估线程定时对模型进行评估
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation
    # 创建评估线程
    eval_env = ROS_env(agent_id=1000)  # 设置特殊id，避免与训练智能体的id冲突
    eval_thread = Thread(target=eval_loop, args=(model, eval_env, eval_scenarios, max_steps, max_epochs))
    eval_thread.start()

     # 创建训练线程
    for i in range(num_agents):
        agent = Thread(
            target=train_agent, 
            args=(i, model)
        )
        agents.append(agent)
        agent.start()

    # 等待所有线程完成
    for agent in agents:
        agent.join()
    eval_thread.join()


def train_agent(agent_id, model):
    # starting epoch number
    epoch = 0  # 当前训练轮次计数器
    # starting episode number
    episode = 0  # 当前轮次的回合数。默认值0.
    # 每个智能体有自己的ROS环境实例
    ros = ROS_env(namespace=f"agent_{agent_id}")
    latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(0.0, 0.0)
    steps = 0
    
    while epoch < max_epochs :  # 全局训练循环控制
        # 局部模型获取当前状态
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        
        # 添加探索噪声的动作选择
        action = model.get_action(state, add_noise=True)
        a_in = [(action[0] + 1) / 2, action[1]]
        
        # 环境交互
        latest_scan, distance, cos, sin, collision, goal, a, reward = ros.step(
            a_in[0], a_in[1]
        )
        
        # 计算下一状态
        next_state, next_terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a
        )
        steps+=1

        # 存储经验 (使用线程安全的方式)
        replay_buffer.add(state, action, reward, terminal or next_terminal, next_state)
        
        if (terminal or steps == max_steps):  
            # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            episode += 1
            steps = 0
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )  # train the model and update its parameters
            if episode % episodes_per_epoch == 0:
                episode = 0
                epoch += 1

def eval_loop(model, env, eval_scenarios, max_steps, max_epochs):
    epoch = 0
    while True:
        # 定期执行评估
        time.sleep(60 * 5)  # 每5分钟评估一次
        print("..............................................")
        print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
        
        # 使用模型最新参数进行冻结评估
        frozen_model = copy.deepcopy(model)
        
        # 执行评估
        avg_reward = 0.0
        col = 0
        gl = 0
        for scenario in eval_scenarios:
            count = 0
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(scenario=scenario)
            while count < max_steps:
                state, terminal = frozen_model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
                if terminal:
                    break
                action = frozen_model.get_action(state, False)
                a_in = [(action[0] + 1) / 2, action[1]]
                latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                    lin_velocity=a_in[0], ang_velocity=a_in[1]
                )
                avg_reward += reward
                count += 1
                col += collision
                gl += goal
                # 记录评估结果
        print(f"Average Reward: {avg_reward}")
        print(f"Average Collision rate: {avg_col}")
        print(f"Average Goal rate: {avg_goal}")
        print("..............................................")
        frozen_model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
        frozen_model.writer.add_scalar("eval/avg_col", col/len(eval_scenarios), epoch)
        frozen_model.writer.add_scalar("eval/avg_goal", gl/len(eval_scenarios), epoch)
        epoch += 1


if __name__ == '__main__':
    mp.set_start_method('spawn')  # 支持CUDA多进程
    main()