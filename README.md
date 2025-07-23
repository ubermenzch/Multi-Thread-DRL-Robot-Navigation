# Multi-Thread-DRL-Robot-Navigation（多线程训练深度强化学习机器人导航）
基于reiniscimurs的DRL-Robot-Navigation-ROS2项目git@github.com:reiniscimurs/DRL-Robot-Navigation-ROS2.git，添加了如多实例gazebo训练、利用栅格图作为输入进行路径规划等功能。

训练示例：
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>


ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav \
TD3 adapted from: https://github.com/reiniscimurs/DRL-robot-navigation \
SAC adapted from: https://github.com/denisyarats/pytorch_sac 


## 安装教程
主要依赖

* [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)

Clone仓库:
```shell
cd ~
git clone https://github.com/ubermenzch/Multi-Thread-DRL-Robot-Navigation.git
# 如果失败，则在自己电脑上生成密钥后，将公钥上传至github账户再使用如下指令进行Clone
git clone git@github.com:ubermenzch/Multi-Thread-DRL-Robot-Navigation.git
```
创建密钥方法（在使用https进行clone失败时使用）：
```shell
# 生成ed25519密钥
ssh-keygen -t ed25519 -C "your_email@example.com"
# 获取公钥内容（路径根据自己存储密钥的位置而定）
cat ~/.ssh/id_ed25519.pub
```

在ubuntu20.04系统中，通过鱼香ROS一键安装完成：ros2 foxy安装、源配置、ros环境配置、rosdep配置：
```shell
wget http://fishros.com/install -O fishros && . fishros
```

安装编译工具：
```shell
cd ~/DRL-Robot-Navigation-ROS2
sudo apt update
sudo apt install python3-colcon-common-extensions # install colcon
colcon build
```

配置环境变量：
```shell
export ROS_DOMAIN_ID=1
# Fill in the path to where you cloned the repository
export DRLNAV_BASE_PATH=~/DRL-Robot-Navigation-ROS2
# Source the path to models
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
# Export the type of robot you want to use
export TURTLEBOT3_MODEL=waffle
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

安装gazebo11：
```shell
sudo apt install gazebo11
sudo apt install ros-foxy-gazebo-ros-pkgs
sudo apt install ros-foxy-gazebo-*
```

安装pytorch：
```shell
# 安装pytorch要根据python版本和CUDA版本进行选择，这里以CUDA12.9、python3.8.10为例
# 先安装network
pip install "networkx<3.0" --no-deps
# 安装pytorch及匹配的CUDA工具包
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
 --index-url https://download.pytorch.org/whl/cu121
```
安装squaternion、tqdm
```shell
pip install squaternion
pip install tqdm
```

安装tensorboard（用于观测训练）：
```shell
pip install tensorboard
pip install tensorboardx
```

下载gazebo模型库（原github项目自带模型不全，会导致训练启动失败）。访问链接下载（如果网络可行也直接在服务器里下，若网络不可行则下载到本地再上传到服务器）：https://github.com/osrf/gazebo_models/archive/refs/heads/master.zip
```shell
# 解压zip文件
unzip gazebo_models-master.zip
# 将gazebo_models-master里的所有文件夹移动到模型文件夹（~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models）
mv ~/gazebo_models-master/* ~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
```

先在一个终端启动gazebo:
```shell
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

再在另一个终端启动训练脚本：
```shell
# 注意，一定要在DRL-Robot-Navigation-ROS2目录下启动脚本
cd ~/DRL-Robot-Navigation-ROS2
python3 src/drl_navigation_ros2/train.py
```

通过tensorboard监控训练
```shell
#运行训练脚本后打开新终端，通过端口映射的方式连接服务器
ssh -p 8899 -L 6006:127.0.0.1:6006 zc@10.130.136.132
# 查看训练数据图
cd ~/DRL-Robot-Navigation-ROS2
tensorboard --logdir runs
#在本地机器（笔记本）上浏览器打开http://localhost:6006/即可进入tensorboard可视化界面
```

后台训练
```shell
# 安装相关依赖
sudo apt-get install expect-dev
# 终端所有输出都保存到train.log日志中
nohup ./start_training.sh > /dev/null 2>&1 &
# 实时监控训练日志
tail -f train.log
```

tensorboard中各个指标的含义：

## 1. 评估指标 (`eval/`)

- `eval/avg_col` & `eval/avg_goal`
    - **含义：** 这些是**特定于环境**的评估指标，**不是 SAC 算法本身的标准指标**。它们的含义取决于你训练任务的环境。
        - `avg_col`: 通常表示在评估周期内，模型“**平均碰撞率**”（Average Collisions）或类似的有害交互事件。
        - `avg_goal`: 通常表示在评估周期内，模型“**平均到达目标率**”（Average Goal Reached）或类似的任务完成指标。
    - **上下文：** 评估模式 (`eval`)。训练器会定期（比如每 `n` 步）暂停训练，使用**当前策略**（不探索，通常是 `actor` 的确定性版本）在测试环境中运行多个回合（Episodes），并计算这些回合的平均值。
    - **目的：**
        - 监控模型在真实（无探索）执行环境下的**实际性能**和**任务完成度**。
        - 判断训练是否收敛以及模型性能是否在提升。
        - 理解模型的行为特性（如是否学会了避免碰撞、达成目标）。
- `eval/avg_reward`
    - **含义：** 在评估周期内（多个测试回合），每个回合（Episode）或每个时间步（取决于日志设置）获得的 **平均累积环境奖励（Reward）**。
    - **上下文：** 评估模式 (`eval`)。
    - **目的：** 这是 **RL 算法性能的最核心指标**。它衡量你的模型在任务中表现得有多好（目标是最大化这个值）。持续上升或维持在一个较高水平通常表明训练有效。它比训练奖励更能反映模型的实际能力，因为训练过程包含了探索扰动。

## 2. 训练指标 (`train/`, `train_actor/`, `train_critic/`, `train_alpha/`)

这些指标来源于训练过程本身的数据批次（Batch）更新。

- **Actor 网络 (Policy Network) 指标 (`train_actor/`)**
    - `train_actor/entropy`
    - `train_actor/entropy_av`
        - **含义：** 策略（Policy）在给定状态下产生的动作分布的**信息熵**。
        - **上下文：** Actor 网络更新计算时基于经验回放池采样的状态批次。
        - **目的：**
            - **衡量策略的随机性（探索性）。** 高熵值表示策略输出动作的随机性高（均匀选择），低熵值表示策略输出动作更确定（偏好某个动作）。
            - SAC **显式鼓励探索**： SAC损失函数中包含一个熵正则化项 `α * H(π(·|s))`，惩罚策略熵过低，从而保持一定的探索性。
            - 监控训练稳定性：熵值异常变化（如陡降）可能预示策略坍塌（Collapse）或数值问题。
            - `entropy_av` 是滑动平均，更能反映整体趋势。
    - `train_actor/loss`
    - `train_actor/loss_av`
        - **含义：** Actor（策略）网络的**损失值**。
        - **上下文：** Actor 网络使用梯度下降更新时的损失值。
        - **目的：**
            - 反映策略网络更新的“**难度**”或“**一致性**”。
            - 在SAC中，Actor损失函数为：`J(π) = E[α * logπ(a|s) - Q(s, a)]`，其中`Q(s, a)`来自于Critic。**目标是最小化这个损失**。它结合了“最小化动作价值的负值”和“最大化动作熵”。
            - 监控训练稳定性：Actor Loss 应保持相对稳定或有小幅波动，剧烈的、无规律的波动或持续上升可能表明训练不稳定（如Critic训练失败、学习率过大）。
    - `train_actor/target_entropy`
    - `train_actor/target_entropy_av`
        - **含义：** SAC算法中用来**自动调整温度参数 `α`** 的**目标熵值**。
        - **上下文：** Actor 更新计算时使用的目标值（但主要由温度网络 `α` 使用）。
        - **目的：**
            - 这是一个**超参数**，但非常关键。它代表了算法希望策略保持的期望熵水平（通常设置为与环境动作空间的维度相关）。
            - 实际的温度 `α` 会通过另一个优化过程（见 `train_alpha`）自动调整，使得 `entropy` 尽量靠近这个 `target_entropy`。它是实现自动熵调节的核心。
            - `target_entropy_av` 是其滑动平均，通常它就是固定的超参数，滑动平均也没什么波动。
- **Critic 网络 (Q-Network) 指标 (`train_critic/`)**
    - `train_critic/loss`
    - `train_critic/loss_av`
        - **含义：** Critic (Q-Value) 网络的**损失值**。
        - **上下文：** Critic 网络使用梯度下降更新时的损失值（例如均方误差）。
        - **目的：**
            - 衡量当前 Critic 预测的 Q 值与目标 Q 值 (`r + γ * (Q_target(s', a') - α logπ(a'|s'))`) 之间的**误差**。**目标是最小化这个损失**。
            - 这是**最重要的训练指标之一**。Critic的任务是准确估计动作价值函数（Q值），其损失的稳定下降是Actor有效学习的前提。
            - 监控训练稳定性：Critic Loss 通常应该随着训练逐渐下降然后稳定在一个较低水平。损失过大或持续波动剧烈通常表示 Critic 学习困难（如bootstrapping误差积累、学习率过大、数据相关性太强、网络容量不足）。持续上升是训练失败的强烈信号。
            - `loss_av` 更能反映整体趋势。
- **温度参数 (α) 指标 (`train_alpha/`)**
    - `train_alpha/loss`
    - `train_alpha/loss_av`
        - **含义：** 温度参数 `α` 本身的**损失值**。
        - **上下文：** 用于优化温度参数 `α`（如果 `α` 是自动调整的）的损失值。
        - **目的：**
            - 在SAC中，`α` 被视为一个可训练的参数（或由一个网络输出）。它的损失函数通常设计为 `J(α) = E[-α * (logπ(a|s) + H_target)]`。
            - 优化这个损失的目标是：当策略熵低于目标熵 `H_target` (即 `logπ(a|s)` 太大导致熵太小)时，增加 `α` 以加强熵惩罚；当策略熵高于目标熵时，减小 `α`。
            - 监控这个损失值有助于了解自动熵调节过程是否正常。
    - `train_alpha/value`
    - `train_alpha/value_av`
        - **含义：** **温度参数 `α` 的当前数值**。
        - **上下文：** 优化后的 `α` 值。
        - **目的：**
            - **直接显示当前用于策略损失的熵正则化的权重。**
            - 监控 `α` 的动态变化是理解 SAC 自动平衡**探索（高`α`/高熵）** 和 **利用（低`α`/低熵）** 的关键。
            - 如果 `α` 持续下降，说明策略倾向于收敛（熵降低）；如果 `α` 稳定在一个正值附近，说明策略保持了适当的随机性。
- **全局训练指标 (`train/`)**
    - `train/batch_reward`
    - `train/batch_reward_av`
        - **含义：** 从经验回放池采样出来的一个训练数据批次（Batch）中所有样本的**即时环境奖励的平均值**。
        - **上下文：** 随机采样的一批训练数据。
        - **目的：**
            - 提供一个训练过程中的（小批量）**环境反馈信号**的粗略快照。但这**不等于策略的性能**。
            - 因为训练批次数据可能来自不同策略阶段（探索性策略采集的老数据 + 优化后策略采集的新数据）以及不同难度/质量的状态，所以波动性通常很大。
            - 主要用于**感知环境反馈的存在**。它上升可能意味着模型开始收集到一些正面奖励的经验，但**不能**据此判断模型性能在提升（性能提升依赖 `eval/avg_reward` 和 `eval/avg_goal`）。过度关注它可能会被误导。

## 总结与监控要点

1. **核心性能指标：** 关注 **`eval/avg_reward`**, **`eval/avg_goal`**, **`eval/avg_col`**。它们代表模型在测试模式下的真实能力。
2. **训练稳定性指标：**
    - **Critic Loss (`train_critic/loss_av`)** 是最关键的。它应稳定下降并保持在一个较低值。
    - **Actor Loss (`train_actor/loss_av`)** 应相对平稳或有小幅波动。
    - **Policy Entropy (`train_actor/entropy_av`)** 应稳定在 `target_entropy` 附近或根据任务需要稳定变化（如任务本身就需要确定性策略，则 `α` 最终会降到很低，熵也会很低）。
    - **Temperature (`train_alpha/value_av`)** 的动态变化揭示了算法对探索/利用的自动平衡过程。
3. **任务特定指标：** `eval/avg_goal` 和 `eval/avg_col` 直接反映了目标任务的具体完成情况和行为特性。
4. **区别 `train/batch_reward` 与 `eval/avg_reward`：** 前者是训练数据批次的奖励均值，**噪音大且不代表策略性能**；后者是评估多个完整回合的平均累积奖励，**是策略性能的真实反映**。
5. **使用滑动平均 (`_av`)**： **务必**关注带 `_av` 后缀的滑动平均指标来观察整体趋势。原始值波动过于剧烈，难以判断真实状况。

通过综合观察这些指标，你可以有效地评估 SAC 模型的训练进展、发现潜在问题（如发散、策略坍塌、Critic 不收敛）、了解算法的探索-利用平衡状态，并最终判断模型是否在任务上成功学习。