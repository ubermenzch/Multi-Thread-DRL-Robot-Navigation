# Multi-Thread-DRL-Robot-Navigation
基于reiniscimurs的DRL-Robot-Navigation-ROS2项目git@github.com:reiniscimurs/DRL-Robot-Navigation-ROS2.git，添加了如多实例gazebo训练、利用栅格图作为输入进行路径规划等功能。

Training example:
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>


ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav \
TD3 adapted from: https://github.com/reiniscimurs/DRL-robot-navigation \
SAC adapted from: https://github.com/denisyarats/pytorch_sac 


## Installation (Under Construction)
Main dependencies: 

* [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)

Clone the repository:
```shell
$ cd ~
### Clone仓库
$ git clone https://github.com/ubermenzch/Multi-Thread-DRL-Robot-Navigation.git
### 如果不行则在自己电脑上生成密钥后，将公钥上传至账户再使用如下指令进行Clone
$ git clone git@github.com:ubermenzch/Multi-Thread-DRL-Robot-Navigation.git
### 生成ed25519密钥
$ ssh-keygen -t ed25519 -C "your_email@example.com"
### 获取公钥内容（路径根据自己存储密钥的位置而定）
$ cat ~/.ssh/id_ed25519.pub
```

在ubuntu20.04系统中，通过鱼香ROS一键安装完成：ros2 foxy安装、源配置、ros环境配置、rosdep配置：
```shell
$ wget http://fishros.com/install -O fishros && . fishros
```
安装编译工具：
```shell
$ cd ~/DRL-Robot-Navigation-ROS2
$ sudo apt update
$ sudo apt install python3-colcon-common-extensions # install colcon
$ colcon build
```

配置环境变量：
```shell
$ export ROS_DOMAIN_ID=1
# Fill in the path to where you cloned the repository
$ export DRLNAV_BASE_PATH=~/DRL-Robot-Navigation-ROS2
# Source the path to models
$ export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/DRL-Robot-Navigation-ROS2/src/turtlebot3_simulations/turtlebot3_gazebo/models
# Export the type of robot you want to use
$ export TURTLEBOT3_MODEL=waffle
$ source /opt/ros/foxy/setup.bash
$ source install/setup.bash
```
安装gazebo11：
```shell
sudo apt install gazebo11
sudo apt install ros-foxy-gazebo-ros-pkgs
sudo apt install ros-foxy-gazebo-*
```

安装pytorch：
```shell
#安装pytorch要根据python版本和CUDA版本进行选择，这里以CUDA12.9、python3.8.10为例
#先安装network
pip install "networkx<3.0" --no-deps
#安装pytorch及匹配的CUDA工具包
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
 --index-url https://download.pytorch.org/whl/cu121
```

To start gazebo simulator open a terminal and set up the sources (commands from above):
```shell
$ ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```

To launch training, open another terminal and set up the same sources:
```shell
$ cd ~/DRL-Robot-Navigation-ROS2
$ python3 src/drl_navigation_ros2/train.py
```

To observe training in rviz, open a new terminal and run:
```shell
$ rviz2
```

To launch the tensorboard, open a new terminal and run:
```shell
$ cd ~/DRL-Robot-Navigation-ROS2
$ tensorboard --logdir runs
```
