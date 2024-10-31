# DRL-Robot-Navigation-ROS2


Deep Reinforcement Learning for mobile robot navigation in ROS2 Gazebo simulator. Using DRL neural network (TD3, SAC), a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch.  Tested with ROS2 Foxy on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.0+cu113.

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
### Clone this repo
$ git clone https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2.git
```

Install rosdep and colcon:
```shell
$ cd ~/DRL-Robot-Navigation-ROS2
$ sudo apt install python3-rosdep2
$ rosdep update
$ rosdep install -i --from-path src --rosdistro foxy -y
$ sudo apt update
$ sudo apt install python3-colcon-common-extensions # install colcon
$ colcon build
```

Setting up the sources:
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