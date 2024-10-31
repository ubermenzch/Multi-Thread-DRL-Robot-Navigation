# DRL-Robot-Navigation-ROS2


Deep Reinforcement Learning for mobile robot navigation in ROS2 Gazebo simulator. Using DRL neural network (TD3, SAC), a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch.  Tested with ROS2 Foxy on Ubuntu 20.04 with python 3.8.10 and pytorch 1.10.0+cu113.

Training example:
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>


ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav \
TD3 adapted from: https://github.com/reiniscimurs/DRL-robot-navigation \
SAC adapted from: https://github.com/denisyarats/pytorch_sac 