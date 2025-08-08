#!/bin/bash
# start_training.sh - 增强版一键启动训练脚本

# ===================== 核心设置 =====================
LOGFILE="$HOME/DRL-Robot-Navigation-ROS2/train.log"  # 统一日志文件
XVFB_DISPLAY=":99"                                   # 虚拟显示端口
GAZEBO_WAIT_TIME=10                                 # Gazebo启动等待时间

# ===================== 初始化日志 =====================
init_logging() {
    # 清空日志文件
    > "$LOGFILE"
    # 添加启动时间戳
    echo "===== 训练启动 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
}

# ===================== 启动虚拟显示器 =====================
start_xvfb() {
    echo "启动虚拟显示器 (Xvfb)..." | tee -a "$LOGFILE"
    Xvfb $XVFB_DISPLAY -screen 0 640x480x16 >> "$LOGFILE" 2>&1 &
    export DISPLAY=$XVFB_DISPLAY
    sleep 2
}

# ===================== 启动Gazebo无头模式 =====================
start_gazebo() {
    echo "启动Gazebo无头服务器..." | tee -a "$LOGFILE"
    # 使用--headless参数明确无头模式
    ros2 launch turtlebot3_gazebo ros2_drl_headless.launch.py >> "$LOGFILE" 2>&1 &
    
    # 等待Gazebo初始化
    echo "等待Gazebo初始化 ($GAZEBO_WAIT_TIME秒)..." | tee -a "$LOGFILE"
    for i in $(seq 1 $GAZEBO_WAIT_TIME); do
        echo -n "." | tee -a "$LOGFILE"
        sleep 1
    done
    echo -e "\nGazebo初始化完成" | tee -a "$LOGFILE"
}

# ===================== 启动训练脚本 =====================
start_training() {
    echo "启动训练脚本..." | tee -a "$LOGFILE"
    cd ~/DRL-Robot-Navigation-ROS2
    # 关键：使用unbuffer和tee确保实时输出
    unbuffer python3 -u src/drl_navigation_ros2/train.py 2>&1 | tee -a "$LOGFILE"
}

# ===================== 清理函数 =====================
cleanup() {
    echo "清理进程..." | tee -a "$LOGFILE"
    killall -9 gzserver Xvfb ros2 python3 2>/dev/null
    echo "===== 训练结束 - $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
}

# ===================== 主执行流程 =====================
main() {
    # 确保捕获中断信号以执行清理
    trap cleanup EXIT SIGINT SIGTERM
    
    init_logging
    start_xvfb
    start_gazebo
    start_training
    cleanup
}

# 执行主函数
main
