#!/bin/bash

# 停止训练脚本（Python）
pkill -f "train.py" && echo "训练脚本train.py已停止"

# 停止Gazebo相关进程
killall -9 gazebo > /dev/null 2>&1
killall -9 gzserver > /dev/null 2>&1
killall -9 gzclient > /dev/null 2>&1

# 停止虚拟显示
killall -9 Xvfb > /dev/null 2>&1

# 查找并停止主脚本及其子进程
found=0
pids=$(pgrep -f "start_training.sh")

if [ -z "$pids" ]; then
    echo "未找到运行中的start_training.sh进程"
else
    for pid in $pids; do
        # 优雅终止
        kill -15 $pid 2>/dev/null
        sleep 2
        
        # 检查并强制终止
        if ps -p $pid > /dev/null; then
            kill -9 $pid
            echo "强制终止PID $pid"
        else
            echo "优雅终止PID $pid"
        fi
        
        # 终止整个进程组
        pgrp=$(ps -o pgid= $pid | tr -d ' ')
        [ -n "$pgrp" ] && kill -9 -$pgrp 2>/dev/null
        
        found=1
    done
fi

# 最终确认
if [ $found -eq 1 ]; then
    echo "所有训练相关进程已停止"
else
    echo "未发现需要停止的进程"
fi
