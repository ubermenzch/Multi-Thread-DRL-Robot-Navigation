#!/bin/bash
# save as delete_obstacles.sh
for i in {6..30}; do
    rm -rf "obstacle$i"
done
echo "已删除obstacle6到obstacle30的文件夹"
