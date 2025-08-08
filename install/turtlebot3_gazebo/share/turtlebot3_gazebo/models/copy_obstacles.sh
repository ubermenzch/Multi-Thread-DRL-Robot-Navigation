#!/bin/bash

# 批量复制障碍物文件夹脚本
# 将 obstacle1~obstacle5 复制为 obstacle6~obstacle30

# 基础文件夹列表
base_folders=("obstacle1" "obstacle2" "obstacle3" "obstacle4" "obstacle5")

# 起始和目标编号
start_index=6
end_index=30

# 循环创建新文件夹
for ((i=start_index; i<=end_index; i++)); do
    # 计算基础文件夹索引 (0-4)
    base_index=$(( (i - start_index) % ${#base_folders[@]} ))
    
    # 获取基础文件夹名称
    base_folder="${base_folders[$base_index]}"
    
    # 新文件夹名称
    new_folder="obstacle$i"
    
    # 复制文件夹
    echo "复制 $base_folder 到 $new_folder"
    cp -r "$base_folder" "$new_folder"
    
    # 更新新文件夹中的文件内容
    find "$new_folder" -type f -exec sed -i "s/$base_folder/$new_folder/g" {} +
    
    # 重命名新文件夹中的文件
    for file in "$new_folder"/*; do
        if [[ -f "$file" ]]; then
            new_file="${file/$base_folder/$new_folder}"
            mv "$file" "$new_file"
        fi
    done
done

echo "复制完成！从 obstacle$start_index 到 obstacle$end_index"
