#!/bin/bash

# 激活conda环境
source ~/miniconda3/bin/activate uvr5
# 或者使用这个命令（取决于您的Conda安装路径）
# source /opt/miniconda3/bin/activate uvr5

# 确保模型目录存在
mkdir -p src/models_dir/mdx/weights
mkdir -p src/models_dir/vr_network/weights

# 启动API服务，使用MPS设备（适用于Mac Silicon芯片）
echo "使用 MPS 设备启动API服务..."
python api.py --device mps 