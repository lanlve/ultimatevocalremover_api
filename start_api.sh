#!/bin/bash

# 激活conda环境
source ~/miniconda3/bin/activate uvr5
# 或者使用这个命令（取决于您的Conda安装路径）
# source /opt/miniconda3/bin/activate uvr5

# 确保模型目录存在
mkdir -p src/models_dir/mdx/weights
mkdir -p src/models_dir/vr_network/weights

# 检查模型文件是否存在，如不存在提示下载
if [ ! -f "models/Kim_Vocal_2.onnx" ] || [ ! -f "models/6_HP-Karaoke-UVR.pth" ] || [ ! -f "models/Reverb_HQ_By_FoxJoy.onnx" ]; then
    echo "检测到部分模型文件不存在，请下载模型文件到models目录"
    echo "模型文件下载链接："
    echo "Kim_Vocal_2.onnx: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx"
    echo "6_HP-Karaoke-UVR.pth: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/6_HP-Karaoke-UVR.pth"
    echo "Reverb_HQ_By_FoxJoy.onnx: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx"
    exit 1
fi

# 启动API服务，使用MPS设备（适用于Mac Silicon芯片）
echo "使用 MPS 设备启动API服务..."
python api.py --device mps 