@echo off
setlocal enabledelayedexpansion

:: 激活conda环境
call conda activate uvr5

:: 确保模型目录存在
if not exist src\models_dir\mdx\weights mkdir src\models_dir\mdx\weights
if not exist src\models_dir\vr_network\weights mkdir src\models_dir\vr_network\weights

:: 检查模型文件是否存在
set missing_models=0
if not exist "models\Kim_Vocal_2.onnx" set missing_models=1
if not exist "models\6_HP-Karaoke-UVR.pth" set missing_models=1
if not exist "models\Reverb_HQ_By_FoxJoy.onnx" set missing_models=1

if %missing_models%==1 (
    echo 检测到部分模型文件不存在，请下载模型文件到models目录
    echo 模型文件下载链接：
    echo Kim_Vocal_2.onnx: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx
    echo 6_HP-Karaoke-UVR.pth: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/6_HP-Karaoke-UVR.pth
    echo Reverb_HQ_By_FoxJoy.onnx: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx
    pause
    exit /b 1
)

:: 确定运行设备
echo 选择运行设备类型:
echo 1) CPU
echo 2) CUDA (NVIDIA GPU)
set /p device_choice=请选择 [1-2]: 

set device=cpu
if "%device_choice%"=="2" set device=cuda

:: 启动API服务
echo 使用 %device% 设备启动API服务...
python api.py --device %device%

pause 