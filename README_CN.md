# Ultimate Vocal Remover API

这是一个用于音频文件分离的API，可以将音频中的人声和伴奏分离出来，也可以去除混响和回声。本项目基于[Ultimate Vocal Remover](https://github.com/NextAudioGen/ultimatevocalremover_api)项目，提供了简单的Web API接口和前端测试界面。

## 功能特点

- 支持多种模型类型（MDX和VR网络）
- 支持多种设备运行（CUDA, MPS, CPU）
- 提供简洁的API接口
- 包含易于使用的Web测试界面
- 处理结果可下载和在线预览

## 支持的模型

本项目使用以下模型进行音频分离：

1. **Kim_Vocal_2.onnx (MDX类型)**
   - 适合分离高品质人声和伴奏
   - [下载链接](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx)

2. **6_HP-Karaoke-UVR.pth (VR类型)**
   - 适合分离卡拉OK音频
   - [下载链接](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/6_HP-Karaoke-UVR.pth)

3. **Reverb_HQ_By_FoxJoy.onnx (MDX类型)**
   - 适合分离人声和混响效果
   - [下载链接](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx)

## 安装和使用

### 准备工作

1. 克隆原始项目并安装
```bash
git clone https://github.com/NextAudioGen/ultimatevocalremover_api.git
cd ultimatevocalremover_api
pip install .
```

2. 下载模型文件到models目录
```bash
mkdir -p models
cd models
# 下载MDX模型
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx
# 下载VR模型
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/6_HP-Karaoke-UVR.pth
cd ..
```

3. 安装其他依赖
```bash
pip install -r requirements.txt
```

### 启动API

对于Mac M系列芯片用户，使用内置脚本启动：
```bash
# 确保脚本有执行权限
chmod +x start_api.sh
# 启动API
./start_api.sh
```

对于Windows系统：
```
start_api.bat
```

API将在http://0.0.0.0:6006上启动，可以通过浏览器访问。

### API接口

1. **获取模型列表**
   - GET `/models`
   - 返回可用的模型列表

2. **音频分离**
   - POST `/separate`
   - 参数：
     - `file`：要处理的音频文件
     - `model`：使用的模型名称
   - 返回处理结果的下载链接

3. **下载处理结果**
   - GET `/download?path=文件路径`
   - 下载处理后的音频文件

4. **清理临时文件**
   - GET `/cleanup?dir_path=目录路径`
   - 清理处理过程中生成的临时文件

## 使用Web界面

在浏览器中访问http://0.0.0.0:6006，将会看到一个简单的Web界面：

1. 从下拉菜单中选择要使用的模型
2. 上传要处理的音频文件（支持mp3、wav、flac等格式）
3. 点击"开始分离"按钮
4. 处理完成后，可以在线预览或下载分离后的人声和伴奏文件
5. 处理完成后，可以点击"清理临时文件"按钮释放磁盘空间

## 注意事项

- 处理大文件可能需要较长时间，请耐心等待
- 不同模型适用于不同类型的音频，可以尝试多个模型找到最佳效果
- 在处理完成后记得清理临时文件，避免占用过多磁盘空间

## 贡献与反馈

如有问题或建议，请提交issue或pull request。

## 致谢

本项目基于[Ultimate Vocal Remover](https://github.com/NextAudioGen/ultimatevocalremover_api)项目，感谢原作者的贡献。同时也感谢[TRvlvr](https://github.com/TRvlvr/model_repo)提供的预训练模型。 