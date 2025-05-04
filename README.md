![Static Badge](https://img.shields.io/badge/passing-tests-blue)
![Static Badge](https://img.shields.io/badge/pre_release-red)
<a href="https://www.buymeacoffee.com/mohannadbarakat" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
<a href="https://colab.research.google.com/drive/1qf17AV5KU_8v0f29zUnPHQBbr3iX8bu6?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/colab-notebook-yellow" alt="Buy Me A Coffee"></a>

# Ultimate Vocal Remover API

这是一个用于从音频文件中分离人声和伴奏的API服务，基于Ultimate Vocal Remover (UVR)项目。

## 功能

- 支持多种模型进行音频分离
- 提供简单的HTTP API接口
- 支持上传音频文件并下载分离结果
- 内置简单的Web界面用于测试

## 支持的模型

当前支持以下几种模型：

1. **Kim_Vocal_2** - MDX类型，用于人声/伴奏分离
2. **Reverb_HQ_By_FoxJoy** - MDX类型，用于混响处理
3. **6_HP-Karaoke-UVR** - VR-Network类型，用于卡拉OK效果

## 安装与运行

### 前提条件

- Python 3.7+
- PyTorch
- CUDA (如需GPU支持)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
python api.py --host 0.0.0.0 --port 6006 --device cpu
```

参数说明：
- `--host`: 指定主机地址（默认为0.0.0.0）
- `--port`: 指定端口号（默认为6006）
- `--device`: 指定计算设备，可选值为cpu、cuda或mps（默认为cpu）

## API使用

### 获取可用模型列表

```
GET /models
```

### 分离音频

```
POST /separate
```

参数：
- `file`: 音频文件
- `model`: 使用的模型名称

### 下载结果

```
GET /download?path={file_path}
```

### 清除缓存

```
GET /clear_cache
```

### 清理临时文件

```
GET /cleanup?dir_path={temp_dir}
```

## 最近更新

### 2023年X月X日

- 重构模型加载逻辑，修复6_HP-Karaoke-UVR模型加载失败的问题
- 改进文件路径处理，支持绝对路径和相对路径
- 增强错误处理和日志记录
- 修复模型数据读取问题
- 添加模型参数文件自动查找功能
- 改进输出文件检测逻辑

## 目录结构

```
├── api.py                  # 主API服务
├── index.html              # 简单的Web界面
├── temp/                   # 临时文件存储
├── src/                    # 源代码
│   └── models_dir/         # 模型目录
│       ├── mdx/            # MDX模型
│       │   └── weights/    # 模型权重文件
│       └── vr_network/     # VR-Network模型
│           └── weights/    # 模型权重文件
└── requirements.txt        # 依赖包列表
```

## 开发说明

若要添加新模型，需要：

1. 在`MODEL_PATHS`中添加模型路径
2. 在`MODEL_CONFIGS`中添加模型配置
3. 确保模型文件放置在正确的位置
