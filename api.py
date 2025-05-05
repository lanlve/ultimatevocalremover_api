import os
# 设置环境变量，使PyTorch在MPS不支持的操作上回退到CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import torch
import json
import argparse
import uvr
from uvr import models
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
import numpy as np
import shutil
from pathlib import Path
import tempfile
from typing import Optional, List
import time
try:
    import onnxruntime as ort
    print("使用标准 onnxruntime")
except ImportError:
    print("错误：无法导入onnxruntime，请安装onnxruntime")
import soundfile as sf
import librosa
import traceback

# 创建FastAPI应用
app = FastAPI(title="Ultimate Vocal Remover API", 
              description="API for separating vocals and instruments from audio files")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 临时文件夹路径
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True, parents=True)  # 确保temp目录存在

# 模型目录路径，改为使用models_dir
MODELS_BASE_DIR = Path("./src/models_dir")

# 确保MDX和VR模型目录存在
MODELS_MDX_DIR = MODELS_BASE_DIR / "mdx" / "weights"
MODELS_MDX_DIR.mkdir(parents=True, exist_ok=True)

MODELS_VR_DIR = MODELS_BASE_DIR / "vr_network" / "weights"
MODELS_VR_DIR.mkdir(parents=True, exist_ok=True)

# 指定模型文件路径
MODEL_PATHS = {
    "Kim_Vocal_2": str(MODELS_MDX_DIR / "Kim_Vocal_2" / "Kim_Vocal_2.onnx"),
    "6_HP-Karaoke-UVR": str(MODELS_VR_DIR / "6_HP-Karaoke-UVR" / "6_HP-Karaoke-UVR.pth"),
    "Reverb_HQ_By_FoxJoy": str(MODELS_MDX_DIR / "Reverb_HQ_By_FoxJoy" / "Reverb_HQ_By_FoxJoy.onnx")
}

# 配置模型参数
MODEL_CONFIGS = {
    "Kim_Vocal_2": {
        "type": "mdx",
        "parameters": {
            "dim_f": 3072,
            "dim_t": 8,
            "n_fft": 8192,
            "denoise": True,
            "margin": 44100,
            "chunks": 15,
            "batch_size": 8
        }
    },
    "Reverb_HQ_By_FoxJoy": {
        "type": "mdx",
        "parameters": {
            "dim_f": 3072,
            "dim_t": 9,
            "n_fft": 8192,
            "denoise": True,
            "margin": 44100,
            "chunks": 15,
            "is_reverb_model": True,
            "batch_size": 8
        }
    },
    "6_HP-Karaoke-UVR": {
        "type": "vr_network",
        "parameters": {"aggressiveness": 0.05}
    }
}

# 全局设备变量
DEVICE = "cpu"

# 模型缓存
MODEL_CACHE = {}
# PyTorch模型缓存目录
TORCH_MODELS_DIR = Path("./src/models_dir/pytorch_cache")
TORCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MDX网络的ConvTDFNet类 (参考自提供的脚本)
class ConvTDFNet:
    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name
        
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2
    
    def to_device(self, device):
        """将窗口和填充移到指定设备"""
        self.window = self.window.to(device)
        self.freq_pad = self.freq_pad.to(device)
        return self

    def stft(self, x):
        # 确保窗口和输入在同一设备上
        if x.device != self.window.device:
            self.window = self.window.to(x.device)
            
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        # 如果维度不匹配，记录日志信息
        if x.shape[-1] != self.dim_t:
            print(f"警告: STFT输出维度 {x.shape} 与预期的时间维度 {self.dim_t} 不匹配")
        return x[:, :, : self.dim_f]

    # Inversed Short-time Fourier transform (STFT).
    def istft(self, x, freq_pad=None):
        # 检查和记录输入维度
        if x.shape[-1] != self.dim_t:
            print(f"警告: ISTFT输入维度 {x.shape} 与预期的时间维度 {self.dim_t} 不匹配")
        
        # 确保窗口和输入在同一设备上
        if x.device != self.window.device:
            self.window = self.window.to(x.device)
            
        if freq_pad is None:
            if x.device != self.freq_pad.device:
                self.freq_pad = self.freq_pad.to(x.device)
            freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])

# MDX预测器类 (参考自提供的脚本)
class MDXPredictor:
    def __init__(self, model_path, params, device="cpu"):
        # 确保torch被导入
        import torch
        
        self.model_path = model_path
        self.params = params
        self.device = device
        
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=params["dim_f"], 
            dim_t=params["dim_t"], 
            n_fft=params["n_fft"]
        )
        
        if device == "cuda" and torch.cuda.is_available():
            self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        elif device == "mps" and torch.backends.mps.is_available():
            try:
                # 首先尝试使用PyTorch+MPS加载ONNX模型
                try:
                    # 确保torch可用
                    import torch.onnx
                    print("尝试使用PyTorch+MPS加载ONNX模型")
                    # 创建一个使用PyTorch的预测器
                    self.use_torch_predictor = True
                    self.torch_predictor = MDXTorchPredictor(model_path, params, device)
                    print("成功使用PyTorch+MPS加载ONNX模型")
                except Exception as e:
                    print(f"无法使用PyTorch+MPS加载ONNX模型: {str(e)}")
                    print(f"错误详情: {traceback.format_exc()}")
                    self.use_torch_predictor = False
                    # 回退到CoreML
                    providers_options = [{
                        'device_id': 0,
                        'device_type': 'mps'
                    }]
                    self.model = ort.InferenceSession(
                        model_path, 
                        providers=['CoreMLExecutionProvider'], 
                        provider_options=providers_options
                    )
                    print("成功使用CoreML/MPS加载ONNX模型")
            except Exception as e:
                print(f"警告：在MPS上加载ONNX模型时出错: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
                print("自动回退到CPU执行")
                self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def demix(self, mix):
        # 如果使用PyTorch预测器
        if hasattr(self, 'use_torch_predictor') and self.use_torch_predictor:
            return self.torch_predictor.demix(mix)
        
        samples = mix.shape[-1]
        margin = self.params["margin"]
        chunk_size = self.params["chunks"] * 44100
        
        assert not margin == 0, "margin cannot be zero!"
        
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.params["chunks"] == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        # 如果使用PyTorch预测器
        if hasattr(self, 'use_torch_predictor') and self.use_torch_predictor:
            return self.torch_predictor.demix_base(mixes, margin_size)
        
        chunked_sources = []
        
        for mix_id, mix in enumerate(mixes):
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)
            
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.params["denoise"]:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                
                if margin_size == 0:
                    end = None
                
                sources.append(tar_signal[:, start:end])

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        
        return _sources

    def predict(self, file_path):
        # 如果使用PyTorch预测器
        if hasattr(self, 'use_torch_predictor') and self.use_torch_predictor:
            return self.torch_predictor.predict(file_path)
            
        print(f"使用MDX模型处理文件: {file_path}")
        try:
            mix, rate = librosa.load(file_path, mono=False, sr=44100)
            
            if mix.ndim == 1:
                mix = np.asfortranarray([mix, mix])
            
            mix = mix.T
            print(f"音频数据形状: {mix.shape}, 采样率: {rate}")
            print(f"使用的dim_t值: {self.model_.dim_t}, 计算得到的时间维度: {2**self.params['dim_t']}")
            print(f"运行设备: {self.device}")
            
            try:
                sources = self.demix(mix.T)
                opt = sources[0].T
                return (mix - opt, opt, rate)
            except Exception as e:
                print(f"MDX模型处理过程中出错: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
                raise
        except Exception as e:
            print(f"加载或处理音频文件时出错: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            raise

# 使用PyTorch+MPS的MDX预测器类
class MDXTorchPredictor:
    def __init__(self, model_path, params, device="mps"):
        # 确保torch被导入
        import torch
        
        self.model_path = model_path
        self.params = params
        self.device_str = device
        
        # 优化批处理大小 - 根据设备可用内存调整
        self.batch_size = params.get("batch_size", 4)  # 默认批处理大小为4
        print(f"使用批处理大小: {self.batch_size}")
        
        # 创建ConvTDFNet模型
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=params["dim_f"], 
            dim_t=params["dim_t"], 
            n_fft=params["n_fft"]
        )
        
        try:
            # 设置设备
            if device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("使用MPS设备加载PyTorch模型")
            elif device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("使用CUDA设备加载PyTorch模型")
            else:
                self.device = torch.device("cpu")
                print("使用CPU设备加载PyTorch模型")
            
            # 将ConvTDFNet模型移动到设备
            self.model_ = self.model_.to_device(self.device)
            print(f"ConvTDFNet模型已移动到{self.device_str}设备")
            
            # 检查是否存在缓存的PyTorch模型
            model_basename = os.path.basename(model_path)
            torch_model_dir = TORCH_MODELS_DIR / model_basename.split('.')[0]
            torch_model_path = torch_model_dir / f"{model_basename}.pt"
            torch_model_dir.mkdir(exist_ok=True)
            
            if os.path.exists(torch_model_path):
                print(f"加载缓存的PyTorch模型: {torch_model_path}")
                try:
                    self.torch_model = torch.load(torch_model_path, map_location=self.device)
                    print("成功加载缓存的PyTorch模型")
                except Exception as e:
                    print(f"加载缓存模型失败: {str(e)}, 将重新转换ONNX模型")
                    self._convert_and_save_model(model_path, torch_model_path)
            else:
                print(f"没有找到缓存的PyTorch模型，将转换ONNX模型并保存")
                self._convert_and_save_model(model_path, torch_model_path)
            
            # 设置为评估模式
            self.torch_model.eval()
            print(f"模型已准备好在{self.device_str}设备上运行")
            
        except Exception as e:
            print(f"在{device}上加载ONNX到PyTorch模型时出错: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            raise
    
    def _convert_and_save_model(self, onnx_path, torch_path):
        """将ONNX模型转换为PyTorch模型并保存"""
        import onnx
        from onnx2torch import convert
        
        print(f"加载ONNX模型: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        self.torch_model = convert(onnx_model)
        self.torch_model.to(self.device)
        
        # 测试一下模型
        print("测试模型转换结果...")
        dummy_input = torch.randn(1, 4, self.params["dim_f"], 2**self.params["dim_t"], device=self.device)
        with torch.no_grad():
            _ = self.torch_model(dummy_input)
        
        # 保存模型
        print(f"保存转换后的PyTorch模型到: {torch_path}")
        torch.save(self.torch_model, torch_path)
        print("模型保存成功")
    
    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.params["margin"]
        chunk_size = self.params["chunks"] * 44100
        
        assert not margin == 0, "margin cannot be zero!"
        
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.params["chunks"] == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources
    
    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        
        for mix_id, mix in enumerate(mixes):
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            
            # 准备数据批处理
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            
            # 使用批处理加速推理
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32).to(self.device)
            batch_results = []
            
            with torch.no_grad():
                # 在这里使用批处理处理数据
                for i in range(0, len(mix_waves), self.batch_size):
                    batch = mix_waves[i:i+self.batch_size]
                    # 使用STFT计算频谱
                    spek = model.stft(batch)
                    spek = spek.to(self.device)
                    
                    if self.params["denoise"]:
                        # 对于去噪操作，需要进行两次推理
                        neg_spek = -spek
                        pos_pred = self.torch_model(spek)
                        neg_pred = self.torch_model(neg_spek)
                        spec_pred = (pos_pred - neg_pred) * 0.5
                        tar_waves_batch = model.istft(spec_pred.cpu())
                    else:
                        # 单次推理
                        spec_pred = self.torch_model(spek)
                        tar_waves_batch = model.istft(spec_pred.cpu())
                    
                    batch_results.append(tar_waves_batch)
                
                # 合并批处理结果
                tar_waves = torch.cat(batch_results, dim=0)
                
                # 后处理
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu().numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                
                if margin_size == 0:
                    end = None
                
                sources.append(tar_signal[:, start:end])

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        
        return _sources
    
    def predict(self, file_path):
        print(f"使用PyTorch+MPS的MDX模型处理文件: {file_path}")
        try:
            mix, rate = librosa.load(file_path, mono=False, sr=44100)
            
            if mix.ndim == 1:
                mix = np.asfortranarray([mix, mix])
            
            mix = mix.T
            print(f"音频数据形状: {mix.shape}, 采样率: {rate}")
            print(f"使用的dim_t值: {self.model_.dim_t}, 计算得到的时间维度: {2**self.params['dim_t']}")
            print(f"运行设备: {self.device_str}, 批处理大小: {self.batch_size}")
            
            try:
                start_time = time.time()
                sources = self.demix(mix.T)
                processing_time = time.time() - start_time
                print(f"处理耗时: {processing_time:.2f}秒")
                
                opt = sources[0].T
                return (mix - opt, opt, rate)
            except Exception as e:
                print(f"PyTorch+MPS的MDX模型处理过程中出错: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
                raise
        except Exception as e:
            print(f"加载或处理音频文件时出错: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            raise

def get_device():
    return DEVICE

def load_model(model_name):
    """加载模型，使用缓存避免重复加载"""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    # 检查模型文件路径
    model_path = MODEL_PATHS.get(model_name)
    
    # 创建目标目录（如果不存在）
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_name}")
    
    # 根据模型类型创建对应的模型实例
    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"模型配置不存在: {model_name}")
    
    device = get_device()
    
    # 创建模型类的自定义实例
    try:
        # 根据模型类型创建不同的模型实例
        if model_config["type"] == "mdx":
            # 创建MDX预测器
            params = model_config["parameters"]
            mdx_predictor = MDXPredictor(
                model_path=model_path,
                params=params,
                device=device
            )
            model = {
                "predictor": mdx_predictor,
                "device": device,
                "model_path": model_path,
                "is_reverb_model": params.get("is_reverb_model", False)
            }
            
        elif model_config["type"] == "vr_network":
            # 为VR模型创建所需的目录结构
            params = model_config["parameters"]
            
            # 从VR_Network类实现类似功能
            from uvr.models_dir.vr_network import vr_interface as vr_api
            
            # 打印vr_api模块中的可用函数
            print(f"vr_api模块内容: {dir(vr_api)}")
            
            try:
                print(f"开始加载VR模型: {model_path}")
                
                # 获取模型hash
                model_hash = vr_api.get_model_hash_from_path(model_path)
                print(f"模型hash: {model_hash}")
                
                # 获取模型目录的绝对路径
                models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models_dir', 'vr_network')
                modelparams_dir = os.path.join(models_dir, 'modelparams')
                
                # 从模型数据中获取模型参数
                model_data = vr_api.MODELS_DATA.get(model_hash)
                if not model_data:
                    print(f"警告: 未在model_data.json中找到模型hash {model_hash}的信息")
                    # 尝试从已有模型中匹配
                    for key, data in vr_api.MODELS_DATA.items():
                        if data.get('vr_model_param') == '3band_44100_msb2' and data.get('is_karaoke', False):
                            model_data = data
                            print(f"使用替代模型数据: {key}")
                            break
                
                if not model_data:
                    raise ValueError(f"无法找到模型 {model_name} 的参数信息")
                
                # 构建参数文件的绝对路径
                param_file = os.path.join(modelparams_dir, f"{model_data['vr_model_param']}.json")
                print(f"参数文件路径: {param_file}")
                
                # 确保参数文件存在
                if not os.path.exists(param_file):
                    raise ValueError(f"参数文件不存在: {param_file}")
                
                # 创建ModelParameters实例
                mp = vr_api.ModelParameters(param_file)
                
                # 获取模型容量和模型类型信息
                is_vr_51_model, model_capacity = vr_api.get_capacity_and_vr_model(model_data)
                
                # 加载模型
                model_run = vr_api._load_model_with_hprams(model_path, mp, is_vr_51_model, model_capacity, device)
                
                # 获取stem信息
                primary_stem = model_data['primary_stem']
                secondary_stem = vr_api.get_secondary_stem(primary_stem)
                stems = {"primary_stem": primary_stem, "secondary_stem": secondary_stem}
                
                print("VR模型加载成功")
                
                # 打印模型参数
                print(f"模型参数(mp): {type(mp)}")
                if hasattr(mp, 'param'):
                    print(f"参数内容: {mp.param.keys()}")
                print(f"is_vr_51_model: {is_vr_51_model}")
                print(f"stems: {stems}")
                
                # 设置攻击性参数
                print("设置攻击性参数")
                aggressiveness = {
                    'value': params.get("aggressiveness", 0.05),
                    'split_bin': mp.param['band'][1]['crop_stop'],
                    'aggr_correction': mp.param.get('aggr_correction')
                }
                print(f"攻击性参数设置完成: {aggressiveness}")
                
                model = {
                    "model_run": model_run,
                    "mp": mp,
                    "is_vr_51_model": is_vr_51_model,
                    "stems": stems,
                    "aggressiveness": aggressiveness,
                    "device": device,
                    "sample_rate": mp.param['sr']
                }
                print("VR模型实例创建完成")
            except Exception as e:
                print(f"加载VR模型时出错: {str(e)}")
                print(f"错误跟踪: {traceback.format_exc()}")
                raise
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_config['type']}")
        
        # 保存到缓存
        MODEL_CACHE[model_name] = model
        return model
        
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}\n{traceback.format_exc()}")

def save_audio(audio_data, path, sample_rate):
    """保存音频文件的辅助函数，替代vr_api.save_audio"""
    try:
        # 确保路径存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 确保音频数据是numpy数组
        if not isinstance(audio_data, np.ndarray):
            print(f"警告: 音频数据不是numpy数组，类型为: {type(audio_data)}")
            audio_data = np.array(audio_data)
        
        # 检查数据形状
        print(f"音频数据形状: {audio_data.shape}, 数据类型: {audio_data.dtype}")
        
        # 确保数据形状正确
        if len(audio_data.shape) == 1:
            # 单通道，转换为2D数组
            print("音频数据是单通道，转换为2D数组")
            audio_data = np.expand_dims(audio_data, axis=0)
        elif len(audio_data.shape) > 2:
            # 维度太多，保留前两个维度
            print(f"警告: 音频数据维度过多: {audio_data.shape}，保留前两个维度")
            audio_data = audio_data[:2]
        
        # 确保是浮点数格式
        if not np.issubdtype(audio_data.dtype, np.floating):
            print(f"警告: 音频数据不是浮点数类型，将转换为float32")
            audio_data = audio_data.astype(np.float32)
        
        # 标准化音频数据（如果需要）
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0 and max_val > 0:
            print(f"标准化音频数据，最大值: {max_val}")
            audio_data = audio_data / max_val
        
        # 保存音频文件
        print(f"开始保存音频到: {path}, 采样率: {sample_rate}")
        
        # 检查转置需求
        if audio_data.shape[0] == 2:  # 如果是立体声格式 [2, n_samples]
            print(f"检测到立体声音频数据，转置为 [n_samples, 2] 格式")
            sf.write(path, audio_data.T, sample_rate)
        else:
            print(f"保存音频数据，形状: {audio_data.shape}")
            sf.write(path, audio_data, sample_rate)
            
        # 验证文件是否成功保存
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"音频已成功保存至 {path}，文件大小: {size_kb:.2f} KB")
            return True
        else:
            print(f"错误: 音频文件似乎未能保存: {path}")
            return False
    except Exception as e:
        print(f"保存音频时出错: {str(e)}")
        print(f"错误跟踪: {traceback.format_exc()}")
        return False

def process_audio(audio_path, model_name, output_path):
    """处理音频文件"""
    try:
        print("\n========== 开始音频处理 ==========")
        print(f"[步骤1] 加载模型: {model_name}")
        model = load_model(model_name)
        model_config = MODEL_CONFIGS.get(model_name)
        
        if model_config["type"] == "mdx":
            print("[步骤2-MDX] 开始使用MDX模型处理")
            # 使用MDX模型处理
            predictor = model["predictor"]
            print(f"[步骤2.1-MDX] 将音频文件传递给MDX预测器: {audio_path}")
            no_vocals, vocals, sr = predictor.predict(audio_path)
            
            # 保存结果文件
            output_dir = os.path.dirname(output_path)
            base_name = os.path.basename(output_path).split('.')[0]
            
            # 判断是否为混响模型
            is_reverb_model = model.get("is_reverb_model", False)
            print(f"[步骤2.2-MDX] 处理完成，是否为混响模型: {is_reverb_model}")
            
            if is_reverb_model:
                # 混响模型输出文件命名
                vocal_path = os.path.join(output_dir, f"{base_name}_(No Reverb).wav")
                reverb_path = os.path.join(output_dir, f"{base_name}_(Reverb).wav")
                
                print(f"[步骤3.1-MDX] 保存无混响人声: {vocal_path}")
                # 保存无混响人声文件（实际上是混响人声）
                sf.write(reverb_path, vocals, sr)
                
                print(f"[步骤3.2-MDX] 保存混响内容: {reverb_path}")
                # 保存混响文件（实际上是无混响人声）
                sf.write(vocal_path, no_vocals, sr)
                
                print(f"[步骤4-MDX] 返回处理结果")
                return {
                    "Vocals": vocal_path,  # 无混响人声作为人声输出
                    "Instrumental": reverb_path  # 混响人声作为伴奏输出
                }
            else:
                # 普通分离模型的处理
                vocal_path = os.path.join(output_dir, f"{base_name}_Vocals.wav")
                print(f"[步骤3.1-MDX] 保存人声: {vocal_path}")
                sf.write(vocal_path, vocals, sr)
                
                # 保存伴奏文件
                instrumental_path = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
                print(f"[步骤3.2-MDX] 保存伴奏: {instrumental_path}")
                sf.write(instrumental_path, no_vocals, sr)
                
                print(f"[步骤4-MDX] 返回处理结果")
                return {
                    "Vocals": vocal_path,
                    "Instrumental": instrumental_path
                }
            
        elif model_config["type"] == "vr_network":
            print("[步骤2-VR] 开始使用VR模型处理")
            from uvr.models_dir.vr_network import vr_interface as vr_api
            
            # 使用VR模型处理
            print(f"[步骤2.1-VR] 音频路径: {audio_path}, 模型: {model_name}")
            inference_params = {
                "wav_type_set": 'PCM_U8',
                "window_size": 512,
                "post_process_threshold": None,
                "batch_size": 4,
                "is_tta": False,
                "normaliz": False,
                "high_end_process": False,
            }
            
            try:
                # 加载混音并进行推理
                print("[步骤2.2-VR] 加载混音文件...")
                inp, input_high_end, input_high_end_h = vr_api.loading_mix(
                    audio_path,
                    model["mp"],
                    model["is_vr_51_model"],
                    wav_type_set=inference_params["wav_type_set"],
                    high_end_process=inference_params["high_end_process"]
                )
                print("[步骤2.3-VR] 混音文件加载成功，开始推理...")
                
                # 进行推理
                y_spec, v_spec = vr_api.inference_vr(
                    X_spec=inp,
                    aggressiveness=model["aggressiveness"],
                    window_size=inference_params["window_size"],
                    model_run=model["model_run"],
                    is_tta=inference_params["is_tta"],
                    batch_size=inference_params["batch_size"],
                    post_process_threshold=inference_params["post_process_threshold"],
                    primary_stem=model["stems"]["primary_stem"],
                    device=model["device"]
                )
                print("[步骤2.4-VR] 推理完成，准备生成音频...")
                
                # 获取音频词典
                audio_res = vr_api.get_audio_dict(
                    y_spec=y_spec,
                    v_spec=v_spec,
                    stems=model["stems"],
                    model_params=model["mp"],
                    normaliz=inference_params["normaliz"],
                    is_vr_51_model=model["is_vr_51_model"],
                    high_end_process=inference_params["high_end_process"],
                    input_high_end=input_high_end,
                    input_high_end_h=input_high_end_h
                )
                
                # 打印调试信息
                print(f"[步骤3-VR] 获取音频数据完成，返回音轨:")
                for key, value in audio_res.items():
                    print(f" - 音轨: {key}, 类型: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"   形状: {value.shape}")
                    elif isinstance(value, list):
                        print(f"   列表长度: {len(value)}")
                        if len(value) > 0:
                            print(f"   列表第一个元素类型: {type(value[0])}")
                
                # 保存结果
                output_dir = os.path.dirname(output_path)
                base_name = os.path.basename(output_path).split('.')[0]
                
                # 创建一个标准化的输出结果字典，使用首字母大写的键名
                standardized_result = {}
                
                # 规范化音轨名称并保存文件
                # 检查各种可能的键名（大小写不同的变体）
                key_mapping = {
                    'vocals': 'Vocals', 
                    'instrumental': 'Instrumental',
                    'Vocals': 'Vocals', 
                    'Instrumental': 'Instrumental',
                    'primary_source': 'Vocals',
                    'secondary_source': 'Instrumental'
                }
                
                print(f"[步骤3.1-VR] 原始音轨名称: {list(audio_res.keys())}")
                
                # 遍历原始结果字典，标准化键名
                for original_key, value in audio_res.items():
                    if original_key in key_mapping:
                        standardized_key = key_mapping[original_key]
                        standardized_result[standardized_key] = value
                
                print(f"[步骤3.2-VR] 标准化后的音轨名称: {list(standardized_result.keys())}")
                
                # 保存人声文件
                if "Vocals" in standardized_result:
                    vocal_path = os.path.join(output_dir, f"{base_name}_Vocals.wav")
                    print(f"[步骤4.1-VR] 保存人声文件: {vocal_path}")
                    save_audio(standardized_result["Vocals"], vocal_path, model["sample_rate"])
                
                # 保存伴奏文件
                if "Instrumental" in standardized_result:
                    instrumental_path = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
                    print(f"[步骤4.2-VR] 保存伴奏文件: {instrumental_path}")
                    save_audio(standardized_result["Instrumental"], instrumental_path, model["sample_rate"])
                
                print("[步骤5-VR] 音频文件保存完成，返回结果")
                
                # 构建返回结果，确保使用文件路径
                result = {}
                
                if "Vocals" in standardized_result:
                    vocal_path = os.path.join(output_dir, f"{base_name}_Vocals.wav")
                    if os.path.exists(vocal_path):
                        print(f"[步骤5.1-VR] 添加人声文件路径到结果: {vocal_path}")
                        result["Vocals"] = vocal_path
                    else:
                        print(f"[步骤5.1-VR] 警告：人声文件不存在: {vocal_path}")
                
                if "Instrumental" in standardized_result:
                    instrumental_path = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
                    if os.path.exists(instrumental_path):
                        print(f"[步骤5.2-VR] 添加伴奏文件路径到结果: {instrumental_path}")
                        result["Instrumental"] = instrumental_path
                    else:
                        print(f"[步骤5.2-VR] 警告：伴奏文件不存在: {instrumental_path}")
                
                print(f"[完成] 返回最终结果: {result}")
                return result
                
            except Exception as e:
                print(f"[错误-VR] 处理VR模型音频时出错: {str(e)}")
                print(f"[错误-VR] 错误跟踪: {traceback.format_exc()}")
                raise
            
    except Exception as e:
        import traceback
        print(f"[严重错误] 处理音频失败: {str(e)}")
        print(f"[严重错误] 错误跟踪: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}\n{traceback.format_exc()}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """返回HTML测试页面"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/models")
async def get_models():
    """获取可用模型列表"""
    return {
        "models": list(MODEL_CONFIGS.keys())
    }

@app.get("/clear_cache")
async def clear_model_cache():
    """清除模型缓存，确保使用最新的模型配置"""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    return {"status": "success", "message": "模型缓存已清除"}

@app.post("/separate")
async def separate_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """分离音频文件"""
    # 检查模型是否支持
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")
    
    # 创建临时目录存储上传和结果文件
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    temp_path = Path(temp_dir)
    
    try:
        # 保存上传的文件
        input_path = temp_path / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 创建输出文件名
        output_base = os.path.splitext(file.filename)[0]
        output_path = temp_path / f"{output_base}_output.wav"
        
        # 处理音频
        print(f"开始处理音频文件: {input_path}, 使用模型: {model}, 输出路径: {output_path}")
        result = process_audio(str(input_path), model, str(output_path))
        print(f"音频处理完成, 结果类型: {type(result)}")
        
        # 返回处理结果文件的路径
        output_files = {}
        
        # 模型类型不同，输出文件名也不同
        model_config = MODEL_CONFIGS.get(model)
        if model_config["type"] == "mdx":
            is_reverb_model = model_config["parameters"].get("is_reverb_model", False)
            
            if is_reverb_model:
                # 混响模型输出文件命名
                vocal_path = temp_path / f"{output_base}_output_(No Reverb).wav"
                reverb_path = temp_path / f"{output_base}_output_(Reverb).wav"
                
                if os.path.exists(vocal_path):
                    output_files["vocals"] = f"/download?path={vocal_path}"
                if os.path.exists(reverb_path):
                    output_files["instrumental"] = f"/download?path={reverb_path}"
            else:
                # 常规MDX模型输出文件命名
                vocal_path = temp_path / f"{output_base}_output_Vocals.wav"
                inst_path = temp_path / f"{output_base}_output_Instrumental.wav"
                
                if os.path.exists(vocal_path):
                    output_files["vocals"] = f"/download?path={vocal_path}"
                if os.path.exists(inst_path):
                    output_files["instrumental"] = f"/download?path={inst_path}"
                
        elif model_config["type"] == "vr_network":
            # VR模型的输出文件路径检测 (对于VR模型，我们使用固定的大写首字母命名)
            vocal_path = str(temp_path / f"{output_base}_output_Vocals.wav")
            inst_path = str(temp_path / f"{output_base}_output_Instrumental.wav")
            
            print(f"检查VR模型输出文件:")
            print(f"  - 人声文件路径: {vocal_path}, 存在: {os.path.exists(vocal_path)}")
            print(f"  - 伴奏文件路径: {inst_path}, 存在: {os.path.exists(inst_path)}")
            
            if os.path.exists(vocal_path):
                output_files["vocals"] = f"/download?path={vocal_path}"
                print(f"找到人声文件: {vocal_path}")
            
            if os.path.exists(inst_path):
                output_files["instrumental"] = f"/download?path={inst_path}"
                print(f"找到伴奏文件: {inst_path}")
        
            # 如果仍然找不到，则列出目录内容寻找匹配文件
            if not output_files:
                print("没有找到标准命名的输出文件，列出目录内容:")
                found_files = {}
                for file_item in os.listdir(temp_path):
                    file_path = str(temp_path / file_item)
                    print(f"  - {file_item}")
                    
                    # 检查是否包含相关关键词的文件
                    file_lower = file_item.lower()
                    if "vocal" in file_lower:
                        found_files["vocals"] = file_path
                        print(f"找到人声文件: {file_path}")
                    elif "instrument" in file_lower:
                        found_files["instrumental"] = file_path
                        print(f"找到伴奏文件: {file_path}")
                
                # 使用找到的文件
                for key, path in found_files.items():
                    output_files[key] = f"/download?path={path}"
        
        # 确保输出文件字典包含有效的文件路径
        print(f"最终输出文件: {output_files}")
        
        # 确保所有的值都是字符串路径，而不是文件对象
        for key in list(output_files.keys()):
            value = output_files[key]
            # 检查是否为字符串
            if not isinstance(value, str):
                # 如果是文件对象，尝试获取文件名
                if hasattr(value, 'filename'):
                    output_files[key] = f"/download?path={value.filename}"
                    print(f"将文件对象转换为路径: {output_files[key]}")
                # 否则，将其删除
                else:
                    print(f"警告: 删除非字符串路径项 {key}: {value} (类型: {type(value)})")
                    del output_files[key]
        
        return {
            "model": model,
            "original_filename": file.filename,
            "files": output_files,
            "temp_dir": str(temp_path)
        }
        
    except Exception as e:
        # 删除临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        import traceback
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}\n{traceback.format_exc()}")

@app.get("/download")
async def download_file(path: str):
    """下载处理后的文件"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(path=path, filename=os.path.basename(path))

@app.get("/cleanup")
async def cleanup_temp_files(dir_path: str):
    """清理临时文件"""
    # 检查路径是否是temp文件夹或其子文件夹
    if dir_path == "temp" or (dir_path.startswith(str(TEMP_DIR)) and os.path.exists(dir_path)):
        # 如果是temp根目录，清空内容但不删除目录本身
        if dir_path == "temp":
            for item in os.listdir(TEMP_DIR):
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path):
                    os.remove(item_path)
            return {"status": "success", "message": "临时文件已清理，保留temp目录"}
        else:
            # 如果是子目录，则可以整个删除
            shutil.rmtree(dir_path, ignore_errors=True)
            return {"status": "success", "message": "临时文件已清理"}
    else:
        raise HTTPException(status_code=400, detail="无效的目录路径")

def rename_audio_res_dict(audio_res:dict, names:dict)->dict:
    """
    Rename the keys of the audio results dictionary.

    Args:
        audio_res (dict): The audio results dictionary.
        names (dict): A dictionary containing the new names for the audio sources.

    Returns:
        dict: The renamed audio results dictionary.
    """
    primary_name = names["primary_stem"]
    secondary_name = names["secondary_stem"]
    audio_res = {primary_name: audio_res["primary_stem"], 
                 secondary_name: audio_res["secondary_stem"]}
    
    # 使用首字母大写的格式 (确保键名大小写一致)
    standard_audio_res = {}
    for k, v in audio_res.items():
        # 将首字母转为大写
        standard_key = k[0].upper() + k[1:] if len(k) > 0 else k
        standard_audio_res[standard_key] = v
        
    return standard_audio_res

@app.post("/separate_from_path")
async def separate_audio_from_path(
    file_path: str = Form(...),
    model: str = Form(...),
    batch_size: Optional[int] = Form(None),
    chunks: Optional[int] = Form(None),
    aggressiveness: Optional[float] = Form(None),
    custom_name: Optional[str] = Form(None)  # 添加自定义文件名参数
):
    """从文件路径分离音频文件"""
    # 检查模型是否支持
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
    
    # 创建临时目录存储结果文件
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    temp_path = Path(temp_dir)
    
    try:
        # 获取输入文件名
        input_filename = os.path.basename(file_path)
        
        # 创建输出文件名
        if custom_name:
            output_path = temp_path / f"{custom_name}.wav"
        else:
            output_base = os.path.splitext(input_filename)[0]
            output_path = temp_path / f"{output_base}_output.wav"
        
        # 根据模型类型设置参数
        model_config = MODEL_CONFIGS.get(model)
        params = model_config["parameters"].copy()
        
        # 更新参数（如果提供）
        if model_config["type"] == "mdx":
            if batch_size is not None:
                params["batch_size"] = batch_size
            if chunks is not None:
                params["chunks"] = chunks
        elif model_config["type"] == "vr_network":
            if aggressiveness is not None:
                params["aggressiveness"] = aggressiveness
        
        # 处理音频
        print(f"开始处理音频文件: {file_path}, 使用模型: {model}, 输出路径: {output_path}")
        result = process_audio(str(file_path), model, str(output_path))
        print(f"音频处理完成, 结果: {result}")
        
        # 如果指定了自定义名称，重命名结果文件
        output_files = {}
        
        if custom_name and "Vocals" in result and "Instrumental" in result:
            # 获取原始文件路径
            original_vocals_path = result["Vocals"]
            original_instrumental_path = result["Instrumental"]
            
            # 根据模型类型决定如何命名文件
            vocals_suffix = ""
            instrumental_suffix = ""
            
            if model == "Reverb_HQ_By_FoxJoy" or model_config.get("parameters", {}).get("is_reverb_model", False):
                vocals_suffix = ".wav"
                instrumental_suffix = ".wav"
            else:
                vocals_suffix = "_Vocals.wav" if not "Vocals.wav" in original_vocals_path else ".wav"
                instrumental_suffix = "_Instrumental.wav" if not "Instrumental.wav" in original_instrumental_path else ".wav"
            
            # 创建新文件路径
            vocals_path = temp_path / f"{custom_name}{vocals_suffix}"
            instrumental_path = temp_path / f"{custom_name}{instrumental_suffix}"
            
            # 复制文件到新路径
            try:
                shutil.copy(original_vocals_path, vocals_path)
                shutil.copy(original_instrumental_path, instrumental_path)
                
                # 更新结果路径
                output_files["vocals"] = f"/download?path={vocals_path}"
                output_files["instrumental"] = f"/download?path={instrumental_path}"
            except Exception as e:
                print(f"重命名文件时出错: {str(e)}")
                # 如果重命名失败，使用原始文件路径
                output_files["vocals"] = f"/download?path={original_vocals_path}"
                output_files["instrumental"] = f"/download?path={original_instrumental_path}"
        else:
            # 检查结果中的各种可能的键值
            # VR和MDX模型输出结构可能不同
            if "Vocals" in result:
                vocal_path = result["Vocals"]
                output_files["vocals"] = f"/download?path={vocal_path}"
            
            if "Instrumental" in result:
                inst_path = result["Instrumental"]
                output_files["instrumental"] = f"/download?path={inst_path}"
        
        return {
            "model": model,
            "original_filename": input_filename,
            "files": output_files,
            "temp_dir": str(temp_path)
        }
        
    except Exception as e:
        # 删除临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        import traceback
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}\n{traceback.format_exc()}")

@app.post("/mix_audio")
async def mix_audio_files(
    file_path1: str = Form(...),
    file_path2: str = Form(...),
    mix_ratio: float = Form(0.5),  # 默认混合比例为0.5
    custom_name: Optional[str] = Form(None)  # 可选的自定义文件名
):
    """混合两个音频文件"""
    # 检查文件是否存在
    if not os.path.exists(file_path1):
        raise HTTPException(status_code=404, detail=f"文件1不存在: {file_path1}")
    
    if not os.path.exists(file_path2):
        raise HTTPException(status_code=404, detail=f"文件2不存在: {file_path2}")
    
    # 创建临时目录存储结果文件
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    temp_path = Path(temp_dir)
    
    try:
        # 获取输入文件名
        input_filename1 = os.path.basename(file_path1)
        original_name = os.path.splitext(input_filename1)[0]
        
        # 如果路径中包含多层目录结构，提取原始文件名
        if "_output_" in original_name:
            original_name = original_name.split("_output_")[0]
        
        # 创建输出文件名
        if custom_name:
            output_filename = f"{custom_name}.wav"
        else:
            output_filename = f"{original_name}-伴奏.wav"
        
        output_path = temp_path / output_filename
        
        # 加载两个音频文件
        print(f"加载音频文件1: {file_path1}")
        audio1, sr1 = librosa.load(file_path1, sr=None, mono=False)
        
        print(f"加载音频文件2: {file_path2}")
        audio2, sr2 = librosa.load(file_path2, sr=None, mono=False)
        
        # 确保采样率相同
        if sr1 != sr2:
            print(f"采样率不同，将文件2重采样到{sr1}Hz")
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        
        # 处理音频数组形状
        if audio1.ndim == 1:
            audio1 = np.expand_dims(audio1, axis=0)
        if audio2.ndim == 1:
            audio2 = np.expand_dims(audio2, axis=0)
        
        # 调整音频长度为较短的一个
        min_length = min(audio1.shape[1], audio2.shape[1])
        audio1 = audio1[:, :min_length]
        audio2 = audio2[:, :min_length]
        
        # 混合音频
        print(f"混合音频文件，比例: {mix_ratio}")
        mixed_audio = audio1 * mix_ratio + audio2 * (1.0 - mix_ratio)
        
        # 归一化
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.9  # 防止音频过大产生剪切
        
        # 检查音频形状
        print(f"混合音频形状: {mixed_audio.shape}, 数据类型: {mixed_audio.dtype}")
        
        # 如果是多通道，需要转置为(n_samples, n_channels)格式
        if mixed_audio.ndim > 1 and mixed_audio.shape[0] == 2:
            print("转置音频数据为(n_samples, n_channels)格式")
            mixed_audio = mixed_audio.T
        
        # 保存混合后的音频 - 显式指定格式为WAV
        print(f"保存混合后的音频到: {output_path}")
        sf.write(str(output_path), mixed_audio, sr1, format='WAV', subtype='PCM_16')
        
        # 返回混合后文件的相对路径
        return {
            "mixed_file": f"/download?path={output_path}",
            "temp_dir": str(temp_path)
        }
        
    except Exception as e:
        # 删除临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        import traceback
        raise HTTPException(status_code=500, detail=f"混合音频失败: {str(e)}\n{traceback.format_exc()}")

def main():
    parser = argparse.ArgumentParser(description="Ultimate Vocal Remover API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API服务主机地址")
    parser.add_argument("--port", type=int, default=6006, help="API服务端口")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "mps", "cpu"],
                        help="使用的设备类型：cuda, mps 或 cpu")
    
    args = parser.parse_args()
    
    # 设置全局设备
    global DEVICE
    DEVICE = args.device
    
    # 打印环境信息
    print("\n=============== Ultimate Vocal Remover API ===============")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    print(f"使用设备: {DEVICE}")
    
    # 检查CoreML支持
    if DEVICE == "mps":
        try:
            providers = ort.get_available_providers()
            print(f"可用的ONNX执行提供程序: {providers}")
            if 'CoreMLExecutionProvider' in providers:
                print("✅ CoreML执行提供程序可用，ONNX模型可以在MPS上加速")
            else:
                print("⚠️ 警告：CoreML执行提供程序不可用，ONNX模型将使用CPU")
                print("  建议安装onnxruntime-silicon以获取MPS加速")
        except Exception as e:
            print(f"检查CoreML支持时出错: {str(e)}")
    
    # 检查临时目录是否存在
    print(f"\n临时目录信息:")
    print(f" - 路径: {TEMP_DIR}")
    print(f" - 是否存在: {os.path.exists(TEMP_DIR)}")
    print(f" - 是否可写: {os.access(TEMP_DIR, os.W_OK) if os.path.exists(TEMP_DIR) else False}")
    
    # 打印模型路径信息
    print("\n模型路径信息:")
    for model_name, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        print(f" - {model_name}: {path} {'[√]' if exists else '[×]'}")
    
    # 检查模型参数文件
    print("\nVR模型参数文件信息:")
    vr_modelparams_dir = os.path.join("src", "models_dir", "vr_network", "modelparams")
    if os.path.exists(vr_modelparams_dir):
        param_files = [f for f in os.listdir(vr_modelparams_dir) if f.endswith('.json')]
        for pf in param_files:
            print(f" - {pf} [{os.path.getsize(os.path.join(vr_modelparams_dir, pf))} 字节]")
    else:
        print(f" 参数目录不存在: {vr_modelparams_dir}")
        print(f" 尝试创建目录...")
        try:
            os.makedirs(vr_modelparams_dir, exist_ok=True)
            print(f" 目录创建成功: {vr_modelparams_dir}")
        except Exception as e:
            print(f" 创建目录失败: {str(e)}")
    
    # 启动服务
    print(f"\n服务启动中... 主机: {args.host}, 端口: {args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    
if __name__ == "__main__":
    main() 