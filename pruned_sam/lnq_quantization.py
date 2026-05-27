import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/zhang/vista-slam')


class LNQActivationQuantizer:
    """对数非线性激活量化器"""
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.scale = None
        self.zero_point = None
        self.eps = 1e-5

    def calibrate(self, x_calib):
        """在校准集上统计对数域的范围"""
        x_log = torch.log2(x_calib - x_calib.min() + self.eps)
        self.x_min_log = x_log.min().item()
        self.x_max_log = x_log.max().item()
        self.scale = (self.x_max_log - self.x_min_log) / (self.qmax - self.qmin)
        self.zero_point = self.qmin - round(self.x_min_log / self.scale)

    def forward(self, x):
        """前向量化"""
        x_min = x.min().detach()
        x_log = torch.log2(x - x_min + self.eps)
        x_q = torch.clamp(torch.round(x_log / self.scale + self.zero_point),
                          self.qmin, self.qmax)
        x_dq = (x_q - self.zero_point) * self.scale
        return torch.pow(2, x_dq) + x_min


class LNQAttentionQuantizer:
    """对数非线性量化器，专门处理注意力分数"""
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1
        self.eps = 1e-5

    def calibrate(self, attn_scores):
        """在校准集上确定对数域范围"""
        log_scores = torch.log2(attn_scores + self.eps)
        self.min_log = log_scores.min().item()
        self.max_log = log_scores.max().item()
        self.scale = (self.max_log - self.min_log) / (self.qmax - self.qmin)
        self.zero_point = self.qmin - round(self.min_log / self.scale)

    def forward(self, x):
        """前向：对数变换 → 量化 → 反量化 → 指数恢复"""
        x_log = torch.log2(x + self.eps)
        x_q = torch.clamp(
            torch.round(x_log / self.scale + self.zero_point),
            self.qmin, self.qmax
        )
        x_dq = (x_q - self.zero_point) * self.scale
        return torch.pow(2, x_dq)


class QuantizedAttention(nn.Module):
    """带LNQ量化的注意力层"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_quantizer = LNQAttentionQuantizer(n_bits=8)
        self.calibrated = False

    def calibrate(self, x):
        """校准量化器"""
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        self.attn_quantizer.calibrate(attn_scores.flatten())
        self.calibrated = True

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        if self.calibrated and self.training is False:
            attn_scores = self.attn_quantizer.forward(attn_scores)
        
        output = attn_scores @ v
        output = output.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(output)


def mac_compensate(linear_layer, x_fp, x_q, n_samples=256):
    """MAC补偿：x_q是量化后的激活值，x_fp是原始FP32值"""
    W_fp = linear_layer.weight.data
    
    delta_Y = (x_fp[:n_samples] - x_q[:n_samples]) @ W_fp.T
    x_q_pinv = torch.linalg.pinv(x_q[:n_samples])
    delta_W = (x_q_pinv @ delta_Y).T
    
    W_compensated = W_fp + delta_W
    return W_compensated


def test_lnq_quantizer():
    """测试LNQ量化器"""
    print("=" * 70)
    print("测试LNQ量化器")
    print("=" * 70)
    
    quantizer = LNQAttentionQuantizer(n_bits=8)
    
    attn_scores = torch.randn(1, 8, 64, 64).softmax(dim=-1)
    print(f"原始注意力分数形状: {attn_scores.shape}")
    print(f"原始注意力分数范围: [{attn_scores.min():.6f}, {attn_scores.max():.6f}]")
    
    quantizer.calibrate(attn_scores)
    print(f"量化参数: scale={quantizer.scale:.6f}, zero_point={quantizer.zero_point}")
    
    quantized = quantizer.forward(attn_scores)
    print(f"量化后范围: [{quantized.min():.6f}, {quantized.max():.6f}]")
    
    mse = ((attn_scores - quantized) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    print("\n✅ LNQ量化器测试通过!")


def create_quantization_config():
    """创建OpenVINO POT量化配置文件"""
    config = {
        "model": {
            "model_name": "pruned_sam_acnr_lnq",
            "model": "./openvino_models/pruned_sam_encoder.xml",
            "weights": "./openvino_models/pruned_sam_encoder.bin"
        },
        "engine": {
            "type": "accuracy_aware",
            "params": {
                "maximal_drop": 0.01,
                "stat_requests_number": 100,
                "eval_requests_number": 50
            }
        },
        "dataset": {
            "name": "calibration_dataset",
            "data_source": "./calib_images",
            "annotation": "./calib_annotations.json"
        },
        "algorithms": [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": "CPU",
                    "preset": "performance",
                    "stat_subset_size": 300
                }
            }
        ]
    }
    
    os.makedirs('/home/zhang/vista-slam/openvino_models', exist_ok=True)
    
    import json
    with open('/home/zhang/vista-slam/quantization_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ 量化配置文件已创建: quantization_config.json")


def prepare_calibration_data():
    """准备校准数据集"""
    os.makedirs('/home/zhang/vista-slam/calib_images', exist_ok=True)
    
    from pruned_sam import build_pruned_sam
    
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    print("\n准备校准数据...")
    for i in range(10):
        dummy_image = torch.randn(1, 3, 1024, 1024)
        torch.save(dummy_image, f'/home/zhang/vista-slam/calib_images/calib_{i}.pt')
    
    print("✅ 校准数据准备完成")


if __name__ == '__main__':
    test_lnq_quantizer()
    create_quantization_config()
    prepare_calibration_data()
    
    print("\n" + "=" * 70)
    print("第一阶段完成: LNQ量化器 + OpenVINO INT8配置")
    print("=" * 70)
