import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam


class GatingNetworkLocal(nn.Module):
    def __init__(self, in_channels=256, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        prob_map = self.conv_layers(x)
        prob_windows = nn.functional.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1), prob_map


class GatingNetworkColab(nn.Module):
    def __init__(self, in_channels=3, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        prob_map = self.net(x)
        prob_windows = nn.functional.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1), prob_map


def test_gating_networks():
    print("=" * 70)
    print("门控网络诊断测试")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    print("\n1. 加载基础模型...")
    base_model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    base_model.eval()
    base_model.to(device)
    
    with torch.no_grad():
        test_input = torch.randn(1, 3, 1024, 1024)
        embedding = base_model.image_encoder(test_input)
        in_channels = embedding.shape[1]
        print(f"   编码器输出形状: {embedding.shape}")
    
    print("\n2. 测试本地训练的门控网络...")
    gating_local = GatingNetworkLocal(in_channels=in_channels)
    gating_local.load_state_dict(torch.load('/home/zhang/vista-slam/pruned_sam/weights/swr_gating.pth', map_location='cpu'))
    gating_local.eval()
    
    with torch.no_grad():
        window_probs, prob_map = gating_local(embedding)
    
    print(f"   窗口概率形状: {window_probs.shape}")
    print(f"   概率图形状: {prob_map.shape}")
    print(f"   窗口概率值: {window_probs.numpy()}")
    print(f"   概率范围: [{window_probs.min().item():.4f}, {window_probs.max().item():.4f}]")
    print(f"   平均值: {window_probs.mean().item():.4f}")
    
    print("\n3. 测试Colab训练的门控网络...")
    gating_colab = GatingNetworkColab(in_channels=3)
    gating_colab.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model.pth', map_location='cpu'))
    gating_colab.eval()
    
    with torch.no_grad():
        window_probs_colab, prob_map_colab = gating_colab(test_input)
    
    print(f"   窗口概率形状: {window_probs_colab.shape}")
    print(f"   概率图形状: {prob_map_colab.shape}")
    print(f"   概率范围: [{window_probs_colab.min().item():.4f}, {window_probs_colab.max().item():.4f}]")
    print(f"   平均值: {window_probs_colab.mean().item():.4f}")
    print(f"   小于0.5的窗口比例: {(window_probs_colab < 0.5).float().mean().item() * 100:.1f}%")
    
    print("\n4. 测试真实图像...")
    test_image_path = '/home/zhang/vista-slam/eval_data/test_100/000000000139.jpg'
    if os.path.exists(test_image_path):
        img = Image.open(test_image_path).convert('RGB')
        img_resized = img.resize((1024, 1024), Image.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding_real = base_model.image_encoder(img_tensor)
            window_probs_local, _ = gating_local(embedding_real)
            window_probs_colab_real, _ = gating_colab(img_tensor)
        
        print(f"   本地门控网络 - 概率范围: [{window_probs_local.min().item():.4f}, {window_probs_local.max().item():.4f}]")
        print(f"   本地门控网络 - 小于0.5的窗口比例: {(window_probs_local < 0.5).float().mean().item() * 100:.1f}%")
        print(f"   Colab门控网络 - 概率范围: [{window_probs_colab_real.min().item():.4f}, {window_probs_colab_real.max().item():.4f}]")
        print(f"   Colab门控网络 - 小于0.5的窗口比例: {(window_probs_colab_real < 0.5).float().mean().item() * 100:.1f}%")
    
    print("\n" + "=" * 70)
    print("诊断测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    test_gating_networks()