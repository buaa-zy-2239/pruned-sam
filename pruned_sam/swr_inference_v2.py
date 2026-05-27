import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam


def check_encoder_output():
    """检查编码器输出形状"""
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    test_input = torch.randn(1, 3, 1024, 1024)
    
    with torch.no_grad():
        output = model.image_encoder(test_input)
    
    print(f"编码器输出形状: {output.shape}")
    return output.shape


class GatingModule(nn.Module):
    """预测每个窗口的前景概率"""
    def __init__(self, in_channels=256, window_size=64):
        super().__init__()
        self.window_size = window_size
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        prob_map = self.gate(x)
        prob_windows = F.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1)


class ShortcutBranch(nn.Module):
    """背景窗口的轻量处理"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.shortcut(x)


class SWRImageEncoder(nn.Module):
    """带稀疏窗口路由的图像编码器"""
    
    def __init__(self, base_encoder, window_size=32, threshold=0.5):
        super().__init__()
        self.base_encoder = base_encoder
        self.window_size = window_size
        self.threshold = threshold
        
        test_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            full_out = base_encoder(test_input)
        
        self.out_channels = full_out.shape[1]
        self.out_height = full_out.shape[2]
        self.out_width = full_out.shape[3]
        
        self.gating_module = GatingModule(in_channels=self.out_channels, window_size=window_size)
        self.shortcut_branch = ShortcutBranch(in_channels=self.out_channels, out_channels=self.out_channels)
    
    def forward(self, x):
        """稀疏窗口路由的前向传播"""
        with torch.no_grad():
            full_out = self.base_encoder(x)
        
        window_probs = self.gating_module(full_out)
        route_mask = (window_probs > self.threshold)
        
        B, C, H, W = full_out.shape
        window_h, window_w = self.window_size, self.window_size
        num_h, num_w = H // window_h, W // window_w
        
        output = full_out.clone()
        
        for b in range(B):
            for i in range(num_h):
                for j in range(num_w):
                    h1, h2 = i * window_h, (i + 1) * window_h
                    w1, w2 = j * window_w, (j + 1) * window_w
                    window_idx = i * num_w + j
                    
                    if not route_mask[b, window_idx]:
                        patch = full_out[b:b+1, :, h1:h2, w1:w2]
                        output[b, :, h1:h2, w1:w2] = self.shortcut_branch(patch).squeeze(0)
        
        return output


def test_swr_module():
    print("=" * 70)
    print("阶段三：稀疏窗口路由(SWR)测试")
    print("=" * 70)
    
    print("\n1. 检查编码器输出形状...")
    out_shape = check_encoder_output()
    print(f"   编码器输出形状: {out_shape}")
    
    print("\n2. 加载裁剪后的SAM模型...")
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    print("   ✅ 模型加载完成")
    
    print("\n3. 创建SWR编码器...")
    swr_encoder = SWRImageEncoder(model.image_encoder, window_size=32, threshold=0.3)
    swr_encoder.eval()
    print("   ✅ SWR编码器创建完成")
    
    print("\n4. 测试前向传播...")
    test_input = torch.randn(1, 3, 1024, 1024)
    
    with torch.no_grad():
        output = swr_encoder(test_input)
    
    print(f"   输入形状: {test_input.shape}")
    print(f"   输出形状: {output.shape}")
    print("   ✅ 前向传播成功")
    
    print("\n5. 测试不同阈值下的路由行为...")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        swr_encoder.threshold = threshold
        with torch.no_grad():
            window_probs = swr_encoder.gating_module(model.image_encoder(test_input))
            routed_ratio = (window_probs > threshold).float().mean().item()
        
        print(f"   阈值 {threshold}: 走完整路径的窗口比例 = {routed_ratio*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("SWR模块测试完成!")
    print("=" * 70)
    
    return swr_encoder


def benchmark_swr():
    print("\n" + "=" * 70)
    print("SWR性能基准测试")
    print("=" * 70)
    
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    swr_encoder = SWRImageEncoder(model.image_encoder, window_size=32, threshold=0.3)
    swr_encoder.eval()
    
    test_input = torch.randn(1, 3, 1024, 1024)
    iterations = 10
    warmup = 3
    
    print("\n原始编码器推理:")
    with torch.no_grad():
        for _ in range(warmup):
            model.image_encoder(test_input)
        
        import time
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            model.image_encoder(test_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        print(f"  平均耗时: {sum(times)/len(times):.2f}ms")
    
    print("\nSWR编码器推理:")
    with torch.no_grad():
        for _ in range(warmup):
            swr_encoder(test_input)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            swr_encoder(test_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        print(f"  平均耗时: {sum(times)/len(times):.2f}ms")


if __name__ == '__main__':
    swr_encoder = test_swr_module()
    benchmark_swr()
