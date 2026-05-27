import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        prob_windows = F.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1)


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
        prob_windows = F.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1)


class LightweightFeatureExtractor(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.stem(x)


class SWRImageEncoder(nn.Module):
    def __init__(self, base_encoder, gating_net, window_size=32, threshold=0.5, gating_on_image=False):
        super().__init__()
        self.base_encoder = base_encoder
        self.gating_net = gating_net
        self.window_size = window_size
        self.threshold = threshold
        self.gating_on_image = gating_on_image
        
        test_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            full_out = base_encoder(test_input)
        
        self.out_channels = full_out.shape[1]
        self.out_height = full_out.shape[2]
        self.out_width = full_out.shape[3]
        
        self.lightweight_extractor = LightweightFeatureExtractor(out_channels=self.out_channels)
    
    def forward(self, x):
        with torch.no_grad():
            light_out = self.lightweight_extractor(x)
            light_out = F.interpolate(light_out, size=(self.out_height, self.out_width), 
                                     mode='bilinear', align_corners=False)
        
        if self.gating_on_image:
            window_probs = self.gating_net(x)
        else:
            window_probs = self.gating_net(light_out)
        
        route_mask = (window_probs > self.threshold)
        
        B, C, H, W = light_out.shape
        window_h, window_w = self.window_size, self.window_size
        num_h, num_w = H // window_h, W // window_w
        
        output = light_out.clone()
        
        foreground_windows = []
        for b in range(B):
            for i in range(num_h):
                for j in range(num_w):
                    window_idx = i * num_w + j
                    if route_mask[b, window_idx]:
                        foreground_windows.append((b, i, j))
        
        if len(foreground_windows) > 0:
            with torch.no_grad():
                full_out = self.base_encoder(x)
            
            for b, i, j in foreground_windows:
                h1, h2 = i * window_h, (i + 1) * window_h
                w1, w2 = j * window_w, (j + 1) * window_w
                output[b, :, h1:h2, w1:w2] = full_out[b, :, h1:h2, w1:w2]
        
        return output, window_probs


def build_swr_model(base_model, gating_net, threshold=0.5, gating_on_image=False):
    class SWRWrapper(nn.Module):
        def __init__(self, base_model, gating_net, gating_on_image):
            super().__init__()
            self.base_model = base_model
            self.swr_encoder = SWRImageEncoder(base_model.image_encoder, gating_net, threshold=threshold, gating_on_image=gating_on_image)
        
        def forward(self, x):
            return self.swr_encoder(x)
    
    swr_model = SWRWrapper(base_model, gating_net, gating_on_image)
    swr_model.eval()
    return swr_model


def benchmark_swr_inference():
    print("=" * 70)
    print("SWR 动态推理性能基准测试 - 比较本地 vs Colab训练")
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
    
    print("\n2. 加载本地训练的门控网络...")
    gating_local = GatingNetworkLocal(in_channels=in_channels)
    gating_local.load_state_dict(torch.load('/home/zhang/vista-slam/pruned_sam/weights/swr_gating.pth', map_location='cpu'))
    gating_local.eval()
    swr_local = build_swr_model(base_model, gating_local, threshold=0.3)
    
    print("\n3. 加载Colab训练的门控网络...")
    gating_colab = GatingNetworkColab(in_channels=3)
    gating_colab.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model.pth', map_location='cpu'))
    gating_colab.eval()
    swr_colab = build_swr_model(base_model, gating_colab, threshold=0.3, gating_on_image=True)
    
    test_input = torch.randn(1, 3, 1024, 1024).to(device)
    iterations = 10
    warmup = 3
    
    print("\n4. 基准测试 - 原始编码器:")
    with torch.no_grad():
        for _ in range(warmup):
            base_model.image_encoder(test_input)
        
        import time
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            base_model.image_encoder(test_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        print(f"   平均耗时: {sum(times)/len(times):.2f}ms")
    
    print("\n5. 基准测试 - 本地SWR编码器:")
    with torch.no_grad():
        for _ in range(warmup):
            swr_local(test_input)
        
        times = []
        routed_ratios = []
        for _ in range(iterations):
            start = time.perf_counter()
            _, probs = swr_local(test_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            routed_ratios.append((probs > 0.3).float().mean().item())
        
        print(f"   平均耗时: {sum(times)/len(times):.2f}ms")
        print(f"   平均前景窗口比例: {sum(routed_ratios)/len(routed_ratios)*100:.1f}%")
    
    print("\n6. 基准测试 - Colab SWR编码器:")
    with torch.no_grad():
        for _ in range(warmup):
            swr_colab(test_input)
        
        times = []
        routed_ratios = []
        for _ in range(iterations):
            start = time.perf_counter()
            _, probs = swr_colab(test_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            routed_ratios.append((probs > 0.3).float().mean().item())
        
        print(f"   平均耗时: {sum(times)/len(times):.2f}ms")
        print(f"   平均前景窗口比例: {sum(routed_ratios)/len(routed_ratios)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("SWR性能基准测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    benchmark_swr_inference()