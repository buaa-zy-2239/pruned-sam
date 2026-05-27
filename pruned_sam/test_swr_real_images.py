import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam


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
    def __init__(self, base_encoder, gating_net, window_size=32, threshold=0.5):
        super().__init__()
        self.base_encoder = base_encoder
        self.gating_net = gating_net
        self.window_size = window_size
        self.threshold = threshold
        
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
            light_out = nn.functional.interpolate(light_out, size=(self.out_height, self.out_width), 
                                     mode='bilinear', align_corners=False)
        
        window_probs, _ = self.gating_net(x)
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


def load_real_images(image_dir, max_images=5):
    images = []
    for f in sorted(os.listdir(image_dir))[:max_images]:
        if f.endswith('.jpg'):
            img_path = os.path.join(image_dir, f)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((1024, 1024), Image.LANCZOS)
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
            images.append(img_tensor)
    return torch.stack(images)


def benchmark_real_images():
    print("=" * 70)
    print("SWR 真实图像性能基准测试")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    print("\n1. 加载基础模型...")
    base_model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    base_model.eval()
    base_model.to(device)
    
    print("\n2. 加载Colab训练的门控网络...")
    gating_colab = GatingNetworkColab(in_channels=3)
    gating_colab.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model.pth', map_location='cpu'))
    gating_colab.eval()
    
    print("\n3. 创建SWR编码器...")
    swr_encoder = SWRImageEncoder(base_model.image_encoder, gating_colab, window_size=32, threshold=0.5)
    swr_encoder.eval()
    
    print("\n4. 加载测试图像...")
    image_dir = '/home/zhang/vista-slam/eval_data/test_100'
    test_images = load_real_images(image_dir, max_images=5).to(device)
    print(f"   加载 {len(test_images)} 张图像")
    
    iterations = 5
    warmup = 2
    
    print("\n5. 基准测试 - 原始编码器:")
    with torch.no_grad():
        for _ in range(warmup):
            base_model.image_encoder(test_images[:1])
        
        import time
        times_orig = []
        for _ in range(iterations):
            start = time.perf_counter()
            base_model.image_encoder(test_images[:1])
            end = time.perf_counter()
            times_orig.append((end - start) * 1000)
        
        orig_time = sum(times_orig) / len(times_orig)
        print(f"   单图平均耗时: {orig_time:.2f}ms")
    
    print("\n6. 基准测试 - SWR编码器:")
    with torch.no_grad():
        for _ in range(warmup):
            swr_encoder(test_images[:1])
        
        times = []
        routed_ratios = []
        for _ in range(iterations):
            start = time.perf_counter()
            _, probs = swr_encoder(test_images[:1])
            end = time.perf_counter()
            times.append((end - start) * 1000)
            routed_ratios.append((probs > 0.5).float().mean().item())
        
        swr_time = sum(times) / len(times)
        print(f"   单图平均耗时: {swr_time:.2f}ms")
        print(f"   平均前景窗口比例: {sum(routed_ratios)/len(routed_ratios)*100:.1f}%")
        print(f"   加速比: {orig_time/swr_time:.2f}x")
    
    print("\n7. 测试不同阈值的效果:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        swr_encoder.threshold = threshold
        with torch.no_grad():
            times = []
            ratios = []
            for _ in range(5):
                start = time.perf_counter()
                _, probs = swr_encoder(test_images[:1])
                end = time.perf_counter()
                times.append((end - start) * 1000)
                ratios.append((probs > threshold).float().mean().item())
            
            avg_time = sum(times) / len(times)
            avg_ratio = sum(ratios) / len(ratios) * 100
            speedup = 591.10 / avg_time
            print(f"   阈值 {threshold}: 耗时={avg_time:.2f}ms, 前景比例={avg_ratio:.1f}%, 加速比={speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("真实图像测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    benchmark_real_images()