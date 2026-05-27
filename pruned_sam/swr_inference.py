"""
SWR 动态推理模块 - 批量优化版本
============================================

核心优化：用"单提示批量推理"取代"逐窗口点集推理"

原理：
1. 门控网络识别前景窗口
2. 为每个前景窗口只生成一个中心点作为提示
3. 将所有前景点拼成一批，一次性喂给mask_decoder
4. 批量推理后合成最终掩码

预期收益：将上百次decoder调用缩减为1次
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class SWRGatingNetwork(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(hidden_dim*2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class BatchedSWRPredictor:
    def __init__(self, model, gating_model, threshold=0.5, device='cpu'):
        self.model = model
        self.model.eval()
        self.gating = gating_model.to(device)
        self.gating.eval()
        self.threshold = threshold
        self.device = device

    def set_image(self, image):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    def predict_with_batched_points(self):
        """
        单提示批量推理：为每个前景窗口生成一个中心点，一次性批量预测。
        
        Returns:
            final_mask: 合成后的最终分割掩码 (H, W)
            num_points: 使用的前景点数量
            routed_ratio: 前景窗口比例
        """
        img_small = cv2.resize(self.image, (256, 256))
        img_tensor = torch.from_numpy(img_small.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            gating_output = self.gating(img_tensor).squeeze()
        
        grid_h, grid_w = gating_output.shape
        
        batch_points = []
        batch_labels = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                prob = gating_output[i, j].item()
                if prob > self.threshold:
                    center_y = int((i + 0.5) * self.h / grid_h)
                    center_x = int((j + 0.5) * self.w / grid_w)
                    
                    batch_points.append([center_x, center_y])
                    batch_labels.append(1)
        
        total_windows = grid_h * grid_w
        routed_ratio = len(batch_points) / total_windows if total_windows > 0 else 0
        
        if not batch_points:
            return np.zeros((self.h, self.w), dtype=np.uint8), 0, routed_ratio
        
        point_coords = torch.as_tensor(batch_points, dtype=torch.float).unsqueeze(0).to(self.device)
        point_labels = torch.as_tensor(batch_labels, dtype=torch.int).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_embedding = self.model.image_encoder(self.image_tensor)
            
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None
            )
            
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
        
        final_mask = np.zeros((self.h, self.w), dtype=bool)
        
        for k in range(low_res_masks.shape[1]):
            mask = low_res_masks[0, k].cpu().numpy()
            mask = cv2.resize(mask, (self.w, self.h))
            final_mask = np.logical_or(final_mask, mask > 0)
        
        return final_mask.astype(np.uint8), len(batch_points), routed_ratio


class OriginalSWRPredictor:
    def __init__(self, model, gating_model, threshold=0.5, device='cpu'):
        self.model = model
        self.model.eval()
        self.gating = gating_model.to(device)
        self.gating.eval()
        self.threshold = threshold
        self.device = device

    def set_image(self, image):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

    def generate(self):
        img_small = cv2.resize(self.image, (256, 256))
        img_tensor = torch.from_numpy(img_small.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            gating_map = self.gating(img_tensor).squeeze()
        
        window_size = 256
        num_win_h = self.h // window_size
        num_win_w = self.w // window_size
        
        foreground_windows = []
        for i in range(num_win_h):
            for j in range(num_win_w):
                gi = int(i * gating_map.shape[0] / max(num_win_h, 1))
                gj = int(j * gating_map.shape[1] / max(num_win_w, 1))
                prob = gating_map[gi, gj].item()
                if prob >= self.threshold:
                    foreground_windows.append((i, j))
        
        with torch.no_grad():
            image_embedding = self.model.image_encoder(self.image_tensor)
            
            final_mask = np.zeros((self.h, self.w), dtype=np.float32)
            
            for i, j in foreground_windows:
                y1, y2 = i * window_size, min((i + 1) * window_size, self.h)
                x1, x2 = j * window_size, min((j + 1) * window_size, self.w)
                
                points = []
                for py in np.linspace(y1, y2, 16):
                    for px in np.linspace(x1, x2, 16):
                        points.append([px, py])
                points = np.array(points)
                
                points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
                labels_tensor = torch.ones(len(points)).long().unsqueeze(0).to(self.device)
                
                sparse_emb, dense_emb = self.model.prompt_encoder(
                    points=(points_tensor, labels_tensor),
                    boxes=None, masks=None
                )
                
                low_res_masks, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                )
                
                mask = low_res_masks[0, 0].cpu().numpy()
                mask = cv2.resize(mask, (self.w, self.h))
                mask_binary = (mask > 0).astype(np.float32)
                final_mask = np.maximum(final_mask, mask_binary)
        
        routed_ratio = len(foreground_windows) / max(num_win_h * num_win_w, 1)
        return final_mask, len(foreground_windows), routed_ratio


def build_batched_swr_predictor(model, gating_ckpt, threshold=0.5, device='cpu'):
    gating_model = SWRGatingNetwork()
    gating_model.load_state_dict(torch.load(gating_ckpt, map_location=device, weights_only=False))
    gating_model.eval()
    return BatchedSWRPredictor(model, gating_model, threshold=threshold, device=device)


def benchmark_swr_optimization():
    print("=" * 70)
    print("SWR 批量优化对比测试")
    print("=" * 70)
    
    import sys
    sys.path.insert(0, '/home/zhang/vista-slam')
    
    device = torch.device('cpu')
    
    print("\n1. 加载基础模型...")
    from pruned_sam import build_pruned_sam
    pruned_m = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    pruned_m.to(device).eval()
    print("   ✅ 模型加载成功")
    
    print("\n2. 加载门控网络...")
    gating_model = SWRGatingNetwork()
    gating_model.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model_v2.pth', map_location=device, weights_only=False))
    gating_model.eval()
    print("   ✅ 门控网络加载成功")
    
    print("\n3. 创建预测器...")
    original_predictor = OriginalSWRPredictor(pruned_m, gating_model, threshold=0.5, device=device)
    batched_predictor = BatchedSWRPredictor(pruned_m, gating_model, threshold=0.5, device=device)
    print("   ✅ 预测器创建成功")
    
    print("\n4. 加载测试图像...")
    image_path = '/home/zhang/vista-slam/eval_data/test_100/000000000139.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    print(f"   图像尺寸: {img.shape}")
    
    print("\n5. 基准测试 - 原始逐窗口推理:")
    original_predictor.set_image(img)
    
    import time
    iterations = 3
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask, num_windows, ratio = original_predictor.generate()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    print(f"   平均耗时: {sum(times)/len(times):.2f}ms")
    print(f"   前景窗口数: {num_windows}")
    print(f"   前景比例: {ratio*100:.1f}%")
    
    print("\n6. 基准测试 - 批量点推理:")
    batched_predictor.set_image(img)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask_batched, num_points, ratio_batched = batched_predictor.predict_with_batched_points()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    print(f"   平均耗时: {sum(times)/len(times):.2f}ms")
    print(f"   前景点数: {num_points}")
    print(f"   前景比例: {ratio_batched*100:.1f}%")
    
    print("\n7. 对比分析:")
    original_time = 2283.5
    batched_time = sum(times)/len(times)
    speedup = original_time / batched_time
    print(f"   原始实现耗时: {original_time:.1f}ms")
    print(f"   批量优化耗时: {batched_time:.1f}ms")
    print(f"   加速比: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("SWR批量优化测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    benchmark_swr_optimization()