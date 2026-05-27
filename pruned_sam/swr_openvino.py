"""
方向二：SWR门控网络 → ONNX导出 → OpenVINO推理集成

流程:
1. 将 SWRGatingNetwork 导出为 ONNX
2. 用 OpenVINO 加载 ONNX 模型
3. 对比 PyTorch vs OpenVINO 推理速度
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import cv2

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam
from pruned_sam.swr_inference import SWRGatingNetwork


def export_gating_to_onnx(output_path='/home/zhang/vista-slam/pruned_sam/weights/swr_gating.onnx'):
    print("\n1. 导出门控网络到 ONNX...")
    
    device = torch.device('cpu')
    model = SWRGatingNetwork()
    ckpt = '/home/zhang/vista-slam/swr_gating_model_v2.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['gating_output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'gating_output': {0: 'batch_size'}},
        opset_version=13,
    )
    print(f"   ✅ ONNX 模型已保存: {output_path}")
    print(f"      输入: (batch, 3, 256, 256)")
    print(f"      输出: (batch, 1, 4, 4)")
    return output_path


def benchmark_openvino_gating(onnx_path, iterations=100):
    print("\n2. OpenVINO 门控网络基准测试...")
    
    from openvino import Core
    
    core = Core()
    model = core.read_model(onnx_path)
    compiled = core.compile_model(model, 'CPU')
    infer_request = compiled.create_infer_request()
    
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
    for _ in range(10):
        infer_request.infer({'input': dummy_input})
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer_request.infer({'input': dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"   OpenVINO 推理: {avg_time:.4f}ms (×{iterations})")
    
    print("\n3. PyTorch 门控网络基准测试（对比）...")
    
    device = torch.device('cpu')
    pt_model = SWRGatingNetwork()
    pt_model.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model_v2.pth', map_location=device, weights_only=False))
    pt_model.eval()
    
    pt_input = torch.randn(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        for _ in range(10):
            pt_model(pt_input)
        
        pt_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            pt_model(pt_input)
            end = time.perf_counter()
            pt_times.append((end - start) * 1000)
    
    pt_avg = sum(pt_times) / len(pt_times)
    speedup = pt_avg / avg_time
    print(f"   PyTorch 推理:  {pt_avg:.4f}ms (×{iterations})")
    print(f"   加速比: {speedup:.2f}x")
    
    return avg_time, pt_avg


def build_swr_openvino_pipeline():
    """
    构建 SWR + OpenVINO 完整推理管线
    """
    print("\n4. 构建 SWR + OpenVINO 推理管线...")
    print("-" * 60)
    
    device = torch.device('cpu')
    
    onnx_path = export_gating_to_onnx()
    ov_gating_time, pt_gating_time = benchmark_openvino_gating(onnx_path)
    
    print("\n5. 测试完整 SWR + OpenVINO 流程...")
    print("-" * 60)
    
    from openvino import Core
    core = Core()
    ov_model = core.read_model(onnx_path)
    ov_compiled = core.compile_model(ov_model, 'CPU')
    ov_infer = ov_compiled.create_infer_request()
    
    print("   加载测试图像...")
    image_path = '/home/zhang/vista-slam/eval_data/test_100/000000000139.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    h, w = img.shape[:2]
    
    img_small = cv2.resize(img, (256, 256))
    img_tensor = img_small.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0
    
    print("   加载 Pruned-M 模型...")
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.to(device).eval()
    
    print("\n   端到端时序测试...")
    iterations = 10
    
    all_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        ov_result = ov_infer.infer({'input': img_tensor})
        gating_output = ov_result[ov_compiled.output(0)]
        
        grid_h, grid_w = 4, 4
        threshold = 0.5
        
        batch_points = []
        for i in range(grid_h):
            for j in range(grid_w):
                prob = gating_output[0, 0, i, j]
                if prob > threshold:
                    center_y = int((i + 0.5) * h / grid_h)
                    center_x = int((j + 0.5) * w / grid_w)
                    batch_points.append([center_x, center_y])
        
        if batch_points:
            image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            
            with torch.no_grad():
                image_embedding = model.image_encoder(image_tensor)
                
                point_coords = torch.as_tensor(batch_points, dtype=torch.float).unsqueeze(0)
                point_labels = torch.ones(len(batch_points)).long().unsqueeze(0)
                
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None, masks=None
                )
                
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                )
                
                final_mask = np.zeros((h, w), dtype=bool)
                for k in range(low_res_masks.shape[1]):
                    mask = low_res_masks[0, k].cpu().numpy()
                    mask = cv2.resize(mask, (w, h))
                    final_mask = np.logical_or(final_mask, mask > 0)
        
        end = time.perf_counter()
        all_times.append((end - start) * 1000)
    
    swr_time = sum(all_times) / len(all_times)
    num_points = len(batch_points)
    
    print(f"   门控推理 (OpenVINO): {ov_gating_time:.3f}ms")
    print(f"   前景窗口数: {num_points} / 16 ({num_points/16*100:.0f}%)")
    print(f"   SWR + OpenVINO 总耗时: {swr_time:.1f}ms")
    
    print("\n6. 对比原始 PyTorch SWR 耗时...")
    
    pt_gating_model = SWRGatingNetwork()
    pt_gating_model.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model_v2.pth', map_location=device, weights_only=False))
    pt_gating_model.eval()
    
    all_pt_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        pt_input = torch.from_numpy(img_tensor)
        with torch.no_grad():
            gating_output = pt_gating_model(pt_input)
        
        batch_points_pt = []
        for i in range(4):
            for j in range(4):
                prob = gating_output[0, 0, i, j].item()
                if prob > 0.5:
                    center_y = int((i + 0.5) * h / 4)
                    center_x = int((j + 0.5) * w / 4)
                    batch_points_pt.append([center_x, center_y])
        
        if batch_points_pt:
            image_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            
            with torch.no_grad():
                image_embedding = model.image_encoder(image_tensor)
                point_coords = torch.as_tensor(batch_points_pt, dtype=torch.float).unsqueeze(0)
                point_labels = torch.ones(len(batch_points_pt)).long().unsqueeze(0)
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None, masks=None
                )
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                )
        
        end = time.perf_counter()
        all_pt_times.append((end - start) * 1000)
    
    pt_swr_time = sum(all_pt_times) / len(all_pt_times)
    
    print(f"   PyTorch SWR 总耗时: {pt_swr_time:.1f}ms")
    print(f"   OpenVINO SWR 总耗时: {swr_time:.1f}ms")
    print(f"   加速比: {pt_swr_time/swr_time:.2f}x")
    print(f"   (门控部分加速: {pt_gating_time:.3f}ms → {ov_gating_time:.3f}ms)")
    
    return {
        'ov_gating_time': ov_gating_time,
        'pt_gating_time': pt_gating_time,
        'swr_time': swr_time,
        'pt_swr_time': pt_swr_time,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("方向二：SWR + OpenVINO 集成")
    print("=" * 70)
    results = build_swr_openvino_pipeline()
    
    print("\n" + "=" * 70)
    print("集成测试完成!")
    print("=" * 70)
