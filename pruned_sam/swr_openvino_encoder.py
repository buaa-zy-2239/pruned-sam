"""
方向二：SAM编码器 OpenVINO 化 + SWR集成

将 Pruned-M 的 image_encoder 导出为 ONNX，
用 OpenVINO 推理，并与 SWR 门控网络集成。
"""

import os
import sys
import time
import torch
import numpy as np
import cv2

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam
from pruned_sam.swr_inference import SWRGatingNetwork


def export_encoder_to_onnx(output_path='/home/zhang/vista-slam/pruned_sam/weights/pruned_m_encoder.onnx'):
    print("\n1. 导出编码器到 ONNX...")
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    encoder = model.image_encoder
    encoder.eval()

    dummy = torch.randn(1, 3, 1024, 1024)
    
    torch.onnx.export(
        encoder,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['image_embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'image_embedding': {0: 'batch_size'}
        },
        opset_version=13,
    )
    print(f"   ✅ ONNX 模型已保存: {output_path}")
    print(f"      输入: (batch, 3, 1024, 1024)")
    print(f"      输出: (batch, 256, 64, 64)")
    return output_path


def benchmark_ov_encoder(onnx_path, iterations=50):
    print("\n2. OpenVINO 编码器基准测试...")
    from openvino import Core
    
    core = Core()
    ov_model = core.read_model(onnx_path)
    compiled = core.compile_model(ov_model, 'CPU')
    infer = compiled.create_infer_request()
    
    dummy = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    
    for _ in range(5):
        infer.infer({'input': dummy})
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        infer.infer({'input': dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"   OpenVINO 编码器: {avg:.2f}ms (×{iterations})")
    
    print("\n3. PyTorch 编码器基准测试（对比）...")
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    encoder = model.image_encoder
    
    pt_dummy = torch.randn(1, 3, 1024, 1024)
    
    with torch.no_grad():
        for _ in range(5):
            encoder(pt_dummy)
        
        pt_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            encoder(pt_dummy)
            end = time.perf_counter()
            pt_times.append((end - start) * 1000)
    
    pt_avg = sum(pt_times) / len(pt_times)
    speedup = pt_avg / avg
    print(f"   PyTorch 编码器:  {pt_avg:.2f}ms (×{iterations})")
    print(f"   加速比: {speedup:.2f}x")
    
    return avg, pt_avg


def test_ov_swr_pipeline(encoder_onnx, gating_onnx):
    print("\n4. SWR + OpenVINO 端到端管线测试...")
    print("-" * 60)
    
    from openvino import Core
    core = Core()
    
    ov_enc = core.read_model(encoder_onnx)
    ov_enc_compiled = core.compile_model(ov_enc, 'CPU')
    ov_enc_infer = ov_enc_compiled.create_infer_request()
    
    ov_gat = core.read_model(gating_onnx)
    ov_gat_compiled = core.compile_model(ov_gat, 'CPU')
    ov_gat_infer = ov_gat_compiled.create_infer_request()
    
    print("   加载 PyTorch prompt_encoder 和 mask_decoder...")
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    print("   加载测试图像...")
    image_path = '/home/zhang/vista-slam/eval_data/test_100/000000000139.jpg'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    h, w = img.shape[:2]
    
    img_preprocessed = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    img_small = (cv2.resize(img, (256, 256)).astype(np.float32).transpose(2, 0, 1)[np.newaxis]) / 255.0
    
    print("\n   时序测试...")
    iterations = 10
    all_times = []
    total_points = 0
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        ov_enc_result = ov_enc_infer.infer({'input': img_preprocessed})
        image_embedding = ov_enc_result[ov_enc_compiled.output(0)]
        
        ov_gat_result = ov_gat_infer.infer({'input': img_small})
        gating_output = ov_gat_result[ov_gat_compiled.output(0)]
        
        batch_points = []
        for i in range(4):
            for j in range(4):
                prob = gating_output[0, 0, i, j]
                if prob > 0.5:
                    center_y = int((i + 0.5) * h / 4)
                    center_x = int((j + 0.5) * w / 4)
                    batch_points.append([center_x, center_y])
        
        total_points = len(batch_points)
        
        if batch_points:
            img_tensor = torch.from_numpy(img_preprocessed)
            emb_tensor = torch.from_numpy(image_embedding)
            
            with torch.no_grad():
                point_coords = torch.as_tensor(batch_points, dtype=torch.float).unsqueeze(0)
                point_labels = torch.ones(len(batch_points)).long().unsqueeze(0)
                
                sparse_emb, dense_emb = model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None, masks=None
                )
                
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=emb_tensor,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                )
                
                final_mask = np.zeros((h, w), dtype=bool)
                for k in range(low_res_masks.shape[1]):
                    mask_np = low_res_masks[0, k].cpu().numpy()
                    mask_np = cv2.resize(mask_np, (w, h))
                    final_mask = np.logical_or(final_mask, mask_np > 0)
        
        end = time.perf_counter()
        all_times.append((end - start) * 1000)
    
    swr_ov_time = sum(all_times) / len(all_times)
    
    print(f"   前景窗口: {total_points}/16 ({total_points/16*100:.0f}%)")
    print(f"   SWR + OV 总耗时: {swr_ov_time:.1f}ms")
    
    print("\n5. 对比纯 PyTorch SWR 耗时...")
    pt_encoder = model.image_encoder
    pt_gating = SWRGatingNetwork()
    pt_gating.load_state_dict(torch.load('/home/zhang/vista-slam/swr_gating_model_v2.pth', map_location='cpu', weights_only=False))
    pt_gating.eval()
    
    all_pt_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        with torch.no_grad():
            image_embedding = pt_encoder(torch.from_numpy(img_preprocessed))
            gating_output = pt_gating(torch.from_numpy(img_small))
        
        batch_points = []
        for i in range(4):
            for j in range(4):
                prob = gating_output[0, 0, i, j].item()
                if prob > 0.5:
                    center_y = int((i + 0.5) * h / 4)
                    center_x = int((j + 0.5) * w / 4)
                    batch_points.append([center_x, center_y])
        
        if batch_points:
            with torch.no_grad():
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
        
        end = time.perf_counter()
        all_pt_times.append((end - start) * 1000)
    
    pt_swr_time = sum(all_pt_times) / len(all_pt_times)
    
    print(f"   PyTorch SWR: {pt_swr_time:.1f}ms")
    print(f"   OpenVINO SWR: {swr_ov_time:.1f}ms")
    print(f"   加速比: {pt_swr_time/swr_ov_time:.2f}x")
    
    return swr_ov_time, pt_swr_time


def main():
    print("=" * 70)
    print("方向二：编码器 OpenVINO 化 + SWR 集成")
    print("=" * 70)
    
    encoder_onnx = '/home/zhang/vista-slam/pruned_sam/weights/pruned_m_encoder.onnx'
    gating_onnx = '/home/zhang/vista-slam/pruned_sam/weights/swr_gating.onnx'
    
    if not os.path.exists(encoder_onnx):
        encoder_onnx = export_encoder_to_onnx()
    
    if not os.path.exists(gating_onnx):
        print("  ⚠️ 门控ONNX不存在，请先运行 swr_openvino.py")
        return
    
    ov_enc_time, pt_enc_time = benchmark_ov_encoder(encoder_onnx)
    
    swr_ov_time, pt_swr_time = test_ov_swr_pipeline(encoder_onnx, gating_onnx)
    
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    print(f"""
┌──────────────────────────────────────────────────────────────┐
│  组件              │ PyTorch     │ OpenVINO    │ 加速比      │
├──────────────────────────────────────────────────────────────┤
│  图像编码器        │ {pt_enc_time:<11.2f}ms │ {ov_enc_time:<11.2f}ms │ {pt_enc_time/ov_enc_time:<8.2f}x     │
│  门控网络          │ 20.64ms     │ 9.47ms      │ 2.18x       │
│  SWR 全管线        │ {pt_swr_time:<11.2f}ms │ {swr_ov_time:<11.2f}ms │ {pt_swr_time/swr_ov_time:<8.2f}x     │
└──────────────────────────────────────────────────────────────┘
""")
    print("=" * 70)
    print("集成测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
