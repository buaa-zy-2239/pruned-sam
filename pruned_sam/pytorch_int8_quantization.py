import os
import sys
import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
from PIL import Image
import time

sys.path.insert(0, '/home/zhang/vista-slam')


def prepare_calibration_data(count=50):
    """准备校准数据"""
    print("=" * 70)
    print("准备校准数据")
    print("=" * 70)
    
    test_dir = '/home/zhang/vista-slam/eval_data/test_100'
    images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:count]
    
    print(f"\n准备 {len(images)} 张校准图片...")
    
    calibration_data = []
    for i, img_file in enumerate(images):
        img_path = os.path.join(test_dir, img_file)
        img = Image.open(img_path).convert('RGB').resize((1024, 1024), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)[np.newaxis])
        calibration_data.append(img_tensor)
        
        if (i + 1) % 10 == 0:
            print(f"   已处理 {i + 1}/{len(images)}")
    
    print(f"\n✅ 校准数据准备完成: {len(calibration_data)} 样本")
    return calibration_data


def quantize_pytorch_model(calibration_data):
    """使用 PyTorch 原生量化"""
    print("\n" + "=" * 70)
    print("使用 PyTorch 原生量化")
    print("=" * 70)
    
    print("\n1. 加载剪枝后的 FP32 模型...")
    from pruned_sam import build_pruned_sam
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    print("\n2. 准备量化配置...")
    backend = "fbgemm"
    qconfig = quant.get_default_qconfig(backend)
    
    print("\n3. 配置模型量化...")
    model.qconfig = qconfig
    
    print("\n4. 插入量化观测器...")
    model_prepared = quant.prepare(model, inplace=False)
    
    print("\n5. 校准（收集统计信息）...")
    with torch.no_grad():
        for i, img_tensor in enumerate(calibration_data):
            model_prepared.image_encoder(img_tensor)
            if (i + 1) % 10 == 0:
                print(f"   已校准 {i + 1}/{len(calibration_data)}")
    
    print("\n6. 转换为量化模型...")
    model_int8 = quant.convert(model_prepared, inplace=False)
    
    print("\n7. 保存量化模型...")
    os.makedirs('/home/zhang/vista-slam/pruned_sam/weights', exist_ok=True)
    torch.save(model_int8.state_dict(), '/home/zhang/vista-slam/pruned_sam/weights/pruned_m_int8_pytorch.pth')
    print("   ✅ PyTorch INT8 模型已保存")
    
    return model_int8


def export_to_onnx(model_int8):
    """导出为 ONNX"""
    print("\n" + "=" * 70)
    print("导出为 ONNX")
    print("=" * 70)
    
    print("\n1. 准备 dummy input...")
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    print("\n2. 导出 ONNX...")
    onnx_path = '/home/zhang/vista-slam/openvino_models/pruned_sam_encoder_int8_pytorch.onnx'
    os.makedirs('/home/zhang/vista-slam/openvino_models', exist_ok=True)
    
    torch.onnx.export(
        model_int8.image_encoder,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=['image'],
        output_names=['features'],
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"   ✅ ONNX 模型已保存到: {onnx_path}")
    return onnx_path


def convert_to_openvino(onnx_path):
    """转换为 OpenVINO IR"""
    print("\n" + "=" * 70)
    print("转换为 OpenVINO IR")
    print("=" * 70)
    
    import openvino as ov
    
    print("\n1. 读取 ONNX 模型...")
    core = ov.Core()
    model = core.read_model(onnx_path)
    
    print("\n2. 保存为 IR 格式...")
    output_dir = '/home/zhang/vista-slam/openvino_models/int8_pytorch'
    os.makedirs(output_dir, exist_ok=True)
    
    ov.save_model(model, f'{output_dir}/pruned_sam_encoder_int8_pytorch.xml')
    print(f"   ✅ OpenVINO IR 已保存到: {output_dir}")
    
    return output_dir


def benchmark_all_models():
    """测试所有模型性能"""
    print("\n" + "=" * 70)
    print("测试所有模型性能")
    print("=" * 70)
    
    import openvino as ov
    core = ov.Core()
    
    test_dir = '/home/zhang/vista-slam/eval_data/test_100'
    images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:50]
    
    test_data = []
    for img_file in images:
        img_path = os.path.join(test_dir, img_file)
        img = Image.open(img_path).convert('RGB').resize((1024, 1024), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        test_data.append(img_np.transpose(2, 0, 1)[np.newaxis])
    
    results = {}
    
    print("\n1. 测试 PyTorch FP32...")
    from pruned_sam import build_pruned_sam
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    times = []
    for data in test_data[:10]:
        img_t = torch.from_numpy(data)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.image_encoder(img_t)
        times.append((time.perf_counter() - start) * 1000)
    results['PyTorch FP32'] = np.mean(times)
    print(f"   平均耗时: {results['PyTorch FP32']:.2f}ms")
    
    print("\n2. 测试 OpenVINO FP32...")
    model_fp32 = core.read_model('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.xml')
    compiled_fp32 = core.compile_model(model_fp32, "CPU")
    
    times = []
    for data in test_data:
        start = time.perf_counter()
        _ = compiled_fp32({"image": data})
        times.append((time.perf_counter() - start) * 1000)
    results['OV FP32'] = np.mean(times)
    print(f"   平均耗时: {results['OV FP32']:.2f}ms")
    
    print("\n3. 测试 OpenVINO INT8 (PyTorch量化)...")
    int8_path = '/home/zhang/vista-slam/openvino_models/int8_pytorch/pruned_sam_encoder_int8_pytorch.xml'
    if os.path.exists(int8_path):
        try:
            model_int8 = core.read_model(int8_path)
            compiled_int8 = core.compile_model(model_int8, "CPU")
            
            times = []
            for data in test_data:
                start = time.perf_counter()
                _ = compiled_int8({"image": data})
                times.append((time.perf_counter() - start) * 1000)
            results['OV INT8 (PyTorch)'] = np.mean(times)
            print(f"   平均耗时: {results['OV INT8 (PyTorch)']:.2f}ms")
            
            fp32_size = os.path.getsize('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.bin') / (1024*1024)
            int8_size = os.path.getsize(int8_path.replace('.xml', '.bin')) / (1024*1024)
            print(f"\n📦 模型大小:")
            print(f"   FP32: {fp32_size:.2f} MB")
            print(f"   INT8: {int8_size:.2f} MB")
            print(f"   压缩比: {fp32_size/int8_size:.2f}x")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
    
    print("\n4. 测试 NNCF compress_weights 量化...")
    int8_compressed_path = '/home/zhang/vista-slam/openvino_models/int8_compressed/pruned_sam_encoder_int8_compressed.xml'
    if os.path.exists(int8_compressed_path):
        try:
            model_int8_compressed = core.read_model(int8_compressed_path)
            compiled_int8_compressed = core.compile_model(model_int8_compressed, "CPU")
            
            times = []
            for data in test_data:
                start = time.perf_counter()
                _ = compiled_int8_compressed({"image": data})
                times.append((time.perf_counter() - start) * 1000)
            results['OV INT8 (compress)'] = np.mean(times)
            print(f"   平均耗时: {results['OV INT8 (compress)']:.2f}ms")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
    
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)
    
    baseline = results['PyTorch FP32']
    print(f"\n{'模型':<25} | {'耗时':<12} | {'加速比':<10}")
    print("-" * 50)
    for name, t in results.items():
        speedup = baseline / t
        print(f"{name:<25} | {t:>8.2f}ms | {speedup:>8.2f}x")
    
    if 'OV INT8 (PyTorch)' in results:
        print(f"\n🚀 OpenVINO INT8 相比 FP32: {results['OV FP32']/results['OV INT8 (PyTorch)']:.2f}x")
        print(f"🚀 OpenVINO INT8 相比 PyTorch: {results['PyTorch FP32']/results['OV INT8 (PyTorch)']:.2f}x")


def main():
    try:
        calibration_data = prepare_calibration_data(count=30)
        model_int8 = quantize_pytorch_model(calibration_data)
        onnx_path = export_to_onnx(model_int8)
        convert_to_openvino(onnx_path)
        benchmark_all_models()
    except Exception as e:
        print(f"\n❌ 量化过程失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  运行已有模型的基准测试...")
        benchmark_all_models()


if __name__ == '__main__':
    main()