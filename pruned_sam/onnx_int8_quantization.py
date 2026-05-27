import os
import sys
import torch
import numpy as np
from PIL import Image
import time

sys.path.insert(0, '/home/zhang/vista-slam')


def prepare_calibration_data(count=50):
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
        img_np_batched = img_np.transpose(2, 0, 1)[np.newaxis]
        calibration_data.append(img_np_batched)
        
        if (i + 1) % 10 == 0:
            print(f"   已处理 {i + 1}/{len(images)}")
    
    print(f"\n✅ 校准数据准备完成: {len(calibration_data)} 样本")
    return calibration_data


def export_pytorch_to_onnx():
    """先将 PyTorch 模型导出为 FP32 ONNX"""
    print("\n" + "=" * 70)
    print("导出 PyTorch 模型为 ONNX")
    print("=" * 70)
    
    print("\n1. 加载剪枝后的模型...")
    from pruned_sam import build_pruned_sam
    model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    model.eval()
    
    print("\n2. 准备 dummy input...")
    dummy_input = torch.randn(1, 3, 1024, 1024)
    
    print("\n3. 导出 ONNX...")
    onnx_path = '/home/zhang/vista-slam/openvino_models/pruned_sam_encoder_fp32.onnx'
    os.makedirs('/home/zhang/vista-slam/openvino_models', exist_ok=True)
    
    torch.onnx.export(
        model.image_encoder,
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


def quantize_onnx_with_ort(onnx_path, calibration_data):
    """使用 ONNX Runtime 进行 INT8 量化"""
    print("\n" + "=" * 70)
    print("使用 ONNX Runtime 进行 INT8 量化")
    print("=" * 70)
    
    print("\n1. 导入 ONNX Runtime 量化工具...")
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
    except ImportError:
        print("   ❌ ONNX Runtime 量化工具不可用")
        return None
    
    print("\n2. 创建校准数据读取器...")
    class SamCalibrationDataReader(CalibrationDataReader):
        def __init__(self, calibration_data):
            self.enum_data = iter(calibration_data)
        
        def get_next(self):
            try:
                data = next(self.enum_data)
                return {'image': data}
            except StopIteration:
                return None
    
    calib_reader = SamCalibrationDataReader(calibration_data)
    
    print("\n3. 执行静态量化...")
    int8_onnx_path = '/home/zhang/vista-slam/openvino_models/pruned_sam_encoder_int8_ort.onnx'
    
    try:
        quantize_static(
            onnx_path,
            int8_onnx_path,
            calib_reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )
        print(f"   ✅ ONNX INT8 模型已保存到: {int8_onnx_path}")
        return int8_onnx_path
    except Exception as e:
        print(f"   ❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    output_dir = '/home/zhang/vista-slam/openvino_models/int8_ort'
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(onnx_path).replace('.onnx', '')
    ov.save_model(model, f'{output_dir}/{model_name}.xml')
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
    
    print("\n3. 测试 OpenVINO INT8 (ONNX Runtime量化)...")
    int8_path = '/home/zhang/vista-slam/openvino_models/int8_ort/pruned_sam_encoder_int8_ort.xml'
    if os.path.exists(int8_path):
        try:
            model_int8 = core.read_model(int8_path)
            compiled_int8 = core.compile_model(model_int8, "CPU")
            
            times = []
            for data in test_data:
                start = time.perf_counter()
                _ = compiled_int8({"image": data})
                times.append((time.perf_counter() - start) * 1000)
            results['OV INT8 (ORT)'] = np.mean(times)
            print(f"   平均耗时: {results['OV INT8 (ORT)']:.2f}ms")
            
            fp32_size = os.path.getsize('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.bin') / (1024*1024)
            int8_size = os.path.getsize(int8_path.replace('.xml', '.bin')) / (1024*1024)
            print(f"\n📦 模型大小:")
            print(f"   FP32: {fp32_size:.2f} MB")
            print(f"   INT8: {int8_size:.2f} MB")
            print(f"   压缩比: {fp32_size/int8_size:.2f}x")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    if 'OV INT8 (ORT)' in results:
        print(f"\n🚀 OpenVINO INT8 相比 FP32: {results['OV FP32']/results['OV INT8 (ORT)']:.2f}x")
        print(f"🚀 OpenVINO INT8 相比 PyTorch: {results['PyTorch FP32']/results['OV INT8 (ORT)']:.2f}x")


def main():
    try:
        calibration_data = prepare_calibration_data(count=30)
        onnx_path = export_pytorch_to_onnx()
        int8_onnx_path = quantize_onnx_with_ort(onnx_path, calibration_data)
        
        if int8_onnx_path:
            convert_to_openvino(int8_onnx_path)
            print("\n🎉 INT8 模型构造完成！")
        else:
            print("\n⚠️  INT8 量化失败")
    except Exception as e:
        print(f"\n❌ 量化过程失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()