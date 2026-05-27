import os
import sys
import torch
import numpy as np
from PIL import Image
import time

sys.path.insert(0, '/home/zhang/vista-slam')


def prepare_calibration_data(count=30):
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


def run_int8_quantization(calibration_data):
    print("\n" + "=" * 70)
    print("执行 INT8 量化 (NNCF + IgnoredScope)")
    print("=" * 70)
    
    print("\n1. 加载 OpenVINO...")
    import openvino as ov
    core = ov.Core()
    
    print("\n2. 读取 FP32 模型...")
    model_fp32 = core.read_model('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.xml')
    print(f"   输入: {[i.any_name for i in model_fp32.inputs]}")
    print(f"   输出: {[o.any_name for o in model_fp32.outputs]}")
    
    print("\n3. 检查 NNCF...")
    import nncf
    print(f"   NNCF 版本: {nncf.__version__}")
    
    print("\n4. 准备校准数据集...")
    from nncf import Dataset
    
    def transform_func(data_item):
        return {"image": data_item}
    
    calib_dataset = Dataset(data_source=calibration_data, transform_func=transform_func)
    print(f"   校准样本数: {len(calibration_data)}")
    
    print("\n5. 执行 INT8 量化...")
    print("   这可能需要几分钟...")
    
    try:
        quantized_model = nncf.quantize(
            model=model_fp32,
            calibration_dataset=calib_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=len(calibration_data),
            target_device=nncf.TargetDevice.CPU
        )
        
        print("   ✅ NNCF INT8 量化成功!")
        
        print("\n7. 保存量化模型...")
        output_dir = '/home/zhang/vista-slam/openvino_models/int8_final'
        os.makedirs(output_dir, exist_ok=True)
        
        ov.save_model(quantized_model, f'{output_dir}/pruned_sam_encoder_int8_final.xml')
        print(f"   ✅ 模型已保存到: {output_dir}")
        
        return True
    except Exception as e:
        print(f"   ❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_model():
    """测试量化模型性能"""
    print("\n" + "=" * 70)
    print("测试量化模型性能")
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
    
    print("\n1. 测试 OpenVINO FP32...")
    model_fp32 = core.read_model('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.xml')
    compiled_fp32 = core.compile_model(model_fp32, "CPU")
    
    times = []
    for data in test_data:
        start = time.perf_counter()
        _ = compiled_fp32({"image": data})
        times.append((time.perf_counter() - start) * 1000)
    results['OV FP32'] = np.mean(times)
    print(f"   平均耗时: {results['OV FP32']:.2f}ms")
    
    print("\n2. 测试 OpenVINO INT8...")
    int8_path = '/home/zhang/vista-slam/openvino_models/int8_final/pruned_sam_encoder_int8_final.xml'
    if os.path.exists(int8_path):
        try:
            model_int8 = core.read_model(int8_path)
            compiled_int8 = core.compile_model(model_int8, "CPU")
            
            times = []
            for data in test_data:
                start = time.perf_counter()
                _ = compiled_int8({"image": data})
                times.append((time.perf_counter() - start) * 1000)
            results['OV INT8'] = np.mean(times)
            print(f"   平均耗时: {results['OV INT8']:.2f}ms")
            
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
    
    if 'OV INT8' in results:
        print(f"\n🚀 OpenVINO INT8 相比 FP32: {results['OV FP32']/results['OV INT8']:.2f}x 加速")


def main():
    calibration_data = prepare_calibration_data(count=30)
    success = run_int8_quantization(calibration_data)
    
    if success:
        benchmark_model()
        print("\n🎉 INT8 量化完成！")
    else:
        print("\n⚠️  INT8 量化失败")


if __name__ == '__main__':
    main()