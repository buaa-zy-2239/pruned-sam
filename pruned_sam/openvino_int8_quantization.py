import os
import sys
import numpy as np
from PIL import Image

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


def run_openvino_quantization(calibration_data):
    print("\n" + "=" * 70)
    print("使用 OpenVINO 原生量化")
    print("=" * 70)
    
    print("\n1. 加载 OpenVINO...")
    import openvino as ov
    core = ov.Core()
    
    print("\n2. 读取 FP32 模型...")
    model_fp32 = core.read_model('/home/zhang/vista-slam/openvino_models/pruned_sam_encoder.xml')
    
    print("\n3. 准备校准数据集...")
    class CalibrationDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {"image": self.data[idx]}
    
    calib_dataset = CalibrationDataset(calibration_data)
    
    print("\n4. 执行 INT8 量化...")
    try:
        from openvino.tools.quantization import quantize_model
        
        quantized_model = quantize_model(
            model=model_fp32,
            calibration_dataset=calib_dataset,
            preset='performance'
        )
        
        print("   ✅ OpenVINO INT8 量化成功!")
        
        print("\n5. 保存量化模型...")
        output_dir = '/home/zhang/vista-slam/openvino_models/int8_openvino'
        os.makedirs(output_dir, exist_ok=True)
        
        ov.save_model(quantized_model, f'{output_dir}/pruned_sam_encoder_int8_openvino.xml')
        print(f"   ✅ 模型已保存到: {output_dir}")
        
        return True
    except ImportError:
        print("   ❌ OpenVINO 量化工具不可用")
        return False
    except Exception as e:
        print(f"   ❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    calibration_data = prepare_calibration_data(count=30)
    success = run_openvino_quantization(calibration_data)
    
    if success:
        print("\n🎉 INT8 模型构造完成！")
    else:
        print("\n⚠️  INT8 量化失败")


if __name__ == '__main__':
    main()