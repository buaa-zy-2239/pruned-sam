import zipfile
import json
import os
import shutil
import re

def extract_partial_dataset():
    """解压部分数据集"""
    print("=" * 70)
    print("解压部分 COCO 数据集")
    print("=" * 70)
    
    base_dir = '/home/zhang/vista-slam/eval_data'
    max_images = 150  # 解压图片数量
    
    # 创建输出目录
    img_dir = os.path.join(base_dir, 'partial_val2017')
    ann_dir = os.path.join(base_dir, 'partial_annotations')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    
    # 1. 解压部分图片
    print(f"\n1. 解压部分图片 (最多 {max_images} 张)...")
    img_zip = os.path.join(base_dir, 'val2017', 'val2017.zip')
    
    if not os.path.exists(img_zip):
        print(f"   ❌ 找不到图片压缩包: {img_zip}")
        return
    
    extracted_ids = []
    with zipfile.ZipFile(img_zip, 'r') as zf:
        image_files = [f for f in zf.namelist() 
                     if f.endswith('.jpg') and re.match(r'val2017/\d+\.jpg', f)]
        image_files.sort(key=lambda x: int(x.split('/')[-1].replace('.jpg', '')))
        print(f"   总共 {len(image_files)} 张图片")
        
        for i, f in enumerate(image_files[:max_images]):
            zf.extract(f, img_dir)
            img_id = int(f.split('/')[-1].replace('.jpg', ''))
            extracted_ids.append(img_id)
            if (i + 1) % 30 == 0:
                print(f"   已解压 {i + 1}/{min(max_images, len(image_files))}")
        
        print(f"   ✅ 完成！共解压 {len(extracted_ids)} 张图片")
    
    # 整理解压后的文件
    extracted_dir = os.path.join(img_dir, 'val2017')
    if os.path.exists(extracted_dir):
        for f in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, f), os.path.join(img_dir, f))
        os.rmdir(extracted_dir)
    
    # 2. 处理标注文件
    print("\n2. 处理标注文件...")
    instances_file = os.path.join(base_dir, 'annotations', 'instances_val2017.json')
    
    if not os.path.exists(instances_file):
        print(f"   ❌ 找不到标注文件: {instances_file}")
        return
    
    print(f"   读取标注文件: {instances_file}")
    with open(instances_file, 'r') as f:
        ann_data = json.load(f)
    
    # 筛选相关图片和标注
    img_ids_set = set(extracted_ids)
    partial_imgs = [img for img in ann_data['images'] if img['id'] in img_ids_set]
    partial_anns = [ann for ann in ann_data['annotations'] if ann['image_id'] in img_ids_set]
    
    partial_data = {
        'images': partial_imgs,
        'annotations': partial_anns,
        'categories': ann_data.get('categories', [])
    }
    
    output_file = os.path.join(ann_dir, 'instances_val2017.json')
    with open(output_file, 'w') as f:
        json.dump(partial_data, f)
    
    print(f"   ✅ 保存到: {output_file}")
    print(f"   - 图片: {len(partial_imgs)} 张")
    print(f"   - 标注: {len(partial_anns)} 条")
    
    # 3. 处理 stuff 标注（如果存在）
    stuff_file = os.path.join(base_dir, 'annotations', 'stuff_val2017.json')
    if os.path.exists(stuff_file):
        print(f"\n3. 处理 stuff 标注...")
        with open(stuff_file, 'r') as f:
            stuff_data = json.load(f)
        
        partial_imgs = [img for img in stuff_data['images'] if img['id'] in img_ids_set]
        partial_anns = [ann for ann in stuff_data['annotations'] if ann['image_id'] in img_ids_set]
        
        partial_stuff = {
            'images': partial_imgs,
            'annotations': partial_anns,
            'categories': stuff_data.get('categories', [])
        }
        
        output_file = os.path.join(ann_dir, 'stuff_val2017.json')
        with open(output_file, 'w') as f:
            json.dump(partial_stuff, f)
        
        print(f"   ✅ 保存到: {output_file}")
        print(f"   - 图片: {len(partial_imgs)} 张")
        print(f"   - 标注: {len(partial_anns)} 条")
    
    print("\n" + "=" * 70)
    print("解压完成!")
    print("=" * 70)
    print(f"\n📁 输出目录:")
    print(f"   图片: {img_dir}/")
    print(f"   标注: {ann_dir}/")
    print(f"\n🖼️  图片数量: {len(extracted_ids)}")
    print(f"📝 标注数量: {len(partial_anns)} 条")


if __name__ == '__main__':
    extract_partial_dataset()
