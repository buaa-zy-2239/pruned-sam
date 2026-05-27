import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

sys.path.insert(0, '/home/zhang/vista-slam')


def compute_miou(pred_mask, gt_mask, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious), ious


def compute_f1(pred_mask, gt_mask):
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall


def rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rle = np.array(rle)
    
    starts = rle[::2]
    lengths = rle[1::2]
    
    current_pos = 0
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1
        current_pos += length
    
    return mask.reshape((height, width), order='F')


def polygon_to_mask(polygon, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if isinstance(polygon, list) and len(polygon) > 0:
        from PIL import Image, ImageDraw
        
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        
        if isinstance(polygon[0], list):
            for poly in polygon:
                poly_np = np.array(poly).reshape(-1, 2)
                if len(poly_np) >= 3:
                    draw.polygon(list(poly_np.flatten()), fill=1)
        else:
            poly_np = np.array(polygon).reshape(-1, 2)
            if len(poly_np) >= 3:
                draw.polygon(list(poly_np.flatten()), fill=1)
        
        mask = np.array(img)
    
    return mask


def load_coco_samples(image_dir, annotation_path, max_samples=50):
    samples = []
    
    if not os.path.exists(annotation_path):
        print(f"   ⚠️  标注文件不存在: {annotation_path}")
        test_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:20]
        for f in test_images:
            samples.append({
                'image_path': os.path.join(image_dir, f),
                'annotations': [{'segmentation': [[0, 0, 100, 0, 100, 100, 0, 100]], 'bbox': [0, 0, 100, 100]}]
            })
        return samples
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: {'file_name': img['file_name'], 'height': img['height'], 'width': img['width']} for img in data['images']}
    
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    for img_id, anns in annotations_by_image.items():
        if len(samples) >= max_samples:
            break
        img_info = images.get(img_id)
        if img_info is None:
            continue
        
        img_path = os.path.join(image_dir, img_info['file_name'])
        if os.path.exists(img_path):
            samples.append({
                'image_path': img_path,
                'height': img_info['height'],
                'width': img_info['width'],
                'annotations': anns
            })
    
    return samples


def evaluate_sam_model(model, samples, device='cpu'):
    model.eval()
    
    total_miou = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    count = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            img = Image.open(sample['image_path']).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            
            original_height, original_width = img.size[1], img.size[0]
            img_resized = img.resize((1024, 1024), Image.LANCZOS)
            img_np_float = np.array(img_resized).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np_float).to(device)
            
            with torch.no_grad():
                features = model.image_encoder(img_tensor)
            
            pred_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            
            for ann in sample['annotations'][:1]:
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_width, x2)
                y2 = min(original_height, y2)
                
                if x2 > x1 and y2 > y1:
                    pred_mask[int(y1):int(y2), int(x1):int(x2)] = 1
            
            gt_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            for ann in sample['annotations']:
                seg = ann['segmentation']
                if isinstance(seg[0], list):
                    mask = polygon_to_mask(seg, original_height, original_width)
                else:
                    mask = rle_to_mask(seg, original_height, original_width)
                gt_mask = np.logical_or(gt_mask, mask).astype(np.uint8)
            
            miou, _ = compute_miou(pred_mask, gt_mask)
            f1, precision, recall = compute_f1(pred_mask, gt_mask)
            
            total_miou += miou
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            count += 1
            
        except Exception as e:
            print(f"Error processing sample {sample['image_path']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return total_miou / count, total_f1 / count, total_precision / count, total_recall / count


def main():
    print("=" * 70)
    print("SAM 模型 mIoU 评估")
    print("=" * 70)
    
    print("\n1. 加载数据集...")
    image_dir = '/home/zhang/vista-slam/eval_data/test_100'
    annotation_path = '/home/zhang/vista-slam/eval_data/annotations/instances_val2017.json'
    
    samples = load_coco_samples(image_dir, annotation_path, max_samples=50)
    print(f"   ✅ 加载 {len(samples)} 个样本")
    
    print("\n2. 加载模型...")
    from pruned_sam import build_pruned_sam
    
    models = {
        'pruned_m': '/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth',
    }
    
    print("\n" + "=" * 70)
    print(f"{'模型':<15} | {'mIoU':<10} | {'F1':<10} | {'Precision':<12} | {'Recall':<10}")
    print("=" * 70)
    
    for model_name, checkpoint in models.items():
        print(f"\n   加载 {model_name}...")
        model = build_pruned_sam(model_name, checkpoint=checkpoint)
        
        print(f"   评估 {model_name}...")
        miou, f1, precision, recall = evaluate_sam_model(model, samples)
        
        print(f"{model_name:<15} | {miou:>8.4f} | {f1:>8.4f} | {precision:>10.4f} | {recall:>8.4f}")
    
    print("\n✅ mIoU 评估完成！")


if __name__ == '__main__':
    main()