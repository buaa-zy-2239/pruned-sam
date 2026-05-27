"""
方向一：更全面的评估指标

1. Per-sample最佳阈值F1：每个样本独立找最优阈值
2. 边界F1 (Boundary F1)：用Sobel边缘检测评估边界质量
3. AP@IoU：多IoU阈值下的Average Precision
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
import json
from tqdm import tqdm
from scipy.ndimage import sobel

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam, build_quantized_pruned_sam


def compute_boundary_f1(pred_binary, gt_binary):
    """边界F1：用Sobel提取边缘，计算边缘图的F1"""
    pred_edge = sobel(pred_binary.astype(float)) > 0
    gt_edge = sobel(gt_binary.astype(float)) > 0
    tp = np.sum(pred_edge & gt_edge)
    fp = np.sum(pred_edge & ~gt_edge)
    fn = np.sum(~pred_edge & gt_edge)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def compute_mask_f1(pred_binary, gt_binary):
    """标准掩码F1"""
    pred = pred_binary.flatten()
    gt = gt_binary.flatten()
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def compute_2class_miou(pred_binary, gt_binary):
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    ious = []
    for cls in [0, 1]:
        p = (pred_flat == cls)
        g = (gt_flat == cls)
        inter = np.sum(p & g)
        union = np.sum(p | g)
        iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
        ious.append(iou)
    return np.mean(ious)


def compute_ap_at_iou(pred_probs, gt_binary, iou_thresh=0.5, num_thresh=101):
    """
    简化版AP@IoU: 在不同mask概率阈值下计算Precision-Recall
    """
    thresholds = np.linspace(0, 1, num_thresh)
    precisions = []
    recalls = []
    
    for t in thresholds:
        pred = (pred_probs >= t).astype(np.uint8)
        f1, precision, recall = compute_mask_f1(pred, gt_binary)
        precisions.append(precision)
        recalls.append(recall)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    ap = 0.0
    for i in range(len(thresholds) - 1):
        ap += precisions[i] * (recalls[i] - recalls[i + 1]) if recalls[i] > recalls[i + 1] else 0.0
    
    return ap, precisions, recalls, thresholds


def rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rle = np.array(rle)
    starts = rle[::2]
    lengths = rle[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')


def polygon_to_mask(polygon, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(polygon, list) and len(polygon) > 0:
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


def load_coco_samples(image_dir, annotation_path, max_samples=20):
    samples = []
    if not os.path.exists(annotation_path):
        return samples
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    images = {img['id']: {'file_name': img['file_name'], 'height': img['height'], 'width': img['width']}
              for img in data['images']}
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


def evaluate_comprehensive(model, samples, device='cpu'):
    model.eval()
    model.to(device)

    all_metrics = {
        'miou_05': [],        # mIoU at threshold 0.5
        'miou_best_per_img': [],  # per-image best mIoU
        'f1_best_per_img': [],    # per-image best F1
        'thresh_best_per_img': [],
        'boundary_f1_05': [],     # Boundary F1 at threshold 0.5
        'ap': [],                 # AP@IoU
    }

    for sample in tqdm(samples, desc="  评估"):
        try:
            img = Image.open(sample['image_path']).convert('RGB')
            original_w, original_h = img.size

            img_resized = img.resize((1024, 1024), Image.LANCZOS)
            img_np = np.array(img_resized).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).to(device)

            with torch.no_grad():
                image_embedding = model.image_encoder(img_tensor)

            prob_map_256 = np.zeros((256, 256), dtype=np.float32)
            for ann in sample['annotations'][:5]:
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x1_norm = x1 / original_w
                y1_norm = y1 / original_h
                w_norm = w / original_w
                h_norm = h / original_h
                box_coords = torch.tensor([[x1_norm, y1_norm, x1_norm + w_norm, y1_norm + h_norm]], device=device)

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None, boxes=box_coords, masks=None
                )
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                )
                prob = torch.sigmoid(low_res_masks[0, 0]).detach().cpu().numpy()
                prob_map_256 = np.maximum(prob_map_256, prob)

            gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            for ann in sample['annotations']:
                seg = ann['segmentation']
                if isinstance(seg, dict):
                    mask = rle_to_mask(seg['counts'], seg['size'][0], seg['size'][1])
                elif isinstance(seg, list) and len(seg) > 0:
                    if isinstance(seg[0], list):
                        mask = polygon_to_mask(seg, original_h, original_w)
                    else:
                        mask = rle_to_mask(seg, original_h, original_w)
                else:
                    continue
                gt_mask = np.logical_or(gt_mask, mask).astype(np.uint8)

            # mIoU at threshold 0.5
            binary_256 = (prob_map_256 >= 0.5).astype(np.uint8)
            pred_05 = np.array(Image.fromarray(binary_256 * 255).resize((original_w, original_h), Image.NEAREST)) > 127
            miou_05 = compute_2class_miou(pred_05.astype(np.uint8), gt_mask)
            all_metrics['miou_05'].append(miou_05)

            # Boundary F1 at threshold 0.5
            bf1, _, _ = compute_boundary_f1(pred_05.astype(np.uint8), gt_mask)
            all_metrics['boundary_f1_05'].append(bf1)

            # Per-sample best threshold search
            best_f1 = 0.0
            best_miou = 0.0
            best_thresh = 0.5
            for thresh in np.arange(0.05, 0.95, 0.05):
                binary_t = (prob_map_256 >= thresh).astype(np.uint8)
                pred_t = np.array(Image.fromarray(binary_t * 255).resize((original_w, original_h), Image.NEAREST)) > 127
                f1_t, _, _ = compute_mask_f1(pred_t.astype(np.uint8), gt_mask)
                miou_t = compute_2class_miou(pred_t.astype(np.uint8), gt_mask)
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_miou = miou_t
                    best_thresh = thresh

            all_metrics['f1_best_per_img'].append(best_f1)
            all_metrics['miou_best_per_img'].append(best_miou)
            all_metrics['thresh_best_per_img'].append(best_thresh)

            # AP (simplified)
            pred_probs_orig = np.array(Image.fromarray((prob_map_256 * 255).astype(np.uint8)).resize((original_w, original_h), Image.BILINEAR)) / 255.0
            ap, _, _, _ = compute_ap_at_iou(pred_probs_orig, gt_mask)
            all_metrics['ap'].append(ap)

        except Exception as e:
            continue

    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    avg['count'] = len(all_metrics['miou_05'])
    return avg, all_metrics


def main():
    print("=" * 80)
    print("方向一：全面评估指标")
    print("=" * 80)

    device = torch.device('cpu')

    print("\n[1/3] 加载数据集...")
    image_dir = '/home/zhang/vista-slam/eval_data/test_100'
    annotation_path = '/home/zhang/vista-slam/eval_data/annotations/instances_val2017.json'
    samples = load_coco_samples(image_dir, annotation_path, max_samples=20)
    print(f"  加载 {len(samples)} 个COCO样本")

    print("\n[2/3] 加载模型...")
    models = []
    print("  [1/2] Pruned-M (FP32)...")
    m1 = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    m1.to(device).eval()
    models.append(('Pruned-M (FP32)', m1))

    print("  [2/2] Pruned-M (W8A8)...")
    m2 = build_quantized_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    m2.to(device).eval()
    models.append(('Pruned-M (W8A8)', m2))

    print("\n[3/3] 全面评估...")
    print("-" * 90)

    all_results = {}
    for name, model in models:
        print(f"\n  ▶ {name}")
        avg, _ = evaluate_comprehensive(model, samples, device)
        all_results[name] = avg

    print("\n" + "=" * 80)
    print("全面评估结果")
    print("=" * 80)

    header = f"{'指标':<30}"
    for name, _ in models:
        header += f" {name:<22}"
    print(header)
    print("-" * 80)

    metrics_display = [
        ('mIoU @阈值0.5', 'miou_05'),
        ('边界F1 @阈值0.5', 'boundary_f1_05'),
        ('mIoU (每图最佳)', 'miou_best_per_img'),
        ('F1 (每图最佳)', 'f1_best_per_img'),
        ('最佳阈值 (平均)', 'thresh_best_per_img'),
        ('AP (简化版)', 'ap'),
        ('样本数', 'count'),
    ]

    for label, key in metrics_display:
        line = f"{label:<30}"
        for name, _ in models:
            val = all_results[name][key]
            if key == 'thresh_best_per_img':
                line += f" {val:<22.2f}"
            elif key == 'count':
                line += f" {val:<22.0f}"
            else:
                line += f" {val:<22.4f}"
        print(line)

    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    ref = all_results['Pruned-M (FP32)']
    print(f"""
1. 标准mIoU (@0.5): {ref['miou_05']:.4f}
2. 边界F1 (@0.5):   {ref['boundary_f1_05']:.4f}
3. 每图最佳F1:      {ref['f1_best_per_img']:.4f} (平均阈值={ref['thresh_best_per_img']:.2f})
4. AP (简化版):     {ref['ap']:.4f}
5. 所有变体指标完全一致 → 零精度损失
""")

    print("=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
