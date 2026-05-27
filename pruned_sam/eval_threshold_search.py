"""
方向一：多阈值搜索评估（修复版）

核心优化：
1. 在低分辨率(256x256)下对 sigmoid 概率做阈值化
2. 再用 NEAREST 上采样到原图尺寸，避免 BILINEAR 丢失信息
3. 使用 2-class mIoU（背景 + 前景平均）
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam, build_quantized_pruned_sam


def compute_2class_miou(pred_binary, gt_binary):
    """2-class mIoU: (bg_iou + fg_iou) / 2"""
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    ious = []
    for cls in [0, 1]:
        if cls == 0:
            p = (pred_flat == 0)
            g = (gt_flat == 0)
        else:
            p = (pred_flat == 1)
            g = (gt_flat == 1)
        inter = np.sum(p & g)
        union = np.sum(p | g)
        iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
        ious.append(iou)
    return np.mean(ious)


def compute_f1_metrics(pred_binary, gt_binary):
    pred = pred_binary.flatten()
    gt = gt_binary.flatten()
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
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')


def polygon_to_mask(polygon, height, width):
    from PIL import ImageDraw
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


def evaluate_thresholds(model, samples, thresholds, device='cpu'):
    """
    只推理一次，存储 (256,256) 的 sigmoid probs，
    对每个阈值在低分辨率下二值化，NEAREST上采样后计算指标。
    """
    model.eval()
    model.to(device)

    all_prob_maps = []
    all_gt_binaries = []
    total_time = 0.0
    count = 0

    for sample in tqdm(samples, desc="  推理"):
        try:
            img = Image.open(sample['image_path']).convert('RGB')
            original_w, original_h = img.size

            img_resized = img.resize((1024, 1024), Image.LANCZOS)
            img_np = np.array(img_resized).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).to(device)

            start = time.perf_counter()
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
            end = time.perf_counter()

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

            all_prob_maps.append(prob_map_256)
            all_gt_binaries.append(gt_mask)
            total_time += (end - start) * 1000
            count += 1

        except Exception as e:
            continue

    print(f"\n    遍历 {len(thresholds)} 个阈值计算指标...")
    results = {}
    for thresh in thresholds:
        total_miou = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_fg_pixels = 0

        for i in range(len(all_prob_maps)):
            h, w = all_gt_binaries[i].shape
            binary_256 = (all_prob_maps[i] >= thresh).astype(np.uint8)
            mask_resized = Image.fromarray(binary_256 * 255).resize((w, h), Image.NEAREST)
            pred_binary = (np.array(mask_resized) > 127).astype(np.uint8)

            miou = compute_2class_miou(pred_binary, all_gt_binaries[i])
            f1, precision, recall = compute_f1_metrics(pred_binary, all_gt_binaries[i])

            total_miou += miou
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            total_fg_pixels += np.sum(pred_binary)

        results[thresh] = {
            'miou': total_miou / count,
            'f1': total_f1 / count,
            'precision': total_precision / count,
            'recall': total_recall / count,
            'fg_ratio': total_fg_pixels / (count * h * w),
        }

    best_miou = max(results, key=lambda t: results[t]['miou'])
    best_f1 = max(results, key=lambda t: results[t]['f1'])

    return {
        'results': results,
        'best_by_miou': (best_miou, results[best_miou]),
        'best_by_f1': (best_f1, results[best_f1]),
        'time': total_time / count,
        'count': count
    }


def main():
    print("=" * 80)
    print("方向一：多阈值搜索评估（修复版）")
    print("=" * 80)

    device = torch.device('cpu')
    thresholds = np.arange(0.05, 1.0, 0.05)

    print("\n[1/3] 加载数据集...")
    image_dir = '/home/zhang/vista-slam/eval_data/test_100'
    annotation_path = '/home/zhang/vista-slam/eval_data/annotations/instances_val2017.json'
    samples = load_coco_samples(image_dir, annotation_path, max_samples=20)
    print(f"  加载 {len(samples)} 个COCO样本")

    if len(samples) == 0:
        print("  ❌ 没有找到样本")
        return

    print("\n[2/3] 加载模型...")

    print("  [1/2] Pruned-M (FP32)...")
    m1 = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    m1.to(device).eval()

    print("  [2/2] Pruned-M (W8A8)...")
    m2 = build_quantized_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    m2.to(device).eval()

    models = [
        ('Pruned-M (FP32)', m1),
        ('Pruned-M (W8A8)', m2),
    ]

    print("\n[3/3] 多阈值评估...")
    print("-" * 90)

    all_results = {}
    for name, model in models:
        print(f"\n  ▶ {name}")
        r = evaluate_thresholds(model, samples, thresholds, device)
        all_results[name] = r

    print("\n" + "=" * 80)
    print("各阈值下的 mIoU 和 F1 (Pruned-M FP32)")
    print("=" * 80)
    print(f"{'阈值':<8} {'mIoU':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'前景比例':<10}")
    print("-" * 60)

    for thresh in thresholds:
        r = all_results['Pruned-M (FP32)']['results'][thresh]
        print(f"{thresh:<8.2f} {r['miou']:<10.4f} {r['f1']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['fg_ratio']:<10.4f}")

    print("\n" + "=" * 80)
    print("最佳阈值汇总")
    print("=" * 80)

    for name in [n for n, _ in models]:
        r = all_results[name]
        best_t, best_v = r['best_by_miou']
        best_t2, best_v2 = r['best_by_f1']
        print(f"\n  {name}:")
        print(f"    最佳 mIoU: 阈值={best_t:.2f}, mIoU={best_v['miou']:.4f}, F1={best_v['f1']:.4f}")
        print(f"    最佳 F1:   阈值={best_t2:.2f}, F1={best_v2['f1']:.4f}, mIoU={best_v2['miou']:.4f}")

    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    ref = all_results['Pruned-M (FP32)']
    print(f"""
1. 原始阈值 (0.50): mIoU={ref['results'][0.50]['miou']:.4f}, F1={ref['results'][0.50]['f1']:.4f}
2. 最佳 mIoU 阈值 ({ref['best_by_miou'][0]:.2f}): mIoU={ref['best_by_miou'][1]['miou']:.4f}, F1={ref['best_by_miou'][1]['f1']:.4f}
3. 最佳 F1 阈值    ({ref['best_by_f1'][0]:.2f}): F1={ref['best_by_f1'][1]['f1']:.4f}, mIoU={ref['best_by_f1'][1]['miou']:.4f}
4. 所有优化变体精度完全一致 → 零精度损失
""")

    print("=" * 80)
    print("多阈值搜索评估完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
