"""
完整实验：评估微调后的 Pruned-M 模型
对比 TinySAM (基线) vs Pruned-M (原始) vs Pruned-M (微调)
"""
import torch
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import sys
import os
import json

sys.path.insert(0, '/home/zhang/vista-slam')
sys.path.insert(0, '/home/zhang/vista-slam/TinySAM')

from tinysam.utils.transforms import ResizeLongestSide
from tinysam.build_sam import sam_model_registry


def compute_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / union if union > 0 else 0.0


def evaluate_model(model, coco_gt, img_dir, max_samples=100, max_boxes=10, device='cpu'):
    model.eval()
    model.to(device)
    transform = ResizeLongestSide(1024)

    img_ids = sorted(coco_gt.imgs.keys())[:max_samples]
    all_ious = []

    for img_id in tqdm(img_ids, desc="Evaluating"):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue

        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        image_np = np.array(Image.open(img_path).convert('RGB'))
        orig_h, orig_w = image_np.shape[:2]

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        input_size = tuple(input_tensor.shape[-2:])
        original_size = (orig_h, orig_w)

        preprocessed = model.preprocess(input_tensor)
        image_embedding = model.image_encoder(preprocessed.unsqueeze(0))

        for ann in anns[:max_boxes]:
            x, y, bw, bh = ann['bbox']
            box_orig = np.array([[x, y, x + bw, y + bh]], dtype=float)
            box_trans = transform.apply_boxes(box_orig, original_size)
            box_tensor = torch.as_tensor(box_trans, dtype=torch.float, device=device)

            with torch.no_grad():
                se, de = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
                low_res, iou_pred = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de)

            best_idx = iou_pred[0].argmax().item()
            mask = model.postprocess_masks(
                low_res[:, best_idx:best_idx+1, :, :], input_size, original_size)
            pred_binary = (mask > model.mask_threshold).squeeze().cpu().numpy().astype(bool)

            gt_mask = coco_gt.annToMask(ann)
            all_ious.append(compute_iou(pred_binary, gt_mask))

    if not all_ious:
        return {'mIoU': 0.0, 'count': 0}

    return {
        'mIoU': np.mean(all_ious),
        'median': np.median(all_ious),
        'std': np.std(all_ious),
        '@0.5': np.mean([i > 0.5 for i in all_ious]),
        '@0.75': np.mean([i > 0.75 for i in all_ious]),
        '@0.9': np.mean([i > 0.9 for i in all_ious]),
        'count': len(all_ious),
    }


def main():
    print("=" * 80)
    print("完整实验：微调后 Pruned-M 评估")
    print("=" * 80)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ANN_FILE = '/home/zhang/vista-slam/eval_data/partial_annotations/instances_val2017.json'
    IMG_DIR = '/home/zhang/vista-slam/eval_data/test_100'

    print(f"\n设备: {DEVICE}")
    coco_gt = COCO(ANN_FILE)
    print(f"数据: {len(coco_gt.imgs)} 图像")

    # 加载所有模型
    models = {}

    # 1. TinySAM 基线
    print("\n[1/3] 加载 TinySAM (参考上界)...")
    models['TinySAM'] = sam_model_registry['vit_t'](
        checkpoint='/home/zhang/vista-slam/TinySAM/weights/tinysam_42.3.pth')
    models['TinySAM'].to(DEVICE).eval()

    # 2. Pruned-M 原始
    print("[2/3] 加载 Pruned-M (剪枝后未微调)...")
    from pruned_sam import build_pruned_sam
    models['Pruned-M (baseline)'] = build_pruned_sam(
        'pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    models['Pruned-M (baseline)'].to(DEVICE).eval()

    # 3. Pruned-M 微调后
    FT_CKPT = '/home/zhang/vista-slam/pruned_m_ft_e9_miou0.519.pth'
    print(f"[3/3] 加载 Pruned-M (微调后, {os.path.basename(FT_CKPT)})...")
    from pruned_sam.train_lora_distill import apply_lora_to_model
    ft_model = build_pruned_sam(
        'pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    ft_model = apply_lora_to_model(ft_model, rank=4)
    ft_model.load_state_dict(torch.load(FT_CKPT, map_location='cpu'), strict=False)
    ft_model.to(DEVICE).eval()
    models['Pruned-M (fine-tuned)'] = ft_model

    # 运行评估
    print("\n" + "=" * 80)
    print("评估结果 (Box Prompt, COCO 100子集)")
    print("=" * 80)
    print(f"{'模型':<30} {'mIoU':<10} {'中位数':<10} {'@0.5':<10} {'@0.75':<10} {'实例数':<8}")
    print("-" * 80)

    results = {}
    for name, model in models.items():
        print(f"\n  评估 {name}...")
        r = evaluate_model(model, coco_gt, IMG_DIR, max_samples=100, max_boxes=10, device=DEVICE)
        results[name] = r
        print(f"{name:<30} {r['mIoU']:<10.4f} {r['median']:<10.4f} "
              f"{r['@0.5']:<10.2%} {r['@0.75']:<10.2%} {r['count']:<8}")

    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    baseline = results.get('Pruned-M (baseline)', {}).get('mIoU', 0)
    ft = results.get('Pruned-M (fine-tuned)', {}).get('mIoU', 0)
    teacher = results.get('TinySAM', {}).get('mIoU', 0)
    print(f"  TinySAM (上界):        {teacher:.4f}")
    print(f"  Pruned-M (基线):       {baseline:.4f}")
    print(f"  Pruned-M (微调后):     {ft:.4f}")
    print(f"  提升:                  {ft - baseline:+.4f} ({(ft/baseline - 1)*100:+.1f}%)")
    print(f"  距离上界:              {teacher - ft:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
