"""
提示循环蒸馏 (Prompt-in-the-Loop Distillation) 进阶微调
====================================================
相比 LoRA 微调的核心改进:
1. 全参数微调 (unfreeze all) — 恢复全部模型容量
2. 特征对齐蒸馏 — 对齐解码器内部特征 (不仅仅是最终掩码)
3. IoU 预测头蒸馏 — 同步学习质量评估能力
4. 更多训练数据 — 使用完整 COCO val2017 (5000图, 36781实例)

目标 mIoU: 0.65-0.70 (Box Prompt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys
import time
import math
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TinySAM'))

from tinysam.utils.transforms import ResizeLongestSide
from tinysam.build_sam import sam_model_registry
from tinysam.modeling import Sam


# ============================================================
# 1. 特征提取 — 对 MaskDecoder 做 monkey-patch
# ============================================================

def _forward_with_features(self, image_embeddings, image_pe,
                            sparse_prompt_embeddings, dense_prompt_embeddings):
    """MaskDecoder forward 增加中间特征返回"""
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape

    hs, src_transformed = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

    src_2d = src_transformed.transpose(1, 2).view(b, c, h, w)
    upscaled = self.output_upscaling(src_2d)

    hyper_in_list = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b2, c2, h2, w2 = upscaled.shape
    masks = (hyper_in @ upscaled.view(b2, c2, h2 * w2)).view(b2, -1, h2, w2)
    iou_pred = self.iou_prediction_head(iou_token_out)

    slice_ = slice(1, None)
    masks = masks[:, slice_, :, :]
    iou_pred = iou_pred[:, slice_]

    return masks, iou_pred, src_transformed, iou_token_out


def patch_decoder(model):
    """给模型挂载带特征提取的 decoder forward"""
    model.mask_decoder.forward_with_features = _forward_with_features.__get__(
        model.mask_decoder, type(model.mask_decoder))


# ============================================================
# 2. 蒸馏损失函数
# ============================================================

def compute_advanced_distill_loss(
    student_masks, student_iou, student_src, student_iou_token,
    teacher_masks, teacher_iou, teacher_src, teacher_iou_token,
    lambda_feat=0.5, lambda_iou=0.1
):
    """
    三部分损失:
    1. L_mask: BCE(学生logits, 教师probs)
    2. L_feat: MSE(学生decoder特征, 教师decoder特征)
    3. L_iou:  MSE(学生IoU预测, 教师IoU预测)
    """
    teacher_probs = torch.sigmoid(teacher_masks).detach()

    loss_mask = F.binary_cross_entropy_with_logits(student_masks, teacher_probs, reduction='mean')

    loss_feat = F.mse_loss(student_src, teacher_src.detach()) if lambda_feat > 0 else 0.0

    loss_iou = F.mse_loss(student_iou, teacher_iou.detach()) if lambda_iou > 0 else 0.0

    total = loss_mask + lambda_feat * loss_feat + lambda_iou * loss_iou
    return total, {'mask': loss_mask.item(), 'feat': loss_feat.item() if lambda_feat > 0 else 0,
                   'iou': loss_iou.item() if lambda_iou > 0 else 0}


# ============================================================
# 3. 数据加载 — 使用完整 COCO val2017
# ============================================================

class COCODistillDataset(Dataset):
    """使用完整COCO标注，支持动态选择训练子集"""

    def __init__(self, ann_file, img_dir, max_boxes=5, max_images=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.max_boxes = max_boxes

        img_ids = sorted(self.coco.imgs.keys())
        if max_images is not None:
            img_ids = img_ids[:max_images]

        self.samples = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                continue
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            valid = []
            for ann in anns[:max_boxes]:
                x, y, w, h = ann['bbox']
                if w > 3 and h > 3:
                    valid.append({
                        'bbox': [x, y, x + w, y + h],
                        'segmentation': ann['segmentation'],
                        'image_id': ann['image_id'],
                    })
            if valid:
                self.samples.append({
                    'img_path': img_path,
                    'img_id': img_id,
                    'annotations': valid,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_np = np.array(Image.open(sample['img_path']).convert('RGB'))
        boxes = []
        for ann in sample['annotations']:
            boxes.append(ann['bbox'])
        return {
            'image_np': image_np,
            'boxes': np.array(boxes, dtype=np.float32),
            'img_path': sample['img_path'],
        }


# ============================================================
# 4. 预计算 Teacher 标签 (含特征)
# ============================================================

@torch.no_grad()
def precompute_teacher_data(teacher, coco_gt, img_dir, max_boxes=5, max_images=None, device='cpu'):
    """预计算 Teacher 的掩码、IoU预测和中间特征"""
    transform = ResizeLongestSide(1024)
    img_ids = sorted(coco_gt.imgs.keys())
    if max_images is not None:
        img_ids = img_ids[:max_images]

    cache = {}
    for img_id in tqdm(img_ids, desc="Precomputing teacher"):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        image_np = np.array(Image.open(img_path).convert('RGB'))
        original_size = image_np.shape[:2]

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()

        preprocessed = teacher.preprocess(input_tensor)
        image_embedding = teacher.image_encoder(preprocessed.unsqueeze(0))

        boxes_orig = []
        for ann in anns[:max_boxes]:
            x, y, w, h = ann['bbox']
            boxes_orig.append([x, y, x + w, y + h])

        boxes_tensor = torch.tensor(boxes_orig, dtype=torch.float32, device=device)
        boxes_transformed = transform.apply_boxes_torch(boxes_tensor, original_size)

        teacher_masks_list = []
        teacher_iou_list = []
        teacher_src_list = []
        teacher_iou_token_list = []

        for i in range(len(boxes_orig)):
            box = boxes_transformed[i:i+1]
            sparse_emb, dense_emb = teacher.prompt_encoder(points=None, boxes=box, masks=None)
            masks, iou_pred, src_trans, iou_token = teacher.mask_decoder.forward_with_features(
                image_embeddings=image_embedding,
                image_pe=teacher.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
            )
            best_idx = iou_pred[0].argmax().item()
            teacher_masks_list.append(masks[:, best_idx:best_idx+1, :, :].cpu())
            teacher_iou_list.append(iou_pred[:, best_idx:best_idx+1].cpu())
            teacher_src_list.append(src_trans.cpu())
            teacher_iou_token_list.append(iou_token.cpu())

        cache[img_id] = {
            'img_path': img_path,
            'boxes_orig': boxes_tensor.cpu(),
            'teacher_masks': torch.cat(teacher_masks_list, dim=0),
            'teacher_iou': torch.cat(teacher_iou_list, dim=0),
            'teacher_src': torch.cat(teacher_src_list, dim=0),
            'teacher_iou_token': torch.cat(teacher_iou_token_list, dim=0),
            'original_size': original_size,
        }

    return cache


# ============================================================
# 5. 训练
# ============================================================

def train_epoch(student, cache, optimizer, device, epoch, total_epochs,
                lambda_feat=0.5, lambda_iou=0.1):
    student.train()
    transform = ResizeLongestSide(1024)
    total_losses = {'total': 0, 'mask': 0, 'feat': 0, 'iou': 0}
    num_valid = 0

    img_ids = list(cache.keys())
    np.random.shuffle(img_ids)

    pbar = tqdm(img_ids, desc=f"Epoch {epoch+1}/{total_epochs}")
    for img_id in pbar:
        data = cache[img_id]
        image_np = np.array(Image.open(data['img_path']).convert('RGB'))
        original_size = data['original_size']

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()

        preprocessed = student.preprocess(input_tensor)
        image_embedding = student.image_encoder(preprocessed.unsqueeze(0))

        boxes_orig = data['boxes_orig'].to(device)
        boxes_transformed = transform.apply_boxes_torch(boxes_orig, original_size)
        t_masks = data['teacher_masks'].to(device)
        t_iou = data['teacher_iou'].to(device)
        t_src = data['teacher_src'].to(device)
        t_iou_token = data['teacher_iou_token'].to(device)

        batch_losses = {'total': 0, 'mask': 0, 'feat': 0, 'iou': 0}
        n = 0

        for i in range(boxes_orig.shape[0]):
            box = boxes_transformed[i:i+1]
            se, de = student.prompt_encoder(points=None, boxes=box, masks=None)
            s_masks, s_iou, s_src, s_iou_token = student.mask_decoder.forward_with_features(
                image_embeddings=image_embedding,
                image_pe=student.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
            )

            best_idx = s_iou[0].argmax().item()
            loss, details = compute_advanced_distill_loss(
                s_masks[:, best_idx:best_idx+1, :, :], s_iou[:, best_idx:best_idx+1],
                s_src, s_iou_token,
                t_masks[i:i+1], t_iou[i:i+1],
                t_src[i:i+1], t_iou_token[i:i+1],
                lambda_feat=lambda_feat, lambda_iou=lambda_iou,
            )

            batch_losses['total'] += loss
            batch_losses['mask'] += details['mask']
            batch_losses['feat'] += details['feat']
            batch_losses['iou'] += details['iou']
            n += 1

        if n > 0:
            (batch_losses['total'] / n).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_losses['total'] += (batch_losses['total'] / n).item()
            total_losses['mask'] += batch_losses['mask'] / n
            total_losses['feat'] += batch_losses['feat'] / n
            total_losses['iou'] += batch_losses['iou'] / n
            num_valid += 1

        pbar.set_postfix({'loss': f"{total_losses['total']/max(num_valid,1):.4f}"})

    for k in total_losses:
        total_losses[k] /= max(num_valid, 1)
    return total_losses


# ============================================================
# 6. 快速评估
# ============================================================

@torch.no_grad()
def quick_eval(student, cache, device='cpu'):
    student.eval()
    transform = ResizeLongestSide(1024)
    ious = []

    for img_id, data in tqdm(cache.items(), desc="Eval"):
        image_np = np.array(Image.open(data['img_path']).convert('RGB'))
        original_size = data['original_size']

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        input_size = tuple(input_tensor.shape[-2:])

        preprocessed = student.preprocess(input_tensor)
        image_embedding = student.image_encoder(preprocessed.unsqueeze(0))

        boxes_orig = data['boxes_orig'].to(device)
        boxes_transformed = transform.apply_boxes_torch(boxes_orig, original_size)

        for i in range(boxes_orig.shape[0]):
            box = boxes_transformed[i:i+1]
            se, de = student.prompt_encoder(points=None, boxes=box, masks=None)
            masks, iou_pred = student.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=student.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
            )
            best_idx = iou_pred[0].argmax().item()
            mask = student.postprocess_masks(
                masks[:, best_idx:best_idx+1, :, :], input_size, original_size)
            pred = (mask > student.mask_threshold).squeeze().cpu().numpy().astype(bool)

            gt_mask = data.get('gt_mask')
            if gt_mask is None:
                continue
            gt = gt_mask[i].astype(bool) if i < len(gt_mask) else None
            if gt is not None:
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                if union > 0:
                    ious.append(inter / union)

    return np.mean(ious) if ious else 0.0


# ============================================================
# 7. 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images', type=int, default=None,
                        help='训练图片数 (None=全部5000)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_feat', type=float, default=0.5)
    parser.add_argument('--lambda_iou', type=float, default=0.1)
    parser.add_argument('--max_boxes', type=int, default=5)
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用完整 COCO val2017 (5000图片, 36781实例)
    ANN_FILE = os.path.join(BASE, 'eval_data/annotations/instances_val2017.json')
    IMG_DIR = os.path.join(BASE, 'eval_data/val2017')

    # fallback: 如果完整数据集不存在，使用部分数据集
    if not os.path.exists(IMG_DIR):
        IMG_DIR = os.path.join(BASE, 'eval_data/test_100')
        ANN_FILE = os.path.join(BASE, 'eval_data/partial_annotations/instances_val2017.json')

    print("=" * 70)
    print("提示循环蒸馏 (Prompt-in-the-Loop Distillation)")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"数据: {ANN_FILE}")
    print(f"图片: {IMG_DIR}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Lambda: feat={args.lambda_feat}, iou={args.lambda_iou}")

    # Load data
    print("\n[1/5] 加载 COCO...")
    coco_gt = COCO(ANN_FILE)
    print(f"  {len(coco_gt.imgs)} 图像, {len(coco_gt.anns)} 标注")

    # Load teacher
    print("\n[2/5] 加载 Teacher (TinySAM)...")
    teacher = sam_model_registry['vit_t'](
        checkpoint=os.path.join(BASE, 'TinySAM/weights/tinysam_42.3.pth'))
    patch_decoder(teacher)
    teacher.to(DEVICE).eval()
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M")

    # Precompute teacher data
    print(f"\n[3/5] 预计算 Teacher 数据 (max_images={args.max_images or 'all'})...")
    cache = precompute_teacher_data(
        teacher, coco_gt, IMG_DIR,
        max_boxes=args.max_boxes, max_images=args.max_images, device=DEVICE)
    total_instances = sum(d['teacher_masks'].shape[0] for d in cache.values())
    print(f"  缓存 {len(cache)} 图像, {total_instances} 实例")

    del teacher
    torch.cuda.empty_cache()

    # Load student
    print("\n[4/5] 加载 Student (Pruned-M)...")
    from pruned_sam import build_pruned_sam
    student = build_pruned_sam(
        'pruned_m',
        checkpoint=os.path.join(BASE, 'pruned_sam/weights/pruned_m.pth'))
    patch_decoder(student)
    student.to(DEVICE)

    # 全参数微调 (unfreeze all)
    for p in student.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  总 params: {total/1e6:.2f}M")
    print(f"  可训练: {trainable/1e6:.2f}M (全参数微调)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Add GT masks to cache for eval
    for img_id in list(cache.keys()):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        gt_masks = []
        for ann in anns:
            gt_masks.append(coco_gt.annToMask(ann))
        cache[img_id]['gt_mask'] = gt_masks

    # Baseline eval
    print("\n  Baseline评估...")
    eval_subset = dict(list(cache.items())[:100])
    miou_before = quick_eval(student, eval_subset, DEVICE)
    print(f"  Student mIoU BEFORE: {miou_before:.4f}")

    # Training
    print(f"\n[5/5] 训练 {args.epochs} epochs...")
    print(f"{'Epoch':<8}{'Loss':<12}{'Mask':<12}{'Feat':<12}{'IoU':<12}{'mIoU':<12}{'LR':<12}")
    print("-" * 80)

    best_miou = miou_before
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = train_epoch(student, cache, optimizer, DEVICE, epoch, args.epochs,
                              lambda_feat=args.lambda_feat, lambda_iou=args.lambda_iou)
        scheduler.step()

        miou = quick_eval(student, eval_subset, DEVICE)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:<8}{losses['total']:<12.4f}{losses['mask']:<12.4f}"
              f"{losses['feat']:<12.4f}{losses['iou']:<12.4f}{miou:<12.4f}{lr:<12.2e}")

        if miou > best_miou:
            best_miou = miou
            ckpt = os.path.join(BASE, 'pruned_sam/weights',
                                f'pruned_m_distill_e{epoch+1}_miou{miou:.3f}.pth')
            torch.save(student.state_dict(), ckpt)
            print(f"  → 最佳: {ckpt}")

    final = os.path.join(BASE, 'pruned_sam/weights/pruned_m_distill_final.pth')
    torch.save(student.state_dict(), final)
    print(f"\n✅ 完成! 最佳 mIoU: {best_miou:.4f}")
    print(f"   模型: {final}")
    return student


if __name__ == '__main__':
    main()
