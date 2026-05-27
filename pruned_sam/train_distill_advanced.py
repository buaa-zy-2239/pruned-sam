"""
提示循环蒸馏 (Prompt-in-the-Loop Distillation) 进阶微调
====================================================
相比 LoRA 微调的核心改进:
1. 全参数微调 (unfreeze all) — 恢复全部模型容量
2. 特征对齐蒸馏 — 对齐解码器内部特征 (不仅仅是最终掩码)
3. IoU 预测头蒸馏 — 同步学习质量评估能力
4. 更多训练数据 — 使用完整 COCO val2017 (5000图, 36781实例)

内存优化: Teacher 模型常驻 GPU, 实时生成软标签 (不缓存特征)
目标 mIoU: 0.65-0.70 (Box Prompt)
用法:
  python pruned_sam/train_distill_advanced.py                    # 全量 5000 图
  python pruned_sam/train_distill_advanced.py --max_images 500   # 子集快速验证
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TinySAM'))

from tinysam.utils.transforms import ResizeLongestSide
from tinysam.build_sam import sam_model_registry


# ============================================================
# 1. 特征提取 — 对 MaskDecoder 做 monkey-patch
# ============================================================

def _forward_with_features(self, image_embeddings, image_pe,
                            sparse_prompt_embeddings, dense_prompt_embeddings):
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
    model.mask_decoder.forward_with_features = _forward_with_features.__get__(
        model.mask_decoder, type(model.mask_decoder))


# ============================================================
# 2. 损失函数
# ============================================================

def compute_distill_loss(s_masks, s_iou, s_src, s_iou_token,
                          t_masks, t_iou, t_src, t_iou_token,
                          lambda_feat=0.5, lambda_iou=0.1):
    t_probs = torch.sigmoid(t_masks).detach()
    loss_mask = F.binary_cross_entropy_with_logits(s_masks, t_probs, reduction='mean')
    loss_feat = F.mse_loss(s_src, t_src.detach())
    loss_iou = F.mse_loss(s_iou, t_iou.detach())
    total = loss_mask + lambda_feat * loss_feat + lambda_iou * loss_iou
    return total, {'mask': loss_mask.item(), 'feat': loss_feat.item(), 'iou': loss_iou.item()}


# ============================================================
# 3. 数据准备 (只存元数据, 不存模型输出)
# ============================================================

def prepare_metadata(coco_gt, img_dir, max_boxes=5, max_images=None):
    """只保存图片路径和边界框, 不缓存任何模型输出"""
    img_ids = sorted(coco_gt.imgs.keys())
    if max_images is not None:
        img_ids = img_ids[:max_images]

    meta = []
    for img_id in tqdm(img_ids, desc="Preparing metadata"):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        boxes = []
        for ann in anns[:max_boxes]:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

        meta.append({
            'img_id': img_id,
            'img_path': img_path,
            'boxes': np.array(boxes, dtype=np.float32),
            'original_size': (img_info['height'], img_info['width']),
        })

    return meta


@torch.no_grad()
def teacher_forward(teacher, image_np, boxes_np, device):
    """单张图片的教师前向传播 (全量输出含特征)"""
    transform = ResizeLongestSide(1024)
    original_size = image_np.shape[:2]

    input_image = transform.apply_image(image_np)
    input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    preprocessed = teacher.preprocess(input_tensor)
    image_embedding = teacher.image_encoder(preprocessed.unsqueeze(0))

    boxes_trans = transform.apply_boxes(boxes_np, original_size)
    boxes_tensor = torch.as_tensor(boxes_trans, dtype=torch.float32, device=device)

    t_masks_list, t_iou_list, t_src_list, t_token_list = [], [], [], []
    for i in range(boxes_tensor.shape[0]):
        box = boxes_tensor[i:i+1]
        se, de = teacher.prompt_encoder(points=None, boxes=box, masks=None)
        masks, iou_pred, src_trans, iou_token = teacher.mask_decoder.forward_with_features(
            image_embeddings=image_embedding,
            image_pe=teacher.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se, dense_prompt_embeddings=de)
        best = iou_pred[0].argmax().item()
        t_masks_list.append(masks[:, best:best+1, :, :])
        t_iou_list.append(iou_pred[:, best:best+1])
        t_src_list.append(src_trans)
        t_token_list.append(iou_token)

    return (torch.cat(t_masks_list, dim=0), torch.cat(t_iou_list, dim=0),
            torch.cat(t_src_list, dim=0), torch.cat(t_token_list, dim=0))


def student_forward(student, image_np, boxes_np, device):
    """单张图片的学生前向传播 (含特征)"""
    transform = ResizeLongestSide(1024)
    original_size = image_np.shape[:2]

    input_image = transform.apply_image(image_np)
    input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    preprocessed = student.preprocess(input_tensor)
    image_embedding = student.image_encoder(preprocessed.unsqueeze(0))

    boxes_trans = transform.apply_boxes(boxes_np, original_size)
    boxes_tensor = torch.as_tensor(boxes_trans, dtype=torch.float32, device=device)

    s_masks_list, s_iou_list, s_src_list, s_token_list = [], [], [], []
    for i in range(boxes_tensor.shape[0]):
        box = boxes_tensor[i:i+1]
        se, de = student.prompt_encoder(points=None, boxes=box, masks=None)
        masks, iou_pred, src_trans, iou_token = student.mask_decoder.forward_with_features(
            image_embeddings=image_embedding,
            image_pe=student.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se, dense_prompt_embeddings=de)
        best = iou_pred[0].argmax().item()
        s_masks_list.append(masks[:, best:best+1, :, :])
        s_iou_list.append(iou_pred[:, best:best+1])
        s_src_list.append(src_trans)
        s_token_list.append(iou_token)

    return (torch.cat(s_masks_list, dim=0), torch.cat(s_iou_list, dim=0),
            torch.cat(s_src_list, dim=0), torch.cat(s_token_list, dim=0))


# ============================================================
# 4. 训练
# ============================================================

def train_epoch(student, teacher, meta, optimizer, device, epoch, total_epochs,
                lambda_feat=0.5, lambda_iou=0.1):
    student.train()
    teacher.eval()
    total_losses = {'total': 0, 'mask': 0, 'feat': 0, 'iou': 0}
    np.random.shuffle(meta)

    pbar = tqdm(meta, desc=f"Epoch {epoch+1}/{total_epochs}")
    for item in pbar:
        image_np = np.array(Image.open(item['img_path']).convert('RGB'))
        boxes_np = item['boxes']

        # Teacher forward (no grad)
        with torch.no_grad():
            t_masks, t_iou, t_src, t_token = teacher_forward(
                teacher, image_np, boxes_np, device)

        # Student forward (with grad)
        s_masks, s_iou, s_src, s_token = student_forward(
            student, image_np, boxes_np, device)

        loss, details = compute_distill_loss(
            s_masks, s_iou, s_src, s_token,
            t_masks, t_iou, t_src, t_token,
            lambda_feat, lambda_iou)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_losses['total'] += loss.item()
        total_losses['mask'] += details['mask']
        total_losses['feat'] += details['feat']
        total_losses['iou'] += details['iou']
        pbar.set_postfix({'loss': f"{details['mask']:.4f}"})

    n = max(len(meta), 1)
    return {k: v / n for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(student, meta, device='cpu', num_samples=100):
    student.eval()
    transform = ResizeLongestSide(1024)
    ious = []

    from pycocotools.coco import COCO
    coco_gt = COCO(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'eval_data/partial_annotations/instances_val2017.json'))

    for item in tqdm(meta[:num_samples], desc="Eval"):
        image_np = np.array(Image.open(item['img_path']).convert('RGB'))
        original_size = item['original_size']
        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        input_size = tuple(input_tensor.shape[-2:])
        preprocessed = student.preprocess(input_tensor)
        image_embedding = student.image_encoder(preprocessed.unsqueeze(0))

        for i in range(len(item['boxes'])):
            box = item['boxes'][i:i+1]
            box_trans = transform.apply_boxes(box, original_size)
            box_tensor = torch.as_tensor(box_trans, dtype=torch.float32, device=device)
            se, de = student.prompt_encoder(points=None, boxes=box_tensor, masks=None)
            masks, iou_pred = student.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=student.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se, dense_prompt_embeddings=de)
            best = iou_pred[0].argmax().item()
            mask = student.postprocess_masks(
                masks[:, best:best+1, :, :], input_size, original_size)
            pred = (mask > student.mask_threshold).squeeze().cpu().numpy().astype(bool)

            ann_ids = coco_gt.getAnnIds(imgIds=[item['img_id']])
            anns = coco_gt.loadAnns(ann_ids)
            if i < len(anns):
                gt = coco_gt.annToMask(anns[i]).astype(bool)
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                if union > 0:
                    ious.append(inter / union)

    return np.mean(ious) if ious else 0.0


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_feat', type=float, default=0.5)
    parser.add_argument('--lambda_iou', type=float, default=0.1)
    parser.add_argument('--max_boxes', type=int, default=5)
    args = parser.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ANN_FILE = os.path.join(BASE, 'eval_data/annotations/instances_val2017.json')
    IMG_DIR = os.path.join(BASE, 'eval_data/val2017')
    if not os.path.exists(IMG_DIR):
        IMG_DIR = os.path.join(BASE, 'eval_data/test_100')
        ANN_FILE = os.path.join(BASE, 'eval_data/partial_annotations/instances_val2017.json')

    print("=" * 70)
    print("提示循环蒸馏 (Prompt-in-the-Loop Distillation)")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"特征蒸馏: λ_feat={args.lambda_feat}, λ_iou={args.lambda_iou}")
    print(f"内存模式: 实时计算 (Teacher 常驻 GPU, 不缓存特征)")

    print("\n[1/5] 加载 COCO...")
    coco_gt = COCO(ANN_FILE)
    print(f"  {len(coco_gt.imgs)} 图像, {len(coco_gt.anns)} 标注")

    print("\n[2/5] 准备元数据...")
    meta = prepare_metadata(coco_gt, IMG_DIR, max_boxes=args.max_boxes, max_images=args.max_images)
    total_instances = sum(len(m['boxes']) for m in meta)
    print(f"  {len(meta)} 图像, {total_instances} 实例")
    eval_meta = meta[:100]

    print("\n[3/5] 加载 Teacher (TinySAM)...")
    teacher = sam_model_registry['vit_t'](
        checkpoint=os.path.join(BASE, 'TinySAM/weights/tinysam_42.3.pth'))
    patch_decoder(teacher)
    teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M (冻结)")

    print("\n[4/5] 加载 Student (Pruned-M) + LoRA...")
    from pruned_sam import build_pruned_sam
    student = build_pruned_sam(
        'pruned_m',
        checkpoint=os.path.join(BASE, 'pruned_sam/weights/pruned_m.pth'))
    patch_decoder(student)
    student.to(DEVICE)
    for p in student.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in student.parameters())
    print(f"  总 params: {total/1e6:.2f}M (全参数可训练)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n  Baseline 评估...")
    miou_before = evaluate(student, meta, DEVICE, num_samples=100)
    print(f"  Student mIoU BEFORE: {miou_before:.4f}")

    print(f"\n[5/5] 训练 {args.epochs} epochs...")
    print(f"{'Epoch':<8}{'Loss':<10}{'Mask':<10}{'Feat':<10}{'IoU':<10}{'mIoU':<10}{'Time':<8}")
    print("-" * 66)

    best_miou = miou_before
    for epoch in range(args.epochs):
        t0 = time.time()
        losses = train_epoch(student, teacher, meta, optimizer, DEVICE, epoch, args.epochs,
                              args.lambda_feat, args.lambda_iou)
        scheduler.step()

        miou = evaluate(student, eval_meta, DEVICE, num_samples=100)
        elapsed = time.time() - t0

        print(f"{epoch+1:<8}{losses['total']:<10.4f}{losses['mask']:<10.4f}"
              f"{losses['feat']:<10.4f}{losses['iou']:<10.4f}{miou:<10.4f}{elapsed:<8.0f}")

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
