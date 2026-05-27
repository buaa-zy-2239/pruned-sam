"""
LoRA + 知识蒸馏微调脚本
======================
目标：在 COCO 子集上对剪枝后的 Pruned-M 进行轻量级微调，恢复分割精度

训练策略:
1. LoRA: 只在 Attention 的 Q/K/V/Out 投影层插入低秩适配器
2. 知识蒸馏: TinySAM Teacher 生成软标签，Student (Pruned-M) 学习
3. 预计算Teacher标签: 训练前一次性生成，避免每个epoch重复推理

适用环境: Colab (GPU) / CPU
使用方法:
  python pruned_sam/train_lora_distill.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys
import time
import math

import os as _os
_BASE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
sys.path.insert(0, _os.path.join(_BASE, 'TinySAM'))

from tinysam.utils.transforms import ResizeLongestSide
from tinysam.build_sam import sam_model_registry


# ============================================================
# 1. LoRA 模块
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.alpha


def apply_lora_to_attention(attn, rank=4):
    for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
        layer = getattr(attn, name, None)
        if layer is not None and isinstance(layer, nn.Linear):
            setattr(attn, name, LoRALinear(layer, rank=rank))


def apply_lora_to_mlp(mlp, rank=4):
    for name in ['lin1', 'lin2']:
        layer = getattr(mlp, name, None)
        if layer is not None and isinstance(layer, nn.Linear):
            setattr(mlp, name, LoRALinear(layer, rank=rank))


def apply_lora_to_model(model, rank=4):
    lora_count = 0
    dec = model.mask_decoder

    for layer in dec.transformer.layers:
        for name in ['self_attn', 'cross_attn_token_to_image', 'cross_attn_image_to_token']:
            apply_lora_to_attention(getattr(layer, name), rank=rank)
            lora_count += 4
        apply_lora_to_mlp(layer.mlp, rank=rank)
        lora_count += 2

    apply_lora_to_attention(dec.transformer.final_attn_token_to_image, rank=rank)
    lora_count += 4

    for mlp in dec.output_hypernetworks_mlps:
        for i, layer in enumerate(mlp.layers):
            if isinstance(layer, nn.Linear):
                mlp.layers[i] = LoRALinear(layer, rank=rank)
                lora_count += 1

    for i, layer in enumerate(dec.iou_prediction_head.layers):
        if isinstance(layer, nn.Linear):
            dec.iou_prediction_head.layers[i] = LoRALinear(layer, rank=rank)
            lora_count += 1

    enc = model.image_encoder
    for layer_idx in range(1, len(enc.layers)):
        for block in enc.layers[layer_idx].blocks:
            if hasattr(block, 'attn'):
                apply_lora_to_attention(block.attn, rank=rank)
                lora_count += 4
            if hasattr(block, 'mlp'):
                apply_lora_to_mlp(block.mlp, rank=rank)
                lora_count += 2

    print(f"  LoRA 适配器: {lora_count}")
    return model


def count_lora_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)


# ============================================================
# 2. 预计算 Teacher 标签
# ============================================================

@torch.no_grad()
def precompute_teacher_labels(teacher_model, coco_gt, img_dir, max_boxes=5, device='cpu'):
    """一次遍历所有样本，生成 Teacher 的软标签并缓存"""
    transform = ResizeLongestSide(1024)
    img_ids = sorted(coco_gt.imgs.keys())
    cache = {}

    for img_id in tqdm(img_ids, desc="Precomputing teacher labels"):
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
        original_size = (orig_h, orig_w)

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()

        preprocessed = teacher_model.preprocess(input_tensor)
        image_embedding = teacher_model.image_encoder(preprocessed.unsqueeze(0))

        boxes_orig = []
        gt_masks = []
        teacher_probs_list = []

        for ann in anns[:max_boxes]:
            x, y, bw, bh = ann['bbox']
            boxes_orig.append([x, y, x + bw, y + bh])
            gt_masks.append(coco_gt.annToMask(ann))

        boxes_tensor = torch.tensor(boxes_orig, dtype=torch.float32, device=device)
        boxes_transformed = transform.apply_boxes_torch(boxes_tensor, original_size).cpu()

        for i in range(len(boxes_orig)):
            box = boxes_transformed[i:i+1].to(device)
            sparse_emb, dense_emb = teacher_model.prompt_encoder(points=None, boxes=box, masks=None)
            low_res, iou_pred = teacher_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=teacher_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
            )
            best_idx = iou_pred[0].argmax().item()
            teacher_probs_list.append(torch.sigmoid(low_res[:, best_idx:best_idx+1, :, :]).cpu())

        cache[img_id] = {
            'img_path': img_path,
            'boxes_orig': boxes_tensor.cpu(),
            'gt_masks': np.stack(gt_masks) if gt_masks else None,
            'teacher_probs': torch.cat(teacher_probs_list, dim=0) if teacher_probs_list else None,
            'original_size': original_size,
        }

    return cache


# ============================================================
# 3. 训练
# ============================================================

def train_epoch(student, cache, optimizer, device, epoch, total_epochs):
    student.train()
    transform = ResizeLongestSide(1024)
    total_loss = 0
    total_distill = 0
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
        teacher_probs = data['teacher_probs'].to(device)
        gt_masks = data['gt_masks']

        batch_loss = 0
        batch_distill = 0
        n = 0

        for i in range(boxes_orig.shape[0]):
            box = boxes_transformed[i:i+1]
            sparse_emb, dense_emb = student.prompt_encoder(points=None, boxes=box, masks=None)
            low_res, iou_pred = student.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=student.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
            )

            best_idx = iou_pred[0].argmax().item()
            logits = low_res[:, best_idx:best_idx+1, :, :]

            t_mask = teacher_probs[i:i+1].to(device)
            if t_mask.shape[-2:] != logits.shape[-2:]:
                t_mask = F.interpolate(t_mask, size=logits.shape[-2:], mode='bilinear', align_corners=False)

            loss_d = F.binary_cross_entropy_with_logits(logits, t_mask, reduction='mean')

            loss_gt = 0.0
            if gt_masks is not None and i < len(gt_masks):
                gt = torch.as_tensor(gt_masks[i], device=device).float().unsqueeze(0).unsqueeze(0)
                if gt.shape[-2:] != logits.shape[-2:]:
                    gt = F.interpolate(gt, size=logits.shape[-2:], mode='nearest')
                loss_gt = F.binary_cross_entropy_with_logits(logits, gt, reduction='mean')

            loss = loss_d + 0.5 * loss_gt
            batch_loss += loss
            batch_distill += loss_d.item()
            n += 1

        if n > 0:
            (batch_loss / n).backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += (batch_loss / n).item()
            total_distill += batch_distill / n
            num_valid += 1

        pbar.set_postfix({'loss': f"{total_loss/max(num_valid,1):.4f}"})

    return total_loss / max(num_valid, 1), total_distill / max(num_valid, 1)


# ============================================================
# 4. 评估 (快速验证)
# ============================================================

@torch.no_grad()
def quick_eval(student, cache, device='cpu'):
    """在缓存数据上快速评估 mIoU"""
    student.eval()
    transform = ResizeLongestSide(1024)
    ious = []

    for img_id, data in tqdm(cache.items(), desc="Evaluating"):
        image_np = np.array(Image.open(data['img_path']).convert('RGB'))
        original_size = data['original_size']

        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
        input_size = tuple(input_tensor.shape[-2:])

        preprocessed = student.preprocess(input_tensor)
        image_embedding = student.image_encoder(preprocessed.unsqueeze(0))

        boxes_orig = data['boxes_orig'].to(device)
        boxes_transformed = transform.apply_boxes_torch(boxes_orig, original_size)
        gt_masks = data['gt_masks']

        for i in range(boxes_orig.shape[0]):
            box = boxes_transformed[i:i+1]
            sparse_emb, dense_emb = student.prompt_encoder(points=None, boxes=box, masks=None)
            low_res, iou_pred = student.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=student.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
            )

            best_idx = iou_pred[0].argmax().item()
            mask = student.postprocess_masks(
                low_res[:, best_idx:best_idx+1, :, :], input_size, original_size
            )
            pred = (mask > student.mask_threshold).squeeze().cpu().numpy().astype(bool)

            if gt_masks is not None and i < len(gt_masks):
                gt = gt_masks[i].astype(bool)
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                if union > 0:
                    ious.append(inter / union)

    return np.mean(ious) if ious else 0.0


# ============================================================
# 5. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("LoRA + 知识蒸馏微调: Pruned-M ← TinySAM Teacher")
    print("=" * 70)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LORA_RANK = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    MAX_BOXES = 5
    SAVE_DIR = _os.path.join(_BASE, 'pruned_sam/weights')
    ANN_FILE = _os.path.join(_BASE, 'eval_data/partial_annotations/instances_val2017.json')
    IMG_DIR = _os.path.join(_BASE, 'eval_data/test_100')

    print(f"\n设备: {DEVICE}")
    print(f"LoRA Rank: {LORA_RANK}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")

    print("\n[1/5] 加载 COCO 标注...")
    coco_gt = COCO(ANN_FILE)
    print(f"  {len(coco_gt.imgs)} 图像, {len(coco_gt.anns)} 标注")

    print("\n[2/5] 加载 Teacher (TinySAM)...")
    teacher = sam_model_registry['vit_t'](
        checkpoint=_os.path.join(_BASE, 'TinySAM/weights/tinysam_42.3.pth')
    )
    teacher.to(DEVICE).eval()
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M params")

    print("\n[3/5] 预计算 Teacher 软标签...")
    cache = precompute_teacher_labels(teacher, coco_gt, IMG_DIR, max_boxes=MAX_BOXES, device=DEVICE)
    print(f"  缓存 {len(cache)} 个样本, "
          f"共 {sum(d['teacher_probs'].shape[0] for d in cache.values())} 个实例")

    del teacher
    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None

    print("\n[4/5] 加载 Student (Pruned-M) + LoRA...")
    from pruned_sam import build_pruned_sam
    student = build_pruned_sam(
        'pruned_m',
        checkpoint=_os.path.join(_BASE, 'pruned_sam/weights/pruned_m.pth')
    )
    student = apply_lora_to_model(student, rank=LORA_RANK)
    student.to(DEVICE)

    lora_params = count_lora_params(student)
    total = sum(p.numel() for p in student.parameters())
    print(f"  总 params: {total/1e6:.2f}M")
    print(f"  可训练 (LoRA): {lora_params/1e3:.1f}K ({lora_params/total*100:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for n, p in student.named_parameters() if 'lora_' in n],
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n[5/5] 训练 {NUM_EPOCHS} epochs...")
    print(f"{'Epoch':<8}{'Loss':<12}{'Distill':<12}{'mIoU':<12}{'LR':<12}{'Time':<8}")
    print("-" * 64)

    best_miou = 0.0
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        loss, distill = train_epoch(student, cache, optimizer, DEVICE, epoch, NUM_EPOCHS)
        scheduler.step()

        miou = quick_eval(student, cache, DEVICE)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:<8}{loss:<12.4f}{distill:<12.4f}{miou:<12.4f}{lr:<12.2e}{elapsed:<8.0f}")

        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(SAVE_DIR, f'pruned_m_ft_e{epoch+1}_miou{miou:.3f}.pth')
            torch.save(student.state_dict(), ckpt_path)
            print(f"  → 最佳模型: {ckpt_path}")

    final_path = os.path.join(SAVE_DIR, 'pruned_m_ft_final.pth')
    torch.save(student.state_dict(), final_path)
    lora_path = os.path.join(SAVE_DIR, 'pruned_m_lora_only.pth')
    torch.save(
        {k: v for k, v in student.state_dict().items() if 'lora_' in k},
        lora_path
    )

    print(f"\n✅ 完成! 最终 mIoU: {best_miou:.4f}")
    print(f"   完整模型: {final_path}")
    print(f"   LoRA 权重: {lora_path}")
    return student


def load_ft_model(config_name='pruned_m', base_ckpt=None, ft_ckpt=None, lora_rank=4):
    from pruned_sam import build_pruned_sam
    model = build_pruned_sam(config_name, checkpoint=base_ckpt)
    model = apply_lora_to_model(model, rank=lora_rank)
    if ft_ckpt:
        model.load_state_dict(torch.load(ft_ckpt, map_location='cpu'), strict=False)
    return model


if __name__ == '__main__':
    main()
