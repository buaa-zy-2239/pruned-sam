"""
LoRA + Knowledge Distillation Fine-tuning for Pruned SAM
=======================================================
Target: Recover segmentation quality of Pruned-M model
Method: LoRA adapters on attention layers + distillation from TinySAM teacher
Hardware: Colab (T4 GPU) or any CUDA device
Time: ~1-2 hours on Colab T4 for 20 epochs on COCO 100 subset

Usage:
  1. Upload this script to Colab
  2. Upload the project zip or clone from repo
  3. Run: python lora_finetune_colab.py
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ======== CONFIG ========
class Config:
    # Paths
    project_root = '/content/vista-slam'
    tinysam_ckpt = '/content/vista-slam/TinySAM/weights/tinysam_42.3.pth'
    pruned_ckpt = '/content/vista-slam/pruned_sam/weights/pruned_m.pth'
    output_dir = '/content/vista-slam/pruned_sam/weights'
    output_name = 'pruned_m_lora_finetuned.pth'

    # Data
    ann_file = '/content/vista-slam/eval_data/partial_annotations/instances_val2017.json'
    img_dir = '/content/vista-slam/eval_data/test_100'

    # Training
    num_epochs = 20
    batch_size = 2
    lr = 3e-4
    weight_decay = 1e-4
    lora_rank = 8
    lora_alpha = 16
    max_images = 100
    max_boxes_per_img = 5
    accumulation_steps = 4

    # Distillation
    distill_weight = 1.0          # weight for distillation loss
    iou_weight = 0.1              # weight for IoU prediction loss
    mask_weight = 1.0             # weight for mask BCE loss
    temperature = 1.0             # distillation temperature

    # Eval
    eval_interval = 5             # evaluate every N epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ======== LORA IMPLEMENTATION ========
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for nn.Linear"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter (merge strategy: add output)"""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.lora = LoRALayer(self.in_features, self.out_features, rank, alpha)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.lora(x)


def inject_lora(module, rank=8, alpha=16, target_modules=None):
    """
    Recursively replace nn.Linear with LoRALinear in target_modules.

    Args:
        module: PyTorch module
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: list of substrings to match module names.
                         If None, applies to all nn.Linear.
    """
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj',
                          'out_proj', 'fc1', 'fc2', 'lin1', 'lin2']

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            replace = any(t in name for t in target_modules)
            if replace:
                setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha, target_modules)

    return module


def get_lora_params(model):
    """Get only LoRA parameters (trainable)"""
    return [p for n, p in model.named_parameters() if 'lora_' in n]


def freeze_all_except_lora(model):
    """Freeze all parameters except LoRA"""
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_lora_params(model):
    return sum(p.numel() for p in get_lora_params(model))


# ======== DATASET ========
class COCOSubsetDataset(Dataset):
    """COCO subset dataset for distillation fine-tuning with box prompts"""

    def __init__(self, ann_file, img_dir, max_images=100, max_boxes=5):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.max_boxes = max_boxes

        img_ids = sorted(self.coco.imgs.keys())[:max_images]
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

            valid_anns = []
            for ann in anns[:max_boxes]:
                x, y, w, h = ann['bbox']
                if w > 5 and h > 5:
                    valid_anns.append({
                        'bbox': [x, y, x + w, y + h],
                        'segmentation': ann['segmentation'],
                    })

            if valid_anns:
                self.samples.append({
                    'img_path': img_path,
                    'img_id': img_id,
                    'orig_h': img_info['height'],
                    'orig_w': img_info['width'],
                    'annotations': valid_anns,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_np = np.array(Image.open(sample['img_path']).convert('RGB'))

        boxes = []
        gt_masks = []
        for ann in sample['annotations']:
            boxes.append(ann['bbox'])
            x, y, x2, y2 = ann['bbox']
            w, h = x2 - x, y2 - y
            mask = self.coco.annToMask({
                'bbox': [x, y, w, h],
                'segmentation': ann['segmentation'],
            })
            gt_masks.append(mask)

        return {
            'image_np': image_np,
            'boxes': np.array(boxes, dtype=np.float32),
            'gt_masks': gt_masks,
            'orig_h': sample['orig_h'],
            'orig_w': sample['orig_w'],
        }


def collate_fn(batch):
    """Custom collate for variable-length boxes per image"""
    return batch


# ======== MODEL SETUP ========
def setup_models(cfg):
    sys.path.insert(0, cfg.project_root)
    sys.path.insert(0, os.path.join(cfg.project_root, 'TinySAM'))

    from tinysam.build_sam import sam_model_registry
    from pruned_sam import build_pruned_sam

    # Teacher: TinySAM (frozen)
    print("Loading teacher (TinySAM)...")
    teacher = sam_model_registry['vit_t'](checkpoint=cfg.tinysam_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(cfg.device)
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M")

    # Student: Pruned-M
    print("Loading student (Pruned-M)...")
    student = build_pruned_sam('pruned_m', checkpoint=cfg.pruned_ckpt)
    student.train()
    student.to(cfg.device)
    print(f"  Student params (before LoRA): {sum(p.numel() for p in student.parameters())/1e6:.2f}M")

    # Inject LoRA into student
    print("Injecting LoRA adapters...")
    target_modules = ['qkv', 'proj', 'q_proj', 'k_proj', 'v_proj',
                      'out_proj', 'fc1', 'fc2', 'lin1', 'lin2']
    student = inject_lora(student, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                          target_modules=target_modules)
    freeze_all_except_lora(student)
    n_lora = count_lora_params(student)
    print(f"  LoRA trainable params: {n_lora/1e3:.1f}K (rank={cfg.lora_rank})")
    print(f"  Total params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")

    return teacher, student


# ======== TRAINING ========
@torch.no_grad()
def prepare_batch(batch_item, model, device):
    """Prepare image embedding and boxes for a single image"""
    from tinysam.utils.transforms import ResizeLongestSide

    image_np = batch_item['image_np']
    boxes_np = batch_item['boxes']
    orig_h, orig_w = image_np.shape[:2]

    transform = ResizeLongestSide(model.image_encoder.img_size)
    input_image = transform.apply_image(image_np)
    input_tensor = torch.as_tensor(input_image, device=device)
    input_tensor = input_tensor.permute(2, 0, 1).contiguous()
    input_size = tuple(input_tensor.shape[-2:])
    original_size = (orig_h, orig_w)

    preprocessed = model.preprocess(input_tensor)
    image_embedding = model.image_encoder(preprocessed.unsqueeze(0))

    boxes_transformed = transform.apply_boxes(boxes_np, original_size)
    boxes_tensor = torch.as_tensor(boxes_transformed, dtype=torch.float, device=device)

    return image_embedding, boxes_tensor, input_size, original_size


@torch.no_grad()
def predict_masks(model, image_embedding, boxes_tensor, input_size, original_size):
    """Run mask prediction for all boxes"""
    all_masks = []
    all_iou_preds = []
    for i in range(boxes_tensor.shape[0]):
        box = boxes_tensor[i:i+1]
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box, masks=None)
        low_res, iou_pred = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb)
        masks_post = model.postprocess_masks(low_res, input_size, original_size)
        all_masks.append(masks_post)
        all_iou_preds.append(iou_pred)
    return torch.cat(all_masks, dim=0), torch.cat(all_iou_preds, dim=0)


def compute_distillation_loss(student_masks, teacher_masks, temperature=1.0):
    """Soft distillation loss using KL divergence"""
    student_probs = torch.sigmoid(student_masks / temperature)
    teacher_probs = torch.sigmoid(teacher_masks / temperature)
    loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction='mean')
    return loss


def compute_mask_bce(pred_masks, gt_masks_binary):
    """BCE loss between predicted masks and ground truth"""
    loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_binary, reduction='mean')
    return loss


def train_epoch(student, teacher, dataloader, optimizer, scaler, cfg, epoch):
    student.train()
    total_loss = 0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")):
        for batch_item in batch:
            # Prepare inputs (shared: image transform + boxes)
            from tinysam.utils.transforms import ResizeLongestSide
            image_np = batch_item['image_np']
            boxes_np = batch_item['boxes']
            orig_h, orig_w = image_np.shape[:2]
            transform = ResizeLongestSide(student.image_encoder.img_size)
            input_image = transform.apply_image(image_np)
            input_tensor = torch.as_tensor(input_image, device=cfg.device)
            input_tensor = input_tensor.permute(2, 0, 1).contiguous()
            input_size = tuple(input_tensor.shape[-2:])
            original_size = (orig_h, orig_w)
            boxes_transformed = transform.apply_boxes(boxes_np, original_size)
            boxes_tensor = torch.as_tensor(boxes_transformed, dtype=torch.float, device=cfg.device)

            # Teacher embedding (frozen)
            with torch.no_grad():
                teacher_preprocessed = teacher.preprocess(input_tensor)
                teacher_embed = teacher.image_encoder(teacher_preprocessed.unsqueeze(0))
                teacher_masks, _ = predict_masks(
                    teacher, teacher_embed, boxes_tensor, input_size, original_size)

            # Student embedding + prediction
            student_preprocessed = student.preprocess(input_tensor)
            student_embed = student.image_encoder(student_preprocessed.unsqueeze(0))
            student_masks, student_iou = predict_masks(
                student, student_embed, boxes_tensor, input_size, original_size)

            # Build ground truth masks
            gt_masks_list = []
            for i, gt_mask in enumerate(batch_item['gt_masks']):
                gt_tensor = torch.from_numpy(gt_mask.astype(np.float32)).to(cfg.device)
                gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)
                gt_resized = F.interpolate(gt_tensor, size=student_masks.shape[-2:],
                                           mode='nearest')
                gt_masks_list.append(gt_resized)
            gt_masks_binary = torch.cat(gt_masks_list, dim=0)

            # Losses
            distill_loss = compute_distillation_loss(
                student_masks, teacher_masks, cfg.temperature)
            mask_loss = compute_mask_bce(student_masks, gt_masks_binary)
            loss = (cfg.distill_weight * distill_loss +
                    cfg.mask_weight * mask_loss)

            loss = loss / cfg.accumulation_steps
            scaler.scale(loss).backward()

            total_loss += loss.item() * cfg.accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(get_lora_params(student), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    avg_loss = total_loss / max(n_batches, 1)
    t = time.time() - t0
    print(f"  Loss: {avg_loss:.4f} | Time: {t:.0f}s")
    return avg_loss


# ======== EVALUATION ========
@torch.no_grad()
def evaluate(model, cfg, num_samples=100):
    """Evaluate mIoU with box prompts"""
    sys.path.insert(0, cfg.project_root)
    sys.path.insert(0, os.path.join(cfg.project_root, 'TinySAM'))
    from pycocotools.coco import COCO
    from tinysam.utils.transforms import ResizeLongestSide

    model.eval()
    coco_gt = COCO(cfg.ann_file)
    img_ids = sorted(coco_gt.imgs.keys())[:num_samples]

    per_instance_ious = []
    for img_id in tqdm(img_ids, desc="Evaluating"):
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        if not anns:
            continue

        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(cfg.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        image_np = np.array(Image.open(img_path).convert('RGB'))
        orig_h, orig_w = image_np.shape[:2]

        transform = ResizeLongestSide(model.image_encoder.img_size)
        input_image = transform.apply_image(image_np)
        input_tensor = torch.as_tensor(input_image, device=cfg.device)
        input_tensor = input_tensor.permute(2, 0, 1).contiguous()
        input_size = tuple(input_tensor.shape[-2:])
        preprocessed = model.preprocess(input_tensor)
        image_embedding = model.image_encoder(preprocessed.unsqueeze(0))

        for ann in anns[:5]:
            x, y, bw, bh = ann['bbox']
            box_orig = np.array([[x, y, x + bw, y + bh]], dtype=float)
            box_trans = transform.apply_boxes(box_orig, (orig_h, orig_w))
            box_tensor = torch.as_tensor(box_trans, dtype=torch.float, device=cfg.device)

            sparse_emb, dense_emb = model.prompt_encoder(
                points=None, boxes=box_tensor, masks=None)
            low_res, iou_pred = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb)

            best_idx = iou_pred[0].argmax().item()
            best_mask = low_res[:, best_idx:best_idx+1, :, :]
            masks_post = model.postprocess_masks(best_mask, input_size, (orig_h, orig_w))
            pred_binary = (masks_post > model.mask_threshold).squeeze().cpu().numpy().astype(bool)

            gt_mask = coco_gt.annToMask(ann)
            inter = np.logical_and(pred_binary, gt_mask).sum()
            union = np.logical_or(pred_binary, gt_mask).sum()
            iou = inter / union if union > 0 else 0.0
            per_instance_ious.append(iou)

    miou = np.mean(per_instance_ious) if per_instance_ious else 0.0
    iou_05 = np.mean([i > 0.5 for i in per_instance_ious]) if per_instance_ious else 0.0
    print(f"  mIoU: {miou:.4f} | @0.5: {iou_05:.2%} | N={len(per_instance_ious)}")
    return miou


# ======== MAIN ========
def main():
    cfg = Config()

    print("=" * 70)
    print("LoRA + Distillation Fine-tuning for Pruned SAM")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"LoRA rank: {cfg.lora_rank}, alpha: {cfg.lora_alpha}")
    print(f"Epochs: {cfg.num_epochs}, LR: {cfg.lr}")
    print(f"Distill weight: {cfg.distill_weight}, Mask weight: {cfg.mask_weight}")
    print()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Setup models
    teacher, student = setup_models(cfg)

    # Setup data
    print("\nLoading dataset...")
    dataset = COCOSubsetDataset(cfg.ann_file, cfg.img_dir,
                                max_images=cfg.max_images,
                                max_boxes=cfg.max_boxes_per_img)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)
    print(f"  {len(dataset)} samples (images)")

    # Setup optimizer
    optimizer = optim.AdamW(get_lora_params(student), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = GradScaler('cuda' if cfg.device == 'cuda' else 'cpu')

    # Baseline evaluation
    print("\n[Baseline] Evaluating student before fine-tuning...")
    miou_before = evaluate(student, cfg, num_samples=100)
    print(f"  Student mIoU BEFORE: {miou_before:.4f}")

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    best_miou = miou_before

    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{cfg.num_epochs} ---")
        loss = train_epoch(student, teacher, dataloader, optimizer, scaler, cfg, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % cfg.eval_interval == 0 or epoch == cfg.num_epochs - 1:
            miou = evaluate(student, cfg, num_samples=100)
            if miou > best_miou:
                best_miou = miou
                save_path = os.path.join(cfg.output_dir, cfg.output_name)
                torch.save(student.state_dict(), save_path)
                print(f"  >>> Saved best model to {save_path} (mIoU: {miou:.4f})")

        print(f"  LR: {current_lr:.2e}")

    # Final results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"  Student mIoU BEFORE: {miou_before:.4f}")
    print(f"  Student mIoU AFTER:  {best_miou:.4f}")
    print(f"  Improvement:         {best_miou - miou_before:+.4f}")
    save_path = os.path.join(cfg.output_dir, cfg.output_name)
    print(f"  Best model saved to: {save_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
