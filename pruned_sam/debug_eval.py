"""
诊断评估：检查模型预测是否生成了非空掩码
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import json

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam

def rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rle = np.array(rle)
    starts = rle[::2]
    lengths = rle[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')

device = torch.device('cpu')

print("=" * 60)
print("诊断评估：检查模型预测输出")
print("=" * 60)

print("\n1. 加载模型...")
model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
model.to(device).eval()

print("\n2. 加载标注...")
annotation_path = '/home/zhang/vista-slam/eval_data/annotations/instances_val2017.json'
image_dir = '/home/zhang/vista-slam/eval_data/test_100'
with open(annotation_path, 'r') as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}

# 筛选test_100目录下的图片
test_files = set(os.listdir(image_dir))

samples = []
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in images:
        continue
    img_info = images[img_id]
    if img_info['file_name'] not in test_files:
        continue
    samples.append((img_info, ann))

print(f"  找到 {len(samples)} 个标注-图片对")

print("\n3. 逐个测试...")
total_masks = 0
non_empty_masks = 0
total_pixels = 0
foreground_pixels = 0

for i, (img_info, ann) in enumerate(samples[:10]):
    img_path = os.path.join(image_dir, img_info['file_name'])
    if not os.path.exists(img_path):
        continue
    
    img = Image.open(img_path).convert('RGB')
    original_w, original_h = img.size
    
    img_resized = img.resize((1024, 1024), Image.LANCZOS)
    img_np = np.array(img_resized).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).to(device)
    
    bbox = ann['bbox']
    x1, y1, w, h = bbox
    x1_norm = x1 / original_w
    y1_norm = y1 / original_h
    w_norm = w / original_w
    h_norm = h / original_h
    box_coords = torch.tensor([[x1_norm, y1_norm, x1_norm + w_norm, y1_norm + h_norm]], device=device)
    
    with torch.no_grad():
        image_embedding = model.image_encoder(img_tensor)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None, boxes=box_coords, masks=None
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
    
    mask_logits = low_res_masks[0, 0].detach().cpu().numpy()
    mask_probs = 1.0 / (1.0 + np.exp(-mask_logits))
    mask_binary = (mask_probs > 0.5).astype(np.uint8)
    
    mask = Image.fromarray(mask_binary * 255).resize((original_w, original_h), Image.NEAREST)
    pred_mask = np.array(mask) > 127
    
    seg = ann['segmentation']
    if isinstance(seg, dict):
        gt_mask = rle_to_mask(seg['counts'], seg['size'][0], seg['size'][1])
    elif isinstance(seg, list):
        from PIL import ImageDraw
        gt_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        if seg and isinstance(seg[0], list):
            for poly in seg:
                poly_np = np.array(poly).reshape(-1, 2)
                if len(poly_np) >= 3:
                    img_draw = Image.new('L', (original_w, original_h), 0)
                    draw = ImageDraw.Draw(img_draw)
                    draw.polygon(list(poly_np.flatten()), fill=1)
                    gt_mask = np.logical_or(gt_mask, np.array(img_draw)).astype(np.uint8)
        else:
            poly_np = np.array(seg).reshape(-1, 2)
            if len(poly_np) >= 3:
                img_draw = Image.new('L', (original_w, original_h), 0)
                draw = ImageDraw.Draw(img_draw)
                draw.polygon(list(poly_np.flatten()), fill=1)
                gt_mask = np.array(img_draw)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    iou_foreground = intersection / union if union > 0 else 0
    
    pred_fg_pixels = np.sum(pred_mask)
    gt_fg_pixels = np.sum(gt_mask)
    
    # Also check raw logits stats
    logit_min, logit_max = mask_logits.min(), mask_logits.max()
    logit_mean = mask_logits.mean()
    
    total_masks += 1
    if pred_fg_pixels > 0:
        non_empty_masks += 1
    
    print(f"\n  [{i+1}] {img_info['file_name']}")
    print(f"    GT框: [{x1:.0f}, {y1:.0f}, {x1+w:.0f}, {y1+h:.0f}]")
    print(f"    Raw logits: min={logit_min:.3f}, max={logit_max:.3f}, mean={logit_mean:.3f}")
    print(f"    Pred前景像素: {pred_fg_pixels} / {original_h * original_w} ({pred_fg_pixels/(original_h*original_w)*100:.1f}%)")
    print(f"    GT前景像素: {gt_fg_pixels} / {original_h * original_w} ({gt_fg_pixels/(original_h*original_w)*100:.1f}%)")
    print(f"    Foreground IoU: {iou_foreground:.4f}")
    
    total_pixels += original_h * original_w
    foreground_pixels += pred_fg_pixels

print(f"\n{'='*60}")
print(f"统计:")
print(f"  总掩码数: {total_masks}")
print(f"  非空掩码数: {non_empty_masks} ({non_empty_masks/total_masks*100:.1f}%)")
print(f"  平均前景像素比例: {foreground_pixels/total_pixels*100:.1f}%")
print(f"{'='*60}")
