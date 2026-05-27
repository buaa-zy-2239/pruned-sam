import torch
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import sys
import os

sys.path.insert(0, '/home/zhang/vista-slam')
sys.path.insert(0, '/home/zhang/vista-slam/TinySAM')

from tinysam.utils.transforms import ResizeLongestSide
from tinysam.build_sam import sam_model_registry
from tinysam.modeling import Sam


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def prepare_image_and_boxes(image_np, boxes_orig, model, device):
    transform = ResizeLongestSide(model.image_encoder.img_size)

    input_image = transform.apply_image(image_np)
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()

    input_size = tuple(input_image_torch.shape[-2:])
    original_size = image_np.shape[:2]

    preprocessed = model.preprocess(input_image_torch)
    image_embedding = model.image_encoder(preprocessed.unsqueeze(0))

    boxes_np = np.array(boxes_orig, dtype=float)
    if boxes_np.ndim == 1:
        boxes_np = boxes_np.reshape(1, 4)
    boxes_transformed = transform.apply_boxes(boxes_np, original_size)
    boxes_tensor = torch.as_tensor(boxes_transformed, dtype=torch.float, device=device)

    return image_embedding, boxes_tensor, input_size, original_size


def evaluate_with_boxes(model, coco_gt, img_dir, max_samples=None, max_boxes_per_img=10, device='cpu'):
    model.eval()
    model.to(device)

    img_ids = sorted(coco_gt.imgs.keys())
    if max_samples is not None:
        img_ids = img_ids[:max_samples]

    per_instance_ious = []
    per_image_mean_ious = []

    for img_id in tqdm(img_ids, desc="Box Prompt Evaluation"):
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

        boxes_orig = []
        gt_masks = []
        for ann in anns[:max_boxes_per_img]:
            x, y, bw, bh = ann['bbox']
            boxes_orig.append([x, y, x + bw, y + bh])
            gt_masks.append(coco_gt.annToMask(ann))

        image_embedding, boxes_tensor, input_size, original_size = prepare_image_and_boxes(
            image_np, boxes_orig, model, device
        )

        instance_ious = []
        for i in range(boxes_tensor.shape[0]):
            box_tensor = boxes_tensor[i:i+1]

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None,
                )
                low_res_masks, iou_pred = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                )

            best_idx = iou_pred[0].argmax().item()
            best_mask = low_res_masks[:, best_idx:best_idx+1, :, :]
            masks_post = model.postprocess_masks(best_mask, input_size, original_size)
            pred_binary = (masks_post > model.mask_threshold).squeeze().cpu().numpy().astype(bool)

            iou = compute_iou(pred_binary, gt_masks[i])
            instance_ious.append(iou)

        if instance_ious:
            per_instance_ious.extend(instance_ious)
            per_image_mean_ious.append(np.mean(instance_ious))

    results = {}
    if per_instance_ious:
        results['mIoU'] = np.mean(per_instance_ious)
        results['mIoU_per_image'] = np.mean(per_image_mean_ious)
        results['num_instances'] = len(per_instance_ious)
        results['num_images'] = len(per_image_mean_ious)
        results['std_iou'] = np.std(per_instance_ious)
        results['median_iou'] = np.median(per_instance_ious)
        results['iou_above_05'] = np.mean([i > 0.5 for i in per_instance_ious])
        results['iou_above_075'] = np.mean([i > 0.75 for i in per_instance_ious])
        results['iou_above_09'] = np.mean([i > 0.9 for i in per_instance_ious])
    else:
        results['mIoU'] = 0.0

    return results


def main():
    print("=" * 70)
    print("Box Prompt mIoU 评估 — COCO GT Bounding Boxes")
    print("(使用 ResizeLongestSide + padding，与 SAM 官方评估一致)")
    print("=" * 70)

    device = torch.device('cpu')

    ANN_FILE = '/home/zhang/vista-slam/eval_data/partial_annotations/instances_val2017.json'
    IMG_DIR = '/home/zhang/vista-slam/eval_data/test_100'

    print(f"\n标注文件: {ANN_FILE}")
    print(f"图片目录: {IMG_DIR}")
    coco_gt = COCO(ANN_FILE)

    MODELS = {
        'TinySAM': '/home/zhang/vista-slam/TinySAM/weights/tinysam_42.3.pth',
        'pruned_m': '/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth',
        'pruned_l': '/home/zhang/vista-slam/pruned_sam/weights/pruned_l.pth',
    }

    NUM_SAMPLES = 100
    MAX_BOXES_PER_IMG = 10

    print(f"\n评估配置:")
    print(f"  最大图片数: {NUM_SAMPLES}")
    print(f"  每图最大实例: {MAX_BOXES_PER_IMG}")

    print("\n" + "=" * 70)
    header = f"{'模型':<15} | {'mIoU':<10} | {'mIoU/图':<10} | {'实例数':<8} | {'中位数IoU':<10} | {'@0.5':<8} | {'@0.75':<8}"
    print(header)
    print("=" * 70)

    for model_name, ckpt in MODELS.items():
        print(f"\n加载 {model_name}...")
        if model_name == 'TinySAM':
            model = sam_model_registry['vit_t'](checkpoint=ckpt)
        else:
            from pruned_sam import build_pruned_sam
            model = build_pruned_sam(model_name, checkpoint=ckpt)

        results = evaluate_with_boxes(
            model, coco_gt, IMG_DIR,
            max_samples=NUM_SAMPLES,
            max_boxes_per_img=MAX_BOXES_PER_IMG,
            device=device
        )

        print(f"{model_name:<15} | {results['mIoU']:>8.4f} | {results['mIoU_per_image']:>8.4f} | "
              f"{results['num_instances']:<8} | {results['median_iou']:>8.4f} | "
              f"{results['iou_above_05']:>6.2%} | {results['iou_above_075']:>6.2%}")

    print("\n" + "=" * 70)
    print("评估完成!")


if __name__ == '__main__':
    main()
