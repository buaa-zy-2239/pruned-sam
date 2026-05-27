import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/zhang/vista-slam')
from pruned_sam import build_pruned_sam


class COCOForegroundDataset(Dataset):
    def __init__(self, image_dir, annotation_path, feature_size=64, window_size=32, max_samples=100):
        self.image_dir = image_dir
        self.feature_size = feature_size
        self.window_size = window_size
        self.samples = []
        
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
            if len(self.samples) >= max_samples:
                break
            img_info = images.get(img_id)
            if img_info is None:
                continue
            
            img_path = os.path.join(image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.samples.append({
                    'image_path': img_path,
                    'height': img_info['height'],
                    'width': img_info['width'],
                    'annotations': anns
                })
    
    def __len__(self):
        return len(self.samples)
    
    def _rle_to_mask(self, rle, height, width):
        mask = np.zeros(height * width, dtype=np.uint8)
        rle = np.array(rle)
        starts = rle[::2]
        lengths = rle[1::2]
        
        for start, length in zip(starts, lengths):
            start -= 1
            mask[start:start + length] = 1
        
        return mask.reshape((height, width), order='F')
    
    def _polygon_to_mask(self, polygon, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(polygon, list) and len(polygon) > 0:
            from PIL import ImageDraw
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
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample['image_path']).convert('RGB')
        img_resized = img.resize((1024, 1024), Image.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        gt_mask = np.zeros((1024, 1024), dtype=np.uint8)
        for ann in sample['annotations']:
            seg = ann['segmentation']
            original_height, original_width = sample['height'], sample['width']
            
            if isinstance(seg, dict):
                mask = self._rle_to_mask(seg['counts'], seg['size'][0], seg['size'][1])
                mask = Image.fromarray(mask * 255).resize((original_width, original_height), Image.NEAREST)
                mask = np.array(mask) > 127
            elif isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    mask = self._polygon_to_mask(seg, original_height, original_width)
                else:
                    mask = self._rle_to_mask(seg, original_height, original_width)
            else:
                continue
            
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_img = mask_img.resize((1024, 1024), Image.NEAREST)
            mask_np = np.array(mask_img) > 127
            gt_mask = np.logical_or(gt_mask, mask_np).astype(np.uint8)
        
        mask_img = Image.fromarray(gt_mask * 255)
        mask_downsampled = mask_img.resize((self.feature_size, self.feature_size), Image.NEAREST)
        gt_mask_feature = np.array(mask_downsampled) > 127
        
        window_step = self.window_size
        num_windows_h = self.feature_size // window_step
        num_windows_w = self.feature_size // window_step
        window_labels = np.zeros(num_windows_h * num_windows_w, dtype=np.float32)
        
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                h1, h2 = i * window_step, (i + 1) * window_step
                w1, w2 = j * window_step, (j + 1) * window_step
                window_mask = gt_mask_feature[h1:h2, w1:w2]
                
                if np.any(window_mask):
                    window_labels[i * num_windows_w + j] = 1.0
        
        return img_tensor, torch.from_numpy(window_labels)


class GatingNetwork(nn.Module):
    def __init__(self, in_channels=256, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        prob_map = self.conv_layers(x)
        prob_windows = F.avg_pool2d(prob_map,
                                    kernel_size=self.window_size,
                                    stride=self.window_size)
        B = prob_windows.size(0)
        return prob_windows.view(B, -1)


class SWRWrapper(nn.Module):
    def __init__(self, base_model, gating_network):
        super().__init__()
        self.base_model = base_model
        self.gating_network = gating_network
    
    def forward(self, x):
        with torch.no_grad():
            image_embedding = self.base_model.image_encoder(x)
        
        window_probs = self.gating_network(image_embedding)
        return image_embedding, window_probs


def train_gating_network():
    print("=" * 70)
    print("SWR 门控网络训练流程")
    print("=" * 70)
    
    device = torch.device('cpu')
    window_size = 32
    num_epochs = 50
    batch_size = 2
    lr = 1e-4
    
    print("\n1. 加载数据集...")
    image_dir = '/home/zhang/vista-slam/eval_data/test_100'
    annotation_path = '/home/zhang/vista-slam/eval_data/annotations/instances_val2017.json'
    
    dataset = COCOForegroundDataset(image_dir, annotation_path, window_size=window_size, max_samples=80)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   ✅ 训练集: {len(train_dataset)} 样本")
    print(f"   ✅ 验证集: {len(val_dataset)} 样本")
    
    print("\n2. 加载基础模型...")
    base_model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    base_model.eval()
    base_model.to(device)
    print("   ✅ 模型加载完成")
    
    print("\n3. 创建门控网络...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 1024, 1024).to(device)
        embedding = base_model.image_encoder(test_input)
        in_channels = embedding.shape[1]
    
    gating_net = GatingNetwork(in_channels=in_channels, window_size=window_size)
    gating_net.to(device)
    print(f"   ✅ 门控网络创建完成 (输入通道: {in_channels})")
    
    print("\n4. 设置训练参数...")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("\n5. 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        gating_net.train()
        train_loss = 0.0
        train_acc = 0.0
        total_train = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                image_embedding = base_model.image_encoder(images)
            
            outputs = gating_net(image_embedding)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            train_acc += (preds == labels).float().sum().item()
            total_train += labels.numel()
        
        train_loss /= len(train_loader.dataset)
        train_acc /= total_train
        
        gating_net.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                image_embedding = base_model.image_encoder(images)
                outputs = gating_net(image_embedding)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = (outputs > 0.5).float()
                val_acc += (preds == labels).float().sum().item()
                total_val += labels.numel()
        
        val_loss /= len(val_loader.dataset)
        val_acc /= total_val
        scheduler.step()
        
        print(f"   Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gating_net.state_dict(), '/home/zhang/vista-slam/pruned_sam/weights/swr_gating.pth')
            print("   ✅ 保存最佳门控网络权重")
    
    print("\n6. 训练完成!")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    print(f"   门控网络权重已保存至: /home/zhang/vista-slam/pruned_sam/weights/swr_gating.pth")
    
    return gating_net


def test_gating_network():
    print("\n" + "=" * 70)
    print("测试训练好的门控网络")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    print("\n1. 加载基础模型...")
    base_model = build_pruned_sam('pruned_m', checkpoint='/home/zhang/vista-slam/pruned_sam/weights/pruned_m.pth')
    base_model.eval()
    base_model.to(device)
    
    print("\n2. 加载门控网络...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 1024, 1024).to(device)
        embedding = base_model.image_encoder(test_input)
        in_channels = embedding.shape[1]
    
    gating_net = GatingNetwork(in_channels=in_channels, window_size=32)
    gating_net.load_state_dict(torch.load('/home/zhang/vista-slam/pruned_sam/weights/swr_gating.pth', map_location=device))
    gating_net.eval()
    gating_net.to(device)
    
    print("\n3. 测试推理...")
    test_input = torch.randn(1, 3, 1024, 1024).to(device)
    
    with torch.no_grad():
        image_embedding = base_model.image_encoder(test_input)
        window_probs = gating_net(image_embedding)
    
    print(f"   输入形状: {test_input.shape}")
    print(f"   嵌入特征形状: {image_embedding.shape}")
    print(f"   窗口概率形状: {window_probs.shape}")
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\n4. 不同阈值下的路由行为:")
    for threshold in thresholds:
        routed_ratio = (window_probs > threshold).float().mean().item()
        print(f"   阈值 {threshold}: 走完整路径的窗口比例 = {routed_ratio*100:.1f}%")
    
    print("\n✅ 门控网络测试完成!")


if __name__ == '__main__':
    gating_net = train_gating_network()
    test_gating_network()