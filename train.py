import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.second import SECOND
from models.center_head import CenterHead
from models.voxel_generator import VoxelGenerator
from models.loss import CenterLoss, generate_heatmap_target
from datasets.kitti import KITTIDataset, kitti_collate_fn
from datasets.dali_kitti import KITTIDALIPipeline, GTDataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from config import DATA_CONFIG, TRAIN_CONFIG


# KITTI CLASSES mapping
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_MAPPING = {c: i for i, c in enumerate(CLASSES)}


def collate_fn(batch):
    """Custom collate function for variable-size point clouds"""
    points_list = [item['points'] for item in batch]
    boxes_list = [item['boxes'] for item in batch]  # boxes: [N, 8] with class_id
    frame_ids = [item['frame_id'] for item in batch]
    
    return {
        'points': points_list,
        'boxes': boxes_list,
        'frame_ids': frame_ids
    }


def build_model(cfg):
    """Build model from config"""
    model_cfg = cfg['model']
    
    # Build SECOND backbone
    second = SECOND(model_cfg)
    
    # Build detection head
    head_cfg = cfg.get('head', {})
    in_channels = model_cfg.get('num_bev_features', 640)
    num_class = model_cfg['num_class']
    feature_stride = head_cfg.get('feature_stride', 16)
    
    center_head = CenterHead(
        in_channels=in_channels,
        num_class=num_class,
        feature_stride=feature_stride
    )
    
    return nn.ModuleDict({
        'second': second,
        'center_head': center_head
    })


def prepare_batch_data(points_list, device, voxel_gen):
    """
    Prepare batch data from point clouds using GPU voxelizer
    """
    batch_voxels = []
    batch_coords = []
    batch_num_points = []
    batch_size = len(points_list)
    
    for i, points in enumerate(points_list):
        # Convert to GPU tensor
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        num_points = len(points_tensor)
        
        if num_points == 0:
            continue
        
        # Generate on GPU
        voxels, coords, num_pts = voxel_gen.generate(points_tensor)
        
        # Adjust batch_idx (pad a column of batch indices)
        batch_idx_col = torch.full((coords.shape[0], 1), i, dtype=coords.dtype, device=device)
        coords = torch.cat([batch_idx_col, coords], dim=1)
        
        batch_voxels.append(voxels)
        batch_coords.append(coords)
        batch_num_points.append(num_pts)
    
    if len(batch_voxels) == 0:
        return None
    
    return {
        'voxels': torch.cat(batch_voxels, dim=0),
        'voxel_coords': torch.cat(batch_coords, dim=0),
        'voxel_num_points': torch.cat(batch_num_points, dim=0),
        'batch_size': batch_size
    }


def prepare_targets(boxes_list, device, heatmap_size=(250, 469), num_classes=3):
    heatmap_list, regression_list, mask_list = [], [], []
    for boxes in boxes_list:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        heatmap, regression, mask = generate_heatmap_target(
            boxes_tensor, heatmap_size, num_classes
        )
        heatmap_list.append(heatmap.unsqueeze(0))
        regression_list.append(regression.unsqueeze(0))
        mask_list.append(mask.unsqueeze(0))
    
    return (
        torch.cat(heatmap_list, dim=0),
        torch.cat(regression_list, dim=0),
        torch.cat(mask_list, dim=0)
    )


from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, epoch, criterion, scaler, voxel_gen, total_batches, scheduler=None):
    model.train()
    
    total_loss_sum = 0.0
    heatmap_loss_sum = 0.0
    reg_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True)
    for batch_idx, batch in pbar:
        # 1. 此时 batch 应该包含 {'points': tensor, 'gt_boxes': list_of_tensors}
        points_list = batch['points'] 
        boxes_list = batch['gt_boxes'] 
        
        # 2. 调用 GPU 上的体素化生成器
        batch_dict = prepare_batch_data(points_list, device, voxel_gen)
        
        # 3. 使用 AMP (自动混合精度) 充分发挥 5090 性能
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model['second'](batch_dict)
            bev_features = preds.get('features')
            
            # 这里的 250, 469 必须与模型输出一致
            head_preds = model['center_head'](bev_features)
            
            # 4. 生成 Target 并计算 Loss
            # 注意：boxes_list 已经是雷达系了
            targets = prepare_targets(boxes_list, device)
            loss_dict = criterion(head_preds, *targets)
            loss = loss_dict['total_loss']

        # 5. 反向传播优化
        scaler.scale(loss).backward()
        
        # 梯度截断，防止 AMP 初期因梯度爆炸产生 NaN/Inf
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
        
        total_loss_sum += loss_dict['total_loss'].item()
        heatmap_loss_sum += loss_dict['heatmap_loss'].item()
        reg_loss_sum += loss_dict['reg_loss'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Heatmap': f"{loss_dict['heatmap_loss'].item():.4f}",
            'Reg': f"{loss_dict['reg_loss'].item():.4f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
    if num_batches == 0:
        return {'total_loss': 0.0, 'heatmap_loss': 0.0, 'reg_loss': 0.0}
        
    return {
        'total_loss': total_loss_sum / num_batches,
        'heatmap_loss': heatmap_loss_sum / num_batches,
        'reg_loss': reg_loss_sum / num_batches
    }


def main():
    # Config
    config = {
        'model': {
            'grid_size': [40, 4000, 7504],
            'num_class': 3,
            'num_bev_features': 640,
        },
        'head': {
            'feature_stride': 8,
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'checkpoint_dir': 'output/ckpt',
        }
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = build_model(config)
    model = model.to(device)
    print("Model built successfully")
    
    # Voxel Generator (Shared for all batches, on GPU)
    voxel_gen = VoxelGenerator(
        voxel_size=config['model'].get('voxel_generator', {}).get('voxel_size', [0.02, 0.02, 0.1]),
        point_cloud_range=config['model'].get('voxel_generator', {}).get('point_cloud_range', [0, -40.0, -3.0, 150.08, 40.0, 1.0]),
        max_num_points=config['model'].get('voxel_generator', {}).get('max_num_points', 5),
        max_voxels=config['model'].get('voxel_generator', {}).get('max_voxels', 80000)
    )
    
    # Loss function
    criterion = CenterLoss(loss_weights={
        'heatmap': 1.0,
        'regression': 0.1,
    })
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0.01))    
    scaler = torch.amp.GradScaler('cuda')  # AMP 训练的梯度缩放器

    # 1. 创建 DALI 原始管线
    dali_pipe = KITTIDALIPipeline(
        data_root=DATA_CONFIG['dataset_path'],
        split="training",
        batch_size=TRAIN_CONFIG['batch_size'],
        num_threads=DATA_CONFIG['num_workers'],
        device_id=0
    )
    dali_pipe.build()

    # 2. 使用 DALI 官方迭代器包装
    # size 设置为原始数据集大小，确保 epoch 正常结束
    dali_iter = DALIGenericIterator(
        [dali_pipe], 
        output_map=["points", "label_idx"], 
        size=len(dali_pipe.file_list),
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )

    print("Loading KITTI dataset...")
    print("Loading KITTI dataset...")
    # 3. 接入你的 GT Sampling 包装器 (CPU 逻辑增强)
    train_loader = GTDataLoader(
        dali_pipe=dali_pipe,
        dali_iter=dali_iter,
        db_info_path=os.path.join(DATA_CONFIG['dataset_path'], 'kitti_dbinfos_train.pkl'),
        label_root=os.path.join(DATA_CONFIG['dataset_path'], 'training/label_2'),
        sample_groups={'Car': 15, 'Pedestrian': 10, 'Cyclist': 10},
        data_root=DATA_CONFIG['dataset_path']
    )
    
    # Create checkpoint dir
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    
    # Calculate approximate total batches per epoch from DALI file list
    total_samples = len(dali_pipe.file_list)
    total_batches = (total_samples + config['training']['batch_size'] - 1) // config['training']['batch_size']
    
    # 学习率调度器 (OneCycleLR) - 提升模型收敛精度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['training']['learning_rate'], 
        steps_per_epoch=total_batches, 
        epochs=num_epochs,
        pct_start=0.4
    )

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        loss_dict = train_epoch(model, train_loader, optimizer, device, epoch, criterion, scaler, voxel_gen, total_batches, scheduler)
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
        print(f"  Heatmap Loss: {loss_dict['heatmap_loss']:.4f}")
        print(f"  Regression Loss: {loss_dict['reg_loss']:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_dict': loss_dict,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['training']['checkpoint_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
