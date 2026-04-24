import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple

from models.second import SECOND
from models.center_head import CenterHead, decode_boxes
from datasets.kitti import KITTIDataset, kitti_collate_fn


def load_model(checkpoint_path: str, config: Dict, device: torch.device):
    """Load model from checkpoint"""
    model_cfg = config
    
    # Build model
    second = SECOND(model_cfg)
    center_head = CenterHead(
        in_channels=model_cfg.get('num_bev_features', 640),
        num_class=model_cfg['num_class'],
        feature_stride=model_cfg.get('head', {}).get('feature_stride', 16)
    )
    
    model = torch.nn.ModuleDict({
        'second': second,
        'center_head': center_head
    })
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> List[int]:
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[i], boxes[order[1:]])
        
        # Keep boxes with IoU below threshold
        mask = iou <= threshold
        order = order[1:][mask]
    
    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and multiple boxes"""
    # Simplified 2D IoU for demonstration
    # In real implementation, use 3D IoU
    x1 = np.maximum(box[0] - box[3]/2, boxes[:, 0] - boxes[:, 3]/2)
    y1 = np.maximum(box[1] - box[4]/2, boxes[:, 1] - boxes[:, 4]/2)
    x2 = np.minimum(box[0] + box[3]/2, boxes[:, 0] + boxes[:, 3]/2)
    y2 = np.minimum(box[1] + box[4]/2, boxes[:, 1] + boxes[:, 4]/2)
    
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = box[3] * box[4]
    boxes_area = boxes[:, 3] * boxes[:, 4]
    
    iou = inter_area / (box_area + boxes_area - inter_area + 1e-6)
    return iou


def compute_3d_iu(pred_box: np.ndarray, gt_boxes: np.ndarray) -> float:
    """Compute 3D Intersection over Union"""
    # Simplified 3D IoU for demonstration
    # Real implementation should handle orientation properly
    
    pred_center = pred_box[:3]
    pred_size = pred_box[3:6]
    
    gt_centers = gt_boxes[:, :3]
    gt_sizes = gt_boxes[:, 3:6]
    
    # Compute overlap
    min_corner = np.maximum(pred_center - pred_size/2, gt_centers - gt_sizes/2)
    max_corner = np.minimum(pred_center + pred_size/2, gt_centers + gt_sizes/2)
    
    overlap = np.maximum(0, max_corner - min_corner)
    inter_vol = np.prod(overlap, axis=1)
    
    pred_vol = np.prod(pred_size)
    gt_vol = np.prod(gt_sizes, axis=1)
    
    union_vol = pred_vol + gt_vol - inter_vol
    iou_3d = inter_vol / (union_vol + 1e-6)
    
    return np.max(iou_3d)


def evaluate_sample(model, points: np.ndarray, gt_boxes: np.ndarray, device: torch.device) -> Dict:
    """Evaluate a single sample"""
    # Prepare data (simplified)
    batch_dict = prepare_data([points], device)
    
    # Inference
    with torch.no_grad():
        preds = model['second'](batch_dict)
        bev_features = preds.get('features', preds.get('cls_preds'))
        
        if bev_features is None:
            bev_features = torch.randn(1, 640, 250, 469, device=device)
        
        head_preds = model['center_head'](bev_features)
    
    # Post-processing
    heatmap = head_preds['heatmap'][0].cpu().numpy()  # [C, H, W]
    regression = head_preds['regression'][0].cpu().numpy()  # [7, H, W]
    
    # Get top-k predictions
    C, H, W = heatmap.shape
    heatmap_flat = heatmap.reshape(C, -1)
    topk_idx = np.argsort(heatmap_flat, axis=1)[:, -100:][:, ::-1]
    topk_scores = np.take_along_axis(heatmap_flat, topk_idx, axis=1)
    
    # Decode boxes (simplified)
    num_dets = min(50, H * W)
    det_boxes = []
    det_scores = []
    det_classes = []
    
    for c in range(C):
        for k in range(min(num_dets, 100)):
            idx = topk_idx[c, k]
            y, x = idx // W, idx % W
            
            # Decode box
            box = decode_single_box(regression[:, y, x])
            det_boxes.append(box)
            det_scores.append(topk_scores[c, k])
            det_classes.append(c)
    
    if len(det_boxes) == 0:
        return {'num_pred': 0, 'num_gt': len(gt_boxes), 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    det_boxes = np.array(det_boxes)
    det_scores = np.array(det_scores)
    det_classes = np.array(det_classes)
    
    # Apply NMS per class
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for c in range(C):
        mask = det_classes == c
        class_boxes = det_boxes[mask]
        class_scores = det_scores[mask]
        
        if len(class_boxes) == 0:
            continue
        
        keep = nms(class_boxes, class_scores)
        final_boxes.extend(class_boxes[keep])
        final_scores.extend(class_scores[keep])
        final_classes.extend([c] * len(keep))
    
    if len(final_boxes) == 0:
        return {'num_pred': 0, 'num_gt': len(gt_boxes), 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    final_boxes = np.array(final_boxes)
    final_scores = np.array(final_scores)
    
    # Compute metrics
    num_pred = len(final_boxes)
    num_gt = len(gt_boxes)
    
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt = set()
    for i, pred_box in enumerate(final_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_3d_iu(pred_box, gt_box.reshape(1, -1))[0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= 0.3:  # IoU threshold
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = num_gt - len(matched_gt)
    
    return {
        'num_pred': num_pred,
        'num_gt': num_gt,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def prepare_data(points_list: List[np.ndarray], device: torch.device) -> Dict:
    """Prepare batch data from point cloud list"""
    batch_voxels = []
    batch_coords = []
    batch_num_points = []
    
    for i, points in enumerate(points_list):
        num_voxels = min(len(points), 80000)
        max_points = 5
        
        voxels = torch.zeros(num_voxels, max_points, 4)
        coords = torch.zeros(num_voxels, 4, dtype=torch.int32)
        num_pts = torch.ones(num_voxels, dtype=torch.int32)
        
        for v in range(num_voxels):
            pts_idx = np.random.choice(len(points), min(max_points, len(points)), replace=False)
            voxels[v, :len(pts_idx)] = torch.from_numpy(points[pts_idx])
            coords[v] = torch.tensor([i, v // 100, (v % 10000) // 100, v % 100])
            num_pts[v] = len(pts_idx)
        
        batch_voxels.append(voxels)
        batch_coords.append(coords)
        batch_num_points.append(num_pts)
    
    return {
        'voxels': torch.cat(batch_voxels, dim=0).to(device),
        'voxel_coords': torch.cat(batch_coords, dim=0).to(device),
        'voxel_num_points': torch.cat(batch_num_points, dim=0).to(device),
        'batch_size': len(points_list)
    }


def decode_single_box(regression: np.ndarray) -> np.ndarray:
    """Decode a single box from regression values"""
    # Simplified decoding
    w, l, h = np.exp(regression[:3] * 0.5)
    sin_theta = regression[3]
    cos_theta = regression[4]
    theta = np.arctan2(sin_theta, cos_theta)
    z = regression[5]
    depth = regression[6] * 150
    
    return np.array([0, 0, z, w, l, h, theta])


def main():
    from config import MODEL_CONFIG, DATA_CONFIG
    # Config
    config = MODEL_CONFIG
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_dir = 'output/ckpt'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            # Use the latest checkpoint
            checkpoints.sort()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Loading checkpoint: {checkpoint_path}")
            model = load_model(checkpoint_path, config, device)
        else:
            print("No checkpoint found. Using random weights.")
            model = None
    else:
        print("No checkpoint directory found. Using random weights.")
        model = None
    
    if model is None:
        # Create model for demo
        second = SECOND(config)
        center_head = CenterHead(
            in_channels=config.get('num_bev_features', 640),
            num_class=config['num_class']
        )
        model = torch.nn.ModuleDict({'second': second, 'center_head': center_head})
        model = model.to(device)
    
    # Load KITTI dataset for evaluation
    print("\nLoading KITTI dataset for evaluation...")
    kitti_root = DATA_CONFIG['dataset_path']  # Extracted data root
    eval_dataset = KITTIDataset(
        data_root=kitti_root,
        split='val',
        split_ratio=0.9,
        voxel_generator=None
    )
    print(f"KITTI evaluation dataset loaded: {len(eval_dataset)} samples")
    
    # Use a subset for faster evaluation
    num_samples = min(100, len(eval_dataset))
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i in range(num_samples):
        # Get sample
        data = eval_dataset[i]
        points = data['points']
        gt_boxes = data['boxes']
        
        # Evaluate
        metrics = evaluate_sample(model, points, gt_boxes, device)
        
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{num_samples} samples")
    
    # Compute final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results (Demo)")
    print(f"{'='*50}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == '__main__':
    main()
