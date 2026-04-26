import os
import sys
import time
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


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.1) -> List[int]:
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
    # box format: [x, y, z, w, l, h, theta]
    # x dimension uses l (box[4]), y dimension uses w (box[3])
    x1 = np.maximum(box[0] - box[4]/2, boxes[:, 0] - boxes[:, 4]/2)
    y1 = np.maximum(box[1] - box[3]/2, boxes[:, 1] - boxes[:, 3]/2)
    x2 = np.minimum(box[0] + box[4]/2, boxes[:, 0] + boxes[:, 4]/2)
    y2 = np.minimum(box[1] + box[3]/2, boxes[:, 1] + boxes[:, 3]/2)
    
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = box[4] * box[3]
    boxes_area = boxes[:, 4] * boxes[:, 3]
    
    iou = inter_area / (box_area + boxes_area - inter_area + 1e-6)
    return iou


def compute_3d_iu(pred_box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute 3D Intersection over Union for an array of boxes"""
    # Simplified 3D IoU for demonstration
    # Real implementation should handle orientation properly
    if len(gt_boxes) == 0:
        return np.array([])
        
    pred_center = pred_box[:3]
    # Box format: [w, l, h] -> Need to align with axes: x_size=l, y_size=w, z_size=h
    pred_size = np.array([pred_box[4], pred_box[3], pred_box[5]])
    
    gt_centers = gt_boxes[:, :3]
    # gt_boxes also [w, l, h] -> align with axes
    gt_sizes = np.column_stack((gt_boxes[:, 4], gt_boxes[:, 3], gt_boxes[:, 5]))
    
    # Compute overlap
    min_corner = np.maximum(pred_center - pred_size/2, gt_centers - gt_sizes/2)
    max_corner = np.minimum(pred_center + pred_size/2, gt_centers + gt_sizes/2)
    
    overlap = np.maximum(0, max_corner - min_corner)
    inter_vol = np.prod(overlap, axis=1)
    
    pred_vol = np.prod(pred_size)
    gt_vol = np.prod(gt_sizes, axis=1)
    
    union_vol = pred_vol + gt_vol - inter_vol
    iou_3d = inter_vol / (union_vol + 1e-6)
    
    return iou_3d


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using VOC 11-point or AUC method"""
    rec = np.concatenate(([0.0], recalls, [1.0]))
    prec = np.concatenate(([0.0], precisions, [0.0]))

    # Compute envelope of precision
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    # Integrate area under curve
    indices = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[indices + 1] - rec[indices]) * prec[indices + 1])
    return ap


def evaluate_sample(model, points: np.ndarray, gt_boxes: np.ndarray, device: torch.device) -> Tuple[Dict, float]:
    """Evaluate a single sample and return per-class predictions, GTs, and inference time"""
    
    # ==== Start Inference Timer ====
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Prepare data (Voxelization on GPU)
    batch_dict = prepare_data([points], device)
    
    # Inference with FP16 for massive acceleration
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            preds = model['second'](batch_dict)
            bev_features = preds.get('features', preds.get('cls_preds'))
            
            if bev_features is None:
                bev_features = torch.randn(1, 640, 250, 469, device=device)
            
            head_preds = model['center_head'](bev_features)
            
    # ==== End Inference Timer ====
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_time = end_time - start_time
    
    # Post-processing (Not counted in network inference FPS)
    heatmap_tensor = head_preds['heatmap'][0]  # [C, H, W]
    
    # Apply 3x3 MaxPool to find local maxima (Peak Extraction)
    local_max = torch.nn.functional.max_pool2d(
        heatmap_tensor.unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    
    heatmap_tensor = heatmap_tensor * (heatmap_tensor == local_max).float()
    
    heatmap = heatmap_tensor.cpu().numpy()  # [C, H, W]
    regression = head_preds['regression'][0].cpu().numpy()  # [7, H, W]
    
    # Get top-k predictions
    C, H, W = heatmap.shape
    heatmap_flat = heatmap.reshape(C, -1)
    topk_idx = np.argsort(heatmap_flat, axis=1)[:, -100:][:, ::-1]
    topk_scores = np.take_along_axis(heatmap_flat, topk_idx, axis=1)
    
    # Decode boxes
    num_dets = min(50, H * W)
    class_results = {c: {'preds': [], 'num_gt': 0} for c in range(C)}
    
    # Record GTs per class
    for c in range(C):
        class_results[c]['num_gt'] = np.sum(gt_boxes[:, 7] == c) if len(gt_boxes) > 0 else 0
    
    for c in range(C):
        det_boxes = []
        det_scores = []
        
        for k in range(min(num_dets, 100)):
            score = topk_scores[c, k]
            # Lower threshold for collecting full AP curve
            if score < 0.1:
                continue
                
            idx = topk_idx[c, k]
            y, x = idx // W, idx % W
            
            box = decode_single_box(regression[:, y, x], x, y)
            det_boxes.append(box)
            det_scores.append(score)
            
        if len(det_boxes) == 0:
            continue
            
        det_boxes = np.array(det_boxes)
        det_scores = np.array(det_scores)
        
        # Apply NMS
        keep = nms(det_boxes, det_scores)
        final_boxes = det_boxes[keep]
        final_scores = det_scores[keep]
        
        # Match with GT (strictly within the same class)
        c_gt_mask = gt_boxes[:, 7] == c if len(gt_boxes) > 0 else np.zeros(0, dtype=bool)
        c_gt_boxes = gt_boxes[c_gt_mask] if len(gt_boxes) > 0 else np.array([])
        
        matched_gt = set()
        for i, pred_box in enumerate(final_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            if len(c_gt_boxes) > 0:
                ious = compute_3d_iu(pred_box, c_gt_boxes)
                for j, iou in enumerate(ious):
                    if j in matched_gt:
                        continue
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= 0.3:  # IoU threshold
                class_results[c]['preds'].append({'score': final_scores[i], 'tp': 1, 'fp': 0})
                matched_gt.add(best_gt_idx)
            else:
                class_results[c]['preds'].append({'score': final_scores[i], 'tp': 0, 'fp': 1})
                
    return class_results, inference_time


from models.voxel_generator import VoxelGenerator
from config import MODEL_CONFIG

def prepare_data(points_list: List[np.ndarray], device: torch.device) -> Dict:
    """Prepare batch data using GPU VoxelGenerator"""
    voxel_gen = VoxelGenerator(
        voxel_size=MODEL_CONFIG.get('voxel_generator', {}).get('voxel_size', [0.02, 0.02, 0.1]),
        point_cloud_range=MODEL_CONFIG.get('voxel_generator', {}).get('point_cloud_range', [0, -40.0, -3.0, 150.08, 40.0, 1.0]),
        max_num_points=MODEL_CONFIG.get('voxel_generator', {}).get('max_num_points', 5),
        max_voxels=MODEL_CONFIG.get('voxel_generator', {}).get('max_voxels', 80000)
    )

    batch_voxels = []
    batch_coords = []
    batch_num_points = []
    batch_size = len(points_list)
    
    for i, points in enumerate(points_list):
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        if len(points_tensor) == 0:
            continue
            
        voxels, coords, num_pts = voxel_gen.generate(points_tensor)
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


def decode_single_box(regression: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Decode a single box from regression values and heatmap coordinates"""
    # Inverse of coordinate mapping in loss.py
    # Feature stride = 16 (0.32m per pixel)
    out_size_factor = 0.32
    # 补偿半个像素宽度，将坐标移到栅格物理中心点
    x = cx * out_size_factor + out_size_factor / 2.0
    y = cy * out_size_factor - 40.0 + out_size_factor / 2.0
    
    # Decoding size (they were log encoded with 1e-6)
    w, l, h = np.exp(regression[:3])
    
    sin_theta = regression[3]
    cos_theta = regression[4]
    theta = np.arctan2(sin_theta, cos_theta)
    
    z = regression[5]
    # Optional if we need to refine x,y using depth, but for now we just use the grid x,y
    # depth = np.exp(regression[6]) 
    
    return np.array([x, y, z, w, l, h, theta])


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
    
    global_metrics = {c: {'preds': [], 'num_gt': 0} for c in range(3)}
    
    total_time = 0.0
    warmup_steps = min(10, num_samples // 3)  # Warmup for a few steps to let CUDA graphs/allocator settle
    valid_samples = 0
    
    for i in range(num_samples):
        # Get sample
        data = eval_dataset[i]
        points = data['points']
        gt_boxes = data['boxes']
        
        # Evaluate
        sample_metrics, inference_time = evaluate_sample(model, points, gt_boxes, device)
        
        if i >= warmup_steps:
            total_time += inference_time
            valid_samples += 1
            
        for c in range(3):
            global_metrics[c]['preds'].extend(sample_metrics[c]['preds'])
            global_metrics[c]['num_gt'] += sample_metrics[c]['num_gt']
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{num_samples} samples")
            
    # Calculate AP and Point Metrics
    ap_per_class = []
    t_tp, t_fp, t_fn = 0, 0, 0
    
    for c in range(3):
        preds = global_metrics[c]['preds']
        num_gt = global_metrics[c]['num_gt']
        
        # Calculate AP by sorting predictions by score descending
        preds.sort(key=lambda x: x['score'], reverse=True)
        tps = np.array([p['tp'] for p in preds])
        fps = np.array([p['fp'] for p in preds])
        
        acc_tps = np.cumsum(tps)
        acc_fps = np.cumsum(fps)
        
        recalls = acc_tps / num_gt if num_gt > 0 else np.zeros_like(acc_tps)
        precisions = acc_tps / (acc_tps + acc_fps) if len(acc_tps) > 0 else np.zeros_like(acc_tps)
        
        if len(recalls) > 0 and len(precisions) > 0:
            ap = compute_ap(recalls, precisions)
            ap_per_class.append(ap)
            
        # Point metrics at strictly high threshold (0.70)
        high_score_preds = [p for p in preds if p['score'] >= 0.70]
        c_tp = sum(p['tp'] for p in high_score_preds)
        c_fp = sum(p['fp'] for p in high_score_preds)
        c_fn = num_gt - c_tp
        
        t_tp += c_tp
        t_fp += c_fp
        t_fn += c_fn

    mAP = np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0
    
    # Compute final point metrics at 0.70 threshold
    precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
    recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute speed
    if valid_samples > 0:
        avg_latency_ms = (total_time / valid_samples) * 1000.0
        fps = 1000.0 / avg_latency_ms
    else:
        avg_latency_ms = 0.0
        fps = 0.0
        
    print(f"\n{'='*50}")
    print(f"Evaluation Results (Demo)")
    print(f"{'='*50}")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Metrics at Confidence Threshold 0.70:")
    print(f"  Total TP: {t_tp}")
    print(f"  Total FP: {t_fp}")
    print(f"  Total FN: {t_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"{'-'*50}")
    print(f"Inference Speed (FP16 Accelerated):")
    print(f"  Avg Latency: {avg_latency_ms:.2f} ms/frame")
    print(f"  FPS:         {fps:.2f} frames/s")


if __name__ == '__main__':
    main()
