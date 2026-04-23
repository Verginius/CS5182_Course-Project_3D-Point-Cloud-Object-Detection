import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss(pred, gt):
    """
    Focal loss for heatmap regression
    pred: [B, C, H, W] predicted heatmap
    gt: [B, C, H, W] ground truth heatmap (0 or 1)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    
    return loss


def _reg_loss(pred, gt, mask):
    """
    L1 regression loss for box parameters
    pred: [B, 7, H, W] predicted regression
    gt: [B, 7, H, W] ground truth regression
    mask: [B, H, W] binary mask for positive locations
    """
    mask = mask.unsqueeze(1).float()  # [B, 1, H, W]
    
    # Only compute loss at positive locations
    loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
    num_pos = mask.sum()
    
    if num_pos == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return loss / num_pos


class CenterLoss(nn.Module):
    """
    Combined loss for CenterHead
    - Heatmap loss (focal loss)
    - Regression loss (L1 loss)
    """
    def __init__(self, loss_weights=None):
        super().__init__()
        self.loss_weights = loss_weights or {
            'heatmap': 1.0,
            'regression': 0.1,
        }
    
    def forward(self, predictions, heatmap_gt, regression_gt, mask):
        """
        Args:
            predictions: dict with 'heatmap', 'regression'
            heatmap_gt: [B, C, H, W] ground truth heatmap
            regression_gt: [B, 7, H, W] ground truth regression values
            mask: [B, H, W] binary mask for positive locations
        """
        heatmap_loss = _neg_loss(predictions['heatmap'], heatmap_gt)
        reg_loss = _reg_loss(predictions['regression'], regression_gt, mask)
        
        total_loss = (
            self.loss_weights['heatmap'] * heatmap_loss +
            self.loss_weights['regression'] * reg_loss
        )
        
        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'reg_loss': reg_loss,
        }


def generate_heatmap_target(boxes, heatmap_size, feature_stride=8, num_classes=3):
    """
    Generate ground truth heatmap from 3D boxes
    
    Args:
        boxes: [N, 8] - x, y, z, w, l, h, rotation_y, class_id
        heatmap_size: (H, W) output heatmap size
        feature_stride: downsampling factor
        num_classes: number of object classes
    
    Returns:
        heatmap: [num_classes, H, W] center heatmap
        regression: [7, H, W] regression targets at center locations
        mask: [H, W] binary mask for positive locations
    """
    device = boxes.device if isinstance(boxes, torch.Tensor) else torch.device('cpu')
    batch_size = 1 if boxes.dim() == 2 else boxes.shape[0]
    
    H, W = heatmap_size
    heatmap = torch.zeros(num_classes, H, W, device=device)
    regression = torch.zeros(7, H, W, device=device)
    mask = torch.zeros(H, W, device=device).bool()
    
    if boxes.numel() == 0:
        return heatmap, regression, mask
    
    # Denormalize boxes to feature map coordinates
    # x, y are in world coordinates, need to project to BEV feature map
    # Assuming feature map covers [-75, 75]m range
    x = boxes[:, 0]  # world x
    y = boxes[:, 1]  # world y
    z = boxes[:, 2]  # world z
    w = boxes[:, 3]  # box width
    l = boxes[:, 4]  # box length
    h = boxes[:, 5]  # box height
    rot = boxes[:, 6]  # rotation_y
    cls = boxes[:, 7].long()  # class_id
    
    # Convert world coords to feature map coords
    # Feature map: [-75, 75]m -> [0, W/H]
    scale = W / 150.0  # Assuming 150m range (-75 to 75)
    
    center_x = (x + 75.0) * scale
    center_y = (y + 75.0) * scale
    
    # Downsample to feature map
    center_x = center_x / feature_stride
    center_y = center_y / feature_stride
    
    for i in range(len(boxes)):
        cx = int(center_x[i].item())
        cy = int(center_y[i].item())
        c = cls[i].item()
        
        if 0 <= cx < W and 0 <= cy < H and c < num_classes:
            # Draw Gaussian heatmap
            radius = max(1, int((l[i].item() + w[i].item()) / 4 / feature_stride))
            draw_gaussian(heatmap[c], (cx, cy), radius)
            
            # Set regression targets
            regression[0, cy, cx] = torch.log(w[i] + 1e-6)
            regression[1, cy, cx] = torch.log(l[i] + 1e-6)
            regression[2, cy, cx] = torch.log(h[i] + 1e-6)
            regression[3, cy, cx] = torch.sin(rot[i])
            regression[4, cy, cx] = torch.cos(rot[i])
            regression[5, cy, cx] = z[i]  # z offset
            regression[6, cy, cx] = torch.log(torch.sqrt(x[i]**2 + y[i]**2) + 1e-6)  # depth
            
            mask[cy, cx] = True
    
    return heatmap, regression, mask


def draw_gaussian(heatmap, center, radius):
    """
    Draw a 2D Gaussian on heatmap
    """
    diameter = 2 * radius + 1
    x = torch.arange(diameter, dtype=torch.float32, device=heatmap.device) - radius
    y = torch.arange(diameter, dtype=torch.float32, device=heatmap.device) - radius
    y, x = torch.meshgrid(y, x)
    
    gaussian = torch.exp(-(x**2 + y**2) / (2 * radius**2))
    
    x, y = center
    
    if radius > 0:
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(heatmap.shape[1], x + radius + 1), min(heatmap.shape[0], y + radius + 1)
        
        if x2 > x1 and y2 > y1:
            gaussian_x1 = max(0, radius - x)
            gaussian_y1 = max(0, radius - y)
            gaussian_x2 = radius + 1 - max(0, x + radius + 1 - heatmap.shape[1])
            gaussian_y2 = radius + 1 - max(0, y + radius + 1 - heatmap.shape[0])
            
            heatmap[y1:y2, x1:x2] = torch.maximum(
                heatmap[y1:y2, x1:x2],
                gaussian[gaussian_y1:gaussian_y2, gaussian_x1:gaussian_x2]
            )
    else:
        if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
            heatmap[y, x] = torch.maximum(heatmap[y, x], torch.tensor(1.0, device=heatmap.device))
