import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterHead(nn.Module):
    """
    Anchor-free detection head based on CenterPoint paradigm.
    Predicts center heatmap, 3D dimensions, depth, and heading angle.
    """
    def __init__(self, in_channels=640, num_class=3, feature_stride=8):
        super().__init__()
        self.num_class = num_class
        self.feature_stride = feature_stride
        
        # Shared conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Heatmap head (center detection)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_class, 1),
        )
        
        # Regression head: (w, l, h, sin(θ), cos(θ), z, depth)
        self.regress_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 7, 1),  # 7 regression parameters
        )
        
    def forward(self, x):
        """
        x: [B, C, H, W] BEV features
        Returns:
            heatmap: [B, num_class, H, W] center heatmap
            regression: [B, 7, H, W] box regression params
        """
        shared_feat = self.conv(x)
        
        heatmap = torch.sigmoid(self.heatmap_head(shared_feat))
        regression = self.regress_head(shared_feat)
        
        return {
            'heatmap': heatmap,
            'regression': regression
        }


def gather_heatmap(heatmap, topk_indices):
    """
    Gather top-k heatmap values and corresponding indices
    """
    batch_size, num_class, height, width = heatmap.shape
    
    # Reshape for topk
    heatmap_flat = heatmap.view(batch_size, num_class, -1)  # [B, C, H*W]
    topk_indices_flat = topk_indices.view(batch_size, num_class, -1)  # [B, C, K]
    
    # Gather values
    topk_scores = torch.gather(heatmap_flat, 2, topk_indices_flat)  # [B, C, K]
    
    # Decode indices to (y, x)
    y_coords = topk_indices_flat // width
    x_coords = topk_indices_flat % width
    
    return topk_scores, y_coords, x_coords


def decode_boxes(regression, y_coords, x_coords):
    """
    Decode box parameters from regression map
    regression: [B, 7, H, W] - (w, l, h, sin, cos, z, depth)
    y_coords: [B, K] heatmap y indices
    x_coords: [B, K] heatmap x indices
    Returns: [B, K, 7] boxes in (x, y, z, w, l, h, heading)
    """
    batch_size = regression.shape[0]
    
    # Extract regression values at predicted centers
    regression = regression.permute(0, 2, 3, 1)  # [B, H, W, 7]
    
    batch_indices = torch.arange(batch_size, device=regression.device).view(-1, 1)
    
    reg_values = regression[batch_indices, y_coords, x_coords]  # [B, K, 7]
    
    # Decode heading angle
    sin_theta = reg_values[..., 3:4]
    cos_theta = reg_values[..., 4:5]
    heading = torch.atan2(sin_theta, cos_theta)
    
    # Decode sizes (log encoded)
    w = torch.exp(reg_values[..., 0:1])
    l = torch.exp(reg_values[..., 1:2])
    h = torch.exp(reg_values[..., 2:3])
    
    # Decode z
    z = reg_values[..., 5:6]
    
    # Decode x, y from grid coordinates
    # Feature stride is 16 (0.32m per pixel) based on current architecture
    out_size_factor = 0.32
    x = x_coords.float().unsqueeze(-1) * out_size_factor
    y = y_coords.float().unsqueeze(-1) * out_size_factor - 40.0
    
    # Build box: (x, y, z, w, l, h, heading)
    boxes = torch.cat([
        x, y, z, w, l, h, heading
    ], dim=-1)
    
    return boxes
