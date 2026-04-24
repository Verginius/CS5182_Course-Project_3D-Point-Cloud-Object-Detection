import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import Dict, Tuple, Optional


class MeanVFE(nn.Module):
    """Voxel Feature Encoder: Extract mean features from points within each voxel"""
    def __init__(self, num_point_features=4):
        super().__init__()
        self.num_point_features = num_point_features

    def forward(self, voxel_features, voxel_num_points):
        """
        voxel_features: [N, MaxPoints, C]
        voxel_num_points: [N]
        """
        points_sum = torch.sum(voxel_features[:, :, :self.num_point_features], dim=1)
        # 防止体素内点数出现 0（通常体素生成器不会有全空体素，但作为安全兜底防 NaN/Inf）
        voxel_num_points = torch.clamp(voxel_num_points, min=1.0)
        voxel_features = points_sum / voxel_num_points.type_as(voxel_features).view(-1, 1)
        return voxel_features


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm'):
    """
    SECOND standard conv-BN-ReLU block
    """
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    else:
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    
    return spconv.SparseSequential(
        conv,
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class VoxelBackbone8x(nn.Module):
    """8x downsampling sparse backbone with enhanced channel width (32/64)"""
    def __init__(self, input_channels, grid_size):
        super().__init__()
        self.grid_size = grid_size
        
        # Stage 1: Maintain resolution
        self.conv1 = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, padding=1, indice_key='subm1'),
        )
        
        # Stage 2
        self.conv2 = spconv.SparseSequential(
            post_act_block(16, 32, 3, stride=2, padding=1, conv_type='sp'), # 32
            post_act_block(32, 32, 3, padding=1, indice_key='subm2'),
            post_act_block(32, 32, 3, padding=1, indice_key='subm2'),
        )
        
        # Stage 3
        self.conv3 = spconv.SparseSequential(
            post_act_block(32, 64, 3, stride=2, padding=1, conv_type='sp'), # 64
            post_act_block(64, 64, 3, padding=1, indice_key='subm3'),
            post_act_block(64, 64, 3, padding=1, indice_key='subm3'),
        )

        # Stage 4
        self.conv4 = spconv.SparseSequential(
            post_act_block(64, 128, 3, stride=2, padding=(1, 1, 1), conv_type='sp'), # 128
            post_act_block(128, 128, 3, padding=1, indice_key='subm4'),
            post_act_block(128, 128, 3, padding=1, indice_key='subm4'),
        )

        # Stage 5
        self.conv5 = spconv.SparseSequential(
            post_act_block(128, 128, 3, stride=(1, 2, 2), padding=1, conv_type='sp'), # 输出 Z=5, H=250, W=469
            post_act_block(128, 128, 3, padding=1, indice_key='subm5'),
        )

    def forward(self, x):
        x = self.conv1(x)
        print("After Stage 1:", x.spatial_shape)
        x = self.conv2(x)
        print("After Stage 2:", x.spatial_shape)
        x = self.conv3(x)
        print("After Stage 3:", x.spatial_shape)
        x = self.conv4(x)
        print("After Stage 4:", x.spatial_shape)
        x = self.conv5(x)

        print(f"Debug: Spatial Shape={x.spatial_shape}, "
              f"Features Shape={x.features.shape}, "
              f"Batch={x.batch_size}")                          # Debug: Check final sparse tensor shape
        return x


class Map2BEV(nn.Module):
    """
    Improved Map2BEV with 1x1 adaptive convolution for efficient height encoding
    """
    def __init__(self):
        super().__init__()

    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adaptive_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    '''

    def forward(self, x_dense):
        """
        x_dense: [B, C, D, H, W] 3D sparse tensor converted to dense
        Returns: [B, C*D, H, W] BEV features
        """
        batch_size, channels, depth, height, width = x_dense.shape
        return x_dense.view(batch_size, channels * depth, height, width)
        # bev_features = self.adaptive_conv(x_dense.view(batch_size, channels * depth, height, width))
        # return bev_features


class SECOND(nn.Module):
    """
    SECOND model with enhanced backbone and improved Map2BEV
    Optimized for RTX 5090 with high-resolution voxelization
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.grid_size = model_cfg['grid_size']  # [Z, Y, X]
        self.num_class = model_cfg['num_class']

        # 1. VFE: Extract mean voxel features
        self.vfe = MeanVFE(num_point_features=4)

        # 2. Backbone3D: Sparse convolution backbone
        # Input 4-dim features, output 128-dim after 8x downsampling
        self.backbone_3d = VoxelBackbone8x(input_channels=4, grid_size=self.grid_size)

        # 3. Map to BEV with adaptive convolution
        # After 4 downsampling stages, Z-axis from 40 becomes 5
        # So input to map2bev is 128 * 5 = 640 channels
        self.num_bev_features = 128 * (self.grid_size[0] // 8)

        # 4. RPN & Head: 2D extraction and detection head
        self.map2bev = Map2BEV()
        self.dense_head = None  # Will be set by config

    def forward(self, batch_dict):
        """
        batch_dict contains:
            voxels: [N, max_points, 4]
            voxel_coords: [N, 4] (batch_index, z, y, x)
            voxel_num_points: [N]
            batch_size: int
        """
        # Step 1: VFE
        voxel_features = self.vfe(batch_dict['voxels'], batch_dict['voxel_num_points'])

        # Step 2: 3D Sparse Convolution
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=self.grid_size,
            batch_size=batch_dict['batch_size']
        )
        x_sp = self.backbone_3d(input_sp_tensor)

        # Step 3: Projection to BEV
        x_dense = x_sp.dense()
        bev_features = self.map2bev(x_dense)

        # Step 4: 2D detection head
        if self.dense_head is not None:
            preds = self.dense_head(bev_features)
        else:
            preds = {'features': bev_features}
        
        return preds
