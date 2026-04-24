import torch
import numpy as np
from typing import List, Tuple, Dict


class VoxelGenerator:
    """
    Dynamic high-resolution voxel generator using spconv.utils.VoxelGeneratorV2
    Enables ultra-fine voxelization (0.02m) for extended perception range (150m)
    """
    def __init__(
        self,
        voxel_size: List[float] = [0.02, 0.02, 0.1],
        point_cloud_range: List[float] = [0, -40.0, -3.0, 150.08, 40.0, 1.0],   # [x_min, y_min, z_min, x_max, y_max, z_max] % 16 == 0
        max_num_points: int = 32,
        max_voxels: int = 300000
    ):
        """
        Args:
            voxel_size: Size of each voxel [x, y, z]
            point_cloud_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points: Maximum points per voxel
            max_voxels: Maximum number of voxels
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        
        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]
        
    def generate(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate voxels from point cloud
        
        Args:
            points: [N, 4] - x, y, z, intensity
            
        Returns:
            voxels: [M, max_points, 4]
            coordinates: [M, 4] - (batch_idx, z, y, x)
            num_points: [M]
        """
        # If it's a PyTorch tensor, use GPU generation!
        is_tensor = isinstance(points, torch.Tensor)
        device = points.device if is_tensor else torch.device('cpu')
        
        if is_tensor and device.type == 'cuda':
            try:
                from spconv.pytorch.utils import PointToVoxel
                if not hasattr(self, '_gpu_generator'):
                    self._gpu_generator = PointToVoxel(
                        vsize_xyz=self.voxel_size,
                        coors_range_xyz=self.point_cloud_range,
                        num_point_features=points.shape[-1],
                        max_num_voxels=self.max_voxels,
                        max_num_points_per_voxel=self.max_num_points,
                        device=device
                    )
                voxels, coordinates, num_points = self._gpu_generator(points)
                return voxels, coordinates, num_points
            except ImportError:
                pass # Fallback to CPU if spconv.pytorch is not available
                
        # CPU Fallback
        if is_tensor:
            points = points.cpu().numpy()

        try:
            from spconv.utils import VoxelGeneratorV2
            voxel_generator = VoxelGeneratorV2(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=points.shape[-1],
                max_num_voxels=self.max_voxels,
                max_num_points_per_voxel=self.max_num_points
            )
            voxels, coordinates, num_points = voxel_generator.generate(points)
        except Exception:
            voxels, coordinates, num_points = self._generate_fallback(points)
            
        if is_tensor:
            return torch.from_numpy(voxels).to(device), torch.from_numpy(coordinates).to(device), torch.from_numpy(num_points).to(device)
            
        return voxels, coordinates, num_points
    
    def _generate_fallback(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fallback voxel generation using pillar-based approach
        """
        # Calculate voxel indices
        x_idx = ((points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]).astype(np.int32)
        y_idx = ((points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]).astype(np.int32)
        z_idx = ((points[:, 2] - self.point_cloud_range[2]) / self.voxel_size[2]).astype(np.int32)
        
        # Filter valid points
        valid_mask = (
            (x_idx >= 0) & (x_idx < self.grid_size[0]) &
            (y_idx >= 0) & (y_idx < self.grid_size[1]) &
            (z_idx >= 0) & (z_idx < self.grid_size[2])
        )
        
        points = points[valid_mask]
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        z_idx = z_idx[valid_mask]
        
        # Create voxel keys
        voxel_keys = x_idx * self.grid_size[1] * self.grid_size[2] + y_idx * self.grid_size[2] + z_idx
        
        # Sort and unique
        sorted_indices = np.argsort(voxel_keys)
        sorted_keys = voxel_keys[sorted_indices]
        
        # Find boundaries
        diff = np.concatenate([[1], sorted_keys[1:] != sorted_keys[:-1]])
        unique_starts = np.where(diff)[0]
        unique_counts = np.diff(np.append(unique_starts, len(sorted_keys)))
        
        # Limit voxels
        num_voxels = min(len(unique_starts), self.max_voxels)
        
        voxels = np.zeros((num_voxels, self.max_num_points, points.shape[1]), dtype=np.float32)
        coordinates = np.zeros((num_voxels, 4), dtype=np.int32)
        num_points = np.zeros(num_voxels, dtype=np.int32)
        
        for i in range(num_voxels):
            start = unique_starts[i]
            count = min(unique_counts[i], self.max_num_points)
            
            voxels[i, :count] = points[sorted_indices[start:start + count]]
            num_points[i] = count
            
            idx = sorted_keys[start]
            z = idx // (self.grid_size[0] * self.grid_size[1])
            y = (idx % (self.grid_size[0] * self.grid_size[1])) // self.grid_size[2]
            x = idx % self.grid_size[2]
            coordinates[i] = [0, z, y, x]
        
        return voxels, coordinates, num_points


def point_cloud_augmentation(points: np.ndarray) -> np.ndarray:
    """
    Data augmentation for point cloud (GT-Sampling style)
    - Random rotation around z-axis
    - Random scaling
    - Random translation
    """
    # Random rotation
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    
    # Random scaling
    scale = np.random.uniform(0.95, 1.05)
    points[:, :3] *= scale
    
    # Random translation
    translation = np.random.uniform(-0.2, 0.2, size=(3,))
    points[:, :3] += translation
    
    return points
