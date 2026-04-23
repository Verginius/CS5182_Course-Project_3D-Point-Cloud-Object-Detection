import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random
import zipfile


class KITTIDataset(Dataset):
    """
    KITTI 3D object detection dataset
    Auto-extracts from /root/autodl-pub/KITTI/object if not found in data_root
    
    Structure:
        training/label_2/    - training labels
        training/velodyne/   - training point clouds
        testing/velodyne/    - testing point clouds (no labels)
    """
    # KITTI 8 classes -> mapped to training classes
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']  # 3 classes
    CLASS_MAPPING = {c: i for i, c in enumerate(CLASSES)}
    
    # Source zip files
    SOURCE_ZIP_DIR = '/root/autodl-pub/KITTI/object'
    
    def __init__(
        self,
        data_root='/root/autodl-tmp/data',
        split='train',  # 'train' or 'val' or 'test'
        split_ratio=0.9,  # train/val split ratio
        voxel_generator=None,
    ):
        self.data_root = data_root
        self.split = split
        self.voxel_generator = voxel_generator
        
        # Paths
        if split == 'test':
            self.velodyne_dir = os.path.join(data_root, 'testing', 'velodyne')
            self.label_dir = None
        else:
            self.velodyne_dir = os.path.join(data_root, 'training', 'velodyne')
            self.label_dir = os.path.join(data_root, 'training', 'label_2')
        
        # Check and extract if needed
        self._maybe_extract()
        
        # Get all sample IDs
        self.all_sample_ids = self._get_sample_ids()
        
        # Split into train/val
        random.seed(42)
        ids = list(self.all_sample_ids)
        random.shuffle(ids)
        split_idx = int(len(ids) * split_ratio)
        
        if split == 'train':
            self.sample_ids = ids[:split_idx]
        elif split == 'val':
            self.sample_ids = ids[split_idx:]
        else:  # test
            self.sample_ids = self.all_sample_ids
        
        print(f"KITTI {split} dataset: {len(self.sample_ids)} samples")
    
    def _maybe_extract(self):
        """Check if data exists, extract from source if not"""
        # Check if training velodyne exists
        if not os.path.exists(self.velodyne_dir):
            velodyne_zip = os.path.join(self.SOURCE_ZIP_DIR, 'data_object_velodyne.zip')
            if os.path.exists(velodyne_zip):
                print(f"Extracting {velodyne_zip} to {self.data_root}...")
                os.makedirs(self.data_root, exist_ok=True)
                with zipfile.ZipFile(velodyne_zip, 'r') as z:
                    z.extractall(self.data_root)
        
        # Check if training label exists
        if self.label_dir is not None and not os.path.exists(self.label_dir):
            label_zip = os.path.join(self.SOURCE_ZIP_DIR, 'data_object_label_2.zip')
            if os.path.exists(label_zip):
                print(f"Extracting {label_zip} to {self.data_root}...")
                with zipfile.ZipFile(label_zip, 'r') as z:
                    z.extractall(self.data_root)
    
    def _get_sample_ids(self) -> List[str]:
        """Get list of sample IDs"""
        if os.path.exists(self.velodyne_dir):
            files = os.listdir(self.velodyne_dir)
            sample_ids = [f.replace('.bin', '') for f in files if f.endswith('.bin')]
            sample_ids.sort()
            return sample_ids
        return []
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx) -> Dict:
        sample_id = self.sample_ids[idx]
        
        # Load point cloud
        velodyne_path = os.path.join(self.velodyne_dir, f'{sample_id}.bin')
        points = self._load_points(velodyne_path)
        
        # Load labels (only for train/val)
        boxes = np.zeros((0, 8), dtype=np.float32)  # [N, 8] with class_id
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, f'{sample_id}.txt')
            boxes = self._load_labels(label_path)
        
        return {
            'points': points,
            'boxes': boxes,
            'frame_id': sample_id
        }
    
    def _load_points(self, path: str) -> np.ndarray:
        """Load point cloud from binary file"""
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def _load_labels(self, path: str) -> np.ndarray:
        """
        Load KITTI label file
        Format: [type, truncated, occluded, alpha, bbox[4], dimensions[3], location[3], rotation_y]
        Returns: boxes [N, 8] - x, y, z, w, l, h, rotation_y, class_id
        """
        if not os.path.exists(path):
            return np.zeros((0, 8), dtype=np.float32)
        
        boxes = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_type = parts[0]
                
                # Map to training class
                if obj_type in ['Van', 'Truck', 'Tram']:
                    class_id = self.CLASS_MAPPING['Car']
                elif obj_type == 'Person_sitting':
                    class_id = self.CLASS_MAPPING['Pedestrian']
                elif obj_type in self.CLASSES:
                    class_id = self.CLASS_MAPPING[obj_type]
                else:
                    continue  # Skip Misc and unknown
                
                # Location (x, y, z)
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                
                # Dimensions (h, w, l) -> KITTI format
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                
                # Rotation
                rotation_y = float(parts[14])
                
                boxes.append([x, y, z, w, l, h, rotation_y, class_id])
        
        if len(boxes) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        
        return np.array(boxes, dtype=np.float32)


def point_cloud_augmentation(points: np.ndarray) -> np.ndarray:
    """Data augmentation for point cloud"""
    # Random rotation around z-axis
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


def kitti_collate_fn(batch):
    """Custom collate function for KITTI dataset"""
    points_list = [item['points'] for item in batch]
    boxes_list = [item['boxes'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    
    return {
        'points': points_list,
        'boxes': boxes_list,
        'frame_ids': frame_ids
    }
