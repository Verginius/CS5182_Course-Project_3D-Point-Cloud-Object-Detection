import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import pickle

class KITTIDALIPipeline(Pipeline):
    def __init__(self, data_root, split, batch_size, num_threads, device_id, augmentation=True):
        super().__init__(batch_size, num_threads, device_id, seed=42)
        self.split = split
        sub_dir = "training" if split == "training" else "testing"
        data_dir = os.path.join(data_root, sub_dir, "velodyne")
        
        self.file_list = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bin")
        ])

    def define_graph(self):
        # raw_bin: 点云数据, label_idx: DALI 内部的文件索引
        raw_bin, label_idx = fn.readers.file(
            files=self.file_list,
            random_shuffle=(self.split == "training"),
            name="Reader",
            pad_last_batch=True # 保证 BS 恒定
        )
        
        points = fn.reinterpret(raw_bin, dtype=types.FLOAT)
        points = fn.reshape(points, shape=[-1, 4])

        # DALI 迭代器要求输出是 dense tensor (相同形状) 才能合并为 batch。
        # 我们使用 pad 将可变长度的点云填充到一个足够大的数值 (比如 150000 点)。
        # 将 fill_value 设为 1000.0，这样越界的点就会在后续体素化时被自动扔弃。
        points = fn.pad(points, fill_value=1000.0, axes=(0,), shape=[150000])

        return points.gpu(), label_idx.gpu()

class GTDataLoader:
    def __init__(self, dali_pipe, dali_iter, db_info_path, label_root, sample_groups, data_root):
        self.dali_pipe = dali_pipe
        self.dali_iter = dali_iter
        self.data_root = data_root
        with open(db_info_path, 'rb') as f:
            self.db_infos = pickle.load(f)
        
        self.label_root = label_root
        self.calib_root = os.path.join(data_root, 'training/calib')
        self.sample_groups = sample_groups # 例如 {'Car': 15, 'Pedestrian': 10}
        # 获取文件名列表，用于索引回找
        self.file_names = [os.path.basename(f).replace('.bin', '') for f in dali_pipe.file_list]
        self.class_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    def _get_calib_matrix(self, calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'Tr_velo_to_cam' in line:
                tr = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                tr = np.concatenate([tr, [[0, 0, 0, 1]]], axis=0)
                return tr
        return None

    def _box_cam_to_lidar(self, box_cam, tr_mat):
        h, w, l, x, y, z, yaw = box_cam
        p_cam = np.array([x, y - h/2, z, 1.0]).reshape(4, 1)
        p_lidar = np.linalg.inv(tr_mat) @ p_cam
        lidar_yaw = -yaw - np.pi / 2
        while lidar_yaw > np.pi: lidar_yaw -= 2 * np.pi
        while lidar_yaw < -np.pi: lidar_yaw += 2 * np.pi
        # Return [x, y, z, w, l, h, yaw] to match what model expects
        return np.array([p_lidar[0,0], p_lidar[1,0], p_lidar[2,0], w, l, h, lidar_yaw])

    def _load_original_boxes(self, file_name):
        label_file = os.path.join(self.label_root, f"{file_name}.txt")
        calib_file = os.path.join(self.calib_root, f"{file_name}.txt")
        if not os.path.exists(label_file) or not os.path.exists(calib_file):
            return np.zeros((0, 8), dtype=np.float32)
        
        tr_mat = self._get_calib_matrix(calib_file)
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) < 15: continue
                cls_type = data[0]
                if cls_type not in self.class_map: continue
                
                box_cam = [float(x) for x in data[8:15]] # h, w, l, x, y, z, yaw
                box_lidar = self._box_cam_to_lidar(box_cam, tr_mat)
                class_id = self.class_map[cls_type]
                
                boxes.append(np.append(box_lidar, class_id))
        
        if len(boxes) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)

    def __iter__(self):
        for data in self.dali_iter:
            points_batch = data[0]['points']
            idx_batch = data[0]['label_idx'].cpu().numpy().flatten()
            
            final_points_batch = []
            final_boxes_batch = []
            
            for i in range(len(idx_batch)):
                pts = points_batch[i].cpu().numpy()
                file_name = self.file_names[idx_batch[i]]
                
                # 1. Load original boxes
                orig_boxes = self._load_original_boxes(file_name)
                
                # 2. GT Sampling
                sampled_pts, sampled_boxes = self.apply_gt_sampling(pts)
                
                # 3. Pad/Slice points
                if len(sampled_pts) > 60000:
                    # 使用无放回随机抽样代替粗暴的顺序切片，保留完整的物体轮廓
                    indices = np.random.choice(len(sampled_pts), 60000, replace=False)
                    sampled_pts = sampled_pts[indices]
                else:
                    pad = np.zeros((60000 - len(sampled_pts), 4), dtype=np.float32)
                    # 将 padding 假点移到 1000m 外，从而让 GPU Voxelizer 自动剔除它们，防止原点处堆积几万个假点导致 inf
                    pad[:, :3] = 1000.0
                    sampled_pts = np.concatenate([sampled_pts, pad], axis=0)
                
                # 4. Combine boxes
                all_boxes = np.concatenate([orig_boxes, sampled_boxes], axis=0) if len(sampled_boxes) > 0 else orig_boxes
                
                # 5. Global Data Augmentation
                if self.dali_pipe.split == 'training':
                    sampled_pts, all_boxes = self._apply_global_augmentation(sampled_pts, all_boxes)
                
                final_points_batch.append(sampled_pts)
                final_boxes_batch.append(all_boxes)
            
            yield {
                'points': final_points_batch, 
                'gt_boxes': final_boxes_batch
            }

    def _apply_global_augmentation(self, points, boxes):
        """Apply random rotation, scaling, and translation to both points and boxes"""
        if len(boxes) == 0:
            return points, boxes
            
        # 0. Random flip along Y-axis (左右翻转)
        if np.random.rand() > 0.5:
            # points (x, y, z, intensity)
            points[:, 1] = -points[:, 1]
            # boxes (x, y, z, w, l, h, yaw, cls)
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]

        # 1. Random rotation around Z
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Apply to points
        valid_mask = points[:, 0] < 900.0  # skip padding points
        points[valid_mask, :3] = points[valid_mask, :3] @ rot_mat.T
        
        # Apply to boxes (x, y, z) and yaw
        boxes[:, :3] = boxes[:, :3] @ rot_mat.T
        boxes[:, 6] += angle
        
        # 2. Random scaling
        scale = np.random.uniform(0.95, 1.05)
        points[valid_mask, :3] *= scale
        boxes[:, :3] *= scale
        boxes[:, 3:6] *= scale  # w, l, h
        
        # 3. Random translation
        trans = np.random.uniform(-0.2, 0.2, size=(3,)).astype(np.float32)
        points[valid_mask, :3] += trans
        boxes[:, :3] += trans
        
        return points, boxes

    def apply_gt_sampling(self, points):
        all_sampled_pts = [points]
        sampled_boxes = []
        for cls, count in self.sample_groups.items():
            if len(self.db_infos[cls]) == 0: continue
            # 安全的 numpy 对象数组抽样方法
            indices = np.random.choice(len(self.db_infos[cls]), count)
            for idx in indices:
                info = self.db_infos[cls][idx]
                file_path = os.path.join(self.data_root, info['path']) if not os.path.isabs(info['path']) else info['path']
                if not os.path.exists(file_path): continue
                obj_p = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                obj_p[:, :3] += info['box3d_lidar'][:3] 
                all_sampled_pts.append(obj_p)
                
                # info['box3d_lidar'] is [x, y, z, l, w, h, yaw]
                # We need [x, y, z, w, l, h, yaw, class_id]
                b = info['box3d_lidar']
                class_id = self.class_map[cls]
                sampled_boxes.append([b[0], b[1], b[2], b[4], b[3], b[5], b[6], class_id])
                
        return np.concatenate(all_sampled_pts, axis=0), np.array(sampled_boxes, dtype=np.float32).reshape(-1, 8)
