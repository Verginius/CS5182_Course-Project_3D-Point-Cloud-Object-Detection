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

        # 在 DALI 内先简单截断到一个较大的值，防止 GPU 显存溢出
        # 此时先不急着 pad 到 60000，因为后面要贴 GT
        points = fn.slice(points, [0, 0], [40000, 4], axes=[0, 1])

        return points.gpu(), label_idx.gpu()

class GTDataLoader:
    def __init__(self, dali_pipe, dali_iter, db_info_path, label_root, sample_groups):
        self.dali_pipe = dali_pipe
        self.dali_iter = dali_iter
        with open(db_info_path, 'rb') as f:
            self.db_infos = pickle.load(f)
        
        self.label_root = label_root
        self.sample_groups = sample_groups # 例如 {'Car': 15, 'Pedestrian': 10}
        # 获取文件名列表，用于索引回找
        self.file_names = [os.path.basename(f).replace('.bin', '') for f in dali_pipe.file_list]

    def __iter__(self):
        for data in self.dali_iter:
            # data[0] 是 {'points': Tensor, 'label_idx': Tensor}
            points_batch = data[0]['points']
            idx_batch = data[0]['label_idx'].cpu().numpy().flatten()
            
            final_points_batch = []
            
            for i in range(len(idx_batch)):
                # 1. 获取当前帧的原始点云和文件名
                pts = points_batch[i].cpu().numpy()
                file_name = self.file_names[idx_batch[i]]
                
                # 2. 执行 GT Sampling 粘贴逻辑
                sampled_pts = self.apply_gt_sampling(pts)
                
                # 3. 最终统一 Pad/Slice 到 60,000 点，匹配 5090 优化需求
                if len(sampled_pts) > 60000:
                    sampled_pts = sampled_pts[:60000]
                else:
                    pad = np.zeros((60000 - len(sampled_pts), 4), dtype=np.float32)
                    sampled_pts = np.concatenate([sampled_pts, pad], axis=0)
                
                final_points_batch.append(sampled_pts)
            
            yield np.array(final_points_batch) # 返回 [BS, 60000, 4]

    def apply_gt_sampling(self, points):
        # 这里的逻辑就是从 db_infos 随机选文件，np.fromfile 读取并 points += box_xyz
        # ... 实现细节参考之前的相对坐标转换 ...
        all_sampled_pts = [points]
        for cls, count in self.sample_groups.items():
            choices = np.random.choice(self.db_infos[cls], count)
            for info in choices:
                obj_p = np.fromfile(info['path'], dtype=np.float32).reshape(-1, 4)
                # 因为是相对坐标，直接加上它原始的 box 中心坐标即可复原到场景中
                obj_p[:, :3] += info['box3d_lidar'][:3] 
                all_sampled_pts.append(obj_p)
        return np.concatenate(all_sampled_pts, axis=0)    