import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm

def get_calib_matrix(calib_file):
    """提取雷达与相机之间的转换矩阵 (Tr_velo_to_cam)"""
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'Tr_velo_to_cam' in line:
            tr = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            tr = np.concatenate([tr, [[0, 0, 0, 1]]], axis=0)
            return tr
    return None

def box_cam_to_lidar(box_cam, tr_mat):
    """
    将 KITTI 相机系的 box 转换到雷达系
    box_cam: [h, w, l, x, y, z, yaw] (相机系下 y 是底部中心)
    """
    h, w, l, x, y, z, yaw = box_cam
    
    # 1. 坐标平移（相机系底部中心 -> 雷达系）
    p_cam = np.array([x, y - h/2, z, 1.0]).reshape(4, 1) # 移到几何中心
    p_lidar = np.linalg.inv(tr_mat) @ p_cam
    
    # 2. 角度转换 (KITTI yaw 在雷达系下需要特殊处理)
    lidar_yaw = -yaw - np.pi / 2
    while lidar_yaw > np.pi: lidar_yaw -= 2 * np.pi
    while lidar_yaw < -np.pi: lidar_yaw += 2 * np.pi
    
    return np.array([p_lidar[0,0], p_lidar[1,0], p_lidar[2,0], l, w, h, lidar_yaw])

def in_hull(points, box):
    """
    判断点是否在 3D Box 内
    box: [x, y, z, l, w, h, yaw]
    """
    x, y, z, l, w, h, yaw = box
    
    # 旋转点云回坐标轴对齐
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    rot_mat = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1]
    ])
    
    local_pts = (points[:, :3] - np.array([x, y, z])) @ rot_mat.T
    
    mask = (local_pts[:, 0] >= -l/2) & (local_pts[:, 0] <= l/2) & \
           (local_pts[:, 1] >= -w/2) & (local_pts[:, 1] <= w/2) & \
           (local_pts[:, 2] >= -h/2) & (local_pts[:, 2] <= h/2)
    return mask

def create_database(data_path):
    root_path = Path(data_path)
    velodyne_path = root_path / "training" / "velodyne"
    label_path = root_path / "training" / "label_2"
    calib_path = root_path / "training" / "calib"
    db_path = root_path / "gt_database"
    db_path.mkdir(exist_ok=True)
    
    all_db_infos = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    
    label_files = sorted(list(label_path.glob("*.txt")))
    
    print(f"Starting to extract objects from {len(label_files)} files...")
    
    for label_file in tqdm(label_files):
        idx = label_file.stem
        # 读取数据
        points = np.fromfile(velodyne_path / f"{idx}.bin", dtype=np.float32).reshape(-1, 4)
        tr_mat = get_calib_matrix(calib_path / f"{idx}.txt")
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.split()
            cls_type = data[0]
            if cls_type not in all_db_infos:
                continue
            
            # 提取相机系参数: [h, w, l, x, y, z, yaw]
            box_cam = np.array([float(x) for x in data[8:15]])
            # 转换为雷达系: [x, y, z, l, w, h, yaw]
            box_lidar = box_cam_to_lidar(box_cam, tr_mat)
            
            # 裁剪点云
            mask = in_hull(points, box_lidar)
            obj_points = points[mask]
            
            # 相对坐标转换（让粘贴时更容易）
            obj_points[:, :3] -= box_lidar[:3]
            
            if len(obj_points) >= 5: # 过滤点数太少的物体
                file_name = f"{idx}_{cls_type}_{len(all_db_infos[cls_type])}.bin"
                rel_path = f"gt_database/{file_name}"
                obj_points.tofile(root_path / rel_path)
                
                info = {
                    'name': cls_type,
                    'path': rel_path,
                    'image_idx': idx,
                    'gt_idx': len(all_db_infos[cls_type]),
                    'box3d_lidar': box_lidar,
                    'num_points_in_gt': len(obj_points),
                    'difficulty': data[2], # KITTI 难度等级
                }
                all_db_infos[cls_type].append(info)

    # 保存索引
    with open(root_path / "kitti_dbinfos_train.pkl", 'wb') as f:
        pickle.dump(all_db_infos, f)
    
    print("\nDatabase creation finished!")
    for cls, infos in all_db_infos.items():
        print(f"{cls}: {len(infos)} objects")

if __name__ == "__main__":
    # 修改为你的数据集存放路径
    DATA_PATH = "/root/autodl-tmp/data"
    create_database(DATA_PATH)