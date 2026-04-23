"""
Configuration for SECOND-DeepOps model
Based on RTX 5090 optimization specifications from 5182.md
"""

# Model Configuration
MODEL_CONFIG = {
    'grid_size': [40, 4000, 7504],  # [Z, Y, X] -KITTI standard dimensions
    'num_class': 3,  # Car, Pedestrian, Cyclist
    'num_bev_features': 640,  # 128 channels * 5 height slices after backbone
    
    # Voxelization settings (High-resolution 0.02m)
    'voxel_generator': {
        'voxel_size': [0.02, 0.02, 0.1],  # Ultra-fine resolution for 150m perception
        'point_cloud_range': [0, -40.0, -3.0, 150.08, 40.0, 1.0],  # KITTI range
        'max_num_points': 5,
        'max_voxels': 80000,  # Increased for high-res voxelization
    },
    
    # Backbone settings (Enhanced 32/64 channels)
    'backbone': {
        'input_channels': 4,
        'channels': [16, 32, 64, 128],  # Increased channels for better feature extraction
    },
    
    # Detection head
    'head': {
        'in_channels': 640,
        'feature_stride': 16,
        'heatmap_max_objects': 200,
        'score_threshold': 0.1,
        'nms_threshold': 0.3,
    },
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 4,  # Adjust based on GPU memory
    'num_epochs': 20,
    'eval_interval': 5, # 前期每 5 轮验证一次，节省时间
    'final_eval_interval': 1, # 最后阶段每轮验证
    'learning_rate': 0.002,
    'weight_decay': 0.0001,
    'checkpoint_dir': 'output/ckpt',
    'log_interval': 10,
    
    # AMP (Automatic Mixed Precision) for RTX 5090
    'use_amp': True,
    'cuda_graphs': False, # 注意：DALI 与 CUDA Graphs 同时开启可能存在兼容性风险，建议初次先设为 False
    'gradient_checkpointing': True, # 显存保镖，防止 500x938 分辨率下 OOM
}

# Data Configuration (集成 DALI)
DATA_CONFIG = {
    'dataset_path': 'data/kitti',
    'num_workers': 8, # 5090 配合多核 CPU，建议加大
    'prefetch_size': 2,
    'max_points_limit': 60000, # 强制 DALI 采样 60,000 个点
    
    # Data augmentation
    'augmentation': {
        'random_rotation': [-0.785, 0.785], # ±45度
        'random_scaling': [0.95, 1.05],
        'random_translation': True,
        'gt_sampling': True,
    },
}

# Inference Configuration
INFERENCE_CONFIG = {
    'checkpoint_path': 'output/ckpt/final_model.pth',
    'batch_size': 1,
    'score_threshold': 0.3,
    'nms_threshold': 0.3,
    'output_dir': 'output/predictions',
}

# Performance targets
PERFORMANCE_TARGETS = {
    'perception_range': 150.08,  # meters
    'voxel_resolution': 0.02,    # meters
    'target_fps': 45,            
    'latency_target': 22,        
}