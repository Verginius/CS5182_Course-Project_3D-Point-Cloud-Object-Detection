# SECOND-DeepOps: Deep Optimization and Refactoring of SECOND based on RTX 5090 and native spconv

## 1. Background

In the 3D point cloud perception tasks of autonomous driving, the SECOND model successfully balances detection accuracy and inference speed via Sparse Convolution. However, in complex high-order autonomous driving scenarios, the traditional SECOND faces the following limitations:

- **Poor Localization of Long-tail Objects**: Anchor-based detection heads lack flexibility when handling obstacles of non-standard sizes and arbitrary orientations.
- **Limited Perception Range**: The traditional coarse voxel division granularity leads to massive loss of features for distant small targets (>100m).
- **Framework Redundancy**: Mainstream frameworks present flexibility barriers in extreme performance optimization and multi-threaded preprocessing scheduling.

Relying on the massive memory bandwidth and computing power redundancy provided by the NVIDIA RTX 5090 (Blackwell architecture, 32GB GDDR7), this project undergoes a low-level refactoring using **Pure PyTorch + spconv 2.x** to build an ultra-high resolution and ultra-low latency 3D detection foundation for autonomous driving.

## 2. Technical Solution

### 2.1 Dynamic High-Resolution Voxelization

- Replaced traditional static voxel division with natively accelerated `spconv.pytorch.utils.PointToVoxel`.
- Base voxel_size: `[0.02, 0.02, 0.1]` (shrunk from the traditional 0.05m).
- Extended effective perception range from 70 meters to 150 meters.

### 2.2 Sparse Backbone Redesign

- Expanded `SubmanifoldSparseConv3d` base feature channels from 16 to 32/64.
- Map2BEV Optimization: Replaced naive dimension flattening with customized feature shaping.

### 2.3 Center-based Head Upgrade

- Removed Anchor-based branches and direction classification loss.
- Introduced the CenterPoint core paradigm: Predicting target center-point heatmaps.
- Regressed 3D dimensions, depth, and orientation angles continuously to eliminate 180-degree ambiguity.

### 2.4 Hardware-level Acceleration for RTX 5090

- Point cloud preprocessing pipeline parallelized via **NVIDIA DALI**.
- Deep integration of PyTorch **AMP** (Automatic Mixed Precision) for FP16 tensor core acceleration.
- Removed out-of-bounds padding points dynamically to maximize sparse convolution throughput.

## 3. Project Structure

```text
CS5182_New/
├── models/
│   ├── __init__.py
│   ├── second.py          # Sparse backbone network
│   ├── center_head.py     # Anchor-free CenterPoint head
│   └── voxel_generator.py # Dynamic voxel generator wrappers
├── data/                  # Data directory (KITTI layout)
├── output/
│   └── ckpt/              # Model checkpoints & Logs
├── train.py               # Main training script (OneCycleLR, DALI pipeline)
├── evaluate.py            # Evaluation & NMS post-processing script
├── run_pipeline.py        # Automated pipeline execution script
├── config.py              # Central configuration file
├── environment.yml        # Conda environment config
├── activate_env.sh        # Environment activation script
└── requirements.txt       # Dependencies (spconv-cu126)
```

## 4. Performance Metrics

| Metric | Baseline (OpenPCDet) | Optimized (RTX 5090) |
|--------|----------------------|----------------------|
| Voxel Resolution | 0.05m | **0.02m** |
| Perception Range | ~70m | **150m** |
| Inference Speed | 25-30 FPS | **74.45 FPS** |
| Underlying Dependency | OpenPCDet | Pure PyTorch + spconv |

### 4.1 Final Evaluation Results

After 80 epochs of fully end-to-end training, we achieved remarkable performance and extreme inference speed on the validation set:

```text
Evaluation Results
Mean Average Precision (mAP): 0.6692

Metrics at Confidence Threshold 0.70:
  Total TP: 344
  Total FP: 309
  Total FN: 213
  Precision: 0.5268
  Recall: 0.6176
  F1 Score: 0.5686
--------------------------------------------------
Inference Speed (FP16 Accelerated):
  Avg Latency: 13.43 ms/frame
  FPS:         74.45 frames/s
```

### 4.2 Training Artifacts

The project includes complete training history monitoring and logging capabilities. Training artifacts are automatically saved in the project:
- **Model Weights**: `output/ckpt/final_model.pth`
- **Raw Training Metrics**: `output/ckpt/training_metrics.csv`
- **Loss / Learning Rate Curve Plot**: `output/training_metrics_plot.png`

## 5. Quick Start

### 5.1 Environment Setup

```bash
# Create Conda environment
conda env create -f environment.yml

# Activate environment
conda activate pointcloud_5090

# Alternatively, use the project script to activate
source activate_env.sh
```

### 5.2 Run Pipeline

```bash
# Run the complete train + evaluation pipeline
python run_pipeline.py

# Or run them separately
python train.py
python evaluate.py
```
