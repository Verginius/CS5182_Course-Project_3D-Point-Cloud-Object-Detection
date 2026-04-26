# SECOND-DeepOps: 基于 RTX 5090 与原生 spconv 的 SECOND 模型深度优化与重构

## 1. 背景 (Background)

在自动驾驶 3D 点云感知任务中，SECOND 模型凭借稀疏卷积（Sparse Convolution）成功平衡了检测精度与推理速度。然而，在复杂的高阶自动驾驶场景中，传统的 SECOND 面临以下局限性：

- **长尾目标定位差**：基于 Anchor 的检测头在处理非标准尺寸障碍物及任意朝向时缺乏柔性
- **感知距离受限**：传统体素划分粒度较粗，导致远距离（>100m）小目标的特征大量丢失
- **框架冗余**：主流框架在极致性能压榨和多线程预处理调度上存在灵活性壁垒

本方案依托 NVIDIA RTX 5090（Blackwell 架构，32GB GDDR7）提供的海量显存带宽与算力冗余，采用**纯 PyTorch + spconv 2.x** 进行底层重构，打造超高分辨率、极低延迟的自动驾驶 3D 检测基座。

## 2. 技术方案 (Technical Solution)

### 2.1 动态高分辨率体素化 (Dynamic High-Res Voxelization)

- 使用 `spconv.utils.VoxelGeneratorV2` 替代传统静态体素划分
- 基础 voxel_size: `[0.02, 0.02, 0.1]`（从传统的 0.05 缩小）
- 有效感知距离从 70 米提升至 150 米

### 2.2 稀疏主干网络重构 (Sparse Backbone Redesign)

- SubmanifoldSparseConv3d 基础特征通道数从 16 扩充至 32/64
- Map2BEV 改进：引入 1×1 自适应卷积代替简单维度展平

### 2.3 Anchor-free 检测头升级 (Center-based Head)

- 移除 Anchor-based 分支与方向分类损失
- 引入 CenterPoint 核心范式：预测目标中心点热力图
- 回归目标的三维尺寸、深度及朝向角

### 2.4 RTX 5090 硬件级加速调优

- NVIDIA DALI 进行点云预处理
- PyTorch AMP（自动混合精度）FP16/BF16 计算
- CUDAGraphs 捕获静态计算图

## 3. 项目结构

```
CS5182_New/
├── models/
│   ├── __init__.py
│   ├── second.py          # 稀疏主干网络
│   ├── center_head.py     # Anchor-free 检测头
│   └── voxel_generator.py # 动态体素生成器
├── data/                  # 数据目录
├── output/
│   └── ckpt/              # 模型checkpoint
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── run_pipeline.py        # 流水线脚本
├── config.py              # 配置文件
├── environment.yml        # Conda 环境配置
├── activate_env.sh        # 环境激活脚本
└── requirements.txt       # 依赖
```

## 4. 性能指标

| 指标 | Baseline (OpenPCDet) | 优化版 (RTX 5090) |
|------|---------------------|------------------|
| 体素分辨率 | 0.05m | 0.02m |
| 有效感知距离 | ~70m | 150m |
| 推理速度 | 25-30 FPS | **74.45 FPS** |
| 底层依赖 | OpenPCDet | 纯 PyTorch + spconv |

### 4.1 最终验证集评估结果 (Evaluation Results)

在完全端到端的 80 轮 (Epochs) 训练后，我们在验证集上取得了卓越的性能表现和极致的推理速度：

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

### 4.2 训练产出日志与模型 (Training Artifacts)

本项目包含完整的训练历史监控与日志留存功能，训练产出将自动保存在项目中：
- **模型权重 (Weights)**: `output/ckpt/final_model.pth`
- **训练指标原始数据**: `output/ckpt/training_metrics.csv`
- **Loss / 学习率下降曲线图**: `output/training_metrics_plot.png`

## 5. 快速开始

### 5.1 环境配置

```bash
# 创建 Conda 环境
conda env create -f environment.yml

# 激活环境
conda activate pointcloud_5090

# 或使用项目脚本激活
source activate_env.sh
```

### 5.2 运行流水线

```bash
# 运行完整训练+评估流水线
python run_pipeline.py

# 或分别运行
python train.py
python evaluate.py
```
