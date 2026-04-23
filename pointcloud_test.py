import numpy as np
pts = np.fromfile('/root/autodl-tmp/data/gt_database/000001_Car_0.bin', dtype=np.float32).reshape(-1, 4)
print(pts.shape) # 看看点数对不对
print(pts[:, :3].mean(axis=0))