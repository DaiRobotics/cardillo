import numpy as np, glob
from os import path

here = path.dirname(path.abspath(__file__))
files = glob.glob(path.join(here, "workspace_samples_straight3driven_*.npy"))
print("found:", files)
s3 = np.asarray(np.load(files[0], allow_pickle=True), float)   # 有多个文件时,把 files[0] 改成你要的那个

v = 1e-3
idx = np.floor(s3 / v).astype(int)
u, cnt = np.unique(idx, axis=0, return_counts=True)
print("points", len(s3), "voxels", len(u), "mean pts/voxel", cnt.mean(), "max", cnt.max())
step_len = np.linalg.norm(np.diff(s3, axis=0), axis=1) * 1000
print("consecutive tip distance (mm): median", np.median(step_len),
      "95%", np.percentile(step_len, 95), "max", step_len.max())