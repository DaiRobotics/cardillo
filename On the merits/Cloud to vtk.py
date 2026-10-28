"""
Convert tip point clouds (.npy, metres) to VTK vertex files (mm) for ParaView.
Each file gets scalar 'z_mm' and 'r_mm' (distance to the rod axis).
Optionally subsample so ParaView stays responsive.
"""
import numpy as np
from os import path

here = path.dirname(path.abspath(__file__))
MAX_POINTS = 150000          # subsample above this

files = {
    "cloud_basic":   "workspace_samples_basic_0_5.8_step_200_3case.npy",
    "cloud_typeI":   "workspace_samples_typeI_beta180_0_5.8_step_37_3case.npy",
    "cloud_typeII":  "workspace_samples_typeII_beta180_0_5.8_step_37_3case.npy",
    "cloud_typeIII": "workspace_samples_typeIII_beta180_0_5.8_step_37_3case.npy",
}

rng = np.random.default_rng(0)#Random Number Generator
for out, fn in files.items():
    fp = path.join(here, fn)
    if not path.exists(fp):
        print("skip (not found):", fn); continue
    P = np.load(fp, allow_pickle=True)
    P = np.vstack([np.asarray(p, float).reshape(-1, 3) for p in P]) if P.dtype == object else np.asarray(P, float)
    P = P[P[:, 2] > 0] * 1000.0#Keep only the points with \\(z>0\\).
    if len(P) > MAX_POINTS:
        P = P[rng.choice(len(P), MAX_POINTS, replace=False)]
    r = np.hypot(P[:, 0], P[:, 1])#The horizontal distance from each point to the \\(z\\)-axis
    with open(path.join(here, out + ".vtk"), "w") as f:
        f.write("# vtk DataFile Version 3.0\ntip cloud\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(P)} float\n")
        for p in P:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write(f"VERTICES {len(P)} {2 * len(P)}\n")
        for i in range(len(P)):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {len(P)}\n")
        for name, arr in (("z_mm", P[:, 2]), ("r_mm", r)):
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{v:.4f}\n")
    print("written", out + ".vtk", len(P), "points")