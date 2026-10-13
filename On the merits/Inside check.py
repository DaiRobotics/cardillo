import numpy as np
import matplotlib.pyplot as plt
from os import path

folder = r"C:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\On the merits"
basic_file = "workspace_samples_basic_0_5.8_step_200_3case.npy"
type_files = {
     "012 (3 straight)":   "workspace_samples_straight3driven_0_5.8_step_53.npy",
    "basic (self-check)": "workspace_samples_basic_0_5.8_step_200_3case.npy",
}
phi_bin = 5.0   # deg, azimuth bin width
z_bin = 1.0     # mm, height bin width
tol = 1.0       # mm, |gap| <= tol counts as "on the shell" (= voxel size)

def polar(P):
    x, y, z = P[:, 0] * 1000, P[:, 1] * 1000, P[:, 2] * 1000
    return np.hypot(x, y), np.rad2deg(np.arctan2(y, x)) % 360, z

nphi = int(360 / phi_bin)

# shell profile r(z) of basic, one per azimuth bin
rb, pb, zb = polar(np.load(path.join(folder, basic_file)))
ib = (pb // phi_bin).astype(int) % nphi 
profiles = []
for k in range(nphi):
    m = ib == k
    if m.sum() == 0:
        profiles.append(None)
        continue
    zi = np.floor(zb[m] / z_bin).astype(int)
    zs = np.unique(zi)
    rs = np.array([rb[m][zi == u].mean() for u in zs]) #Sort by height level
    profiles.append((zs * z_bin + z_bin / 2, rs)) #get r

def gap_to_shell(P):
    # r_tip - r_basic at the same azimuth and height (negative = inside the shell)
    r, p, z = polar(P)
    ip = (p // phi_bin).astype(int) % nphi
    d = np.full(len(r), np.nan) #create a gap array, length = len(r), fill with nan
    for k in range(nphi):
        m = ip == k
        if profiles[k] is None or m.sum() == 0:
            continue
        zc, rs = profiles[k]
        ok = (z[m] >= zc.min()) & (z[m] <= zc.max())
        dk = np.full(m.sum(), np.nan)
        dk[ok] = r[m][ok] - np.interp(z[m][ok], zc, rs)
        d[m] = dk
    return d

fig, ax = plt.subplots(1, len(type_files), figsize=(5 * len(type_files), 4))
print(f"{'':10s}{'inside':>9s}{'on shell':>10s}{'outside':>9s}{'no ref':>8s}{'median gap':>12s}")
for a, (name, f) in zip(np.atleast_1d(ax), type_files.items()):
    d = gap_to_shell(np.load(path.join(folder, f)))
    v = d[~np.isnan(d)]
    n = len(d)
    print(f"{name:10s}{(v < -tol).sum() / n:9.1%}{(np.abs(v) <= tol).sum() / n:10.1%}"
          f"{(v > tol).sum() / n:9.1%}{np.isnan(d).sum() / n:8.1%}{np.median(v):10.1f} mm")
    a.hist(v, bins=60, color="C0")
    a.axvline(0, c="k", lw=1)
    a.set_xlabel("r_tip - r_basic at same azimuth and z (mm)")
    a.set_ylabel("number of points")
    a.set_title(name)
plt.tight_layout()
plt.savefig(path.join(folder, "inside_shell_check.png"), dpi=300, bbox_inches="tight")
plt.show()