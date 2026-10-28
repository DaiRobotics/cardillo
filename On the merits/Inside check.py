import numpy as np
import matplotlib.pyplot as plt
from os import path

folder = r"C:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\On the merits"
basic_file = "workspace_samples_straight3driven_0_5.8_step_53.npy"
type_files = {
    "012 (3 straight)":   "workspace_samples_straight3driven_0_5.8_step_53.npy",
    "Type I":   "workspace_samples_typeI_beta180_0_5.8_step_37_3case.npy",
    "Type II":  "workspace_samples_typeII_beta180_0_5.8_step_37_3case.npy",
    "Type III": "workspace_samples_typeIII_beta180_0_5.8_step_37_3case.npy",
}
phi_bin = 5.0   # deg, azimuth bin width total = 72
z_bin = 1.0     # mm, height bin width
tol = 1.0       # mm, gap < 1 is considered "on shell" 

def polar(P):
    x, y, z = P[:, 0] * 1000, P[:, 1] * 1000, P[:, 2] * 1000
    return np.hypot(x, y), np.rad2deg(np.arctan2(y, x)) % 360, z #keep z in mm, r in mm, p in deg

nphi = int(360 / phi_bin)

# shell profile r(z) of basic, one per azimuth bin
rb, pb, zb = polar(np.load(path.join(folder, basic_file)))
ib = (pb // phi_bin).astype(int) % nphi #which azimuth bin each point belongs to
profiles = []
for k in range(nphi):#loop
    m = ib == k #choose points in the k-th azimuth bin
    if m.sum() == 0:
        profiles.append(None)
        continue
    zi = np.floor(zb[m] / z_bin).astype(int) #which height bin each point belongs to
    zs = np.unique(zi)#Integer layer index
    rs = np.array([rb[m][zi == u].mean() for u in zs]) #average r in each height bin
    profiles.append((zs * z_bin + z_bin / 2, rs)) #72 curves

def gap_to_shell(P):
    # r_tip - r_basic at the same azimuth and height (negative = inside the shell)
    r, p, z = polar(P)
    ip = (p // phi_bin).astype(int) % nphi #which azimuth bin each point belongs to
    d = np.full(len(r), np.nan) #fill with nan
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
    print(f"{name:10s}{(v < -tol).sum() / n:9.1%}{(np.abs(v) <= tol).sum() / n:10.1%}"#inside, on shell, outside, no ref, median gap
          f"{(v > tol).sum() / n:9.1%}{np.isnan(d).sum() / n:8.1%}{np.median(v):10.1f} mm")
    a.hist(v, bins=60, color="C0")
    a.axvline(0, c="k", lw=1)
    a.set_xlabel("r_tip - r_basic at same azimuth and z (mm)")
    a.set_ylabel("number of points")
    a.set_title(name)
plt.tight_layout()
plt.savefig(path.join(folder, "inside_shell_check.png"), dpi=300, bbox_inches="tight")
plt.show()