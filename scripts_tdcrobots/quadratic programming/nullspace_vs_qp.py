"""Why W_tau-nullspace redistribution cannot do what the QP does.

Both methods start from the same unconstrained allocation F_p and both claim to preserve the
generalized force W_tau @ F_p. They differ in ONE respect, and it decides everything:

    wtau_nullspace : searches the LINE   {F_p + alpha*v : alpha in R},  W_tau @ v ~ 0
    QP             : searches the CONE   {x : x >= f_min}

The line is a feasibility problem. The cone is an optimization problem. A feasibility problem
can have an empty feasible set; `min ||A(x-F_p)|| s.t. x >= f_min` never can, because f_min*1 is
always feasible. That is the whole difference, and this script measures its consequences.

THE CONSERVED QUANTITY
----------------------
The W_tau null direction is an antagonistic pair swap, v ~ (1,-1,1,-1)/2 (two tendons up, two
down), so 1^T v ~ 0. Therefore along the entire line

    1^T (F_p + alpha*v) = 1^T F_p     for every alpha

The total tendon force is INVARIANT under nullspace redistribution. But positivity requires
F >= 0, which implies 1^T F >= 0. So whenever 1^T F_p < 0 the whole line is infeasible -- not
hard to search, provably empty. No choice of alpha, no smarter solver, no extra iterations.

The QP is not bound by this because it is allowed to leave the line: it changes 1^T x freely and
pays a residual ||W_tau(x - F_p)|| instead.

Run:  python nullspace_vs_qp.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# The user's reorganization split these into sibling folders, and
# wtau_nullspace_controller.py still does a bare `from nullspace_controller import ...`.
# Put both on the path rather than editing their files.
sys.path.insert(0, str(HERE.parent / "nullspace redistribution"))
sys.path.insert(0, str(HERE.parent / "lateral and axial shifting"))

from cardillo.solver import ScipyDAE

from qp_positive_controller import solve_positive_qp
from qp_positive_test import build
from wtau_nullspace_controller import achievable_floor, wtau_alloc, wtau_null_dir

CACHE = HERE / "qp_cache" / "nullspace_vs_qp.npz"


def collect(t_sim=10.0, dt=1e-4, t_hold=2.0, stride=25):
    """Run the unconstrained baseline once, then analyse both allocators along it."""
    if CACHE.exists():
        print(f"cached: {CACHE.name}")
        return dict(np.load(CACHE))

    model, ctrl, _ = build(metric="W_tau", f_min=0.0, bypass=True, t_hold=t_hold)
    print(f"baseline (unconstrained) run: t_sim={t_sim}, dt={dt}", flush=True)
    sol = ScipyDAE(model.system, t_sim, dt).solve()

    qD, uD = ctrl.qDOF, ctrl.uDOF
    rec = {k: [] for k in (
        "t", "ones_v", "sum_Fp", "sum_F", "floor_w", "floor_qp", "min_Fp",
        "sigma4", "sigma1", "sigma2", "leak_W", "leak_J", "feasible", "e_qp_W",
    )}

    idx = range(0, len(sol.t), stride)
    print(f"analysing {len(idx)} of {len(sol.t)} samples", flush=True)
    for i in idx:
        t, q, u = sol.t[i], sol.q[i][qD], sol.u[i][uD]
        J, b = ctrl._Jb(t, q, u)
        W = ctrl.W_tau(t, q)

        al = wtau_alloc(J, b, W, F_min=0.0, inv_damping=ctrl.inv_damping)
        # wtau_null_dir returns `simple` (True = smallest eigenvalue separated, v usable)
        v, sigma, _, simple = wtau_null_dir(W)

        # the QP, same anchor, same metric
        x, _, _ = solve_positive_qp(W, al.F_p, 0.0)

        rec["t"].append(t)
        rec["ones_v"].append(float(np.sum(v)) if simple else np.nan)
        rec["sum_Fp"].append(float(al.F_p.sum()))
        rec["sum_F"].append(float(al.F.sum()))
        rec["min_Fp"].append(float(al.F_p.min()))
        rec["floor_w"].append(achievable_floor(al))
        rec["floor_qp"].append(float(x.min()))
        rec["sigma1"].append(float(sigma[0]))
        rec["sigma2"].append(float(sigma[1]))
        rec["sigma4"].append(float(sigma[-1]))
        rec["leak_W"].append(float(al.leak_W))
        rec["leak_J"].append(float(al.leak_J))
        rec["feasible"].append(bool(al.feasible))
        rec["e_qp_W"].append(float(np.linalg.norm(W @ (x - al.F_p))))

    out = {k: np.asarray(v) for k, v in rec.items()}
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def report(d):
    n = len(d["t"])
    print(f"\n{n} samples analysed over t = [{d['t'][0]:.2f}, {d['t'][-1]:.2f}] s")

    print("\n--- 1. the null direction conserves total tendon force ---")
    print(f"  |1^T v|                     : max {np.nanmax(np.abs(d['ones_v'])):.2e}")
    print(f"  |sum(F) - sum(F_p)|         : max {np.max(np.abs(d['sum_F'] - d['sum_Fp'])):.2e}")
    print("  => sum of tendon forces is invariant along the nullspace line.")

    print("\n--- 2. so the sign of sum(F_p) decides feasibility outright ---")
    neg = d["sum_Fp"] < 0
    print(f"  samples with sum(F_p) < 0   : {neg.sum():5d} / {n}  ({100 * neg.mean():.1f}%)")
    print(f"    of those, nullspace floor < 0 : {int((d['floor_w'][neg] < 0).sum())} / {int(neg.sum())}"
          f"   <-- must be ALL of them, positivity is impossible there")
    print(f"  samples with sum(F_p) >= 0  : {(~neg).sum():5d} / {n}")
    print(f"    of those, nullspace floor < 0 : {int((d['floor_w'][~neg] < 0).sum())} / {int((~neg).sum())}"
          f"   <-- sum >= 0 is necessary, not sufficient")

    print("\n--- 3. what each method actually achieves ---")
    fw, fq = d["floor_w"], d["floor_qp"]
    print(f"  nullspace redistribution, best achievable min force:")
    print(f"    min over run {fw.min():9.4f} N | median {np.median(fw):8.4f} N | max {fw.max():8.4f} N")
    print(f"    fraction of samples that CANNOT reach 0 N : {100 * np.mean(fw < 0):.1f}%")
    print(f"  QP, achieved min force:")
    print(f"    min over run {fq.min():9.4f} N | fraction below 0 N : {100 * np.mean(fq < -1e-12):.1f}%")

    print("\n--- 4. the invariance the nullspace method claims is itself approximate ---")
    print(f"  sigma(W_tau): sigma_1 {d['sigma1'].mean():.4f} | sigma_2 {d['sigma2'].mean():.5f}"
          f" | sigma_4 {d['sigma4'].mean():.2e} (mean over run)")
    print(f"  sigma_4 / sigma_1           : max {np.max(d['sigma4'] / d['sigma1']):.2e}")
    print(f"  sigma_4 / sigma_2 (lateral) : max {np.max(d['sigma4'] / d['sigma2']):.2e}"
          f"   <-- vs the bending authority that steers the tip")
    print(f"  leaked wrench ||W(F-F_p)||  : mean {d['leak_W'].mean():.4e}  max {d['leak_W'].max():.4e}")
    print(f"  leaked tip accel ||J(F-F_p)||: mean {d['leak_J'].mean():.4e}  max {d['leak_J'].max():.4e}")
    print(f"  QP residual  ||W(x-F_p)||   : mean {d['e_qp_W'].mean():.4e}  max {d['e_qp_W'].max():.4e}")
    print("  => neither is exactly free; the QP prices its deviation explicitly, the")
    print("     nullespace method leaks it silently and still misses positivity.")


def plot(d):
    import matplotlib.pyplot as plt

    t = d["t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].plot(t, d["sum_Fp"], lw=0.9, label=r"$\mathbf{1}^\top F_p$ (conserved by the shift)")
    ax[0].fill_between(t, 0, d["sum_Fp"], where=d["sum_Fp"] < 0, color="r", alpha=0.25,
                       label="positivity provably impossible")
    ax[0].set_ylabel("Total force [N]")
    ax[0].set_title("The invariant: nullspace redistribution cannot change this sum")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.4)

    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].plot(t, d["floor_w"], lw=0.9, label="nullspace: best achievable min force")
    ax[1].plot(t, d["floor_qp"], lw=1.2, label="QP: achieved min force")
    ax[1].fill_between(t, d["floor_w"], 0, where=d["floor_w"] < 0, color="r", alpha=0.2)
    ax[1].set_ylabel("Min tendon force [N]")
    ax[1].set_title("What each method can deliver")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.4)

    ax[2].semilogy(t, np.maximum(d["leak_W"], 1e-18), lw=0.9, label=r"nullspace leak $\|W(F-F_p)\|$")
    ax[2].semilogy(t, np.maximum(d["e_qp_W"], 1e-18), lw=0.9, label=r"QP residual $\|W(x-F_p)\|$")
    ax[2].semilogy(t, d["sigma4"], lw=0.8, ls="--", label=r"$\sigma_4(W_\tau)$ (kernel is not exact)")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Generalized force error")
    ax[2].set_title("Neither preserves the wrench exactly")
    ax[2].legend(fontsize=8)
    ax[2].grid(True, alpha=0.4, which="both")

    fig.tight_layout()
    p = HERE / "nullspace_vs_qp.png"
    fig.savefig(p, dpi=130)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    d = collect()
    report(d)
    plot(d)
