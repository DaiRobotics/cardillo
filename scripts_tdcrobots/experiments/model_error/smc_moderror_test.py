# SMC with a wrong model in the controller, plant stays exact
# err_h scales h, err_w scales W_tau, err_m scales M_tilde_inv, 1.0 = exact

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from experiments.smc.smc_p2p_test import run, report, SETPOINT_TABLE


def settled_error(d, point):
    return float(np.linalg.norm(d["r"][-1] - SETPOINT_TABLE[point]))


def sweep_err_h(gammas, point="A", mode="qp", alpha=10.0, k=10.0, c=40.0,
                t_hold=6.0, dt=1e-4, **kwargs):
    # settled error at one held setpoint against err_h
    print(f"\nSMC '{mode}', alpha={alpha:g} k={k:g} c={c:g}, {point} held {t_hold:g} s")
    rows = []
    for g in gammas:
        d = run(mode=mode, alpha=alpha, k=k, c=c, names=[point], t_hold=t_hold,
                t_sim=t_hold, dt=dt, err_h=g, **kwargs)
        e = settled_error(d, point)
        la = d["la"][-1]
        print(f"  err_h = {g:.2f}: |e| = {e * 1e3:.4f} mm, la = {np.array2string(la, precision=4)}",
              flush=True)
        rows.append((g, e, la))
    return rows


if __name__ == "__main__":
    # ---- parameters ----
    point = "A"  # setpoint held in the sweep
    mode = "qp"
    # mode = "plain"
    # mode = "nullspace"
    # mode = "both"
    alpha = 10.0
    k = 10.0
    c = 40.0
    t_hold = 5.0
    dt = 1e-4
    start = "E"

    # ---- model error (1.0 = exact) ----
    err_h = 0.9
    err_w = 1.0
    err_m = 1.0

    # sweep err_h at one setpoint, or one run over the stair with figures
    sweep = False
    gammas = (0.9, 1.0, 1.05, 1.1, 1.2, 1.5)

    if sweep:
        sweep_err_h(gammas, point=point, mode=mode, alpha=alpha, k=k, c=c,
                    t_hold=t_hold, dt=dt, start=start)
        raise SystemExit

    # ---- single run: exact model alongside the error model ----
    names = ["A", "B", "C", "D", "E"]
    runs = []
    for eh, ew, em in [(1.0, 1.0, 1.0), (err_h, err_w, err_m)]:
        d = run(mode=mode, alpha=alpha, k=k, c=c, names=names, t_hold=t_hold,
                t_sim=t_hold * len(names), dt=dt, start=start,
                err_h=eh, err_w=ew, err_m=em)
        lbl = "exact model" if (eh, ew, em) == (1.0, 1.0, 1.0) else "error model"
        print(f"\n--- {lbl} ---")
        report(d)
        runs.append((lbl, d))

    # ---- visualization ----
    import matplotlib.pyplot as plt

    out = Path(__file__).parent
    tag = (f"_a{alpha:g}_k{k:g}_c{c:g}_{mode}_start{start}"
           + f"_hold{t_hold:g}"
           + f"_errh{err_h:g}w{err_w:g}m{err_m:g}")
    marks = [j * t_hold for j in range(1, len(names))]

    fig, axs = plt.subplots(3, 1, num="XYZ", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(runs[-1][1]["t"], runs[-1][1]["r_ref"][:, i] * 1e2, "b--",
                    label="desired")
        for label, d_k in runs:
            axs[i].plot(d_k["t"], d_k["r"][:, i] * 1e2, lw=1.0, label=label)
        for tm in marks:
            axs[i].axvline(tm, color="0.7", ls=":", lw=0.8)
        axs[i].set_ylabel(f"{lbl} [cm]")
        axs[i].legend(fontsize=8)
        axs[i].grid(True)
    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out / f"smc_moderr_tip_tracking{tag}.png", dpi=150)

    fig, ax = plt.subplots(num="Error", figsize=(9, 4.0))
    for label, d_k in runs:
        ax.semilogy(d_k["t"], np.maximum(d_k["e"], 1e-12) * 1e3, lw=1.0, label=label)
    for tm in marks:
        ax.axvline(tm, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$||\boldsymbol{r}_{OP}^{*} - \boldsymbol{r}_{OP}||$ [mm]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out / f"smc_moderr_tracking_error{tag}.png", dpi=150)

    fig, axs = plt.subplots(len(runs), 1, num="TendonForces", figsize=(9, 4.0 * len(runs)),
                            sharex=True, squeeze=False)
    for ax, (label, d_k) in zip(axs[:, 0], runs):
        for j in range(d_k["la"].shape[1]):
            ax.plot(d_k["t"], d_k["la"][:, j], label=f"tendon {j + 1}")
        ax.axhline(0.0, color="k", lw=0.8)
        for tm in marks:
            ax.axvline(tm, color="0.7", ls=":", lw=0.8)
        ax.set_ylabel(r"$\lambda_{\tau,i}$ [N]")
        ax.set_title(label, fontsize=9)
        ax.grid(True)
        ax.legend(ncol=4, fontsize=8)
    axs[-1, 0].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out / f"smc_moderr_tendon_forces{tag}.png", dpi=150)

    print("figures written to", out)
    plt.show()
