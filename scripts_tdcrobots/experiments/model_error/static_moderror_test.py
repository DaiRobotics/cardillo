# static controller with a wrong model in its static twin, plant stays exact
# err_stiff scales the twin's E and G, err_rho its density, err_g its g_accel, 1.0 = exact

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from model.common_model import SETPOINT_TABLE
from experiments.static.static_controller_test_main import applied_tensions, report, simulate

# best reachable error at A with la >= 0
A_FEASIBLE_FLOOR_MM = 0.1139


def uniform_error(gamma):
    # scales h_elastic and h_gravity by the same factor, same as err_h = gamma in the dynamic tests
    return dict(err_stiff=gamma, err_rho=gamma, err_g=1.0)


def settled_error(sol, rod, point):
    r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
    return np.linalg.norm(r_OP[-1] - SETPOINT_TABLE[point])


def sweep_error(gammas, point="A", variant="plain", t_hold=5.0, **kwargs):
    # settled error at one held setpoint against a uniform twin error
    print(f"\nstatic '{variant}', uniform twin error, {point} held {t_hold:g} s")
    if variant == "qp":
        print(f"  floor with la >= 0 at A: {A_FEASIBLE_FLOOR_MM:.4f} mm")
    rows = []
    for g in gammas:
        sol, controller, rod, refs, model = simulate(
            [point], smooth=False, t_hold=t_hold, t_sim=t_hold, variant=variant,
            verbose=False, **uniform_error(g), **kwargs
        )
        e = settled_error(sol, rod, point)
        la = applied_tensions(controller, sol)[-1]
        print(f"  err = {g:.2f}: |e| = {e * 1e3:.4f} mm, la = {np.array2string(la, precision=4)}",
              flush=True)
        rows.append((g, e, la))
    return rows


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    point = "A"  # setpoint held in the sweep
    variant = "qp"
    # variant = "plain"
    smooth = False
    t_move = 0.5
    t_hold = 5.0
    t_sim = 25.0
    dt = 1e-3
    Kp = 2.0
    Kd = 0.0
    damping_ratio = 0.1
    use_feedforward = False
    start_point = "E"
    J_stat_check_dt = 1e-3

    # ---- model error (1.0 = exact) ----
    # err_stiff and one of err_rho / err_g at the same value gives err_h of the dynamic tests
    err_stiff = 0.9
    err_rho = 0.9
    err_g = 1.0

    # sweep the uniform error at one setpoint, or one run over the stair with figures
    sweep = False
    gammas = (0.9, 1.0, 1.05, 1.1, 1.2, 1.5)

    common = dict(dt=dt, Kp=Kp, Kd=Kd, damping_ratio=damping_ratio,
                  use_feedforward=use_feedforward, start_point=start_point,
                  J_stat_check_dt=J_stat_check_dt)

    if sweep:
        sweep_error(gammas, point=point, variant=variant, t_hold=t_hold, **common)
        raise SystemExit

    # ---- single run: exact model alongside the error model ----
    runs = []
    for es, er, eg in [(1.0, 1.0, 1.0), (err_stiff, err_rho, err_g)]:
        sol, controller, rod, refs, model = simulate(
            names, smooth=smooth, t_move=t_move, t_hold=t_hold, t_sim=t_sim, variant=variant,
            err_stiff=es, err_rho=er, err_g=eg, **common
        )
        lbl = "exact model" if (es, er, eg) == (1.0, 1.0, 1.0) else "error model"
        r_OP, r_ref, e, la_ts = report(f"static '{variant}', {lbl}", sol, controller, rod,
                                       refs, names, t_hold, smooth, t_move)
        runs.append((lbl, sol, r_OP, r_ref, e, la_ts))

    # ---- visualization ----
    import matplotlib.pyplot as plt

    out = Path(__file__).parent
    tag = (f"_{variant}" + ("_ff" if use_feedforward else "_noff")
           + (f"_smooth{t_move:g}" if smooth else "_step")
           + (f"_start{start_point}" if start_point else "")
           + f"_hold{t_hold:g}"
           + "".join(f"_{n}{v:g}" for n, v in (("stiff", err_stiff), ("rho", err_rho),
                                               ("g", err_g)) if v != 1.0))
    marks = [k * t_hold for k in range(1, len(names))]

    fig, axs = plt.subplots(3, 1, num="XYZ", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(runs[-1][1].t, runs[-1][3][:, i] * 1e2, "b--", label="desired")
        for label, sol_k, r_OP_k, _, _, _ in runs:
            axs[i].plot(sol_k.t, r_OP_k[:, i] * 1e2, lw=1.0, label=label)
        for tm in marks:
            axs[i].axvline(tm, color="0.7", ls=":", lw=0.8)
        axs[i].set_ylabel(f"{lbl} [cm]")
        axs[i].legend(fontsize=8)
        axs[i].grid(True)
    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out / f"static_moderr_tip_tracking{tag}.png", dpi=150)

    fig, ax = plt.subplots(num="Error", figsize=(9, 4.0))
    for label, sol_k, _, _, e_k, _ in runs:
        ax.semilogy(sol_k.t, np.maximum(e_k, 1e-12) * 1e3, lw=1.0, label=label)
    for tm in marks:
        ax.axvline(tm, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$||\boldsymbol{r}_{OP}^{*} - \boldsymbol{r}_{OP}||$ [mm]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out / f"static_moderr_tracking_error{tag}.png", dpi=150)

    fig, axs = plt.subplots(len(runs), 1, num="TendonForces", figsize=(9, 4.0 * len(runs)),
                            sharex=True, squeeze=False)
    for ax, (label, sol_k, _, _, _, la_k) in zip(axs[:, 0], runs):
        for k in range(la_k.shape[1]):
            ax.plot(sol_k.t, la_k[:, k], label=f"tendon {k + 1}")
        ax.axhline(0.0, color="k", lw=0.8)
        for tm in marks:
            ax.axvline(tm, color="0.7", ls=":", lw=0.8)
        ax.set_ylabel(r"$\lambda_{\tau,i}$ [N]")
        ax.set_title(label, fontsize=9)
        ax.grid(True)
        ax.legend(ncol=4, fontsize=8)
    axs[-1, 0].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out / f"static_moderr_tendon_forces{tag}.png", dpi=150)

    print("figures written to", out)
    plt.show()
