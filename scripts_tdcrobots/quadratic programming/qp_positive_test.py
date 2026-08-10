"""Test harness for QPPositiveController.

Three modes, cheapest first:

    python qp_positive_test.py --check
        No simulation. Verifies the QP actually solves the problem (KKT) and that the analytic
        la_tau_q / la_tau_u match central differences. Run this before trusting any simulation:
        a wrong Jacobian degrades ScipyDAE's Newton convergence quietly rather than loudly.

    python qp_positive_test.py --replay --t-sim 10
        One baseline (unconstrained) simulation, then every sample is projected offline. Gives
        the positivity-vs-accuracy trade-off for both metrics without re-simulating.

    python qp_positive_test.py --run --t-sim 2 --metric W_tau
        Closed loop, QP in the control path. --baseline runs the unconstrained law under
        identical conditions for comparison.

All modes use the plant, gains and solver of dynamic_control_test.py, which is imported
unmodified.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# dynamic_control_test.py and its dependencies live one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_control_test import CommonModel, SETPOINT_TABLE
from dynamic_controller import DynamicControllerPD
from qp_positive_controller import (
    METRICS,
    QPPositiveController,
    metric_matrix,
    replay_qp,
    solve_positive_qp,
)
from cardillo.solver import ScipyDAE

OUT = Path(__file__).parent

# Same gains and plant settings as dynamic_control_test.py
KP, KD, INV_DAMPING, DAMPING_RATIO = 200.0, 20.0, 1e-3, 0.1


def p2p_sequence(names, t_hold=2.0):
    """Un-smoothed setpoint stair, as used in dynamic_control_test.py:215."""
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        return pts[min(int(t // t_hold), len(pts) - 1)]

    return r_OP_ref_fn


def build(metric="W_tau", f_min=0.0, bypass=False, names=("A", "B", "C", "D", "E"), t_hold=2.0):
    model = CommonModel(damping_ratio=DAMPING_RATIO, la_pre=0.0)
    r_OP_ref_fn = p2p_sequence(list(names), t_hold=t_hold)
    controller = QPPositiveController(
        model.system,
        model.rod,
        model.tendons,
        r_OP_ref_fn,
        v_P_ref_fn=lambda t: np.zeros(3),
        a_P_ref_fn=lambda t: np.zeros(3),
        Kp=KP,
        Kd=KD,
        inv_damping=INV_DAMPING,
        metric=metric,
        f_min=f_min,
        bypass=bypass,
    )
    model.system.add(controller)
    model.system.assemble()
    return model, controller, r_OP_ref_fn


## ----------------------------------------------------------------- check mode


def kkt_residual(A, la_ref, x, f_min):
    """First-order optimality of  min 1/2||A(x-la_ref)||^2  s.t.  x >= f_min.

    Returns (primal, stationarity, complementarity), all of which should be ~0. The gradient
    g = A^T A (x - la_ref) must vanish on the free set and be >= 0 where x sits at the bound
    (pushing further down is blocked).
    """
    g = A.T @ (A @ (x - la_ref))
    at_bound = x <= f_min + 1e-9
    primal = max(0.0, float(np.max(f_min - x)))
    stationarity = float(np.max(np.abs(g[~at_bound]))) if (~at_bound).any() else 0.0
    complementarity = float(np.max(np.maximum(0.0, -g[at_bound]))) if at_bound.any() else 0.0
    return primal, stationarity, complementarity


def fd_jacobian(fn, z, eps_rel=1e-6):
    """Central-difference Jacobian of fn at z, plus the per-column step used."""
    n = len(z)
    steps = eps_rel * np.maximum(1.0, np.abs(z))
    cols = []
    for k in range(n):
        dz = np.zeros(n)
        dz[k] = steps[k]
        cols.append((fn(z + dz) - fn(z - dz)) / (2.0 * steps[k]))
    return np.array(cols).T


def check(metric, f_min, n_states, seed):
    model, ctrl, _ = build(metric=metric, f_min=f_min)
    sys = model.system
    q0, u0 = sys.q0[ctrl.qDOF].copy(), sys.u0[ctrl.uDOF].copy()
    rng = np.random.default_rng(seed)

    print(f"\n=== check: metric={metric}  f_min={f_min}  n_q={ctrl._nq}  n_u={ctrl._nu} ===")

    # A spread of perturbation scales, so both the unconstrained regime (QP inactive) and the
    # strongly-clamped regime are exercised.
    scales = np.geomspace(1e-4, 3e-2, n_states)
    n_inactive = 0

    for i, s in enumerate(scales):
        t = 0.3 + 0.7 * i
        q = q0 + s * rng.standard_normal(len(q0))
        u = u0 + s * rng.standard_normal(len(u0))

        la_ref = DynamicControllerPD.la_tau(ctrl, t, q, u)
        x = ctrl.la_tau(t, q, u)
        W_tau = ctrl.W_tau(t, q)
        A, _ = metric_matrix(W_tau, ctrl.M_tilde_inv @ W_tau, metric)
        primal, stat, comp = kkt_residual(A, la_ref, x, f_min)
        n_clamped = int(np.sum(x <= f_min + 1e-9))

        # When nothing binds the QP must reproduce the baseline bit-for-bit.
        exact = np.max(np.abs(x - la_ref)) if n_clamped == 0 else np.nan
        if n_clamped == 0:
            n_inactive += 1

        # Analytic vs finite-difference Jacobians. NNLS is piecewise smooth; where the active
        # set differs between the + and - perturbations the derivative genuinely does not
        # exist, so those columns are counted and excluded rather than reported as error.
        def report(analytic, z, fn, active_of):
            fd = fd_jacobian(fn, z)
            base_set = active_of(z)
            steps = 1e-6 * np.maximum(1.0, np.abs(z))
            kink = []
            for k in range(len(z)):
                dz = np.zeros(len(z))
                dz[k] = steps[k]
                if not (
                    np.array_equal(active_of(z + dz), base_set)
                    and np.array_equal(active_of(z - dz), base_set)
                ):
                    kink.append(k)
            keep = np.setdiff1d(np.arange(len(z)), kink)
            if len(keep) == 0:
                return np.nan, len(kink)
            scale = max(np.max(np.abs(fd[:, keep])), 1e-12)
            return float(np.max(np.abs(analytic[:, keep] - fd[:, keep])) / scale), len(kink)

        act_q = lambda qq: ctrl.la_tau(t, qq, u) <= f_min + 1e-9
        act_u = lambda uu: ctrl.la_tau(t, q, uu) <= f_min + 1e-9
        err_q, kink_q = report(
            ctrl.la_tau_q(t, q, u), q, lambda qq: ctrl.la_tau(t, qq, u), act_q
        )
        err_u, kink_u = report(
            ctrl.la_tau_u(t, q, u), u, lambda uu: ctrl.la_tau(t, q, uu), act_u
        )

        print(
            f"  s={s:8.1e}  min(la_ref)={la_ref.min():9.3f}  min(x)={x.min():8.3f}  "
            f"clamped={n_clamped}  KKT=({primal:.1e},{stat:.1e},{comp:.1e})  "
            f"dq={err_q:.2e}({kink_q} kinks)  du={err_u:.2e}({kink_u} kinks)"
            + ("" if n_clamped else f"  |x-la_ref|={exact:.1e}")
        )

        # Inactive regime: drop the bound below the baseline minimum. The QP must then be a
        # no-op and both Jacobians must reduce exactly to DynamicControllerPD's.
        if n_inactive == 0:
            f_lo, ctrl.f_min = ctrl.f_min, la_ref.min() - 1.0
            d_x = np.max(np.abs(ctrl.la_tau(t, q, u) - la_ref))
            d_q = np.max(np.abs(ctrl.la_tau_q(t, q, u) - ctrl._base_la_tau_q(t, q, u)))
            d_u = np.max(np.abs(ctrl.la_tau_u(t, q, u) - ctrl._base_la_tau_u(t, q, u)))
            ctrl.f_min = f_lo
            # Only metric="W_tau" has full column rank, so only it has a unique unconstrained
            # minimizer that must coincide with the parent's damped pseudo-inverse. The 3-row
            # metrics leave the 1-D nullspace free by design, so a mismatch there is the
            # intended behaviour, not an error.
            verdict = "must be ~0" if metric == "W_tau" else "nullspace-free by design"
            print(
                f"    inactive-bound regime (f_min={la_ref.min() - 1.0:.3f}, {verdict}): "
                f"|x-la_ref|={d_x:.1e}  |dq-base|={d_q:.1e}  |du-base|={d_u:.1e}"
            )
            n_inactive += 1


## ---------------------------------------------------------------- replay mode


def replay(t_sim, dt, f_min):
    """Simulate the unconstrained law once, then project every sample offline."""
    model, ctrl, r_OP_ref_fn = build(f_min=f_min, bypass=True)
    print(f"baseline (unconstrained) run: t_sim={t_sim}, dt={dt}")
    sol = ScipyDAE(model.system, t_sim, dt).solve()

    ctrl.bypass = False
    results = {}
    for metric in METRICS:
        r = replay_qp(ctrl, sol, metric=metric)
        results[metric] = r
        la_ref, la_qp = r["la_ref"], r["la_qp"]
        neg = np.mean(la_ref.min(axis=1) < 0.0) * 100.0
        print(f"\n--- metric={metric} ---")
        print(f"  baseline min force over run : {la_ref.min():.4f} N ({neg:.1f}% of samples negative)")
        print(f"  projected min force over run: {la_qp.min():.4f} N")
        print(f"  generalized-force error ||W_tau (x - la_ref)||: mean {r['e_gen'].mean():.4e}  max {r['e_gen'].max():.4e}")
        print(f"  tip-acceleration error  ||J (x - la_ref)||    : mean {r['e_task'].mean():.4e}  max {r['e_task'].max():.4e}")
        print(f"    axial   component: mean {r['e_ax'].mean():.4e}  max {r['e_ax'].max():.4e}")
        print(f"    lateral component: mean {r['e_lat'].mean():.4e}  max {r['e_lat'].max():.4e}")
        print(f"  tendons clamped at f_min: mean {r['n_clamped'].mean():.2f} of {ctrl.nla_tau}")

    replay_plot(sol, results, ctrl.nla_tau, f_min)
    return sol, results


def replay_plot(sol, results, n_tendons, f_min):
    import matplotlib.pyplot as plt

    ts = sol.t
    fig, axes = plt.subplots(3, 1, num="QP replay", figsize=(9, 9), sharex=True)

    ax = axes[0]
    for k in range(n_tendons):
        ax.plot(ts, results["W_tau"]["la_ref"][:, k], lw=0.8, label=f"tendon {k+1}")
    ax.axhline(f_min, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Force [N]")
    ax.set_title("Unconstrained la_tau (baseline)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True)

    ax = axes[1]
    for k in range(n_tendons):
        ax.plot(ts, results["W_tau"]["la_qp"][:, k], lw=0.8, label=f"tendon {k+1}")
    ax.axhline(f_min, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Force [N]")
    ax.set_title("QP-projected la_tau (metric = W_tau)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True)

    ax = axes[2]
    for metric in results:
        ax.plot(ts, results[metric]["e_task"], lw=0.9, label=f"{metric}: total")
        ax.plot(ts, results[metric]["e_ax"], lw=0.7, alpha=0.5, ls="--", label=f"{metric}: axial")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|J(x-\lambda_\tau)\|$  [m/s$^2$]")
    ax.set_yscale("log")
    ax.set_title("Induced tip-acceleration error (dashed = axial component)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4, which="both")

    fig.tight_layout()
    path = OUT / "qp_positive_replay.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")
    plt.show()


## ------------------------------------------------------------------- run mode


def one_run(label, t_sim, dt, metric, f_min, baseline, names, t_hold, cache=True):
    """Simulate one configuration and return everything the plots need.

    Results are cached to .npz: a 10 s run costs several minutes, and comparison plots are
    usually re-made several times with different config sets.
    """
    tag = f"{'baseline' if baseline else metric}_f{f_min:g}_t{t_sim:g}_h{t_hold:g}_{'-'.join(names)}"
    cache_file = OUT / "qp_cache" / f"{tag}.npz"
    if cache and cache_file.exists():
        d = dict(np.load(cache_file, allow_pickle=True))
        d["label"] = label
        d["f_min"] = None if baseline else f_min
        d["t_sim"] = t_sim  # caches written before t_sim was stored
        print(f"[{label}] cached: {cache_file.name}", flush=True)
        _report(d)
        return d

    model, ctrl, r_OP_ref_fn = build(
        metric=metric, f_min=f_min, bypass=baseline, names=names, t_hold=t_hold
    )
    print(f"[{label}] solving: t_sim={t_sim}, dt={dt}, Kp={KP}, Kd={KD}", flush=True)
    sol = ScipyDAE(model.system, t_sim, dt).solve()

    la_ts = np.array(
        [ctrl.la_tau(t, q[ctrl.qDOF], u[ctrl.uDOF]) for t, q, u in zip(sol.t, sol.q, sol.u)]
    )
    r_OP = sol.q[:, model.rod.qDOF].reshape((-1, model.rod.nnode, 7))[:, -1, 0:3]
    r_ref = np.array([r_OP_ref_fn(t) for t in sol.t])
    e = np.linalg.norm(r_ref - r_OP, axis=1)

    # Settled error: last 25% of each hold window, which excludes the step transients that
    # otherwise dominate the mean and makes the configurations comparable.
    seg = np.floor(sol.t / t_hold).astype(int)
    settled = np.concatenate(
        [
            np.where(seg == s)[0][-max(1, int(0.25 * np.sum(seg == s))):]
            for s in np.unique(seg)
        ]
    )

    out = dict(
        label=label, t=sol.t, la=la_ts, r_OP=r_OP, r_ref=r_ref, e=e,
        f_min=f_min if not baseline else None, settled=settled,
        n_tendons=model.n_tendons, t_hold=t_hold, t_sim=t_sim,
    )
    if cache:
        cache_file.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache_file, **{k: v for k, v in out.items() if k != "label"})
    _report(out)
    return out


def truncated(r):
    """Did the solver stop before t_sim? ScipyDAE returns a short solution rather than raising,
    so without this check a run that died at t=4 is silently averaged as if it had finished."""
    return float(r["t"][-1]) < 0.999 * float(r["t_sim"])


def _report(r):
    warn = f"  *** TRUNCATED at t={r['t'][-1]:.3f} of {float(r['t_sim']):.3g} ***" if truncated(r) else ""
    print(
        f"[{r['label']}] min force {r['la'].min():8.4f} N | "
        f"negative samples {np.mean(r['la'].min(axis=1) < -1e-9) * 100:5.1f}% | "
        f"error mean {r['e'].mean() * 1e3:7.3f} mm, "
        f"settled {r['e'][r['settled']].mean() * 1e3:7.3f} mm{warn}",
        flush=True,
    )


def compare_plot(runs, tag):
    """Overlay tracking error and show each configuration's tendon forces."""
    import matplotlib.pyplot as plt

    n = len(runs)
    fig, axes = plt.subplots(
        n + 2, 1, figsize=(10, 3 + 2.4 * (n + 1)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.2] + [1.6] * n},
    )

    ax = axes[0]
    for i, name in enumerate("XYZ"):
        ax.plot(runs[0]["t"], runs[0]["r_ref"][:, i], "k--", lw=1.0,
                label="reference" if i == 0 else None)
    for r in runs:
        ax.plot(r["t"], r["r_OP"][:, 1], lw=1.0, label=f"{r['label']} (Y)")
    ax.set_ylabel("Tip position [m]")
    ax.set_title("Tip tracking (all reference components dashed; Y component shown per run)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    for r in runs:
        ax.plot(r["t"], r["e"] * 1e3, lw=1.0, label=r["label"])
    ax.set_ylabel("Tracking error [mm]")
    ax.set_yscale("log")
    ax.set_title("Tip tracking error")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4, which="both")

    for ax, r in zip(axes[2:], runs):
        for k in range(r["n_tendons"]):
            ax.plot(r["t"], r["la"][:, k], lw=0.7, label=f"tendon {k+1}")
        ax.axhline(0.0, color="k", lw=0.8)
        if r["f_min"] is not None:
            ax.axhline(r["f_min"], color="r", ls="--", lw=0.9,
                       label=f"f_min = {r['f_min']:g} N")
        lo = min(r["la"].min(), 0.0)
        ax.set_ylim(lo - 0.15 * abs(lo) - 0.1, r["la"].max() * 1.1 + 0.1)
        ax.set_ylabel("Force [N]")
        if truncated(r):
            ax.axvspan(r["t"][-1], float(r["t_sim"]), color="0.85", zorder=0)
            ax.text(
                0.5 * (r["t"][-1] + float(r["t_sim"])), ax.get_ylim()[1] * 0.5,
                "solver died", ha="center", fontsize=8, color="0.35",
            )
        ax.set_title(
            f"{r['label']}   (min {r['la'].min():.3f} N, "
            f"{np.mean(r['la'].min(axis=1) < -1e-9) * 100:.1f}% negative"
            + (f", DIED at t={r['t'][-1]:.2f}" if truncated(r) else "") + ")",
            fontsize=9,
        )
        ax.legend(fontsize=6, ncol=5)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()
    path = OUT / f"qp_positive_{tag}.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")
    plt.show()
    return path


def run(t_sim, dt, metric, f_min, baseline, names, t_hold, vis):
    label = "baseline (unconstrained)" if baseline else f"QP {metric}, f_min={f_min:g}"
    r = one_run(label, t_sim, dt, metric, f_min, baseline, names, t_hold)
    compare_plot([r], f"run_{'baseline' if baseline else metric}")
    return r


def compare(t_sim, dt, names, t_hold, configs):
    runs = [one_run(lab, t_sim, dt, m, f, b, names, t_hold) for lab, m, f, b in configs]
    print(f"\n{'configuration':<34}{'min force':>11}{'% neg':>8}{'mean err':>11}{'settled err':>13}")
    for r in runs:
        print(
            f"{r['label']:<34}{r['la'].min():>10.4f}N"
            f"{np.mean(r['la'].min(axis=1) < -1e-9) * 100:>7.1f}%"
            f"{r['e'].mean() * 1e3:>9.3f}mm{r['e'][r['settled']].mean() * 1e3:>11.3f}mm"
            + (f"   DIED at t={r['t'][-1]:.2f}" if truncated(r) else "")
        )
    compare_plot(runs, f"compare_t{t_sim:g}")
    return runs


## ----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="KKT + finite-difference checks")
    mode.add_argument("--replay", action="store_true", help="offline projection of a baseline run")
    mode.add_argument("--run", action="store_true", help="closed loop with the QP in the path")
    mode.add_argument("--compare", action="store_true", help="baseline + every metric, one plot")

    p.add_argument("--metric", default="W_tau", choices=METRICS)
    p.add_argument("--f-min", type=float, default=0.0)
    p.add_argument("--t-sim", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--baseline", action="store_true", help="--run: unconstrained law instead")
    p.add_argument("--setpoints", default="A,B,C,D,E")
    p.add_argument("--t-hold", type=float, default=2.0)
    p.add_argument("--vis", action="store_true", help="--run: also open the VTK visualization")
    p.add_argument("--n-states", type=int, default=6, help="--check: number of test states")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = tuple(args.setpoints.split(","))

    if args.check:
        check(args.metric, args.f_min, args.n_states, args.seed)
    elif args.replay:
        replay(args.t_sim, args.dt, args.f_min)
    elif args.compare:
        compare(
            args.t_sim,
            args.dt,
            names,
            args.t_hold,
            [
                # metric="axial" is deliberately absent: its Jacobian freezes u_ax and is
                # wrong by 1-2 orders of magnitude, so it is --replay/--check only.
                ("baseline (unconstrained)", "W_tau", 0.0, True),
                ("QP W_tau, f_min=0", "W_tau", 0.0, False),
                ("QP W_tau, f_min=0.5", "W_tau", 0.5, False),
                ("QP J, f_min=0.5", "J", 0.5, False),
                ("QP whitened, f_min=0.5", "whitened", 0.5, False),
            ],
        )
    else:
        run(
            args.t_sim,
            args.dt,
            args.metric,
            args.f_min,
            args.baseline,
            names,
            args.t_hold,
            args.vis,
        )
