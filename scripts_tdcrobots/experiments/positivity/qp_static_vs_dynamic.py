# static vs dynamic controller, 10 s and 25 s, both with the positivity QP
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from controllers.dynamic_controller import DynamicControllerPD
from controllers.static_controller_qp import StaticController

OUT = Path(__file__).parent / "static vs dynamic"
CACHE = Path(__file__).parent / "cache"
NAMES = ["A", "B", "C", "D", "E"]

# each controller's own established gains
KP_DYN, KD_DYN, INV_DAMPING, DT_DYN = 200.0, 20.0, 1e-3, 1e-4
KP_STA, KD_STA, DT_STA = 2.0, 0.0, 1e-3
LA_T0 = np.array([0.5, 0.0, 0.0, 0.0])
J_STAT_CHECK_DT = 1e-2
DAMPING_RATIO = 0.1


def p2p(t_hold):
    pts = [SETPOINT_TABLE[n] for n in NAMES]
    return lambda t: pts[min(int(t // t_hold), len(pts) - 1)]


def run(tag, host, t_sim, t_hold, cache=True):
    f = CACHE / f"svd_{tag}.npz"
    if cache and f.exists():
        d = dict(np.load(f, allow_pickle=True))
        print(f"[{tag}] cached", flush=True)
        _report(tag, d)
        return d

    model = CommonModel(damping_ratio=DAMPING_RATIO, la_pre=0.0)
    ref = p2p(t_hold)
    if host == "dynamic":
        dt = DT_DYN
        c = DynamicControllerPD(model.system, model.rod, model.tendons, ref,
                                v_P_ref_fn=lambda t: np.zeros(3), a_P_ref_fn=lambda t: np.zeros(3),
                                Kp=KP_DYN, Kd=KD_DYN, inv_damping=INV_DAMPING, positive=True)
        model.system.add(c)
        model.system.assemble()
    else:
        dt = DT_STA
        c = StaticController(model.system, model.rod, model.tendons, ref,
                             v_P_ref_fn=lambda t: np.zeros(3), Kp=KP_STA, Kd=KD_STA, la_t0=LA_T0,
                             model_factory=CommonModel, g_accel=G_ACCEL,
                             J_stat_check_dt=J_STAT_CHECK_DT, positive=True)
        model.system.add(c)
        model.system.assemble()
        # start from the static equilibrium at la_t0
        q0 = np.concatenate((c.static_model_twin.q_eq(), c.q0))
        model.system.set_new_initial_state(q0, np.zeros(model.system.nu))

    print(f"[{tag}] solving t_sim={t_sim}, dt={dt}", flush=True)
    sol = ScipyDAE(model.system, t_sim, dt).solve()

    la = np.array([c.la_tau(t, q[c.qDOF], u[c.uDOF]) for t, q, u in zip(sol.t, sol.q, sol.u)])
    r = sol.q[:, model.rod.qDOF].reshape((-1, model.rod.nnode, 7))[:, -1, 0:3]
    rr = np.array([ref(t) for t in sol.t])
    e = np.linalg.norm(rr - r, axis=1)
    seg = np.floor(sol.t / t_hold).astype(int)
    # settled = last 25 % of each hold
    settled = np.concatenate(
        [np.where(seg == s)[0][-max(1, int(0.25 * np.sum(seg == s))):] for s in np.unique(seg)]
    )
    d = dict(t=sol.t, la=la, r=r, r_ref=rr, e=e, settled=settled, t_sim=t_sim,
             t_hold=t_hold, host=host, n_tendons=model.n_tendons, dt=dt)
    if cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(f, **d)
    _report(tag, d)
    return d


def _report(tag, d):
    print(f"[{tag}] t_end = {float(d['t'][-1]):.3f}/{float(d['t_sim']):g}, "
          f"min force {d['la'].min():.4f} N, "
          f"neg {np.mean(d['la'].min(axis=1) < -1e-9) * 100:.1f}%, "
          f"settled err {d['e'][d['settled']].mean() * 1e3:.3f} mm", flush=True)


def plot(tag, d, title):
    import matplotlib.pyplot as plt

    t = d["t"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})

    for i, name in enumerate("XYZ"):
        a = ax[i]
        a.plot(t, d["r_ref"][:, i] * 1e2, "b--", lw=1.2, label="desired")
        a.plot(t, d["r"][:, i] * 1e2, "r", lw=1.2, label="actual")
        a.set_ylabel(f"{name} [cm]")
        a.legend(fontsize=8, loc="best")
        a.grid(True, alpha=0.4)
    ax[0].set_title(title, fontsize=11)

    a = ax[3]
    for k in range(int(d["n_tendons"])):
        a.plot(t, d["la"][:, k], lw=1.0, label=f"tendon {k+1}")
    a.axhline(0.0, color="k", lw=1.0)
    neg = np.mean(d["la"].min(axis=1) < -1e-9) * 100
    a.set_title(f"tendon forces   (min {d['la'].min():.4f} N, {neg:.1f}% negative)", fontsize=10)
    a.set_xlabel("Time [s]")
    a.set_ylabel("Tendon force [N]")
    a.legend(fontsize=8, ncol=4)
    a.grid(True, alpha=0.4)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("  wrote", p)


CONFIGS = {
    "dynamic_10s": ("dynamic", 10.0, 2.0, "Dynamic PD + QP, 10 s (2 s per setpoint)"),
    "dynamic_25s": ("dynamic", 25.0, 5.0, "Dynamic PD + QP, 25 s (5 s per setpoint)"),
    "static_10s": ("static", 10.0, 2.0, "Static (Newton) + QP, 10 s (2 s per setpoint)"),
    "static_25s": ("static", 25.0, 5.0, "Static (Newton) + QP, 25 s (5 s per setpoint)"),
}

if __name__ == "__main__":
    # ---- parameters ----
    want = set(CONFIGS)
    # want = {"static_10s"}

    results = {}
    for tag, (host, t_sim, t_hold, title) in CONFIGS.items():
        if tag not in want:
            continue
        d = run(tag, host, t_sim, t_hold)
        plot(tag, d, title)
        results[tag] = d

    if len(results) == len(CONFIGS):
        print()
        for tag in CONFIGS:
            d = results[tag]
            print(f"{tag}: min force {d['la'].min():.4f} N, "
                  f"neg {np.mean(d['la'].min(axis=1) < -1e-9) * 100:.1f}%, "
                  f"settled err {d['e'][d['settled']].mean() * 1e3:.3f} mm, "
                  f"dt {float(d['dt']):.0e}")
        print("figures in", OUT)
