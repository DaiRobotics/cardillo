# SMC, closed loop, with and without the W_tau positivity QP
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE
from controllers.sliding_mode_controller import DynamicControllerSMC

CACHE = Path(__file__).parent / "cache"
OUT = Path(__file__).parent
NAMES = ["A", "B", "C", "D", "E"]
DAMPING_RATIO = 0.1
C_ERF = 40.0


def p2p(t_hold):
    pts = [SETPOINT_TABLE[n] for n in NAMES]
    return lambda t: pts[min(int(t // t_hold), len(pts) - 1)]


def run(tag, alpha, k, positive, t_sim=10.0, t_hold=2.0, dt=1e-4, cache=True,
        inv_damping=1e-3, c_erf=None, la_pre=0.0):
    # inv_damping and c belong in the cache key
    suffix = "" if (inv_damping == 1e-3 and c_erf is None and la_pre == 0.0) else f"_d{inv_damping:g}_c{(c_erf or C_ERF):g}_p{la_pre:g}"
    f = CACHE / f"smcqp_{tag}{suffix}_t{t_sim:g}_h{t_hold:g}_dt{dt:g}.npz"
    if cache and f.exists():
        d = dict(np.load(f, allow_pickle=True))
        print(f"[{tag}] cached", flush=True)
        _report(tag, d)
        return d

    model = CommonModel(damping_ratio=DAMPING_RATIO, la_pre=la_pre)
    ref = p2p(t_hold)
    # physical constraint is la_pre + la_tau >= 0, so the bound on la_tau is -la_pre
    c = DynamicControllerSMC(model.system, model.rod, model.tendons, ref,
                             v_P_ref_fn=lambda t: np.zeros(3), a_P_ref_fn=lambda t: np.zeros(3),
                             alpha=alpha, k=k, c=(c_erf or C_ERF), inv_damping=inv_damping,
                             positive=positive, f_min=-la_pre)
    model.system.add(c)
    model.system.assemble()

    print(f"[{tag}] solving alpha={alpha:g} k={k:g} inv_damping={inv_damping:g} "
          f"c={(c_erf or C_ERF):g} positive={positive}", flush=True)
    sol = ScipyDAE(model.system, t_sim, dt).solve()

    la = np.array([c.la_tau(t, q[c.qDOF], u[c.uDOF])
                   for t, q, u in zip(sol.t, sol.q, sol.u)])
    r = sol.q[:, model.rod.qDOF].reshape((-1, model.rod.nnode, 7))[:, -1, 0:3]
    rr = np.array([ref(t) for t in sol.t])
    e = np.linalg.norm(rr - r, axis=1)
    seg = np.floor(sol.t / t_hold).astype(int)
    # settled = last 25 % of each hold
    settled = np.concatenate(
        [np.where(seg == s)[0][-max(1, int(0.25 * np.sum(seg == s))):] for s in np.unique(seg)]
    )
    dtv = np.diff(sol.t)
    chatter = float(np.mean(np.abs(np.diff(la, axis=0)) / dtv[:, None])) if len(dtv) else 0.0

    d = dict(t=sol.t, la=la, r=r, r_ref=rr, e=e, settled=settled, t_sim=t_sim,
             alpha=alpha, k=k, positive=positive, chatter=chatter, la_pre=la_pre,
             n_tendons=model.n_tendons)
    if cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(f, **d)
    _report(tag, d)
    return d


def _report(tag, d):
    print(f"[{tag}] t_end = {float(d['t'][-1]):.2f}/{float(d['t_sim']):g}, "
          f"min force {d['la'].min():.4f} N, "
          f"neg {np.mean(d['la'].min(axis=1) < -1e-9) * 100:.1f}%, "
          f"mean err {d['e'].mean() * 1e3:.3f} mm, "
          f"settled err {d['e'][d['settled']].mean() * 1e3:.3f} mm, "
          f"chatter {float(d['chatter']):.1f} N/s", flush=True)


def plot(tag, d, title):
    import matplotlib.pyplot as plt

    t = d["t"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})
    for i, nm in enumerate("XYZ"):
        ax[i].plot(t, d["r_ref"][:, i] * 1e2, "b--", lw=1.2, label="desired")
        ax[i].plot(t, d["r"][:, i] * 1e2, "r", lw=1.0, label="actual")
        ax[i].set_ylabel(f"{nm} [cm]")
        ax[i].legend(fontsize=8)
        ax[i].grid(True, alpha=0.4)
    ax[0].set_title(title, fontsize=11)

    a = ax[3]
    lp = float(d.get("la_pre", 0.0))
    tot = d["la"] + lp
    for kk in range(int(d["n_tendons"])):
        a.plot(t, tot[:, kk], lw=0.9, label=f"tendon {kk+1}")
    a.axhline(0.0, color="k", lw=1.0)
    neg = np.mean(tot.min(axis=1) < -1e-9) * 100
    a.set_title(
        f"total tendon tension (la_pre={lp:g} N)  (min {tot.min():.4f} N, {neg:.1f}% negative, "
        f"chatter {float(d['chatter']):.1f} N/s)", fontsize=10)
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
    "a40_k80_unc": (40.0, 80.0, False, "SMC alpha=40 k=80 c=40, UNCONSTRAINED"),
    "a40_k80_qp": (40.0, 80.0, True, "SMC alpha=40 k=80 c=40 + W_tau QP"),
    "a20_k80_qp": (20.0, 80.0, True, "SMC alpha=20 k=80 c=40 + W_tau QP"),
    "a40_k160_qp": (40.0, 160.0, True, "SMC alpha=40 k=160 c=40 + W_tau QP"),
}

if __name__ == "__main__":
    # ---- parameters ----
    want = set(CONFIGS)
    # want = {"a20_k80_qp"}
    t_sim = 10.0
    t_hold = 2.0
    dt = 1e-4

    res = {}
    for tag, (al, kk, pos, title) in CONFIGS.items():
        if tag not in want:
            continue
        d = run(tag, al, kk, pos, t_sim, t_hold, dt)
        plot(tag, d, title)
        res[tag] = d

    if len(res) == len(CONFIGS):
        print()
        for tag in CONFIGS:
            d = res[tag]
            print(f"{tag}: min force {d['la'].min():.4f} N, "
                  f"neg {np.mean(d['la'].min(axis=1) < -1e-9) * 100:.1f}%, "
                  f"mean err {d['e'].mean() * 1e3:.3f} mm, "
                  f"settled err {d['e'][d['settled']].mean() * 1e3:.3f} mm, "
                  f"chatter {float(d['chatter']):.1f} N/s")
        print("figures in", OUT)
