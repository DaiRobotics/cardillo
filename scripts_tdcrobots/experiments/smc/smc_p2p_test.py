# SMC, p2p A-B-C-D-E, starting from the equilibrium pose at E
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.special import erf

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, DATA_DIR
from controllers.sliding_mode_controller import DynamicControllerSMC

NAMES = ["A", "B", "C", "D", "E"]
START = "E"
MODES = ("plain", "qp")


class _WrongH:
    # proxy over the system that hands the controller a scaled h (plant stays true)
    def __init__(self, system, gamma):
        self._sys = system
        self._g = gamma

    def __getattr__(self, name):
        return getattr(self._sys, name)

    def h(self, t, q, u):
        return self._g * self._sys.h(t, q, u)

    def h_q(self, t, q, u):
        return self._g * self._sys.h_q(t, q, u)

    def h_u(self, t, q, u):
        return self._g * self._sys.h_u(t, q, u)


class SMCPositive(DynamicControllerSMC):
    # SMC with selectable positivity handling: plain or qp

    def __init__(self, *args, mode="qp", floor=0.0, err_h=1.0, err_w=1.0, err_m=1.0, **kwargs):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        kwargs["positive"] = mode == "qp"
        kwargs["f_min"] = floor
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.floor = floor
        # model error in the controller only
        self.err_h = err_h  # drift term
        self.err_w = err_w  # tendon force directions
        self.err_m = err_m  # inertia map
        self._wrong = _WrongH(self.system, err_h) if err_h != 1.0 else None

    def system_state(self, q, u=None):
        out = super().system_state(q, u)
        if self._wrong is None:
            return out
        return (self._wrong,) + tuple(out[1:])

    def build_M_tilde_inv(self, t, q_sys):
        super().build_M_tilde_inv(t, q_sys)
        if self.err_m != 1.0 and not getattr(self, "_m_scaled", False):
            self.M_tilde_inv = self.err_m * self.M_tilde_inv
            self._m_scaled = True  # base class caches, so scale once

    def W_tau(self, t, q):
        return self.err_w * super().W_tau(t, q)

    def W_tau_q(self, t, q):
        return self.err_w * super().W_tau_q(t, q)


def load_q0(name):
    # equilibrium rod configuration at a setpoint (columns q0_A .. q0_E)
    return pd.read_csv(DATA_DIR / "p2p_q0_gamma0.csv")[f"q0_{name}"].to_numpy().copy()


def p2p_sequence(names, t_hold):
    pts = [SETPOINT_TABLE[n] for n in names]
    return lambda t: pts[min(int(t // t_hold), len(pts) - 1)]


def run(mode="qp", alpha=40.0, k=80.0, c=40.0, f_min=0.0, la_pre=0.0, damping_ratio=0.1,
        inv_damping=1e-3, t_hold=2.0, t_sim=10.0, dt=1e-4, start=START, names=NAMES,
        err_h=1.0, err_w=1.0, err_m=1.0):
    model = CommonModel(damping_ratio=damping_ratio, la_pre=la_pre)
    if start is not None:
        # must precede system.assemble()
        model.rod.q0 = load_q0(start)

    r_OP_ref_fn = p2p_sequence(names, t_hold)
    floor = f_min - la_pre  # bound on la_tau, the physical bound is on la_pre + la_tau
    controller = SMCPositive(
        model.system, model.rod, model.tendons, r_OP_ref_fn,
        v_P_ref_fn=lambda t: np.zeros(3),
        a_P_ref_fn=lambda t: np.zeros(3),
        alpha=alpha, k=k, c=c, inv_damping=inv_damping,
        mode=mode, floor=floor, err_h=err_h, err_w=err_w, err_m=err_m,
    )
    model.system.add(controller)
    model.system.assemble()

    r_start = model.rod._view_nodal_q(model.rod.q0)[-1, :3]
    if start is not None:
        print("start pose", start, ": tip =", r_start,
              f", err = {np.linalg.norm(r_start - SETPOINT_TABLE[start]) * 1e3:.4f} mm")
    if (err_h, err_w, err_m) != (1.0, 1.0, 1.0):
        print(f"model error (controller only): h x{err_h:g}, W_tau x{err_w:g}, M_tilde_inv x{err_m:g}")
    print(f"solving mode={mode} alpha={alpha:g} k={k:g} c={c:g} floor={floor:g} N "
          f"la_pre={la_pre:g} N, t_sim={t_sim:g} t_hold={t_hold:g} dt={dt:g}", flush=True)

    sol = ScipyDAE(model.system, t_sim, dt).solve()

    # replay the allocation over the solved trajectory
    la = np.array([controller.la_tau(t, q[controller.qDOF], u[controller.uDOF])
                   for t, q, u in zip(sol.t, sol.q, sol.u)])

    r = sol.q[:, model.rod.qDOF].reshape((-1, model.rod.nnode, 7))[:, -1, 0:3]
    r_ref = np.array([r_OP_ref_fn(t) for t in sol.t])
    e = np.linalg.norm(r_ref - r, axis=1)

    # sliding variable, v_ref = a_ref = 0 here
    v_P = np.array([model.rod._view_nodal_u(u[model.rod.uDOF])[-1, :3] for u in sol.u])
    s = -v_P + alpha * (r_ref - r)
    sw = erf(c * s)  # switching term, +-1 is full authority

    seg = np.minimum(np.floor(sol.t / t_hold).astype(int), len(names) - 1)
    dtv = np.diff(sol.t)
    chatter = float(np.mean(np.abs(np.diff(la, axis=0)) / dtv[:, None])) if len(dtv) else 0.0

    return dict(t=sol.t, la=la, r=r, r_ref=r_ref, e=e, seg=seg, names=list(names),
                t_hold=t_hold, t_sim=t_sim, la_pre=la_pre, f_min=f_min, floor=floor,
                chatter=chatter, n_tendons=model.n_tendons, alpha=alpha, k=k, c=c, mode=mode,
                s=s, sw=sw,
                start=start, err_h=err_h, err_w=err_w, err_m=err_m)


def report(d):
    t = d["t"]
    print(f"\nreached t = {t[-1]:.3f} / {d['t_sim']:g} s")
    for s in np.unique(d["seg"]):
        idx = np.where(d["seg"] == s)[0]
        tail = idx[-max(1, int(0.25 * len(idx))):]  # settled = last 25 % of the hold
        print(f"  {d['names'][s]}: t = {t[idx[0]]:.1f}-{t[idx[-1]]:.1f} s, "
              f"mean err {d['e'][idx].mean() * 1e3:.3f} mm, "
              f"settled {d['e'][tail].mean() * 1e3:.3f} mm, "
              f"final {d['e'][idx[-1]] * 1e3:.3f} mm")

    tot = d["la"] + d["la_pre"]
    print(f"mean err {d['e'].mean() * 1e3:.3f} mm")
    print(f"min tension {tot.min():.4f} N, "
          f"{np.mean(tot.min(axis=1) < d['f_min'] - 1e-9) * 100:.1f}% of steps below the floor "
          f"({d['f_min']:g} N)")
    print(f"chatter {d['chatter']:.1f} N/s")


def _subtitle(d):
    err = ""
    if (d["err_h"], d["err_w"], d["err_m"]) != (1.0, 1.0, 1.0):
        err = f", model err h x{d['err_h']:g} W x{d['err_w']:g} M x{d['err_m']:g}"
    return (f"SMC alpha={d['alpha']:g} k={d['k']:g} c={d['c']:g}, mode={d['mode']}, "
            f"floor={d['f_min']:g} N, start {d['start']}{err}")


def _chatter_window(d, width=0.05):
    # window with the largest mean |d la / dt|
    t, la = d["t"], d["la"]
    if len(t) < 3:
        return float(t[0]), float(t[-1])
    rate = np.abs(np.diff(la, axis=0)).max(axis=1) / np.diff(t)
    n = max(2, int(width / max(np.median(np.diff(t)), 1e-12)))
    if n >= len(rate):
        return float(t[0]), float(t[-1])
    csum = np.concatenate([[0.0], np.cumsum(rate)])
    means = (csum[n:] - csum[:-n]) / n
    i = int(np.argmax(means))
    return float(t[i]), float(t[i + n])


def _mark_segments(ax, d):
    for k in range(1, len(d["names"])):
        ts = k * d["t_hold"]
        if ts > d["t"][-1]:
            break
        ax.axvline(ts, color="0.7", ls=":", lw=0.8)


def plot(d, tag, show=True, zoom=None):
    import matplotlib.pyplot as plt

    # keep labels as text in the svg
    plt.rcParams["svg.fonttype"] = "none"

    out = Path(__file__).parent
    t = d["t"]
    figs = []

    # ---- tendon forces ----
    fig, ax = plt.subplots(num="TendonForces", figsize=(9, 4.5))
    for k in range(int(d["n_tendons"])):
        ax.plot(t, d["la"][:, k], label=f"tendon {k + 1}")
    ax.axhline(0.0, color="k", ls="-", lw=0.8)
    if d["floor"]:
        ax.axhline(d["floor"], color="k", ls="--", lw=0.8, label=f"floor ({d['floor']:g} N)")
    _mark_segments(ax, d)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$\lambda_{\tau,i}$ [N]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    figs.append((fig, out / f"smc_p2p_tendon_forces_{tag}.png"))

    # ---- xyz tip tracking ----
    fig, axs = plt.subplots(3, 1, num="xyz", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(t, d["r_ref"][:, i], "b--", label="desired")
        axs[i].plot(t, d["r"][:, i], "r", label="actual")
        axs[i].set_ylabel(f"{lbl} [m]")
        axs[i].legend()
        axs[i].grid(True)
        _mark_segments(axs[i], d)
    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    figs.append((fig, out / f"smc_p2p_tip_tracking_{tag}.png"))

    # ---- tracking error ----
    fig, ax = plt.subplots(num="Error", figsize=(9, 4.0))
    ax.semilogy(t, np.maximum(d["e"], 1e-12) * 1e3, "r")
    _mark_segments(ax, d)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$||\boldsymbol{r}_{OP}^{*} - \boldsymbol{r}_{OP}||$ [mm]")
    ax.grid(True, which="both")
    fig.tight_layout()
    figs.append((fig, out / f"smc_p2p_tracking_error_{tag}.png"))

    # ---- chattering zoom ----
    t0, t1 = zoom if zoom else _chatter_window(d)
    w = (t >= t0) & (t <= t1)
    fig, ax = plt.subplots(3, 1, num="Chattering", figsize=(9, 9), sharex=True)
    for k in range(int(d["n_tendons"])):
        ax[0].plot(t[w], d["la"][w, k], marker=".", ms=2, label=f"tendon {k + 1}")
    ax[0].axhline(0.0, color="k", ls="-", lw=0.8)
    ax[0].set_ylabel(r"$\lambda_{\tau,i}$ [N]")
    ax[0].legend()
    ax[0].set_title(f"t = {t0:.4f}..{t1:.4f} s, {_subtitle(d)}", fontsize=10)

    for i, lbl in enumerate("xyz"):
        ax[1].plot(t[w], d["sw"][w, i], label=rf"$\mathrm{{erf}}(c\,s_{lbl})$")
    ax[1].axhline(0.0, color="k", ls="-", lw=0.8)
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_ylabel(r"$\mathrm{erf}(c\,s_i)$")
    ax[1].legend()
    # |s| <= 1/c is the boundary layer
    inside = np.mean(np.abs(d["s"][w]) < 1.0 / d["c"]) * 100
    ax[1].set_title(f"boundary layer $|s| < 1/c$ = {1.0 / d['c']:.2e} m/s: "
                    f"{inside:.1f}% of samples inside", fontsize=9)

    rate = np.abs(np.diff(d["la"], axis=0)).max(axis=1) / np.diff(t)
    ax[2].semilogy(t[1:][w[1:]], np.maximum(rate[w[1:]], 1e-6), "r")
    ax[2].set_ylabel(r"$\max_i |\dot{\lambda}_{\tau,i}|$ [N/s]")
    ax[2].set_xlabel(r"$t$ [s]")
    ax[2].set_title(f"run mean {d['chatter']:.1f} N/s, window peak {rate[w[1:]].max():.1f} N/s",
                    fontsize=9)
    for a_ in ax:
        a_.grid(True)
    fig.tight_layout()
    figs.append((fig, out / f"smc_p2p_chattering_{tag}.png"))

    for fig, path in figs:
        # png plus svg for inkscape
        fig.savefig(path, dpi=150)
        fig.savefig(path.with_suffix(".svg"))
        print("  wrote", path, "(+ .svg)")
    if show:
        plt.show()
    else:
        for fig, _ in figs:
            plt.close(fig)


if __name__ == "__main__":
    # ---- parameters ----
    mode = "qp"  # "plain" or "qp"
    alpha = 40.0  # alpha = 10 for qp
    k = 80.0  # k = 15 for qp
    c = 40.0  # erf steepness, 1/c = boundary layer
    f_min = 0.0  # floor on the total tension [N]
    la_pre = 0.0  # pretension per tendon [N]
    damping_ratio = 0.1
    t_hold = 2.0
    t_sim = 10.0
    dt = 1e-4
    start = START  # None starts from the straight rod
    err_h = 1.0  # controller's h scaled by this
    err_w = 1.0  # controller's W_tau scaled by this
    err_m = 1.0  # controller's M_tilde_inv scaled by this
    zoom = None  # (t0, t1) window for the chattering figure
    show = True

    d = run(mode=mode, alpha=alpha, k=k, c=c, f_min=f_min, la_pre=la_pre,
            damping_ratio=damping_ratio, t_hold=t_hold, t_sim=t_sim, dt=dt,
            start=start, err_h=err_h, err_w=err_w, err_m=err_m)
    report(d)

    # every knob that changes the result goes into the file name
    perfect = (err_h, err_w, err_m) == (1.0, 1.0, 1.0)
    etag = "" if perfect else f"_errh{err_h:g}w{err_w:g}m{err_m:g}"
    if damping_ratio != 0.1:
        etag += f"_damp{damping_ratio:g}"
    if la_pre:
        etag += f"_pre{la_pre:g}"
    if f_min:
        etag += f"_fmin{f_min:g}"
    plot(d, f"a{alpha:g}_k{k:g}_c{c:g}_{mode}_start{start}{etag}", show=show, zoom=zoom)
