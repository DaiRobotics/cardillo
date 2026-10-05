# shared helpers for the tip disturbance runs: hidden force, summary, plots

import numpy as np

from cardillo.forces import Force


class _HiddenForce:
    # proxy over the system that removes one force contribution from h

    def __init__(self, system, dist):
        self._sys = system
        self._d = dist

    def __getattr__(self, name):
        return getattr(self._sys, name)

    def h(self, t, q, u):
        out = np.array(self._sys.h(t, q, u), dtype=float, copy=True)
        d = self._d
        out[d.uDOF] -= d.h(t, q[d.qDOF], u[d.uDOF])
        return out


def disturbance_fn(shape="pulse", direction=(0.0, -1.0, 0.0), f_mag=0.003,
                   t_on=2.3, t_off=2.9, f_hz=5.0):
    # pulse: constant in [t_on, t_off], step: constant from t_on, sine: f_mag sin(2 pi f_hz (t - t_on))
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)

    def f(t):
        if t < t_on or (shape != "step" and t > t_off):
            return np.zeros(3)
        if shape == "sine":
            return f_mag * np.sin(2 * np.pi * f_hz * (t - t_on)) * d
        return f_mag * d

    return f


def add_tip_disturbance(system, rod, **kwargs):
    # point force at the rod tip, returns (contribution, f(t))
    f = disturbance_fn(**kwargs)
    dist = Force(f, rod, xi=1.0, name="tip_disturbance")
    system.add(dist)
    return dist, f


def summarize(label, t, r_OP, r_ref, la, t_on, t_off, shape, setpoints=None, names=None,
              t_hold=None, recover_tol=1e-3, smooth=False, t_move=0.0):
    e = np.linalg.norm(r_OP - r_ref, axis=1)
    pre = t < t_on
    during = (t >= t_on) & (t <= t_off)
    after = t > t_off

    print(f"\n--- {label} ---")
    if pre.any():
        print(f"  |e| before the disturbance: {e[pre][-1] * 1e3:.4f} mm")
    if during.any():
        print(f"  peak |e| while it acts: {e[during].max() * 1e3:.4f} mm")
        print(f"  |e| at the end of the window: {e[during][-1] * 1e3:.4f} mm")
    if shape != "step" and after.any():
        idx = np.where(e[after] < recover_tol)[0]
        if len(idx):
            t_rec = t[after][idx[0]]
            print(f"  recovered below {recover_tol * 1e3:g} mm at t = {t_rec:.3f} s "
                  f"({t_rec - t_off:.3f} s after release)")
            if t_hold is not None and t_rec > (np.floor(t_off / t_hold) + 1) * t_hold:
                print("    (warning: a setpoint change happened first)")
        else:
            print(f"  never returned below {recover_tol * 1e3:g} mm within the run")
    print(f"  tension range: [{la.min():.3f}, {la.max():.3f}] N")
    if setpoints is not None and names is not None and t_hold is not None:
        print("  settled error at the end of each hold:")
        for k, name in enumerate(names):
            # the blend is centred on the setpoint change, so the hold ends t_move/2 early
            t_end = (k + 1) * t_hold - (0.5 * t_move if smooth else 0.0)
            i = min(np.searchsorted(t, t_end) - 1, len(t) - 1)
            err = np.linalg.norm(r_OP[i] - setpoints[name])
            print(f"    {name} (t = {t[i]:.2f} s): {err * 1e3:.4e} mm")
    return e


def plot_disturbance(out_dir, prefix, tag, runs, f_dist, t_on, t_hi, marks):
    # runs is a list of (label, t, r_OP, r_ref, e, la)
    import matplotlib.pyplot as plt

    def shade(ax):
        ax.axvspan(t_on, t_hi, color="orange", alpha=0.15, lw=0)
        for tm in marks:
            ax.axvline(tm, color="0.7", ls=":", lw=0.8)

    fig, axs = plt.subplots(3, 1, num=f"{prefix}XYZ", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("XYZ"):
        axs[i].plot(runs[-1][1], runs[-1][3][:, i] * 1e2, "b--", lw=1.0, label="desired")
        for label, t_k, r_OP_k, _, _, _ in runs:
            axs[i].plot(t_k, r_OP_k[:, i] * 1e2, lw=1.0, label=label)
        shade(axs[i])
        axs[i].set_ylabel(f"{lbl} [cm]")
        axs[i].legend(fontsize=8)
        axs[i].grid(True)
    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}tip_tracking{tag}.png", dpi=150)

    fig, axs = plt.subplots(2, 1, num=f"{prefix}Error", figsize=(9, 6),
                            sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for label, t_k, _, _, e_k, _ in runs:
        axs[0].semilogy(t_k, np.maximum(e_k, 1e-12) * 1e3, lw=1.0, label=label)
    shade(axs[0])
    axs[0].set_ylabel(r"$||\boldsymbol{r}_{n}^{*} - \boldsymbol{r}_n||$ [mm]")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, which="both")
    f_prof = np.array([f_dist(t) for t in runs[-1][1]])
    for i, lbl in enumerate("XYZ"):
        axs[1].plot(runs[-1][1], f_prof[:, i], lw=1.0, label=f"$f_{{{lbl}}}$")
    shade(axs[1])
    axs[1].set_ylabel("disturbance [N]")
    axs[1].set_xlabel(r"$t$ [s]")
    axs[1].legend(fontsize=8)
    axs[1].grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}tracking_error{tag}.png", dpi=150)

    fig, axs = plt.subplots(len(runs), 1, num=f"{prefix}Forces",
                            figsize=(9, 4.0 * len(runs)), sharex=True, squeeze=False)
    for ax, (label, t_k, _, _, _, la_k) in zip(axs[:, 0], runs):
        for k in range(la_k.shape[1]):
            ax.plot(t_k, la_k[:, k], label=f"tendon {k + 1}")
        ax.axhline(0.0, color="k", lw=0.8)
        shade(ax)
        ax.set_ylabel(r"$\lambda_{\tau,i}$ [N]")
        ax.set_title(label, fontsize=9)
        ax.grid(True)
        ax.legend(ncol=4, fontsize=8)
    axs[-1, 0].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}tendon_forces{tag}.png", dpi=150)
    print("figures written to", out_dir)
