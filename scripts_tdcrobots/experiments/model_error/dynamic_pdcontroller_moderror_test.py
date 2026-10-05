# dynamic PD on the five setpoint stair with a wrong model in the controller
# err_h scales h, err_w scales W_tau, err_m scales M_tilde_inv, 1.0 = exact, plant stays exact

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from experiments.dynamic.dynamic_pdcontroller_test_main import (
    DynamicControllerPD, rod_q0_at, p2p_sequence, smooth_p2p_sequence, report,
)


class _WrongH:
    # proxy over the system that hands the controller a scaled h (and matching h_q, h_u)

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


class DynamicControllerPDError(DynamicControllerPD):
    # same control law, only the controller's h, W_tau and M_tilde_inv are scaled

    def __init__(self, *args, err_h=1.0, err_w=1.0, err_m=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.err_h = err_h
        self.err_w = err_w
        self.err_m = err_m
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
            self._m_scaled = True  # base class caches, scale once

    def W_tau(self, t, q):
        return self.err_w * super().W_tau(t, q)

    def W_tau_q(self, t, q):
        return self.err_w * super().W_tau_q(t, q)


def simulate(
    names=("A", "B", "C", "D", "E"),
    smooth=False,
    t_move=0.5,
    t_hold=1.5,
    t_sim=None,
    dt=1e-3,
    Kp=45.0,
    Kd=13.416,
    damping_ratio=0.1,
    la_pre=0.0,
    inv_damping=1e-3,
    positive=True,
    start_point="E",
    err_h=1.0,
    err_w=1.0,
    err_m=1.0,
    verbose=True,
):
    names = list(names)
    t_sim = t_hold * len(names) if t_sim is None else t_sim
    model = CommonModel(damping_ratio=damping_ratio, la_pre=la_pre)
    system, rod, tendons = model.system, model.rod, model.tendons

    if start_point is not None:
        rod.q0 = rod_q0_at(start_point).copy()
    r_start = rod.q0[rod.nodalDOF_r[-1]]

    r_fn, v_fn, a_fn = p2p_sequence(names, t_hold=t_hold)
    refs = {"r": r_fn, "v": v_fn, "a": a_fn}
    if smooth:
        refs["r"], refs["v"], refs["a"] = smooth_p2p_sequence(
            names, t_hold=t_hold, t_move=t_move, r_start=r_start
        )

    if verbose:
        print("trajectory: " + " -> ".join(names)
              + (f", quintic blends t_move={t_move}" if smooth else ", stepped")
              + f", t_hold={t_hold}, t_sim={t_sim}")
        print(f"  model error: err_h={err_h}, err_w={err_w}, err_m={err_m}")

    controller = DynamicControllerPDError(
        system, rod, tendons,
        lambda t: refs["r"](t),
        v_P_ref_fn=lambda t: refs["v"](t),
        a_P_ref_fn=lambda t: refs["a"](t),
        Kp=Kp, Kd=Kd, inv_damping=inv_damping, positive=positive,
        err_h=err_h, err_w=err_w, err_m=err_m,
    )
    system.add(controller)
    system.assemble()

    sol = ScipyDAE(system, t_sim, dt).solve()
    return sol, controller, rod, refs, model


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    smooth = False
    t_move = 0.5
    t_hold = 5.0
    t_sim = t_hold * len(names)
    dt = 1e-3
    Kp = 45.0
    Kd = 2 * np.sqrt(Kp)  # critical damping
    damping_ratio = 0.1
    la_pre = 0.0
    inv_damping = 1e-3
    positive = True
    start_point = "E"

    # ---- model error (1.0 = exact) ----
    err_h = 0.9
    err_w = 1.0
    err_m = 1.0
    compare_exact = True  # run the exact model alongside

    show_3d = False

    kwargs = dict(
        names=names, smooth=smooth, t_move=t_move, t_hold=t_hold, t_sim=t_sim, dt=dt,
        Kp=Kp, Kd=Kd, damping_ratio=damping_ratio, la_pre=la_pre,
        inv_damping=inv_damping, positive=positive, start_point=start_point,
    )

    print(f"dynamic PD, positive={positive}, Kp={Kp}, Kd={Kd}, g={G_ACCEL}")

    runs = []
    for eh, ew, em in ([(1.0, 1.0, 1.0), (err_h, err_w, err_m)] if compare_exact
                       else [(err_h, err_w, err_m)]):
        sol, c, rod, refs, model = simulate(err_h=eh, err_w=ew, err_m=em, **kwargs)
        lbl = "exact model" if (eh, ew, em) == (1.0, 1.0, 1.0) else "error model"
        r_OP, r_ref, e, la = report(lbl, sol, c, rod, refs, names, t_hold, smooth, t_move)
        runs.append((lbl, sol, r_OP, r_ref, e, la))

    # ---- visualization ----
    import matplotlib.pyplot as plt

    out = Path(__file__).parent
    tag = (("_qp" if positive else "_plain")
           + (f"_smooth{t_move:g}" if smooth else "_step")
           + (f"_start{start_point}" if start_point else "")
           + f"_hold{t_hold:g}"
           + f"_errh{err_h:g}w{err_w:g}m{err_m:g}")
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
    fig.savefig(out / f"moderr_tip_tracking{tag}.png", dpi=150)

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
    fig.savefig(out / f"moderr_tracking_error{tag}.png", dpi=150)

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
    fig.savefig(out / f"moderr_tendon_forces{tag}.png", dpi=150)

    print("figures written to", out)

    if show_3d:
        from model.tdcm_li2023 import rod_visualization

        rod_visualization(model, runs[-1][1])

    plt.show()
