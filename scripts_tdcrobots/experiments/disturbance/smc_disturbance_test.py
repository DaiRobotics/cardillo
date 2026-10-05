# tip disturbance on the SMC, exact model, force hidden from the controller

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from experiments.smc.smc_p2p_test import SMCPositive, load_q0, p2p_sequence
from experiments.disturbance.disturbance_common import (
    _HiddenForce, add_tip_disturbance, summarize, plot_disturbance,
)


class SMCBlind(SMCPositive):
    # SMC that cannot see one force contribution in h, wraps the model error proxy if any

    def __init__(self, *args, disturbance=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist = disturbance

    def system_state(self, q, u=None):
        out = super().system_state(q, u)
        if self._dist is None:
            return out
        return (_HiddenForce(out[0], self._dist),) + tuple(out[1:])


def simulate(names=("A", "B", "C", "D", "E"), t_hold=1.5, t_sim=None, dt=1e-3,
             alpha=40.0, k=80.0, c=40.0, mode="plain", f_min=0.0, la_pre=0.0,
             damping_ratio=0.1, inv_damping=1e-3, start_point="E",
             shape="pulse", direction=(0.0, -1.0, 0.0), f_mag=0.003,
             t_on=2.3, t_off=2.9, f_hz=5.0, known=False, verbose=True):
    names = list(names)
    t_sim = t_hold * len(names) if t_sim is None else t_sim
    model = CommonModel(damping_ratio=damping_ratio, la_pre=la_pre)
    system, rod, tendons = model.system, model.rod, model.tendons

    if start_point is not None:
        rod.q0 = load_q0(start_point)  # before assemble()

    r_OP_ref_fn = p2p_sequence(names, t_hold)
    refs = {"r": r_OP_ref_fn}

    dist, f_dist = add_tip_disturbance(
        system, rod, shape=shape, direction=direction, f_mag=f_mag,
        t_on=t_on, t_off=t_off, f_hz=f_hz,
    )
    if verbose:
        window = f"[{t_on}, {t_off}]" if shape != "step" else f"[{t_on}, end]"
        print(f"disturbance: {shape}, |f| = {f_mag} N along {np.asarray(direction)}, "
              f"t in {window}, {'known to' if known else 'hidden from'} the controller")

    controller = SMCBlind(
        system, rod, tendons, r_OP_ref_fn,
        v_P_ref_fn=lambda t: np.zeros(3),
        a_P_ref_fn=lambda t: np.zeros(3),
        alpha=alpha, k=k, c=c, inv_damping=inv_damping,
        mode=mode, floor=f_min - la_pre,
        disturbance=(None if known else dist),
    )
    system.add(controller)
    system.assemble()

    sol = ScipyDAE(system, t_sim, dt).solve()
    return sol, controller, rod, refs, model, f_dist


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    t_hold = 1.5
    t_sim = t_hold * len(names)
    dt = 1e-3
    alpha, k, c = 40.0, 80.0, 40.0
    mode = "plain"
    # mode = "qp"
    # mode = "nullspace"
    # mode = "both"
    f_min, la_pre = 0.0, 0.0
    start_point = "E"

    # ---- disturbance ----
    shape = "pulse"
    # shape = "step"
    # shape = "sine"
    direction = (0.0, -1.0, 0.0)
    f_mag = 0.003  # same force as the PD test
    t_on, t_off = 2.3, 2.9
    f_hz = 5.0
    known = False
    compare_clean = True

    kwargs = dict(names=names, t_hold=t_hold, t_sim=t_sim, dt=dt, alpha=alpha, k=k, c=c,
                  mode=mode, f_min=f_min, la_pre=la_pre, start_point=start_point,
                  shape=shape, direction=direction, t_on=t_on, t_off=t_off, f_hz=f_hz,
                  known=known)

    print(f"SMC mode='{mode}', exact model, alpha={alpha}, k={k}, c={c}, g={G_ACCEL}")

    runs, f_dist = [], None
    for fm in ([0.0, f_mag] if compare_clean else [f_mag]):
        sol, ctrl, rod, refs, model, f_dist = simulate(f_mag=fm, **kwargs)
        r_OP = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))[:, -1, 0:3]
        r_ref = np.array([refs["r"](t) for t in sol.t])
        la = np.array([ctrl.la_tau(t, q[ctrl.qDOF], u[ctrl.uDOF])
                       for t, q, u in zip(sol.t, sol.q, sol.u)])
        lbl = "no disturbance" if fm == 0.0 else f"disturbance {fm:g} N"
        e = summarize(lbl, sol.t, r_OP, r_ref, la, t_on, t_off, shape,
                      setpoints=SETPOINT_TABLE, names=names, t_hold=t_hold)
        runs.append((lbl, sol.t, r_OP, r_ref, e, la))

    # ---- visualization ----
    out = Path(__file__).parent
    tag = (f"_{mode}_{shape}_f{f_mag:g}" + ("_known" if known else "_hidden")
           + f"_a{alpha:g}k{k:g}c{c:g}")
    plot_disturbance(out, "smc_", tag, runs, f_dist, t_on,
                     t_sim if shape == "step" else t_off,
                     [k_ * t_hold for k_ in range(1, len(names))])

    import matplotlib.pyplot as plt
    plt.show()
