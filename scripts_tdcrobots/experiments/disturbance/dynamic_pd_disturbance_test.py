# tip disturbance on the dynamic PD controller, exact model, force hidden from the controller

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL, compute_la_ts
from experiments.dynamic.dynamic_pdcontroller_test_main import (
    DynamicControllerPD, rod_q0_at, p2p_sequence, smooth_p2p_sequence,
)
from experiments.disturbance.disturbance_common import (
    _HiddenForce, add_tip_disturbance, summarize, plot_disturbance,
)


class DynamicControllerPDBlind(DynamicControllerPD):
    # PD controller that cannot see one force contribution in h

    def __init__(self, *args, disturbance=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._dist = disturbance

    def system_state(self, q, u=None):
        out = super().system_state(q, u)
        if self._dist is None:
            return out
        return (_HiddenForce(out[0], self._dist),) + tuple(out[1:])


def simulate(names=("A", "B", "C", "D", "E"), smooth=False, t_move=0.5, t_hold=1.5,
             t_sim=None, dt=1e-3, Kp=45.0, Kd=None, damping_ratio=0.1, la_pre=0.0,
             inv_damping=1e-3, positive=False, start_point="E",
             shape="pulse", direction=(0.0, -1.0, 0.0), f_mag=0.003,
             t_on=2.3, t_off=2.9, f_hz=5.0, known=False, verbose=True):
    names = list(names)
    Kd = 2.0 * np.sqrt(Kp) if Kd is None else Kd
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

    dist, f_dist = add_tip_disturbance(
        system, rod, shape=shape, direction=direction, f_mag=f_mag,
        t_on=t_on, t_off=t_off, f_hz=f_hz,
    )
    if verbose:
        window = f"[{t_on}, {t_off}]" if shape != "step" else f"[{t_on}, end]"
        print(f"disturbance: {shape}, |f| = {f_mag} N along {np.asarray(direction)}, "
              f"t in {window}, {'known to' if known else 'hidden from'} the controller")

    controller = DynamicControllerPDBlind(
        system, rod, tendons,
        lambda t: refs["r"](t),
        v_P_ref_fn=lambda t: refs["v"](t),
        a_P_ref_fn=lambda t: refs["a"](t),
        Kp=Kp, Kd=Kd, inv_damping=inv_damping, positive=positive,
        disturbance=(None if known else dist),
    )
    system.add(controller)
    system.assemble()

    sol = ScipyDAE(system, t_sim, dt).solve()
    return sol, controller, rod, refs, model, f_dist


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    smooth = False
    t_move = 0.5
    t_hold = 1.5
    t_sim = t_hold * len(names)
    dt = 1e-3
    Kp = 45.0
    Kd = 2 * np.sqrt(Kp)
    positive = False
    start_point = "E"

    # ---- disturbance ----
    shape = "pulse"
    # shape = "step"
    # shape = "sine"
    direction = (0.0, -1.0, 0.0)
    f_mag = 0.003
    t_on, t_off = 2.3, 2.9
    f_hz = 5.0
    known = False
    compare_clean = True

    kwargs = dict(names=names, smooth=smooth, t_move=t_move, t_hold=t_hold, t_sim=t_sim,
                  dt=dt, Kp=Kp, Kd=Kd, positive=positive, start_point=start_point,
                  shape=shape, direction=direction, t_on=t_on, t_off=t_off, f_hz=f_hz,
                  known=known)

    print(f"dynamic PD, exact model, Kp={Kp}, Kd={Kd:.4f}, positive={positive}, g={G_ACCEL}")

    runs, f_dist = [], None
    for fm in ([0.0, f_mag] if compare_clean else [f_mag]):
        sol, c, rod, refs, model, f_dist = simulate(f_mag=fm, **kwargs)
        r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
        r_ref = np.array([refs["r"](t) for t in sol.t])
        la = compute_la_ts(c, sol)
        lbl = "no disturbance" if fm == 0.0 else f"disturbance {fm:g} N"
        e = summarize(lbl, sol.t, r_OP, r_ref, la, t_on, t_off, shape,
                      setpoints=SETPOINT_TABLE, names=names, t_hold=t_hold)
        runs.append((lbl, sol.t, r_OP, r_ref, e, la))

    # ---- visualization ----
    out = Path(__file__).parent
    tag = (f"_{shape}_f{f_mag:g}" + ("_known" if known else "_hidden")
           + (f"_smooth{t_move:g}" if smooth else "_step"))
    plot_disturbance(out, "pd_", tag, runs, f_dist, t_on,
                     t_sim if shape == "step" else t_off,
                     [k * t_hold for k in range(1, len(names))])

    import matplotlib.pyplot as plt
    plt.show()
