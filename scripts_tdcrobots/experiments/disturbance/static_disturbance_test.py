# tip disturbance on the static controller, exact model
# the static controller never reads h, so no hidden force proxy is needed

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from experiments.static.static_controller_test_main import (
    load_controller, p2p_sequence, smooth_p2p_sequence, applied_tensions,
)
from experiments.disturbance.disturbance_common import add_tip_disturbance, summarize, plot_disturbance


def simulate(names=("A", "B", "C", "D", "E"), smooth=False, t_move=0.5, t_hold=1.5,
             t_sim=None, dt=1e-3, Kp=2.0, Kd=0.0, damping_ratio=0.1,
             start_point="E", variant="plain", J_stat_check_dt=1e-3,
             shape="pulse", direction=(0.0, -1.0, 0.0), f_mag=0.003,
             t_on=2.3, t_off=2.9, f_hz=5.0, verbose=True):
    names = list(names)
    t_sim = t_hold * len(names) if t_sim is None else t_sim
    Controller, _ = load_controller(variant)

    model = CommonModel(damping_ratio=damping_ratio, la_pre=0.0)
    system, rod, tendons = model.system, model.rod, model.tendons

    # refs dict so the smooth reference can be swapped in once the start pose is known
    r_fn, v_fn, _ = p2p_sequence(names, t_hold=t_hold)
    refs = {"r": r_fn, "v": v_fn}

    controller = Controller(
        system, rod, tendons,
        lambda t: refs["r"](t),
        v_P_ref_fn=lambda t: refs["v"](t),
        Kp=Kp, Kd=Kd,
        la_t0=np.array([0.5, 0.0, 0.0, 0.0]),
        model_factory=CommonModel,
        g_accel=G_ACCEL,
        J_stat_check_dt=J_stat_check_dt,
    )

    # ---- start pose ----
    r_start = None
    if start_point is not None:
        la_t_start = controller.inverse_statics(
            SETPOINT_TABLE[start_point], la_t0=np.array([0.5, 0.0, 0.0, 0.0])
        )
        controller.q0 = la_t_start - controller.la_t_ff(0.0)
        controller.set_J_stat(controller.static_model_twin.solve_and_eval_J_stat(la_t_start))
        r_start = controller.static_model_twin.r_OP_eq()
        if verbose:
            print(f"start pose {start_point}: la_t = "
                  f"{np.array2string(la_t_start, precision=4)}, err = "
                  f"{np.linalg.norm(r_start - SETPOINT_TABLE[start_point]) * 1e3:.4f} mm")

    if smooth:
        refs["r"], refs["v"], _ = smooth_p2p_sequence(
            names, t_hold=t_hold, t_move=t_move,
            r_start=(r_start if r_start is not None
                     else controller.static_model_twin.r_OP_eq()),
        )

    _, f_dist = add_tip_disturbance(
        system, rod, shape=shape, direction=direction, f_mag=f_mag,
        t_on=t_on, t_off=t_off, f_hz=f_hz,
    )
    if verbose:
        window = f"[{t_on}, {t_off}]" if shape != "step" else f"[{t_on}, end]"
        print(f"disturbance: {shape}, |f| = {f_mag} N along {np.asarray(direction)}, "
              f"t in {window}")

    system.add(controller)
    system.assemble()

    q0 = np.concatenate((controller.static_model_twin.q_eq(), controller.q0))
    system.set_new_initial_state(q0, np.zeros(system.nu))

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
    Kp = 2.0
    Kd = 0.0
    variant = "plain"
    # variant = "qp"
    start_point = "E"

    # ---- disturbance ----
    shape = "pulse"
    # shape = "step"
    # shape = "sine"
    direction = (0.0, -1.0, 0.0)
    f_mag = 0.003  # tuned for the PD test at Kp = 45, the static controller is much more sensitive
    t_on, t_off = 2.3, 2.9
    f_hz = 5.0
    compare_clean = True

    kwargs = dict(names=names, smooth=smooth, t_move=t_move, t_hold=t_hold, t_sim=t_sim,
                  dt=dt, Kp=Kp, Kd=Kd, variant=variant, start_point=start_point,
                  shape=shape, direction=direction, t_on=t_on, t_off=t_off, f_hz=f_hz)

    print(f"static controller '{variant}', exact model, Kp={Kp}, Kd={Kd}, g={G_ACCEL}")

    runs, f_dist = [], None
    for fm in ([0.0, f_mag] if compare_clean else [f_mag]):
        sol, c, rod, refs, model, f_dist = simulate(f_mag=fm, **kwargs)
        r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
        r_ref = np.array([refs["r"](t) for t in sol.t])
        la = applied_tensions(c, sol)
        lbl = "no disturbance" if fm == 0.0 else f"disturbance {fm:g} N"
        e = summarize(lbl, sol.t, r_OP, r_ref, la, t_on, t_off, shape,
                      setpoints=SETPOINT_TABLE, names=names, t_hold=t_hold)
        runs.append((lbl, sol.t, r_OP, r_ref, e, la))

    # ---- visualization ----
    out = Path(__file__).parent
    tag = (f"_{variant}_{shape}_f{f_mag:g}"
           + (f"_smooth{t_move:g}" if smooth else "_step"))
    plot_disturbance(out, "static_", tag, runs, f_dist, t_on,
                     t_sim if shape == "step" else t_off,
                     [k * t_hold for k in range(1, len(names))])

    import matplotlib.pyplot as plt
    plt.show()
