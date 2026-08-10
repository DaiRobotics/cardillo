import sys
from pathlib import Path

import numpy as np

# dynamic_control_test.py and its dependencies live one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cardillo.solver import ScipyDAE

from dynamic_control_test import CommonModel, SETPOINT_TABLE, G_ACCEL

from static_controller_li2023_newton import (
    StaticControllerLi2023,
    la_t_from_solution,
)


def p2p_sequence(names, t_hold=5.0):
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        idx = min(int(t // t_hold), len(pts) - 1)
        return pts[idx]

    return r_OP_ref_fn


if __name__ == "__main__":
    # ---- parameters ----
    damping_ratio = 0.1
    Kp = 2.0  
    Kd = 0  
    use_feedforward = True
    la_t0 = np.zeros(4) if use_feedforward else np.array([0.5, 0.0, 0.0, 0.0])
    J_stat_check_dt = 1e-2 
    t_hold = 2.0
    t_sim = 10
    dt = 1e-3
    show_3d = False

    # ----- model -----
    model = CommonModel(damping_ratio=damping_ratio, la_pre=0.0)
    system, rod, tendons = model.system, model.rod, model.tendons

    # ----- reference trajectory -----
    names = ["A", "B", "C", "D", "E"]
    r_OP_ref_fn = p2p_sequence(names, t_hold=t_hold)
    v_P_ref_fn = lambda t: np.zeros(3)

    # ----- controller -----
    controller = StaticControllerLi2023(
        system,
        rod,
        tendons,
        r_OP_ref_fn,
        v_P_ref_fn=v_P_ref_fn,
        Kp=Kp,
        Kd=Kd,
        la_t0=la_t0,
        model_factory=CommonModel,
        g_accel=G_ACCEL,
        J_stat_check_dt=J_stat_check_dt,
    )

    # ---- feedforward: one inverse statics solve per setpoint ----
    if use_feedforward:
        controller.feedforward_from_setpoints(
            [SETPOINT_TABLE[n] for n in names],
            t_hold=t_hold,
            la_t0=np.array([0.5, 0.0, 0.0, 0.0]),
        )

    system.add(controller)
    system.assemble()

    # start from the static equilibrium at la_t0 instead of the straight rod
    q0 = np.concatenate((controller.static_model_twin.q_eq(), controller.q0))
    system.set_new_initial_state(q0, np.zeros(system.nu))

    print(f"J_stat(la_t0) =\n{controller.J_stat}")
    print(f"cond(J_stat)  = {np.linalg.cond(controller.J_stat):.3e}")
    print(f"r_OP(la_t0)  = {controller.static_model_twin.r_OP_eq()}")

    # ---- solve ----
    solver = ScipyDAE(system, t_sim, dt)
    sol = solver.solve()
    print(
        f"Kp = {Kp}, Kd = {Kd}, damping ratio = {damping_ratio}, "
        f"t_sim = {t_sim}, dt = {dt}, J_stat_check_dt = {J_stat_check_dt}, "
        f"static solves = {controller.static_model_twin.n_solves} "
        f"({controller.static_model_twin.n_retries} load stepped retries)"
    )

    # ---- tracking summary ----
    r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
    la_ts = la_t_from_solution(controller, sol)
    print("steady state tip error at the end of each hold:")
    for k, name in enumerate(names):
        i = min(np.searchsorted(sol.t, (k + 1) * t_hold) - 1, len(sol.t) - 1)
        e = np.linalg.norm(r_OP[i] - SETPOINT_TABLE[name])
        print(f"  {name}: {e * 1e3:8.4f} mm")
    print(f"tendon tension range: [{la_ts.min():.3f}, {la_ts.max():.3f}] N")

    # ---- visualization ----
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).parent
    r_OP_ref = np.array([r_OP_ref_fn(ti) for ti in sol.t])
    setpoint_times = [k * t_hold for k in range(1, len(names))]

    la_ff_ts = np.array([controller.la_t_ff(ti) for ti in sol.t])

    fig, ax = plt.subplots(num="TendonForces", figsize=(9, 4.5))
    for k in range(model.n_tendons):
        (line,) = ax.plot(sol.t, la_ts[:, k], label=f"tendon {k + 1}")
        if use_feedforward:
            ax.plot(sol.t, la_ff_ts[:, k], color=line.get_color(), ls="--", lw=1.0)
    ax.axhline(0.0, color="k", ls="-", lw=0.8)
    for ts in setpoint_times:
        ax.axvline(ts, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tendon force [N]")
    ff_txt = "solid: total, dashed: feedforward" if use_feedforward else "no feedforward"
    ax.set_title(f"Tendon forces, static controller (Kp={Kp}, Kd={Kd}; {ff_txt})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "static_tendon_forces.png", dpi=150)

    fig, axs = plt.subplots(3, 1, num="XYZ", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("XYZ"):
        axs[i].plot(sol.t, r_OP_ref[:, i], "b--", label="desired")
        axs[i].plot(sol.t, r_OP[:, i], "r", label="actual")
        axs[i].set_ylabel(f"{lbl} [m]")
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Tip trajectory tracking, static controller")
    fig.tight_layout()
    fig.savefig(out_dir / "static_tip_tracking.png", dpi=150)
    print(f"figures written to {out_dir}")

    if show_3d:
        from espedal_control_test import p2p_vis_plot

        p2p_vis_plot(model, sol, r_OP_ref_fn)
    plt.show()
