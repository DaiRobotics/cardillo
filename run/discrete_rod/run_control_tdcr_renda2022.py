import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_renda2022")

from cardillo.solver import Newton, ScipyDAE, Riks
from cardillo.visualization import Plotter
from cardillo.math import quat2axis_angle

from cardillo_example_systems.tdcr_renda2022 import gen_tdcr_renda2022


def forward_statics(la_t, sol_last=None, verbose=False, solver="riks"):
    if sol_last is None:
        t0 = 0
    else:
        t0 = sol_last.t[-1]

    la_t0 = np.array([td.la_tau(t0, None, None) for td in tendons_stat])

    if np.allclose(la_t0, la_t, atol=1e-2) and sol_last is not None:
        sol = sol_last
    else:
        for td, la0, la1 in zip(tendons_stat, la_t0, la_t):
            td.la_tau = lambda t, q, u, la0=la0, la1=la1: la0 + (la1 - la0) * t
        if solver == "riks":
            riks.verbose = verbose
            riks.reset(x0=riks.xk)
            sol = riks.solve()
        elif solver == "newton":
            newton.verbose = verbose
            x0 = newton.x[-1] if sol_last is not None else None
            newton.reset(x0=x0)
            sol = newton.solve()
        else:
            raise ValueError(f"Solver {solver} not supported. Use 'newton' or 'riks'.")

        assert (
            sol.success
        ), f"Forward statics failed: {np.round(la_t0, 2)} ==> {np.round(la_t, 2)}"

    t = sol.t[-1]
    q = sol.q[-1]
    if isinstance(sol.solver, Riks):
        x = sol.solver.xk
        df_dx = sol.solver.J(x).tocsc()[:-1, :-1]
    else:
        x = newton.x[-1]
        df_dx = newton.jac(x, t)

    r_OP = q[-7:-4]

    # compute jacobian
    df_dla_t = np.zeros((newton.nx, system_stat.nla_tau), dtype=float)
    df_dla_t[: system_stat.nu] = system_stat.W_tau(t, q, format="Coo").toarray(
        fix_size=True
    )
    dx_dla_t = scipy.sparse.linalg.spsolve(df_dx, -df_dla_t)
    dq_dla_t = dx_dla_t[: system_stat.nq][-7:]
    dy_dla_t = np.concatenate([dq_dla_t[:3], dq_dla_t[4:]])
    return sol, r_OP, dy_dla_t


############
# controller
############
def wrap_step_callback(step_callback):
    system_dyn.t_jac_last = -np.inf
    newton.r_OP_pred = []
    newton.t_pred = []
    sol_last = sol_stat

    def _step_callback(t, q, u):
        nonlocal sol_last
        t = float(t)
        q, u = step_callback(t, q, u)
        if t - system_dyn.t_jac_last >= dt_jacobian:
            la_t = system_dyn.la_tau(t, q, u)
            sol_last, r_OP, dy_dla_t = forward_statics(
                la_t, sol_last=sol_last, verbose=False, solver=solver_stat
            )

            newton.t_pred.append(t)
            newton.r_OP_pred.append(r_OP)
            controller_dyn.dla_t_dy = np.linalg.pinv(dy_dla_t, rcond=1e-6)
            system_dyn.t_jac_last = t
        return q, u

    return _step_callback


if __name__ == "__main__":
    # ---- simulation setup ----
    G_ACCEL = 9.81 * 0
    rod_nelement = 29 * 2
    damping_ratio = 10e-2
    la_t_stat = np.array([0, 0, 0, 0, 0, 0])
    # static solver
    n_load_steps = 10
    solver_stat = "riks"
    la_arc0 = 0.1
    # dynamic solver
    dt_sim = 1e-3
    max_step = np.inf
    rtol = 1.0e-3
    atol = 1.0e-6
    # controller
    Kp_r = 0.8
    Kp_p = 180 / np.pi * 1e-3 * 100
    feedforward = False
    t_circle = 20
    t_spiral = 20
    dt_jacobian = 1e-2
    t_sim = t_spiral + t_circle
    #
    n = t_sim / dt_jacobian
    assert (
        abs(n - round(n)) < 1e-8
    ), f"t_sim ({t_sim} s) must be a multiple of dt_jacobian ({dt_jacobian} s)"

    # ---- initial condition ----
    # load system
    ret = gen_tdcr_renda2022(rod_nelement=rod_nelement, g_accel=G_ACCEL, statics=True)
    system_stat = ret["system"]
    tendons_stat = ret["tendons"]
    rod_gravity_stat = ret["rod_gravity"]

    # solve static problem
    newton = Newton(
        system_stat,
        n_load_steps=n_load_steps,
    )

    sol_stat, _, _ = forward_statics(la_t_stat, verbose=True, solver="newton")

    system_stat.set_new_initial_state(sol_stat.q[-1], sol_stat.u[-1])
    riks = Riks(system_stat, la_arc0=la_arc0, compute_init_ds=False)

    rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(
        sol_stat.t[-1], xi
    )

    # ---- trajectory generation ----
    def r_OP_traj(t):
        omg = 2 * np.pi / t_circle
        theta = omg * t

        if t < t_spiral:
            x = 0.58 - 0.28 * (1 - np.cos(np.pi * t / t_spiral)) / 2
            r = 0.175 * (1 - np.cos(np.pi * t / t_spiral)) / 2
            dx = -0.28 * np.pi / t_spiral * np.sin(np.pi * t / t_spiral) / 2
            dr = 0.175 * np.pi / t_spiral * np.sin(np.pi * t / t_spiral) / 2
        else:
            x = 0.3
            r = 0.175
            dr = 0
            dx = 0
        r_OP = np.array([x, r * np.cos(theta), r * np.sin(theta)])
        v_P = (
            np.array(
                [
                    dx,
                    dr * np.cos(theta) - r * omg * np.sin(theta),
                    dr * np.sin(theta) + r * omg * np.cos(theta),
                ]
            )
            * feedforward
        )
        return np.concatenate([r_OP, v_P])

    # ---- dynamical simulation ----
    # load system
    ret = gen_tdcr_renda2022(
        rod_nelement=rod_nelement,
        g_accel=G_ACCEL,
        damping_ratio=damping_ratio,
        statics=False,
        controller=True,
    )
    system_dyn = ret["system"]
    rod_dyn = ret["rod"]
    controller_dyn = ret["controller"]

    # set initial state
    q0_dyn = np.concatenate((sol_stat.q[-1], la_t_stat))
    u0_dyn = np.zeros_like(sol_stat.u[-1])
    system_dyn.set_new_initial_state(q0_dyn, u0_dyn)

    # full gravity for jacobian computation
    rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

    # set controller
    controller_dyn.Kp_r = Kp_r
    controller_dyn.Kp_p = Kp_p
    system_dyn.set_tau(r_OP_traj)
    system_dyn.step_callback = wrap_step_callback(system_dyn.step_callback)

    # solve the dynamic problem
    solver = ScipyDAE(
        system_dyn,
        t1=t_sim,
        dt=dt_sim,
        method="Radau",
        atol=atol,
        rtol=rtol,
        max_step=max_step,
    )

    sol = solver.solve()

    # ---- visualization ----
    t, q, u = sol.t, sol.q, sol.u
    la_t = np.array([system_dyn.la_tau(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    q_end = q[:, rod_dyn.qDOF][:, -7:]
    r_OP = q_end[:, :3] * 1000
    psi = np.rad2deg([quat2axis_angle(pi) for pi in q_end[:, 3:]])
    r_OP_ref = np.array([r_OP_traj(ti) for ti in t]) * 1000
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(6, 2)

    # r_OP
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, r_OP_ref[:, i], "-r", label=f"{['x', 'y', 'z'][i]}_ref")
        ax.plot(t, r_OP[:, i], label=f"{['x', 'y', 'z'][i]}")
        ax.set_ylabel("mm")
        ax.legend()
        ax.grid(True)

    for i in range(3):
        ax = fig.add_subplot(gs[i + 3, 0])
        ax.plot(t, psi[:, i], label=f"psi_{['x', 'y', 'z'][i]}")
        ax.set_ylabel("deg")
        ax.legend()
        ax.grid(True)

    # la_t
    for i in range(system_dyn.nla_tau):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, la_t[:, i], label=f"la_{i}")
        ax.set_ylabel("N")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    plotter = Plotter(system_dyn, window_size=(960, 540))
    plotter.add_ground(*[0.6, -0.6, 0.6, -0.6, -0.15], 10, 10)
    plotter.render_solution(sol, speed_up=2)
