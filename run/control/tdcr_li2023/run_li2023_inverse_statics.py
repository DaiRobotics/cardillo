import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache


from tqdm import tqdm

configure_cache("run_tdcr_li2023")

from cardillo.solver.radau_dae import solve_dae
from cardillo.solver import Newton, ScipyDAE, Solution
from cardillo.visualization import Plotter

from cardillo_example_systems.tdcr_li2023 import gen_tdcr_li2023

from cardillo.math import Exp_SO3_quat, Log_SO3, Exp_SO3

from cardillo.utility.check_time_derivatives import check_time_derivatives


def forward_statics(la_t, active_gravity, sol_last=None, verbose=False):
    if sol_last is None:
        t0 = 0.0
    else:
        t0 = sol_last.t[-1]

    tau0 = system_stat.tau(t0)
    tau1 = np.concatenate([la_t, [active_gravity]])
    system_stat.set_tau(lambda t, tau0=tau0, tau1=tau1: tau0 + (tau1 - tau0) * t)

    newton.verbose = verbose
    if sol_last is not None:
        x0 = newton.x[-1]
        f0 = newton.f[-1]
        newton.reset(x0=x0, f0=f0)
    sol = newton.solve()

    assert (
        sol.success
    ), f"Forward statics failed: {np.round(tau0, 2)} ==> {np.round(la_t, 2)}"

    t = sol.t[-1]
    q = sol.q[-1]

    x = newton.x[-1]
    df_dx = newton.jac(x, t)

    r_OP = q[-7:-4]
    # compute jacobian
    df_dla_t = np.zeros((newton.nx, 4), dtype=float)
    df_dla_t[: system_stat.nu] = system_stat.W_tau(t, q, format="Coo").toarray(
        fix_size=True
    )[:, :-1]

    dx_dla_t = scipy.sparse.linalg.spsolve(df_dx, -df_dla_t)
    dq_dla_t = dx_dla_t[: system_stat.nq]
    dr_OP_dla_t = dq_dla_t[-7:-4]
    return sol, r_OP, dr_OP_dla_t


############
# controller
############
def make_piecewise_C2_interp_fun(ts, ys):
    def f(t):
        n = np.searchsorted(ts, t, "right") - 1
        if n == len(ts) - 1:
            n -= 1
        y0 = ys[n]
        t0 = ts[n]
        y1 = ys[n + 1]
        t1 = ts[n + 1]
        alpha = (t - t0) / (t1 - t0)
        y = y0 + (y1 - y0) * (10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5)
        y_dot = (y1 - y0) * (30 * alpha**2 - 60 * alpha**3 + 30 * alpha**4) / (t1 - t0)
        return y, y_dot

    return f


def set_points_to_traj(ys, t_trans, t_hold, hold_init_point=False):
    dt = t_trans + t_hold
    _ys = ys[1:]
    ts = np.array([dt * i for i in range(len(_ys))])
    ts = np.repeat(ts, 2, axis=0)
    ts[1::2] += dt
    ts[:-1:2] += t_trans
    _ys = np.repeat(_ys, 2, axis=0)
    if hold_init_point:
        ts += dt
        ts = np.concatenate(([0, dt], ts))
        ys = np.concatenate((ys[None, 0], ys[None, 0], _ys))
    else:
        ts = np.concatenate(([0], ts))
        ys = np.concatenate((ys[None, 0], _ys))
    return make_piecewise_C2_interp_fun(ts, ys)


if __name__ == "__main__":
    # ---- simulation setup ----
    active_gravity = True
    rod_nelement = 12
    damping_ratio = 5e-2
    la_t_stat = np.array([0, 0, 0, 0])
    # dynamic solver
    t_sim = 25
    dt_sim = 0.02
    rtol = 1.0e-3
    atol = 1.0e-6
    # controller
    Kp = 2
    dt_control = 5e-2
    la_t_skip_tol = 1e-3
    active_controller = True
    # trajectory
    t_traj_trans = 2
    t_traj_hold = 5 - t_traj_trans
    #
    n = t_sim / dt_control
    assert (
        abs(n - round(n)) < 1e-8
    ), f"t_sim ({t_sim} s) must be a multiple of dt_jacobian ({dt_control} s)"

    # ---- initial condition ----
    # load system
    ret = gen_tdcr_li2023(rod_nelement=rod_nelement, statics=True)
    system_stat = ret["system"]
    tendons_stat = ret["tendons"]
    assert system_stat._jaxed
    # system_stat._jaxed = False
    # solve static problem
    newton = Newton(system_stat)

    sol_stat, _, dr_OP_dla_t = forward_statics(
        la_t_stat, active_gravity=active_gravity, verbose=True
    )

    q_stat = sol_stat.q[-1]
    u_stat = sol_stat.u[-1]

    # ---- trajectory generation ----
    # fmt: off
    r_OP_set_points = np.array([
        sol_stat.q[-1, -7:-4] * 1e2, # initial point
        [15.438,  4.335,  3.399],    # A
        [15.272, -5.114, -0.463],    # B
        [10.888,  9.106, -5.492],    # C
        [14.615, -4.486, -6.375],    # D
        [13.951,  0.   , -9.842],    # E
        ]) * 1e-2
    # fmt: on
    r_OP_traj = set_points_to_traj(
        r_OP_set_points, t_traj_trans, t_traj_hold, hold_init_point=False
    )

    # ---- dynamical simulation ----
    # load system
    ret = gen_tdcr_li2023(
        rod_nelement=rod_nelement,
        damping_ratio=damping_ratio,
        statics=False,
    )
    system_dyn = ret["system"]
    rod_dyn = ret["rod"]
    assert system_dyn._jaxed

    # set initial state
    system_dyn.set_new_initial_state(q_stat, u_stat)

    # solve the dynamic problem
    solver = ScipyDAE(
        system_dyn,
        t1=t_sim,
        dt=dt_sim,
        method="Radau",
        atol=atol,
        rtol=rtol,
        verbose=False,
    )

    # solver_summary = SolverSummary(f"Scipy solve_dae with method '{self.method}'")
    y0 = solver.y0
    yp0 = solver.y_dot0

    t_stat_list, r_OP_stat_list, dr_OP_dla_t_list = [], [], []
    t_list, q_list, u_list, la_t_list = [], [], [], []

    pbar = tqdm(total=int(t_sim / dt_control), unit="pct")

    la_t = np.zeros(4)
    dla_t = np.zeros(4)
    sol_last = sol_stat
    from cProfile import Profile

    cprofile = Profile()
    solver.warmup_jax()
    cprofile.enable()

    for i in range(int(t_sim / dt_control)):
        pbar.update(1)
        t0 = i * dt_control
        t1 = (i + 1) * dt_control

        split = solver.split
        q = y0[: split[0]]
        u = y0[split[0] : split[1]]

        # static jacobian estimation
        if (
            i == 0
            or np.sqrt(np.mean((la_t - system_stat.tau(1)[:-1]) ** 2)) > la_t_skip_tol
        ):
            sol_last, r_OP_pred, dr_OP_dla_t = forward_statics(
                la_t,
                active_gravity=active_gravity,
                sol_last=sol_last,
                verbose=False,
            )
        # compute reference trajectory
        r_OP_ref, v_P_ref = r_OP_traj(t0)

        # compute control input
        r_OP = q[-7:-4]
        dla_t = (
            np.linalg.lstsq(dr_OP_dla_t, v_P_ref + Kp * (r_OP_ref - r_OP), rcond=1e-6)[
                0
            ]
            * dt_control
            * active_controller
        )
        la_t += dla_t

        # set control input
        tau0 = system_dyn.tau(t0).copy()
        tau0[-1] = 1  # full gravity
        tau1 = np.concatenate([la_t, [1]])
        system_dyn.set_tau(
            lambda t, t0=t0, tau0=tau0.copy(), tau1=tau1.copy(): tau0
            + (tau1 - tau0) * (t - t0) / dt_control
        )

        t_stat_list.append(t0)
        r_OP_stat_list.append(r_OP_pred)
        dr_OP_dla_t_list.append(dr_OP_dla_t)

        sol = solve_dae(
            solver.fun,
            [t0, t1],
            y0,
            yp0,
            # t_eval=[t0, t1],
            method=solver.method,
            rtol=solver.rtol,
            atol=solver.atol,
            events=[solver.event],
            jac=solver.jac,
            fun_vec=solver._fun_vec,
            **solver.kwargs,
        )

        # extract solution
        k0 = 0 if i == 0 else 1
        ts = sol.t[k0:]
        qs = sol.y[: split[0], k0:].T
        us = sol.y[split[0] : split[1], k0:].T

        # append solution
        t_list.append(ts)
        q_list.append(qs)
        u_list.append(us)
        la_t_list.append(
            np.array(
                [
                    system_dyn.la_tau(ti, qi, ui, system_dyn.tau(ti))
                    for ti, qi, ui in zip(ts, qs, us)
                ]
            )
        )

        # update initial condition for next step
        y0 = sol.y[:, -1]
        yp0 = sol.yp[:, -1]

    # close progress bar
    pbar.close()
    cprofile.disable()
    cprofile.dump_stats("run_li2023_inverse_statics.prof")

    sol = Solution(
        system_dyn,
        t=np.concatenate(t_list),
        q=np.vstack(q_list),
        u=np.vstack(u_list),
        la_t=np.vstack(la_t_list),
    )

    # system_dyn.export("vtk", f"li2023_Kp_{Kp}", sol, fps=50)
    # ---- visualization ----
    t, q, u, la_t = sol.t, sol.q, sol.u, sol.la_t
    r_OP = q[:, rod_dyn.qDOF][:, -7:-4]
    r_OP_ref = np.array([r_OP_traj(ti)[0] for ti in t])
    t_stat = np.array(t_stat_list)
    r_OP_stat = np.array(r_OP_stat_list)
    dr_OP_dla_t = np.array(dr_OP_dla_t_list)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2)

    # r_OP
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, r_OP_ref[:, 0], "-r", label="x_ref")
    ax2.plot(t, r_OP_ref[:, 1], "-r", label="y_ref")
    ax3.plot(t, r_OP_ref[:, 2], "-r", label="z_ref")
    ax1.plot(t_stat, r_OP_stat[:, 0], "--g", label="x_pred")
    ax2.plot(t_stat, r_OP_stat[:, 1], "--g", label="y_pred")
    ax3.plot(t_stat, r_OP_stat[:, 2], "--g", label="z_pred")

    ax1.plot(t, r_OP[:, 0], label="x")
    ax2.plot(t, r_OP[:, 1], label="y")
    ax3.plot(t, r_OP[:, 2], label="z")

    # la_t
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax4)
    ax6 = fig.add_subplot(gs[2, 1], sharex=ax4)
    ax7 = fig.add_subplot(gs[3, 1], sharex=ax4)

    ax4.plot(t, la_t[:, 0], label="la1")
    ax5.plot(t, la_t[:, 1], label="la2")
    ax6.plot(t, la_t[:, 2], label="la3")
    ax7.plot(t, la_t[:, 3], label="la4")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    plotter = Plotter(system_dyn, window_size=(960, 540))
    plotter.add_ground(*[0.2, -0.2, 0.2, -0.2, -0.15], 10, 10)
    plotter.render_solution(sol, speed_up=1)
