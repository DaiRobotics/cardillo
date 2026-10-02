import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_li2023_dyn")

from cardillo.solver import Newton, ScipyDAE, SolverOptions, Riks, Solution
from cardillo.visualization import Plotter

from cardillo_example_systems.tdcr_li2023 import gen_tdcr_li2023


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
        assert t1 > t0, "t1 must be greater than t0"
        alpha = (t - t0) / (t1 - t0)
        y = y0 + (y1 - y0) * (10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5)
        y_dot = (y1 - y0) * (30 * alpha**2 - 60 * alpha**3 + 30 * alpha**4) / (t1 - t0)
        # TODO:
        y_ddot = (
            (y1 - y0)
            * (60 * alpha - 180 * alpha**2 + 120 * alpha**3)
            / ((t1 - t0) ** 2)
        )
        return np.concatenate([y, y_dot, y_ddot])

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


def wrap_step_callback(step_callback):
    system_dyn.t_jac_last = -np.inf
    newton.r_OP_pred = []
    newton.t_pred = []
    sol_last = sol_stat

    def _step_callback(t, q, u):
        nonlocal sol_last
        t = float(t)
        q, u = step_callback(t, q, u)

        # self.system.M(t, q) @ u_dot
        # - self.system.h(t, q, u)
        # - self.system.W_tau(t, q, format="csr") @ self.system.la_tau(t, q, u)
        # - self.system.W_c(t, q, format="csr") @ la_c
        h = system_dyn.h(t, q, u)[-6:-3]
        W_tau = system_dyn.W_tau(t, q, format="csr")[-6:-3].toarray()
        # TODO: test only
        # W_tau[:, -1] = 0  # remove the last column corresponding to the last tendon
        W_tau_inv = np.linalg.pinv(W_tau, rcond=1e-6)
        W_c = system_dyn.W_c(t, q, format="csr")[-6:-3]
        M = system_dyn.M(t, q, format="csr")[-6:-3, -6:-3]
        la_c = system_dyn.la_c(t, q, u)

        f0 = h + W_c[:, -12:] @ la_c[-12:]  # + W_tau @ la_t_comp
        la_t_comp = -W_tau_inv @ f0
        J_inv = W_tau_inv @ M

        # J_inv2 = np.linalg.pinv(np.linalg.inv(M.toarray()) @ W_tau)
        # assert np.allclose(J_inv, J_inv2, atol=1e-6)
        # TODO
        controller_dyn.la_t_comp = la_t_comp * 0
        controller_dyn.dla_t_dbeta = J_inv
        system_dyn.t_jac_last = t
        return q, u

    return _step_callback


if __name__ == "__main__":
    # ---- simulation setup ----
    G_ACCEL = 9.81 * 0
    rod_nelement = 12
    damping_ratio = 5e-2
    la_t_stat = np.array([0, 0, 0, 0])
    # static solver
    n_load_steps = 10
    solver_stat = "newton"
    la_arc0 = 0.1
    # dynamic solver
    t_sim = 25
    dt_sim = 1e-3
    max_step = np.inf
    rtol = 1.0e-3
    atol = 1.0e-6
    # controller
    Kd = 50
    # Kp = 400
    Kp = (Kd / 2) ** 2
    t_traj_trans = 2
    t_traj_hold = 5 - t_traj_trans
    dt_control = 1e-2

    # ---- initial condition ----
    # load system
    ret = gen_tdcr_li2023(rod_nelement=rod_nelement, g_accel=G_ACCEL, statics=True)
    system_stat = ret["system"]
    tendons_stat = ret["tendons"]
    rod_gravity_stat = ret["rod_gravity"]

    # #################################################
    # la_t1 = np.array([6.47, 7.59, 7.1, 5.98])
    # la_t2 = np.array([5.31, 6.61, 5.85, 4.54])

    # newton = Newton(
    #     system_stat,
    #     n_load_steps=10,
    # )
    # print("============")
    # sol_stat, _, _ = forward_statics(la_t1, verbose=True, solver="newton")

    # system_stat.set_new_initial_state(sol_stat.q[-1], sol_stat.u[-1])
    # riks = Riks(system_stat, la_arc0=0.05, compute_init_ds=False)

    # # full gravity for jacobian computation
    # rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(
    #     sol_stat.t[-1], xi
    # )

    # sol_stat2, _, _ = forward_statics(
    #     la_t2, sol_last=sol_stat, verbose=True, solver="riks"
    # )
    # print(sol_stat2.t)

    # from cardillo.visualization.vtk_render2 import Plotter

    # plt = Plotter(system_stat, window_size=(960, 540))
    # plt.render_solution(sol_stat2, speed_up=0.3)

    # exit()
    # #################################################

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
        g_accel=G_ACCEL,
        damping_ratio=damping_ratio,
        statics=False,
        controller="FeedbackLinearization",
    )
    system_dyn = ret["system"]
    rod_dyn = ret["rod"]
    controller_dyn = ret["controller"]

    # set controller
    controller_dyn.Kd = Kd
    controller_dyn.Kp = Kp

    # set initial state
    q0_dyn = sol_stat.q[-1]
    u0_dyn = np.zeros_like(sol_stat.u[-1])
    system_dyn.set_new_initial_state(q0_dyn, u0_dyn)

    # full gravity for jacobian computation
    rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

    system_dyn.set_tau(r_OP_traj)
    # system_dyn.step_callback = wrap_step_callback(system_dyn.step_callback)

    # solve the dynamic problem
    solver = ScipyDAE(
        system_dyn,
        t1=t_sim,
        dt=dt_sim,
        method="Radau",
        atol=atol,
        rtol=rtol,
        max_step=max_step,
        verbose=False
    )


    from scipy_dae.integrate import solve_dae
    from tqdm import tqdm


    y0 = solver.y0
    yp0 = solver.y_dot0

    
    from cProfile import Profile
    prof = Profile()
    prof.enable()
    first_step = None     
    t_list, q_list, u_list = [], [], []
    pbar =tqdm(total=int(t_sim / dt_control), unit="pct")

    for i in range(int(t_sim / dt_control)):
        pbar.update(1)
        t0 = i * dt_control
        t1 = (i + 1) * dt_control

        split = solver.split
        q = y0[: split[0]]
        u = y0[split[0] : split[1]]


        # self.system.M(t, q) @ u_dot
        # - self.system.h(t, q, u)
        # - self.system.W_tau(t, q, format="csr") @ self.system.la_tau(t, q, u)
        # - self.system.W_c(t, q, format="csr") @ la_c
        h = system_dyn.h(t0, q, u)[-6:-3]
        W_tau = system_dyn.W_tau(t0, q, format="csr")[-6:-3].toarray()
        # TODO: test only
        # W_tau[:, -1] = 0  # remove the last column corresponding to the last tendon
        W_tau_inv = np.linalg.pinv(W_tau, rcond=1e-6)
        W_c = system_dyn.W_c(t0, q, format="csr")[-6:-3]
        M = system_dyn.M(t0, q, format="csr")[-6:-3, -6:-3]
        la_c = system_dyn.la_c(t0, q, u)

        la = la_c[-12:]
        # la[6:] = 0
        f0 = h + W_c[:, -12:] @ la # + W_tau @ la_t_comp
        la_t_comp = -W_tau_inv @ f0
        J_inv = W_tau_inv @ M

        # J_inv2 = np.linalg.pinv(np.linalg.inv(M.toarray()) @ W_tau)
        # assert np.allclose(J_inv, J_inv2, atol=1e-6)
        # TODO
        controller_dyn.la_t_comp = la_t_comp * 0
        controller_dyn.dla_t_dbeta = J_inv
        # system_dyn.t_jac_last = t

        
        # solver.pbar = tqdm(total=100, unit="pct")
        # print(f"Solving DAE from t={t0:.2f} to t={t1:.2f}...")
        sol = solve_dae(
            solver.fun,
            [t0, t1],
            y0,
            yp0,
            t_eval=[t0, t1],
            method=solver.method,
            rtol=solver.rtol,
            atol=solver.atol,
            events=[solver.event],
            jac=solver.jac,
            dense_output=True,
            first_step=first_step,
            **solver.kwargs
        )
        # dt = np.diff(sol.sol.ts[-3:])
        # first_step = sol.sol.ts[-1] - sol.sol.ts[-2]        # store segment solution (skip first point except for the first segment)
        k0 = 0 if i == 0 else 1
        t_list.append(sol.t[k0:])
        q_list.append(sol.y[: split[0], k0:].T)
        u_list.append(sol.y[split[0] : split[1], k0:].T)
        y0 = sol.y[:, -1]
        yp0 = sol.yp[:, -1]
    pbar.close()
    # sol = solver.solve()
    sol = Solution(
        system_dyn,
        t=np.concatenate(t_list),
        q=np.vstack(q_list),
        u=np.vstack(u_list),
    )


    prof.disable()
    prof.dump_stats("run_control_tdcr_li2023_dyn.prof")# 
    # system_dyn.export("vtk", f"li2023_dyn", sol, fps=50)
    # ---- visualization ----
    t, q, u = sol.t, sol.q, sol.u
    la_t = np.array([system_dyn.la_tau(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    r_OP = q[:, rod_dyn.qDOF][:, -7:-4]
    r_OP_ref = np.array([r_OP_traj(ti)[:3] for ti in t])
    # t_pred = np.array(newton.t_pred)
    # r_OP_pred = np.array(newton.r_OP_pred)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2)

    # r_OP
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(t, r_OP_ref[:, 0], "-r", label="x_ref")
    ax2.plot(t, r_OP_ref[:, 1], "-r", label="y_ref")
    ax3.plot(t, r_OP_ref[:, 2], "-r", label="z_ref")
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
    plotter.render_solution(sol, speed_up=2)
