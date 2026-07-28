import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo import system
from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_li2023")

from cardillo.math.fsolve import fsolve
from cardillo.solver import Newton, ScipyDAE, Solution
from cardillo.solver.runge_kutta import (
    runge_kutta_3_8,
    runge_kutta_4,
    solve_ivp_sequence,
)
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.visualization import Plotter

from cardillo_example_systems.tdcr_li2023 import gen_tdcr_li2023

##############
# Setup system
##############
Kp = 1
G_ACCEL = 9.81
damping_ratio = 0.01
t_sim = 10
method = "BDF"
dt = 1e-3
dt_sequence = 1e-1
max_step = 1e-1
rtol = 1.0e-3
atol = 1.0e-6

la_t_0 = np.array([0, 0, 0, 0], dtype=np.float64)
ret = gen_tdcr_li2023(rod_nelement=10, g_accel=G_ACCEL, statics=True)
system_stat = ret["system"]
tendons_stat = ret["tendons"]
for td, la in zip(tendons_stat, la_t_0):
    td.la_tau = lambda t, q, u, la=la: t * la

ret = gen_tdcr_li2023(
    rod_nelement=10, g_accel=G_ACCEL, damping_ratio=damping_ratio, statics=False
)
system_dyn = ret["system"]
tendons_dyn = ret["tendons"]
rod_dyn = ret["rod"]
##################
# desired position
##################
points = [
    
]
def r_OP_traj(t):
    return np.array([0.1 * t, 0.1 * t, 0.1 * t], dtype=np.float64)

###############
# static solver
###############
newton = Newton(
    system_stat,
    n_load_steps=4,
    verbose=True,
)

sol_stat = newton.solve()

#######################
# static initial states
#######################
x0 = newton.x[-1]
q0_stat = sol_stat.q[-1]

q0_dyn = np.concatenate((q0_stat, la_t_0))
u0 = np.zeros_like(sol_stat.u[-1])
system_dyn.set_new_initial_state(q0_dyn, u0)

#################
# ScipyDAE solver
#################
# scipy_dae = ScipyDAE(dyn_system, t1=1, dt=1e-2)

# sol_dyn = scipy_dae.solve()

####################
# Runge Kutta solver
####################
nq = system_dyn.nq
nu = system_dyn.nu
n_tau = system_dyn.nla_tau
# q0_dyn, u0 = system_dyn.q0, system_dyn.u0
y0 = np.concatenate((q0_dyn, u0))
M_inv = scipy.sparse.linalg.inv(system_dyn.M(0, q0_dyn).tocsc()).tocsr()
x = newton.x[-1]
f = newton.fun(x, 1)


def compute_dr_OP_dla_t(t, y):
    global x, f, dr_OP_dla_t_inv
    t = float(t)
    # static solver
    la_t = y[nq : nq + n_tau]
    for td, la in zip(tendons_stat, la_t):
        td.la_tau = lambda t, q, u, la=la: la
    sol = fsolve(
        newton.fun,
        x,
        f0=f,
        jac=newton.jac,
        fun_args=(1,),
        jac_args=(1,),
        options=newton.options,
    )
    assert sol.success, f"Static solver failed to converge: {la_t}"
    x = sol.x
    q_stat = x[: system_stat.nq]
    f = newton.fun(x, 1)
    df_dx = newton.jac(x, 1)
    df_dla_t = np.zeros((newton.nx, system_stat.nla_tau), dtype=np.float64)
    df_dla_t[: system_stat.nu] = system_stat.W_tau(t, q_stat, format="Coo").toarray(
        fix_size=True
    )
    dx_dla_t = scipy.sparse.linalg.spsolve(df_dx, -df_dla_t)
    dq_dla_t = dx_dla_t[: system_stat.nq]
    dr_OP_dla_t_inv = scipy.linalg.pinv(
        dq_dla_t[-7:-4]
    )  # (pseudo-) inverse of dr_OP_dla_t
    return dr_OP_dla_t_inv


W_tau_coo = CooMatrix((nu, n_tau), manual_sync=True)
W_c_coo = CooMatrix((nu, system_dyn.nla_c), manual_sync=True)


def ydot(t, y):
    global dr_OP_dla_t_inv
    global W_tau_coo, W_c_coo
    t = float(t)
    q, u = y[:nq], y[-nu:]
    la_tau = y[nq:-nu]
    h = system_dyn.h(t, q, u)
    W_c_coo = system_dyn.W_c(t, q, format="Coo", coo=W_c_coo)
    la_c = system_dyn.la_c(t, q, u)
    W_tau_coo = system_dyn.W_tau(t, q, format="Coo", coo=W_tau_coo)

    # control input
    r_OP = q[-7:-4]
    r_OP_des = r_OP_traj(t)
    la_t_dot = dr_OP_dla_t_inv @ (r_OP_des - r_OP) * Kp

    # update u_dot
    W_c_coo.manual_sync()
    W_c = W_c_coo.tocsr(fix_size=True)
    W_tau_coo.manual_sync()
    W_tau = W_tau_coo.tocsr(fix_size=True)
    u_dot = M_inv @ (h + W_c @ la_c + W_tau @ la_tau)

    ydot = np.concatenate((system_dyn.q_dot(t, q, u), la_t_dot, u_dot))
    # fix the first node
    ydot[:7] = 0.0
    ydot[-nu : -nu + 6] = 0.0
    return ydot


dr_OP_dla_t_inv = compute_dr_OP_dla_t(0, q0_dyn)


def step_callback(t, y):
    t = float(t)
    q, u = y[:nq], y[nq:]
    system_dyn.step_callback(t, q, u)


t, y = solve_ivp_sequence(
    ydot,
    y0,
    system_dyn.t0,
    t_sim,
    dt,
    method=method,
    step_callback=step_callback,
    dt_sequence=dt_sequence,
    sequence_callback=compute_dr_OP_dla_t,
    rtol=rtol,
    atol=atol,
    max_step=max_step,
)

# def step_callback(t, y):
#     t = float(t)
#     q, u = y[:nq], y[nq:]
#     system_dyn.step_callback(t, q, u)

#     # update jacobian dr_OP_dla_t
#     compute_dr_OP_dla_t(t, y)


# t, y = runge_kutta_3_8(dydt, y0, 0, t_sim, 1e-3, step_callback=step_callback)

###############
# visualization
###############
q = y[:, :nq]
la_t = y[:, nq:-nu]
r_OP = q[:, rod_dyn.qDOF].reshape((-1, rod_dyn.nnode, 7))[:, -1, :3]
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(4, 2)

# r_OP
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

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
plotter.render_solution(Solution(system_dyn, t, q), True)
