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
    solve_ivp,
)
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.visualization import Plotter

from cardillo_example_systems.tdcr_li2023 import gen_tdcr_li2023

##############
# Setup system
##############
Kp = 100
G_ACCEL = 9.81
damping_ratio = 0.01
t_sim = 0.1
method = "BDF"
dt = 1e-3
max_step = 1e-3
rtol = 1.0e-3
atol = 1.0e-6

la_t_0 = np.array([1, 1, 1, 1], dtype=np.float64) * 0
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
r_des = q0_stat[-7:-4] + 0.1 * (q0_stat[:3] - q0_stat[-7:-4])
# r_des = np.array([0.0, 0.0, 0.2])
print(f"r_des: {r_des}")

q0_dyn = q0_stat
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
n_tau = system_dyn.nla_tau
y0 = np.concatenate((q0_dyn, u0))
M_inv = scipy.sparse.linalg.inv(system_dyn.M(0, q0_dyn).tocsc()).tocsr()
x = newton.x[-1]
f = newton.fun(x, 1)

W_tau_coo = CooMatrix((system_dyn.nu, system_dyn.nla_tau), manual_sync=True)


def dydt(t, y):
    global dr_OP_dla_t_inv
    global W_tau_coo
    t = float(t)
    q, u = y[:nq], y[nq:]
    h = system_dyn.h(t, q, u)
    W_c = rod_dyn.W_c(t, q).tocsr(fix_size=True)
    la_c = rod_dyn.la_c(t, q, u)
    W_tau_coo = system_dyn.W_tau(t, q, format="Coo", coo=W_tau_coo)
    f_int = W_c @ la_c
    f = h + f_int

    # control input
    r_OP = q[-7:-4]
    v_P = u[-6:-3]
    e = r_des - r_OP
    e_dot = -v_P
    W_tau_coo.manual_sync()
    W_tau = W_tau_coo.tocsr(fix_size=True)
    M_inv_W_tau = M_inv @ W_tau
    # force compensation
    # la_tau_comp = scipy.sparse.linalg.lsqr(W_tau[-6:-3, :], -f[-6:-3])[0]
    # la_tau_comp = scipy.optimize.lsq_linear(W_tau[-6:-3, :], -f[-6:-3], bounds=(0, np.inf))['x']
    la_tau_comp = (
        np.linalg.lstsq(W_tau[-6:-3, :].toarray(), -f[-6:-3], rcond=None)[0] * 1
    )
    # feedback control
    J = M_inv_W_tau[-6:-3, :]
    la_tau_ctrl = scipy.sparse.linalg.lsqr(J, 2 * Kp * e_dot + Kp**2 * e)[0] * 1
    # la_tau_ctrl = scipy.optimize.lsq_linear(J, 2 * Kp * e_dot + Kp**2 * e, bounds=(0, np.inf))['x'] * 0

    la_tau = la_tau_ctrl + la_tau_comp

    # print(np.round(e, 3), "\t", np.round(la_tau_ctrl, 5),"\t", np.round(la_tau_comp, 5),"\t", np.round(la_tau, 5))

    # update q_dot
    q_dot = np.array(rod_dyn.q_dot(t, q, u))
    q_dot[:7] = 0.0  # fix the first node

    # update u_dot
    u_dot = M_inv @ f + M_inv_W_tau @ la_tau
    u_dot[:6] = 0.0  # fix the first node

    dydt = np.concatenate((q_dot, u_dot))
    return dydt


def step_callback(t, y):
    t = float(t)
    q, u = y[:nq], y[nq:]
    system_dyn.step_callback(t, q, u)


t, y = solve_ivp(
    dydt,
    y0,
    system_dyn.t0,
    t_sim,
    dt,
    method=method,
    step_callback=step_callback,
    rtol=rtol,
    atol=atol,
    max_step=max_step,
)

# t, y = runge_kutta_3_8(dydt, y0, 0, t_sim, 1e-3, step_callback=step_callback)


###############
# visualization
###############
q = y[:, :nq]
r_OP = q[:, rod_dyn.qDOF].reshape((-1, rod_dyn.nnode, 7))[:, -1, :3]
la_t = q[:, -4:]
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
