import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_li2023")

from cardillo.solver import Newton, ScipyDAE, Solution
from cardillo.visualization import Plotter

from cardillo_example_systems.tdcr_li2023 import gen_tdcr_li2023

############
# parameters
############
G_ACCEL = 9.81
rod_nelement = 24
damping_ratio = 5e-2
la_t_0 = np.array([0.5, 0, 0, 0], dtype=np.float64)
# static solver
n_load_steps = 4
# dynamic solver
dt = 1e-3
t_sim = 25
rtol = 1.0e-3
atol = 1.0e-6
# controller
Kp = 2
t_traj_hold = 5
dt_jacobian = 1e-2
#
n = t_sim / dt_jacobian
assert (
    abs(n - round(n)) < 1e-8
), f"t_sim ({t_sim} s) must be a multiple of dt_jacobian ({dt_jacobian} s)"


##############
# load systems
##############
# statics
ret = gen_tdcr_li2023(rod_nelement=rod_nelement, g_accel=G_ACCEL, statics=True)
system_stat = ret["system"]
tendons_stat = ret["tendons"]
rod_gravity_stat = ret["rod_gravity"]

# dynamics
ret = gen_tdcr_li2023(
    rod_nelement=rod_nelement,
    g_accel=G_ACCEL,
    damping_ratio=damping_ratio,
    statics=False,
    controller=True,
)
system_dyn = ret["system"]
rod_dyn = ret["rod"]
controller_dyn = ret["controller"]

nla_tau = system_dyn.nla_tau


###################
# initial condition
###################
for td, la in zip(tendons_stat, la_t_0):
    td.la_tau = lambda t, q, u, la=la: t * la

newton = Newton(
    system_stat,
    n_load_steps=n_load_steps,
    verbose=True,
)
sol0_stat = newton.solve()

q0_dyn = np.concatenate((sol0_stat.q[-1], la_t_0))
u0_dyn = sol0_stat.u[-1]
system_dyn.set_new_initial_state(q0_dyn, u0_dyn)


############
# controller
############
# fmt: off
points = np.array([
    [15.438,  4.335,  3.399],    # A
    [15.272, -5.114, -0.463],    # B
    [10.888,  9.106, -5.492],    # C
    [14.615, -4.486, -6.375],    # D
    [13.951,  0.   , -9.842],    # E
    ], dtype=np.float64) * 1e-2
# fmt: on


def r_OP_traj(t):
    n = int(np.floor(t / t_traj_hold))
    n = min(n, len(points) - 1)
    return points[n]


def compute_dr_OP_dla_t(t, q):
    t = float(t)

    # interpolation of la_tau for the static solver
    la_t = q[-nla_tau:]
    for td, la in zip(tendons_stat, la_t):
        td.la_tau = lambda t, q, u, la1=td.la_tau(1, None, None), la2=la: la1 + t * (
            la2 - la1
        )
    # static soluton with warm start from the previous solution
    sol = newton.solve(x0=newton.x[-1])
    assert sol.success, f"Static solver failed to converge: {la_t}"

    # compute jacobian for the controller
    q = sol.q[-1, : system_stat.nq]
    df_dx = newton.jac(newton.x[-1], 1)
    df_dla_t = np.zeros((newton.nx, system_stat.nla_tau), dtype=np.float64)
    df_dla_t[: system_stat.nu] = system_stat.W_tau(t, q, format="Coo").toarray(
        fix_size=True
    )
    dx_dla_t = scipy.sparse.linalg.spsolve(df_dx, -df_dla_t)
    dq_dla_t = dx_dla_t[: system_stat.nq]
    dr_OP_dla_t_inv = scipy.linalg.pinv(
        dq_dla_t[-7:-4]
    )  # (pseudo-) inverse of dr_OP_dla_t
    return dr_OP_dla_t_inv


def wrap_step_callback(step_callback):
    system_dyn.t_jac_last = -np.inf

    def _step_callback(t, q, u):
        q, u = step_callback(t, q, u)
        # update jacobian every dt_jacobian_update seconds
        if t - system_dyn.t_jac_last >= dt_jacobian:
            controller_dyn.dr_OP_dla_t_inv = compute_dr_OP_dla_t(t, q)
            system_dyn.t_jac_last = t
        return q, u

    return _step_callback


# full gravity
rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

# disable output for the dynamical simulation
newton.verbose = False

controller_dyn.Kp = Kp
controller_dyn.r_OP_traj = r_OP_traj


######################
# dynamical simulation
######################
# set step callback to update the jacobian for the controller
system_dyn.step_callback = wrap_step_callback(system_dyn.step_callback)

solver = ScipyDAE(system_dyn, t1=t_sim, dt=dt, method="Radau", atol=atol, rtol=rtol)
sol = solver.solve()


###############
# visualization
###############
t, q = sol.t, sol.q
la_t = q[:, -nla_tau:]
r_OP = q[:, rod_dyn.qDOF][:, -7:-4]
r_OP_ref = np.array([r_OP_traj(ti) for ti in t], dtype=np.float64)
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
plotter.render_solution(Solution(system_dyn, t, q), True, play_speed_up=2)
