import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_renda2022")

from cardillo.solver import Newton, ScipyDAE, Solution
from cardillo.visualization import Plotter
from cardillo.math import quat2axis_angle

from cardillo_example_systems.tdcr_renda2022 import gen_tdcr_renda2022

############
# parameters
############
G_ACCEL = 9.81 * 0
rod_nelement = 29 * 2
damping_ratio = 10e-2
la_t_0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
# static solver
n_load_steps = 6
# dynamic solver
dt = 1e-3
t_sim = 90
rtol = 1.0e-3
atol = 1.0e-6
# controller
Kp = 0.5
t_circle = 45
t_spiral = t_circle / 2
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
ret = gen_tdcr_renda2022(rod_nelement=rod_nelement, g_accel=G_ACCEL, statics=True)
system_stat = ret["system"]
tendons_stat = ret["tendons"]
rod_gravity_stat = ret["rod_gravity"]

# dynamics
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
def r_OP_traj(t):
    omg = 2 * np.pi / t_circle
    theta = omg * t

    R = 0.175
    if t < t_spiral:
        x = 0.58 - 0.28 * (1 - np.cos(np.pi * t / t_spiral)) / 2
        r = 0.175 * (1 - np.cos(np.pi * t / t_spiral)) / 2
    else:
        x = 0.3
        r = 0.175
    return np.array([x, r * np.cos(theta), r * np.sin(theta)])


def compute_dq_dla_t(t, la_t):
    t = float(t)

    # interpolation of la_tau for the static solver
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
    dq_dla_t = dx_dla_t[: system_stat.nq][-7:]

    # dq_dla_t_inv = scipy.linalg.pinv(dq_dla_t)  # (pseudo-) inverse of dq_dla_t
    return dq_dla_t


def wrap_step_callback(step_callback):
    system_dyn.t_jac_last = -np.inf

    def _step_callback(t, q, u):
        q, u = step_callback(t, q, u)
        # update jacobian every dt_jacobian_update seconds
        if t - system_dyn.t_jac_last >= dt_jacobian:
            la_t = q[-nla_tau:]
            controller_dyn.dq_dla_t = compute_dq_dla_t(t, la_t)
            system_dyn.t_jac_last = t
        return q, u

    return _step_callback


# full gravity
rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

# disable output for the dynamical simulation
newton.verbose = False

controller_dyn.Kp = Kp
controller_dyn.r_OP_traj = r_OP_traj
controller_dyn.dq_dla_t = compute_dq_dla_t(0, la_t_0)

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
    ax.legend()
    ax.grid(True)

for i in range(3):
    ax = fig.add_subplot(gs[i + 3, 0])
    ax.plot(t, psi[:, i], label=f"psi_{['x', 'y', 'z'][i]}")
    ax.legend()
    ax.grid(True)

# la_t
for i in range(nla_tau):
    ax = fig.add_subplot(gs[i, 1])
    ax.plot(t, la_t[:, i], label=f"la_{i}")
    ax.legend()
    ax.grid(True)


plt.tight_layout()
plt.show(block=False)

plotter = Plotter(system_dyn, window_size=(960, 540))
plotter.add_ground(*[0.6, -0.6, 0.6, -0.6, -0.15], 10, 10)
plotter.render_solution(sol, speed_up=2)
