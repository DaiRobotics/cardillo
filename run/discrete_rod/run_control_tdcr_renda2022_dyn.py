import numpy as np
import scipy
from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_tdcr_renda2022")

from cardillo.solver import Newton, ScipyDAE
from cardillo.visualization import Plotter
from cardillo.math import quat2axis_angle, Exp_SO3_quat, Exp_SO3_quat_P, Log_SO3_A

from cardillo_example_systems.tdcr_renda2022_dyn import gen_tdcr_renda2022

############
# parameters
############
G_ACCEL = 9.81 * 0
rod_nelement = 29 * 2
damping_ratio = 1e-2
la_t_0 = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64) * 0
# static solver
n_load_steps = 6
# dynamic solver
dt = 1e-3
t_sim = 10
rtol = 1.0e-3
atol = 1.0e-6
# controller
Kp_r = np.diag([100, 5, 5])
Kp_p = 180 / np.pi * 1e-3 * 100 * 0
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

q0_dyn = sol0_stat.q[-1]
u0_dyn = sol0_stat.u[-1]
system_dyn.set_new_initial_state(q0_dyn, u0_dyn)


############
# controller
############
def r_OP_traj(t):
    ret = sol0_stat.q[-1, -7:-4].copy()
    # ret *= 0.95
    # return ret
    ret[1] += ret[0] * 0.0
    ret[2] += ret[0] * 0.0
    ret[0] *= 0.99
    return ret
    omg = 2 * np.pi / t_circle
    theta = omg * t

    if t < t_spiral:
        x = 0.58 - 0.28 * (1 - np.cos(np.pi * t / t_spiral)) / 2
        ret = 0.175 * (1 - np.cos(np.pi * t / t_spiral)) / 2
    else:
        x = 0.3
        ret = 0.175
    return np.array([x, ret * np.cos(theta), ret * np.sin(theta)])


def compute_force(t, q, u):
    t = float(t)
    q_rod = q[rod_dyn.qDOF]
    u_rod = u[rod_dyn.uDOF]

    # return np.array([0, 0, 0, 0, 1, 0], dtype=np.float64)
    r_OP_ref = r_OP_traj(t)
    r_OP = q_rod[-7:-4]
    v_P = u_rod[-6:-3]

    # try:
    #     M_inv = self._M_inv
    # except AttributeError:
    M = rod_dyn.M(t, q_rod).tocsr()[-6:-3, -6:-3]
    W_c = system_dyn.W_c(t, q, format="Coo")
    la_c = system_dyn.la_c(t, q, u)
    # la_c = la_c.reshape((rod_dyn.nelement, -1))
    # la_c[:, 6:] = 0
    # la_c = la_c.flatten()
    h = system_dyn.h(t, q, u)[rod_dyn.uDOF][-6:-3]
    W_tau = controller_dyn.W_tau(t, q).toarray(fix_size=True)[-6:-3, 3:]

    Wla_c = (W_c.tocsr(fix_size=True) @ la_c)[rod_dyn.uDOF][-6:-3]
    la_tau_comp = np.linalg.lstsq(W_tau, -(Wla_c + h * 1), rcond=1e-6)[0]
    la_tau_feedback = np.linalg.lstsq(
        W_tau, M @ ((Kp_r**2) @ (r_OP_ref - r_OP) + 2 * Kp_r @ (-v_P)), rcond=1e-6
    )[0]
    la_tau = la_tau_comp * 1 + la_tau_feedback * 1
    return np.concatenate((np.zeros(3), la_tau))


def wrap_step_callback(step_callback):
    system_dyn.t_jac_last = -np.inf

    def _step_callback(t, q, u):
        q, u = step_callback(t, q, u)
        # update jacobian every dt_jacobian_update seconds
        if t - system_dyn.t_jac_last >= dt_jacobian:
            controller_dyn._la_t = compute_force(t, q, u)
            system_dyn.t_jac_last = t
        return q, u

    return _step_callback


# full gravity
rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

# disable output for the dynamical simulation
newton.verbose = False

controller_dyn.Kp_r = Kp_r
controller_dyn.Kp_p = Kp_p
controller_dyn.r_OP_traj = r_OP_traj

######################
# dynamical simulation
######################
# set step callback to update the jacobian for the controller
system_dyn.step_callback = wrap_step_callback(system_dyn.step_callback)

solver_dyn = ScipyDAE(
    system_dyn, t1=t_sim, dt=dt, method="Radau", atol=atol, rtol=rtol, max_step=1e-1
)
sol = solver_dyn.solve()


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
for i in range(nla_tau):
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
