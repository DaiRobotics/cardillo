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
G_ACCEL = 9.81 * 1
rod_nelement = 24
damping_ratio = 5e-2
la_t_stat = np.array([2.0351404, 1.92957053, 1.81885301, 1.92957053], dtype=np.float64)
# static solver
n_load_steps = 10
# dynamic solver
dt = 1e-3
max_step = 1e-1
t_sim = 25
rtol = 1.0e-3
atol = 1.0e-6
# controller
Kp = 2
traj_type = "points"
t_transfer = 1
t_traj_hold = 5 - t_transfer
# traj_type = "circle"
# t_traj_hold = 20
dt_ref_points = 1e-1
n_steps_inverse_statics = 10


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
newton = Newton(
    system_stat,
    n_load_steps=n_load_steps,
    verbose=True,
)


def forward_statics(la_t, warm_start=True, verbose=False):
    newton.verbose = verbose
    if warm_start:
        la_t0 = np.array([td.la_tau(1, None, None) for td in tendons_stat])
    else:
        la_t0 = np.array([td.la_tau(0, None, None) for td in tendons_stat])

    for td, la0, la1 in zip(tendons_stat, la_t0, la_t):
        td.la_tau = lambda t, q, u, la1=la0, la2=la1: la1 + t * (la2 - la1)

    # static soluton with warm start from the previous solution
    sol = newton.solve(x0=newton.x[-1] if warm_start else None)
    assert (
        sol.success
    ), f"Forward statics failed: {np.round(la_t0, 2)} ==> {np.round(la_t, 2)}"

    q = sol.q[-1]
    x = newton.x[-1]

    r_OP = q[-7:-4]

    # compute jacobian
    df_dx = newton.jac(x, 1)
    df_dla_t = np.zeros((newton.nx, system_stat.nla_tau), dtype=np.float64)
    df_dla_t[: system_stat.nu] = system_stat.W_tau(1, q, format="Coo").toarray(
        fix_size=True
    )
    dx_dla_t = scipy.sparse.linalg.spsolve(df_dx, -df_dla_t)
    dq_dla_t = dx_dla_t[: system_stat.nq]
    dr_OP_dla_t = dq_dla_t[-7:-4]
    return sol, r_OP, dr_OP_dla_t


sol_stat, _, _ = forward_statics(la_t_stat, warm_start=False, verbose=True)
q0_stat = sol_stat.q[-1]

# after initialization, set the rod gravity to be the same as the static solution
rod_gravity_stat.force = lambda t, xi, f=rod_gravity_stat.force: f(1, xi)

q0_dyn = np.concatenate((q0_stat, la_t_stat * 0))
u0_dyn = sol_stat.u[-1]
system_dyn.set_new_initial_state(q0_dyn, u0_dyn)


############
# trajectory
############
# fmt: off
r_OP_targets = np.array([
    [13.951,  0.   , -9.842],    # E
    [15.438,  4.335,  3.399],    # A
    [15.272, -5.114, -0.463],    # B
    [10.888,  9.106, -5.492],    # C
    [14.615, -4.486, -6.375],    # D
    # [13.951,  0.   , -9.842],    # E
    ], dtype=np.float64) * 1e-2
# fmt: on


def inverse_statics(r_OP_target, la_t_init):
    def con(x):
        sol, r_OP, dr_OP_dla_t = forward_statics(x)
        return r_OP - r_OP_target

    def jac(x):
        sol, r_OP, dr_OP_dla_t = forward_statics(x)
        return dr_OP_dla_t

    nlc = scipy.optimize.NonlinearConstraint(
        con,
        lb=np.zeros(3),
        ub=np.zeros(3),
        jac=jac,
        # hess="2-point",
    )

    result = scipy.optimize.minimize(
        lambda x: x @ x * 0.5,
        la_t_init,
        jac=lambda x: x,
        # method="trust-constr",
        # hess=lambda x: np.eye(len(la_t_init), dtype=np.float64),
        method="SLSQP",
        constraints=[nlc],
    )
    assert result.success, f"Inverse statics failed: {result.message}"
    la_t = result.x
    return la_t, *(forward_statics(la_t)[1:])


# solution from the inverse statics
la_t_targets = np.array(
    [
        [2.0351404, 1.92957053, 1.81885301, 1.92957053],
        [-0.82373167, 1.11966336, 2.35559171, 0.40791725],
        [1.30489802, 1.86640674, 3.4706309, 2.90251597],
        [1.44490565, 3.10742688, 2.30291534, 0.651404],
        [2.22942324, 2.06963233, 2.92758557, 3.09179046],
        # [ 2.0351404 ,  1.92957053,  1.81885301,  1.92957053]
    ]
)

do_inverse_statics = False
if do_inverse_statics:
    print("Computing inverse statics for target points...")
    la_t = la_t_stat
    target_data = []
    for i in range(len(r_OP_targets)):
        print(f"Planning trajectory at {i}th point: {r_OP_targets[i]}")
        r_OP1 = r_OP_targets[i]
        if i == 0:
            r_OP0 = q0_stat[-7:-4]
        else:
            r_OP0 = r_OP_targets[i - 1]
        for j in range(n_steps_inverse_statics):
            alpha = j / (n_steps_inverse_statics - 1)
            r_OP_target = r_OP0 + alpha * (r_OP1 - r_OP0)
            la_t, r_OP, dr_OP_dla_t = inverse_statics(r_OP_target, la_t)
        target_data.append((r_OP, la_t, dr_OP_dla_t))

    la_t_targets = np.array([la_t for _, la_t, _ in target_data])

# dr_OP_dla_t_ref_pts = np.array([dr_OP_dla_t for _, _, dr_OP_dla_t in target_data])


def la_t_traj(t):
    n = int(np.floor(t / (t_traj_hold + t_transfer)))
    n = min(n, len(la_t_targets) - 1)
    la_t1 = la_t_targets[n]
    if n == 0:
        la_t0 = la_t_stat
    else:
        la_t0 = la_t_targets[n - 1]
    la_t = la_t0 + (la_t1 - la_t0) * min(
        (t - n * (t_traj_hold + t_transfer)) / t_transfer, 1
    )
    return la_t


t_ref_pts = np.linspace(0, t_sim, int(t_sim / dt_ref_points) + 1)
la_t_ref_pts = np.array([la_t_traj(ti) for ti in t_ref_pts])

print("Computing forward statics for reference points...")
r_OP_ref_pts = []
dla_t_dr_OP_ref_pts = []
for la_t in (reversed(la_t_ref_pts) if do_inverse_statics else la_t_ref_pts):
    sol, r_OP, dr_OP_dla_t = forward_statics(la_t, verbose=False)
    r_OP_ref_pts.append(r_OP)
    dla_t_dr_OP_ref_pts.append(scipy.linalg.pinv(dr_OP_dla_t))
r_OP_ref_pts = np.array(reversed(r_OP_ref_pts) if do_inverse_statics else r_OP_ref_pts)
dla_t_dr_OP_ref_pts = np.array(
    reversed(dla_t_dr_OP_ref_pts) if do_inverse_statics else dla_t_dr_OP_ref_pts
)


def interpolate_ref_points(y):
    def f(t):
        x = t / dt_ref_points
        n = int(np.floor(x))
        n = min(n, len(y) - 1)
        y0 = y[n]
        try:
            y1 = y[n + 1]
        except IndexError:
            y1 = y[n]
        y_interp = y0 + (y1 - y0) * min((x - n), 1)
        return y_interp

    return f


r_OP_traj = interpolate_ref_points(r_OP_ref_pts)
dla_t_dr_OP_traj = interpolate_ref_points(dla_t_dr_OP_ref_pts)

system_dyn.set_tau(
    lambda t: np.concatenate(
        [r_OP_traj(t), la_t_traj(t), dla_t_dr_OP_traj(t).flatten()]
    )
)

controller_dyn.Kp = Kp * 1


######################
# dynamical simulation
######################
# set step callback to update the jacobian for the controller
solver = ScipyDAE(
    system_dyn, t1=t_sim, dt=dt, method="Radau", atol=atol, rtol=rtol, max_step=max_step
)
sol = solver.solve()


###############
# visualization
###############
t, q, u = sol.t, sol.q, sol.u
la_t = q[:, -nla_tau:]
la_t = np.array([system_dyn.la_tau(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
r_OP = q[:, rod_dyn.qDOF][:, -7:-4]
r_OP_des = np.array([r_OP_traj(ti) for ti in t], dtype=np.float64)
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(4, 2)

# r_OP
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

ax1.plot(t, r_OP_des[:, 0], "-r", label="x_ref")
ax2.plot(t, r_OP_des[:, 1], "-r", label="y_ref")
ax3.plot(t, r_OP_des[:, 2], "-r", label="z_ref")
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
