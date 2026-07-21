from time import perf_counter
import numpy as np
import scipy
from scipy.integrate import solve_ivp


from matplotlib import pyplot as plt

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_rk_flying_spaghetti")

from cardillo.solver import ScipyDAE, Solution
from cardillo.solver.runge_kutta import runge_kutta_4, runge_kutta_3_8
from cardillo_example_systems.flying_spaghetti import gen_flying_spaghetti, x_ref

##############
# Setup system
##############
tend = 7.0
dt = 1e-2
nelement = 10
ret = gen_flying_spaghetti(nelement=nelement)
system = ret["system"]
rod = ret["rod"]
t0 = system.t0

#################
# ScipyDAE solver
#################
solver = ScipyDAE(system, tend, dt, rtol=1e-3, atol=1e-6)
solver.fun(system.t0, solver.y0, solver.y0)
solver.jac(system.t0, solver.y0, solver.y0)
t1 = perf_counter()
sol1 = solver.solve()
print(f"ScipyDAE time: {perf_counter() - t1:.2f} s")


####################
# Runge Kutta solver
####################
nq = system.nq
q0, u0 = system.q0, system.u0
y0 = np.concatenate((q0, u0))
M_inv = scipy.sparse.linalg.inv(system.M(t0, q0).tocsc())


def step_callback(t, y):
    q = y[:nq]
    u = y[nq:]
    system.step_callback(t, q, u)


def dydt(t, y):
    q, u = y[:nq], y[nq:]
    t = float(t)

    q_dot = rod.q_dot(t, q, u)

    W_c = rod.W_c(t, q).tocsr(fix_size=True)
    la_c = rod.la_c(t, q, u)
    h = W_c @ la_c + system.h(t, q, u)

    u_dot = M_inv @ h
    return np.concatenate((q_dot, u_dot))


# warm start
dydt(t0, y0)

t1 = perf_counter()
# t, y =runge_kutta_4(dydt, y0, 0, tsim, dt, step_callback=step_callback)
t, y = runge_kutta_3_8(dydt, y0, 0, tend, dt, step_callback=step_callback)

# sol_ivp = solve_ivp(dydt, (0, tsim), y0, method='RK45', t_eval=np.arange(0, tsim, dt))
# t, y = sol_ivp.t, sol_ivp.y.T

print(f"Runge Kutta time: {perf_counter() - t1:.2f} s")
sol2 = Solution(system, t, y[:, : system.nq])


##############
# Plot results
##############
t = sol1.t

# analytical solution of center of mass
r_OC_ref = np.array([[x_ref(ti), 0, 4] for ti in t])

# center of mass
weights = np.ones(nelement + 1)
weights[1:-1] = 2
weights /= np.sum(weights)

r_OC1 = sol1.q[:, rod.qDOF].reshape((-1, nelement + 1, 7))[..., :3]
r_OC1_com = np.tensordot(r_OC1, weights, axes=(1, 0))

r_OC2 = sol2.q[:, rod.qDOF].reshape((-1, nelement + 1, 7))[..., :3]
r_OC2_com = np.tensordot(r_OC2, weights, axes=(1, 0))

dr = np.abs(r_OC1 - r_OC2)

plt.figure("center of mass")
plt.subplot(2, 2, 1)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i], "r")
    plt.plot(t, r_OC1_com[:, i], "--")
plt.grid(True)

plt.subplot(2, 2, 2)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i], "r")
    plt.plot(t, r_OC2_com[:, i], "--")
plt.grid(True)

plt.subplot(2, 2, 3)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i] - r_OC1_com[:, i], label=f"dr_{i}")
plt.grid(True)
plt.yscale("log")

plt.subplot(2, 2, 4)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i] - r_OC2_com[:, i], label=f"dr_{i}")
plt.grid(True)
plt.yscale("log")
plt.legend()

# configurations
plt.figure("configurations")
plt.subplot(2, 3, 1)
plt.plot(r_OC1[:, 0, 0] - 6, r_OC1[:, 0, 2], "k")
plt.plot(r_OC1[:, -1, 0] - 6, r_OC1[:, -1, 2], "--k")
for ti in [0, 2, 3, 3.8, 4.4, 5, 5.5, 5.8, 6.1, 6.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(r_OC1[i, :, 0] - 6, r_OC1[i, :, 2])
plt.grid(True)
plt.axis("equal")

plt.subplot(2, 3, 2)
plt.plot(r_OC2[:, 0, 0] - 6, r_OC2[:, 0, 2], "k")
plt.plot(r_OC2[:, -1, 0] - 6, r_OC2[:, -1, 2], "--k")
for ti in [0, 2, 3, 3.8, 4.4, 5, 5.5, 5.8, 6.1, 6.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(r_OC2[i, :, 0] - 6, r_OC2[i, :, 2])
plt.grid(True)
plt.axis("equal")

plt.subplot(2, 3, 3)
plt.plot(dr[:, 0, 0], dr[:, 0, 2], "k")
plt.plot(dr[:, -1, 0], dr[:, -1, 2], "--k")
for ti in [0, 2, 3, 3.8, 4.4, 5, 5.5, 5.8, 6.1, 6.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(dr[i, :, 0], dr[i, :, 2])
plt.grid(True)
plt.axis("equal")
plt.xscale("log")
plt.yscale("log")

plt.subplot(2, 3, 4)
plt.plot(r_OC1[:, 0, 1], r_OC1[:, 0, 2], "k")
plt.plot(r_OC1[:, -1, 1], r_OC1[:, -1, 2], "--k")
for ti in [0, 2.5, 3.5, 3.8, 4.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(r_OC1[i, :, 1], r_OC1[i, :, 2])
plt.grid(True)
plt.axis("equal")

plt.subplot(2, 3, 5)
plt.plot(r_OC2[:, 0, 1], r_OC2[:, 0, 2], "k")
plt.plot(r_OC2[:, -1, 1], r_OC2[:, -1, 2], "--k")
for ti in [0, 2.5, 3.5, 3.8, 4.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(r_OC2[i, :, 1], r_OC2[i, :, 2])
plt.grid(True)
plt.axis("equal")


plt.subplot(2, 3, 6)
plt.plot(dr[:, 0, 1], dr[:, 0, 2], "k")
plt.plot(dr[:, -1, 1], dr[:, -1, 2], "--k")
for ti in [0, 2.5, 3.5, 3.8, 4.5]:
    i = int(ti // (t[1] - t[0]))
    plt.plot(dr[i, :, 1], dr[i, :, 2])
plt.grid(True)
plt.axis("equal")
plt.xscale("log")
plt.yscale("log")


plt.show()
