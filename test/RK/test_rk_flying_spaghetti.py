from time import perf_counter
import numpy as np
import scipy
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

from cardillo.solver import ScipyDAE, Solution
from cardillo.example_systems.flying_spaghetti import *


def runge_kutta_4(dydt, y0, t0, tf, h):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * dydt(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t[i + 1] = t[i] + h

        q = y[i + 1][:nq]
        u = y[i + 1][nq:]
        system.step_callback(t, q, u)

    return t, y


def runge_kutta_3_8(dydt, y0, t0, tf, h):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    f13 = 1 / 3
    f23 = 2 / 3
    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + f13 * h, y[i] + f13 * k1)
        k3 = h * dydt(t[i] + f23 * h, y[i] - f13 * k1 + k2)
        k4 = h * dydt(t[i] + h, y[i] + k1 - k2 + k3)

        y[i + 1] = y[i] + (k1 + 3 * k2 + 3 * k3 + k4) / 8
        t[i + 1] = t[i] + h

        q = y[i + 1][:nq]
        u = y[i + 1][nq:]
        system.step_callback(t, q, u)

    return t, y


def dydt(t, y):
    q, u = y[:nq], y[nq:]
    t = float(t)

    q_dot = rod.q_dot(t, q, u)

    W_c = rod.W_c(t, q).asformat("csr")
    la_c = rod.la_c(t, q, u)
    h = W_c @ la_c + system.h(t, q, u)

    u_dot = M_inv @ h
    return np.concatenate((q_dot, u_dot))


nq = system.nq
t0, q0, u0 = system.t0, system.q0, system.u0
y0 = np.concatenate((q0, u0))
M_inv = scipy.sparse.linalg.inv(system.M(t0, q0).tocsc())


# warm start
dydt(t0, y0)

tsim = 7.0
dt = 1e-2
#################
# ScipyDAE solver
#################
solver = ScipyDAE(system, tsim, dt, rtol=1e-3, atol=1e-6)
solver.fun(system.t0, solver.y0, solver.y0)
solver.jac(system.t0, solver.y0, solver.y0)
t1 = perf_counter()
sol1 = solver.solve()
print(f"ScipyDAE time: {perf_counter() - t1:.2f} s")


####################
# Runge Kutta solver
####################
t1 = perf_counter()
# t, y =runge_kutta_4(dydt, y0, 0, tsim, dt)
t, y = runge_kutta_3_8(dydt, y0, 0, tsim, dt)

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
