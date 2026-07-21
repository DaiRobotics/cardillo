import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt
import scipy
from scipy.integrate import solve_ivp

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_RK_cantilever_beam")

from cardillo.solver import ScipyDAE, Solution
from cardillo.solver.runge_kutta import runge_kutta_3_8, runge_kutta_4
from cardillo_example_systems.dynamic_cantilever import gen_cantilever_beam

dt = 8e-4
tend = 3.0

ret = gen_cantilever_beam(rod="DiscreteRod")
rod = ret["rod"]
system = ret["system"]
##########
# ScipyDAE
##########
solver = ScipyDAE(system, tend, dt)

t0 = perf_counter()
sol = solver.solve()
print(f"Scipy DAE time: {perf_counter() - t0:.2f} s")

t1, q1 = sol.t, sol.q[:, rod.qDOF]

####################
# Runge Kutta solver
####################
nq = system.nq
q0, u0 = system.q0, system.u0
y0 = np.concatenate((q0, u0))
M_inv = scipy.sparse.linalg.inv(system.M(t0, q0).tocsc())

q_node0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
u_node0 = np.zeros(6)


def step_callback(t, y):
    q = y[:nq]
    u = y[nq:]
    system.step_callback(t, q, u)


def dydt(t, y):
    t = float(t)
    q, u = y[:nq], y[nq:]

    q_dot = rod.q_dot(t, q, u)

    W_c = rod.W_c(t, q).tocsr(fix_size=True)
    la_c = rod.la_c(t, q, u)
    h = W_c @ la_c + system.h(t, q, u)

    u_dot = M_inv @ h
    dydt = np.concatenate((q_dot, u_dot))

    # fixed the first node
    dydt[:7] = 0.0
    dydt[nq : nq + 6] = 0.0
    return dydt


t0 = perf_counter()
# t, y =runge_kutta_4(dydt, y0, 0, tend, dt, step_callback=step_callback)
t, y = runge_kutta_3_8(dydt, y0, 0, tend, dt, step_callback=step_callback)

# sol_ivp = solve_ivp(
#     dydt, (0, tend), y0, method="RK45", t_eval=np.arange(0, tend + dt, dt)
# )
# t, y = sol_ivp.t, sol_ivp.y.T

print(f"Runge Kutta time: {perf_counter() - t0:.2f} s")


sol = Solution(system, t, y[:, : system.nq])

t2, q2 = sol.t, sol.q[:, rod.qDOF]

#############
# plot result
#############
qs1 = q1.reshape((t1.shape[0], -1, 7))

qs2 = q2.reshape((t2.shape[0], -1, 7))


# plot result
nelement = qs1.shape[1]
for n in np.arange(nelement + 1)[:: nelement // int(nelement / 5)]:
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t1, qs1[:, n, 0], "--.")
    plt.plot(t1, qs2[:, n, 0], "-")
    plt.grid()
    plt.subplot(4, 1, 2)
    plt.plot(t1, qs1[:, n, 2], "--.")
    plt.plot(t1, qs2[:, n, 2], "-")
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.plot(t1, qs1[:, n, 0] - qs2[:, n, 0], "-r")
    plt.yscale("log")
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.plot(t1, qs1[:, n, 1] - qs2[:, n, 1], "-r")
    plt.yscale("log")
    plt.grid()
plt.show(block=True)
