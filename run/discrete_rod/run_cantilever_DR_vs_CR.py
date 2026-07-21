import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt


from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_comparison_dynamic_cantilever_beam")

from cardillo.solver import ScipyDAE
from cardillo_example_systems.dynamic_cantilever import gen_cantilever_beam

dt = 1e-2
tend = 3.0

##############
# discrete rod
##############
ret = gen_cantilever_beam(rod="DiscreteRod")
rod = ret["rod"]
system = ret["system"]

solver = ScipyDAE(system, tend, dt)

t0 = perf_counter()
sol = solver.solve()
print(f"Discrete rod time: {perf_counter() - t0:.2f} s")

t1, q1 = sol.t, sol.q[:, rod.qDOF]


##############
# Cosserat rod
##############
ret = gen_cantilever_beam(rod="CosseratRod")
rod = ret["rod"]
system = ret["system"]

solver = ScipyDAE(system, tend, dt)

t0 = perf_counter()
sol = solver.solve()
print(f"Cosserat rod time: {perf_counter() - t0:.2f} s")

t2, q2 = sol.t, sol.q[:, rod.qDOF]


#############
# plot result
#############
qs1 = q1.reshape((t1.shape[0], -1, 7))

qs2 = q2.reshape((t2.shape[0], 7, -1)).swapaxes(1, 2)


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
