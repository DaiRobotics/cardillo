from time import perf_counter
from cProfile import Profile

import numpy as np
from cardillo.solver import ScipyDAE, Moreau, BackwardEuler, Radau


from cardillo.example_systems.flying_spaghetti import *

#################
# ScipyDAE solver
#################

solver = ScipyDAE(system, 7.0, 1e-2, rtol=1e-3, atol=1e-6)
solver.fun(system.t0, solver.y0, solver.y0)
solver.jac(system.t0, solver.y0, solver.y0)

# prof = Profile()
# prof.enable()

t0 = perf_counter()
sol1 = solver.solve()
print(f"Simulation time: {perf_counter() - t0:.2f} s")

# prof.disable()
# prof.dump_stats("prof.prof")


###############
# Radau solver
###############
# solver = Radau(system, 7, 1e-2, rtol=1e-3, atol=1e-6, stages=3)
# solver.fun(system.t0, solver.y0, solver.y0)
# solver.jac(system.t0, solver.y0, solver.y0)

# prof = Profile()
# prof.enable()

# t0 = perf_counter()
# sol2 = solver.solve()
# print(f"Simulation time: {perf_counter() - t0:.2f} s")

# prof.disable()
# prof.dump_stats("prof.prof")


for sol in [sol1]:
    # for sol in [sol1, sol2]:
    t = sol.t

    # analytical solution of center of mass
    r_OC_ref = np.array([[x_ref(ti), 0, 4] for ti in t])

    # center of mass
    weights = np.ones(nelement + 1)
    weights[1:-1] = 2
    weights /= np.sum(weights)
    r_OC = sol.q[:, rod.qDOF].reshape((-1, nelement + 1, 7))[..., :3]
    r_OC_com = np.tensordot(r_OC, weights, axes=(1, 0))

    # plot
    # https://www.sciencedirect.com/science/article/pii/S0045794912001368
    # analytical solution of center of mass
    from matplotlib import pyplot as plt

    plt.figure("center of mass")
    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(t, r_OC_ref[:, i], "r")
        plt.plot(t, r_OC_com[:, i], "--")
    plt.grid()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(t, r_OC_ref[:, i] - r_OC_com[:, i], label=f"dr_{i}")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

    # configurations
    plt.figure("configurations")
    plt.subplot(2, 1, 1)
    plt.plot(r_OC[:, 0, 0] - 6, r_OC[:, 0, 2], "k")
    plt.plot(r_OC[:, -1, 0] - 6, r_OC[:, -1, 2], "--k")
    for ti in [0, 2, 3, 3.8, 4.4, 5, 5.5, 5.8, 6.1, 6.5]:
        i = int(ti // (t[1] - t[0]))
        plt.plot(r_OC[i, :, 0] - 6, r_OC[i, :, 2])
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(2, 1, 2)
    plt.plot(r_OC[:, 0, 1], r_OC[:, 0, 2], "k")
    plt.plot(r_OC[:, -1, 1], r_OC[:, -1, 2], "--k")
    for ti in [0, 2.5, 3.5, 3.8, 4.5]:
        i = int(ti // (t[1] - t[0]))
        plt.plot(r_OC[i, :, 1], r_OC[i, :, 2])
    plt.grid(True)
    plt.axis("equal")


plt.show()

# export
# from pathlib import Path
# import sys

# dir_name = Path(sys.argv[0]).parent
# system.export(dir_name, f"vtk", sol, fps=10)
