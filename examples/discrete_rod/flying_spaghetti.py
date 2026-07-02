from time import perf_counter

import numpy as np
from jax import numpy as jnp

from cardillo import System
from cardillo.math import A_IB_basic
from cardillo.forces import Force, Moment
from cardillo.rods import (
    DiscreteRod,
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.solver import ScipyDAE, Moreau

nelement = 10
L = 10
radius = 0.03
cross_section = CircularCrossSection(radius)
cross_section_inertias = CrossSectionInertias(
    A_rho0=1, B_I_rho0=np.diag([20, 10, 10])
)  # Hesse
cross_section_inertias = CrossSectionInertias(
    A_rho0=1, B_I_rho0=np.diag([20, 10, 10])
)  # Boyer

EI = 500
EA = 1e4
GA = 1e4
GJ = 500
material_model = Simo1986(
    np.array([EA, GA, GA]),
    np.array([GJ, EI, EI]),
)
Q = DiscreteRod.straight_configuration(
    nelement, L, r_OP0=np.array([6, 0, 0]), A_IB0=A_IB_basic(-np.pi + np.atan(8 / 6)).y
)
rod = DiscreteRod(
    cross_section,
    material_model,
    nelement,
    Q=Q,
    cross_section_inertias=cross_section_inertias,
    damping_ratio=0,
)


def f(t):
    return 80 * t * (t <= 2.5) + (t > 2.5) * (t <= 5) * (400 - 80 * t)


force = Force(lambda t: jnp.array([f(t) / 10, 0, 0]), rod, xi=0)
moment = Moment(lambda t: jnp.array([0, f(t), -f(t) / 2]), rod, xi=0)


system = System()
system.add(rod, force, moment)
system.assemble()


solver = ScipyDAE(system, 7.0, 1e-1, rtol=1e-3, atol=1e-6)
system.t0 = 0.0
solver.fun(system.t0, solver.y0, solver.y0)
solver.jac(system.t0, solver.y0, solver.y0)

# from cProfile import Profile
# prof = Profile()
# prof.enable()

t0 = perf_counter()
sol = solver.solve()
print(f"Simulation time: {perf_counter() - t0:.2f} s")

# prof.disable()
# prof.dump_stats("prof.prof")


t = sol.t
weights = np.ones(nelement + 1)
weights[1:-1] = 2
weights /= np.sum(weights)
r_OC = sol.q[:, rod.qDOF].reshape((-1, nelement + 1, 7))[..., :3]
# center of mass
r_OC_com = np.tensordot(r_OC, weights, axes=(1, 0))


def x_ref(t):
    if t <= 2.5:
        return 3 + 2 / 15 * t**3
    elif t <= 5:
        return 43 / 6 - 5 * t + 2 * t**2 - 2 / 15 * t**3
    else:
        return -19 / 2 + 5 * t


r_OC_ref = np.array([[x_ref(ti), 0, 4] for ti in t])

# plot
# https://www.sciencedirect.com/science/article/pii/S0045794912001368
# analytical solution of center of mass
from matplotlib import pyplot as plt

plt.figure()
plt.subplot(1, 2, 1)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i], "r")
    plt.plot(t, r_OC_com[:, i], "--")
plt.grid()

plt.subplot(1, 2, 2)
for i in range(3):
    plt.plot(t, r_OC_ref[:, i] - r_OC_com[:, i], label=f"dr_{i}")
plt.grid()
plt.yscale("log")
plt.legend()

# configurations
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(r_OC[:, 0, 0] - 6, r_OC[:, 0, 2], "k")
plt.plot(r_OC[:, -1, 0] - 6, r_OC[:, -1, 2], "--k")
for i in [0, 20, 30, 38, 44, 50, 55, 58, 61, 65]:
    plt.plot(r_OC[i, :, 0] - 6, r_OC[i, :, 2])
plt.grid()
plt.axis("equal")

plt.subplot(2, 1, 2)
plt.plot(r_OC[:, 0, 1], r_OC[:, 0, 2], "k")
plt.plot(r_OC[:, -1, 1], r_OC[:, -1, 2], "--k")
for i in [0, 25, 35, 38, 45]:
    plt.plot(r_OC[i, :, 1], r_OC[i, :, 2])
plt.grid()
plt.axis("equal")


plt.show()

# export
# from pathlib import Path
# import sys

# dir_name = Path(sys.argv[0]).parent
# system.export(dir_name, f"vtk", sol, fps=10)
