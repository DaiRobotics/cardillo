import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.constraints import RigidConnection

from cardillo.solver import *
from cardillo.forces import *
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.system import System

from cardillo.rods import DiscreteRod

import cProfile


def cantilever_beam(Solver, Rod, profile=False):
    nelement = 20
    L = 1
    mass = 0.4
    gravity = 9.81
    radius = 0.03
    density = mass / (L * np.pi * radius**2)
    cross_section = CircularCrossSection(radius)
    cross_section_inertias = CrossSectionInertias(
        density=density, cross_section=cross_section
    )
    E, G = 7e5, 2e5
    EI = E * cross_section.second_moment[1, 1]
    EA = E * cross_section.area
    GA = G * cross_section.area
    GJ = G * cross_section.second_moment[0, 0]
    material_model = Simo1986(
        np.array([EA, GA, GA]),
        np.array([GJ, EI, EI]),
    )
    Q = Rod.straight_configuration(nelement, L)
    rod = Rod(
        cross_section,
        material_model,
        nelement,
        Q=Q,
        cross_section_inertias=cross_section_inertias
    )

    # nodes = rod.nodes

    system = System()
    f_fun = lambda t: t * np.array([0, -0.5, 0])
    force = Force(f_fun, rod, xi=1)
    force_gravity = Force_line_distributed(
        lambda t, xi: t * np.array([0, 0, mass * gravity / L]), rod
    )
    rc = RigidConnection(system.origin, rod, xi2=0)

    system.add(rod)
    system.add(force)
    system.add(force_gravity)
    system.add(rc)
    system.assemble()

    solver = Newton(system, n_load_steps=10)
    sol_statics = solver.solve()
    system.set_new_initial_state(sol_statics.q[-1], sol_statics.u[-1])

    system.remove(force)
    system.remove(force_gravity)
    force_gravity = Force_line_distributed(
        lambda t, xi: np.array([0, 0, mass * gravity / L]), rod
    )
    system.add(force_gravity)
    system.assemble()

    t1 = 3
    solver = Solver(system, t1, 1e-2)

    if profile:
        solver.fun(0.0, solver.y0, solver.y0)
        solver.jac(0.0, solver.y0, solver.y0)
        prof = cProfile.Profile()
        prof.enable()

    t0 = perf_counter()
    sol = solver.solve()
    print("time:", perf_counter() - t0)

    if profile:
        prof.disable()
        prof.dump_stats("prof.prof")

    return sol.t, sol.q[:, rod.qDOF]


Rod = make_CosseratRod(polynomial_degree=1)

t1, q_rod1 = cantilever_beam(ScipyDAE, DiscreteRod, profile=False)
t2, q_rod2 = cantilever_beam(ScipyDAE, Rod, profile=False)


qs1 = q_rod1.reshape((t1.shape[0], -1, 7))

qs2 = q_rod2.reshape((t2.shape[0], 7, -1)).swapaxes(1, 2)


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

# # render solution
# step = int(len(q)//1000)
# sol = Solution(system, t_eval[::step], q[::step], u[::step])
# ren = Renderer(system)
# ren.render_solution(sol, repeat=True)
