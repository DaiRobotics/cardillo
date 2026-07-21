import numpy as np

from cardillo.system import System

from cardillo.forces import Force
from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    DiscreteRod,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.constraints import RigidConnection
from cardillo.solver import Newton


def gen_cantilever_beam(nelement=20, rod="DiscreteRod"):
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

    if rod == "DiscreteRod":
        Rod = DiscreteRod
    elif rod == "CosseratRod":
        Rod = make_CosseratRod(polynomial_degree=1)

    Q = Rod.straight_configuration(nelement, L)
    rod = Rod(
        cross_section,
        material_model,
        nelement,
        Q=Q,
        cross_section_inertias=cross_section_inertias,
    )

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

    solver = Newton(system, n_load_steps=2)
    sol_statics = solver.solve()
    system.set_new_initial_state(sol_statics.q[-1], sol_statics.u[-1])

    system.remove(force)
    system.remove(force_gravity)
    force_gravity = Force_line_distributed(
        lambda t, xi: np.array([0, 0, mass * gravity / L]), rod
    )
    system.add(force_gravity)
    system.assemble()

    return {"system": system, "rod": rod}
