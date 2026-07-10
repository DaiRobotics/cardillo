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


# https://www.sciencedirect.com/science/article/pii/S0045794912001368
# analytical solution of center of mass
def x_ref(t):
    if t <= 2.5:
        return 3 + 2 / 15 * t**3
    elif t <= 5:
        return 43 / 6 - 5 * t + 2 * t**2 - 2 / 15 * t**3
    else:
        return -19 / 2 + 5 * t
