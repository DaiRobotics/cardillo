import sys
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import B_Moment
from cardillo.math import e1, e3
from cardillo.rods import CircularCrossSection, animate_beam, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions
from cProfile import Profile
from cardillo.rods.discreteRod import DiscreteRod


def gen_helix(
    nelements: int = 100,
    n_coil: float = 2,
    slenderness: float = 1.0e2,
    rod="DiscreteRod",
):
    # geometry of the rod
    R0 = 10  # radius of the helix
    h = 50  # height of the helix
    c = h / (2 * R0 * np.pi * n_coil)  # pitch of the helix
    length = np.sqrt(1 + c**2) * R0 * 2 * np.pi * n_coil
    cc = 1 / (np.sqrt(1 + c**2))

    alpha = lambda xi: 2 * np.pi * n_coil * xi
    alpha_xi = 2 * np.pi * n_coil

    # cross section properties
    width = length / slenderness
    radius = width / 2
    cross_section = CircularCrossSection(radius=radius)
    A = cross_section.area
    Ip, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    E = 1.0  # Young's modulus
    G = 0.5  # shear modulus
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ip, E * I2, E * I3])
    material_model = Simo1986(Ei, Fi)

    # initialize system
    system = System()

    # initial positions and orientations at xi=0
    alpha_0 = alpha(0)

    r_OP0 = R0 * np.array([np.sin(alpha_0), -np.cos(alpha_0), c * alpha_0])

    e_x = cc * np.array([np.cos(alpha_0), np.sin(alpha_0), c])
    e_y = np.array([-np.sin(alpha_0), np.cos(alpha_0), 0])
    e_z = cc * np.array([-c * np.cos(alpha_0), -c * np.sin(alpha_0), 1])

    A_IB0 = np.vstack((e_x, e_y, e_z))
    A_IB0 = A_IB0.T

    #####
    # rod
    #####
    if rod == "DiscreteRod":
        Rod = DiscreteRod
    elif rod == "CosseratRod":
        Rod = make_CosseratRod(polynomial_degree=1)
    # generate position coordinates for straight initial configuration
    q0 = Rod.straight_configuration(
        nelements,
        length,
        r_OP0=r_OP0,
        A_IB0=A_IB0,
    )
    # create rod
    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
    )
    system.add(rod)

    ##########
    # clamping
    ##########
    clamping = RigidConnection(system.origin, rod, xi2=0)
    system.add(clamping)

    ################
    # applied moment
    ################
    Fi = material_model.Fi
    M = lambda t: (R0 * alpha_xi**2) / (length**2) * (c * e1 * Fi[0] + e3 * Fi[2]) * t
    moment = B_Moment(M, rod, 1)
    system.add(moment)

    # assemble system
    system.assemble()
    return {"system": system, "rod": rod}
