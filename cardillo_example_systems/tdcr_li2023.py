from cardillo.constraints import RigidConnection
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    RodTendonForce,
    DiscreteRod,
    RodTendonForceIntegrator,
)
from cardillo.system import System

import numpy as np


def gen_tdcr_li2023(rod_nelement=10, g_accel=9.81, damping_ratio=0, statics=True):
    # ---- pysical parameters ----
    rod_l0 = 0.192  # [m] length of rod
    rod_r_base = 14e-3  # [m] radius at bottom of rod
    rod_r_tip = 8.5e-3  # [m] radius at tip of rod
    rod_density = 1.41e3  # density of material
    rod_r_OC0 = np.array([0, 0, 0.2], dtype=np.float64)
    rod_A_IB0 = np.eye(3, dtype=np.float64)
    E, G = 2.563e5, 8.543e4

    # ---- rod ----
    radius = lambda xi: rod_r_base * (1 - xi) + rod_r_tip * xi
    cross_section = CircularCrossSection(radius)
    EA = lambda xi: E * cross_section.area(xi)
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
    GA = lambda xi: G * cross_section.area(xi)
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
    material_model = Simo1986(
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
    )

    # ---- system ----
    system = System()

    # ---- inital configuration ----
    def r_OP(xi):
        return np.array([xi * rod_l0, 0, 0], dtype=np.float64)

    A_IB = lambda xi: np.eye(3, dtype=np.float64)
    q0 = DiscreteRod.pose_configuration(
        rod_nelement,
        r_OP,
        A_IB,
        r_OC0=rod_r_OC0,
        A_IB0=rod_A_IB0,
    )
    Q = q0.copy()

    rod = DiscreteRod(
        cross_section,
        material_model,
        rod_nelement,
        Q=Q,
        q0=q0,
        cross_section_inertias=CrossSectionInertias(rod_density, cross_section),
        damping_ratio=damping_ratio,
    )

    # ---- rigid connections ----
    rc = RigidConnection(rod, system.origin, xi1=0)

    # ---- tendons ----
    n_tendons = 4
    tendons = []
    B_r_CP_lists = [
        [
            np.array(
                [
                    0,
                    radius(xi) * np.sin(phi),
                    -radius(xi) * np.cos(phi),
                ]
            )
            for xi in np.linspace(0, 1, rod_nelement + 1)
        ]
        for phi in np.linspace(0, 2 * np.pi, n_tendons, endpoint=False)
    ]
    for B_r_CP_list in B_r_CP_lists:
        n = len(B_r_CP_list)
        if statics:
            tendon = RodTendonForce(
                rod,
                [i / (n - 1) for i in range(n)],
                B_r_CPs=B_r_CP_list,
            )
            tendons.append(tendon)
        else:
            tendon = RodTendonForceIntegrator(
                rod,
                [i / (n - 1) for i in range(n)],
                B_r_CPs=B_r_CP_list,
            )
            tendons.append(tendon)

    system.add(rod, rc, *tendons)

    # ---- external forces ----
    if statics:
        gravity = Force_line_distributed(
            lambda t, xi: rod_density
            * cross_section.area(xi)
            * g_accel
            * t
            * np.array([0, 0, -1], dtype=np.float64),
            rod,
        )
    else:
        gravity = Force_line_distributed(
            lambda t, xi: rod_density
            * cross_section.area(xi)
            * g_accel
            * np.array([0, 0, -1], dtype=np.float64),
            rod,
        )
    system.add(gravity)

    system.assemble()

    return {
        "system": system,
        "tendons": tendons,
        "rod": rod,
    }
