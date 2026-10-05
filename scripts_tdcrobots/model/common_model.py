from abc import ABC
from pathlib import Path

from cardillo.constraints import RigidConnection
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    DiscreteRod,
    RodTendonForce,
)

from cardillo.system import System

import numpy as np

# csv / stl inputs
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# G_ACCEL = -1
G_ACCEL = 9.81
# G_ACCEL = 7
# G_ACCEL = 0 # Test

SETPOINT_TABLE = {
    "A": np.array([15.438e-2, 4.335e-2, 3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2, 9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2, 0.000e-2, -9.842e-2]),
}


def paper_to_cardillo(u):
    X, Y, Z = u
    return np.array([Y, Z, X])


SETPOINT_TABLE = {k: paper_to_cardillo(u) for k, u in SETPOINT_TABLE.items()}

class CommonModel(ABC):
    def __init__(self, damping_ratio=0, la_pre=0.0, stiff_scale=1.0):
        super().__init__()
        # ---- pysical parameters ----
        rod_nelement = 10  # 1000
        rod_l0 = 0.192  # [m] length of rod
        rod_r0_base = 1.4e-2  # [m] radius at bottom of rod
        rod_r0_tip = 8.5e-3  # [m] radius at tip of rod (original with 60% tip to base ratio)
        # rod_r0_tip = 8.5e-3 * 0.5  # [m] radius at tip of rod for 30% tip to base ratio
        # rod_r0_tip = 8.5e-3 * 1.5  # [m] radius at tip of rod for 90% tip to base ratio
        # rod_r0_tip = 1.4e-2 * 0.95  # [m] radius at tip of rod for 100% tip to base ratio
        self.rod_density = 1.41e3  # density of material
        rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
        rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
        # stiff_scale != 1 gives a wrong rod for the model error tests
        E, G = 2.563e5 * stiff_scale, 8.543e4 * stiff_scale

        # ---- rod ----
        radius = lambda xi: rod_r0_base * (1 - xi) + rod_r0_tip * xi
        self.cross_section = CircularCrossSection(radius)
        EA = lambda xi: E * self.cross_section.area(xi)
        EI = lambda xi: E * self.cross_section.second_moment(xi)[1, 1]
        GA = lambda xi: G * self.cross_section.area(xi)
        GJ = lambda xi: G * self.cross_section.second_moment(xi)[0, 0]
        material_model = Simo1986(
            lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
            lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
        )

        # ---- system ----
        self.system = System()

        # ---- inital configuration ----
        def r_OP(xi):
            return np.array([xi * rod_l0, 0, 0], dtype=np.float64)

        A_IB = lambda xi: np.eye(3, dtype=np.float64)
        q0 = DiscreteRod.pose_configuration(
            rod_nelement,
            r_OP,
            A_IB,
            A_IB0=rod_A_IB0,
        )
        Q = q0.copy()

        self.rod = DiscreteRod(
            self.cross_section,
            material_model,
            rod_nelement,
            Q=Q,
            q0=q0,
            cross_section_inertias=CrossSectionInertias(
                self.rod_density, self.cross_section
            ),
            damping_ratio=damping_ratio,
        )

        # ---- rigid connections ----
        rc = RigidConnection(self.rod, self.system.origin, xi1=0)

        # ---- tendons ----
        self.n_tendons = 4
        # self.n_tendons = 3
        self.tendons = []
        B_r_CP_lists = [
            [
                rod_A_IB0.T
                @ np.array(
                    [
                        radius(xi) * np.cos(phi),
                        radius(xi) * np.sin(phi),
                        0,
                    ]
                )
                for xi in np.linspace(0, 1, rod_nelement + 1)
            ]
            for phi in np.linspace(0, 2 * np.pi, self.n_tendons, endpoint=False)
        ]
        for B_r_CP_list in B_r_CP_lists:
            n = len(B_r_CP_list)
            tendon = RodTendonForce(
                self.rod,
                [i / (n - 1) for i in range(n)],
                B_r_CPs=B_r_CP_list,
            )
            self.tendons.append(tendon)

        # ---- tendon pretension ----
        # goes into sys.h, the controller compensates it, real tension = la_pre + la_tau
        self.la_pre = la_pre
        for tendon in self.tendons:
            tendon.set_force(la_pre)

        self.system.add(self.rod, rc, *self.tendons)

        # ---- external forces ----
        self.gravity = Force_line_distributed(
            lambda t, xi: self.rod_density
            * self.cross_section.area(xi)
            * G_ACCEL
            * np.array([0, -1.0, 0], dtype=np.float64),
            self.rod,
        )
        self.system.add(self.gravity)

def la_t_plot(model, la_ts, sol):
    import matplotlib.pyplot as plt
    ts = sol.t

    f_pre = getattr(model, "pretension", 0.0)
    fig, ax = plt.subplots(num="TendonForces", figsize=(8, 4))
    for k in range(model.n_tendons):
        ax.plot(ts, la_ts[:, k] + f_pre, label=f"tendon {k+1}")
    if f_pre > 0:
        ax.axhline(f_pre, color="k", ls="--", lw=0.8, label=f"pretension ({f_pre:g} N)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Tendon Force [N]")
    ax.set_title("Tendon Forces"); ax.legend(); ax.grid(True)
    plt.show()

def compute_la_ts(controller, sol):
    la_ts = np.array([
        controller.la_tau(t, q[controller.qDOF], u[controller.uDOF])
        for t, q, u in zip(sol.t, sol.q, sol.u)
    ])
    return la_ts
