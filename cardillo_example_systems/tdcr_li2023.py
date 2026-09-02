from cardillo.constraints import RigidConnection
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    RodTendonForce,
    RodTendonKinematics,
    DiscreteRod,
)
from cardillo.system import System
from cardillo.utility.coo_matrix import CooMatrix

import numpy as np


class Controller:
    def __init__(
        self,
        rod,
        tendons: list[RodTendonKinematics],
        Kp=0.0,
        name="controller",
    ) -> None:
        self.rod = rod
        self.tendons = tendons
        self.name = name
        self.nq = self.nla_tau = len(tendons)
        self.ntau = 3 * 2
        self.q0 = np.zeros(self.nq, dtype=np.float64)

        self.Kp = Kp
        self.dr_OP_dla_t_inv = np.zeros((self.nla_tau, 3))

    def q_dot(self, t, q, u):
        r_OP = q[-7 - self.nla_tau : -4 - self.nla_tau]
        tau = self.tau(t)
        r_OP_des, v_P_def = tau[:3], tau[3:6]
        return self.dr_OP_dla_t_inv @ (v_P_def + (r_OP_des - r_OP) * self.Kp)

    def q_dot_q(self, t, q, u):
        coo = CooMatrix((self.nq, self._nq))
        coo[:, -self.nla_tau - 7 : -self.nla_tau - 4] = self.dr_OP_dla_t_inv * (
            -self.Kp
        )
        return coo

    def W_tau(self, t, q):
        W_tau = CooMatrix((self._nu, self._nq))
        for i, td, qDOF, uDOF in zip(
            range(self.nla_tau), self.tendons, self._td_qDOF, self._td_uDOF
        ):
            W_tau[i, uDOF, i] = td.W_t(q[qDOF])
        return W_tau

    def Wla_tau_q(self, t, q, u):
        coo = CooMatrix((self._nu, self._nq))
        for i, td, la_tau, qDOF, uDOF in zip(
            range(self.nla_tau),
            self.tendons,
            self.la_tau(t, q, u),
            self._td_qDOF,
            self._td_uDOF,
        ):
            q_td = q[qDOF]
            W_t_q = td.W_t_q(q_td)
            coo[2 * i, uDOF, qDOF] = W_t_q * la_tau
            coo[2 * i + 1, uDOF, -self.nla_tau + i] = td.W_t(q[qDOF])
        return coo

    def Wla_tau_u(self, t, q, u):
        return None

    def tau(self, t):
        return np.zeros(self.ntau, dtype=np.float64)

    def la_tau(self, t, q, u):
        return q[-self.nla_tau :]

    def assembler_callback(self):
        qDOF = self.rod.qDOF
        uDOF = self.rod.uDOF

        self._td_qDOF = []
        self._td_uDOF = []
        for td in self.tendons:
            assert min(qDOF) <= min(td.qDOF) and max(qDOF) >= max(td.qDOF)
            self._td_qDOF.append(np.searchsorted(qDOF, td.qDOF))
            assert min(uDOF) <= min(td.uDOF) and max(uDOF) >= max(td.uDOF)
            self._td_uDOF.append(np.searchsorted(uDOF, td.uDOF))

        self.qDOF = np.concatenate([qDOF, self.my_qDOF])
        self.uDOF = uDOF
        self._nq = len(self.qDOF)
        self._nu = len(self.uDOF)


def gen_tdcr_li2023(
    rod_nelement=10, g_accel=9.81, damping_ratio=0, statics=True, controller=False
):
    # ---- pysical parameters ----
    rod_l0 = 0.192  # [m] length of rod
    rod_r_base = 14e-3  # [m] radius at bottom of rod
    rod_r_tip = 8.5e-3  # [m] radius at tip of rod
    rod_density = 1.41e3  # density of material
    rod_r_OC0 = np.array([0, 0, 0], dtype=np.float64)
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

    # ---- external forces ----
    if statics:
        rod_gravity = Force_line_distributed(
            lambda t, xi: rod_density
            * cross_section.area(xi)
            * g_accel
            * t
            * np.array([0, 0, -1], dtype=np.float64),
            rod,
        )
    else:
        rod_gravity = Force_line_distributed(
            lambda t, xi: rod_density
            * cross_section.area(xi)
            * g_accel
            * np.array([0, 0, -1], dtype=np.float64),
            rod,
        )
    system.add(rod_gravity)

    # ---- tendons ----
    n_tendons = 4
    tendons = []
    assert rod_nelement % 12 == 0, "rod_nelement must be a multiple of 12"
    B_r_CP_lists = [
        [
            np.array(
                [
                    0,
                    radius(xi) * np.sin(phi),
                    -radius(xi) * np.cos(phi),
                ]
            )
            for xi in np.linspace(0, 1, 13)
        ]
        for phi in np.linspace(0, 2 * np.pi, n_tendons, endpoint=False)
    ]
    for B_r_CP_list in B_r_CP_lists:
        n_vert = len(B_r_CP_list)
        if statics:
            tendon = RodTendonForce(
                rod,
                xis=[i / (n_vert - 1) for i in range(n_vert)],
                B_r_CPs=B_r_CP_list,
            )
        else:
            tendon = RodTendonKinematics(
                rod,
                xis=[i / (n_vert - 1) for i in range(n_vert)],
                B_r_CPs=B_r_CP_list,
            )
        tendons.append(tendon)

    system.add(rod, rc, *tendons)

    # --- controller ---
    if controller:
        controller = Controller(rod, tendons)
        system.add(controller)
    system.assemble()

    return {
        "system": system,
        "tendons": tendons,
        "rod": rod,
        "rod_gravity": rod_gravity,
        "controller": controller,
    }
