from abc import ABC

from cardillo.math import A_IB_basic
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.forces import Force
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    DiscreteRod,
    RodTendonForce,
)

from cardillo.solver import ScipyDAE, BackwardEuler, Newton, SolverOptions, Solution

from cardillo.system import System

from cardillo.interactions import nPointInteraction

import numpy as np
from scipy.sparse.linalg import splu

from espedal_control_test import p2p_vis_plot
from runge_kutta import RungeKutta


G_ACCEL = 9.81

class CommonModel(ABC):
    def __init__(self, damping_ratio=0):
        super().__init__()
        # ---- pysical parameters ----
        rod_nelement = 10  # 1000
        rod_l0 = 0.192  # [m] length of rod
        rod_r0_base = 1.4e-2  # [m] radius at bottom of rod
        rod_r0_tip = 8.5e-3  # [m] radius at tip of rod
        self.rod_density = 1.41e3  # density of material
        rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
        rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
        E, G = 2.563e5, 8.543e4

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
        # self.n_tendons = 4
        self.n_tendons = 3
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

class DynamicController():
    def __init__(
        self,
        system,
        r_OP_ref,
        v_P_ref,
        Kp,
        Kd,
        inv_damping,
        rod,
        tendons: list[RodTendonForce],
        name="tendon_force_control",
    ) -> None:
        self.system = system
        self.r_OP_ref = r_OP_ref
        self.v_P_ref = v_P_ref
        self.Kp = Kp
        self.Kd = Kd
        self.inv_damping = inv_damping
        self.rod = rod
        self.tendons = tendons
        self.name = name
        
        self.M_inv2 = None
        self.nq = len(tendons)
        self.q0 = np.zeros(self.nq)
        self._la_t_dot = np.zeros(self.nq)
        self._la_t = np.zeros(self.nq)


    def assembler_callback(self):
        self.qDOF = np.concatenate([self.my_qDOF, self.rod.qDOF])
        self._nq1 = len(self.my_qDOF)
        self.uDOF = self.rod.uDOF
        # self.uDOF_free = np.setdiff1d(self.uDOF, self.uDOF[self.rod.nodalDOF_u[0]]) # Dont work with free DOFs in Controller, instead in solver!
        self.C_1 = np.zeros((3,self.rod.nq))
        self.C_1[:, self.rod.nodalDOF_r[-1]] = np.eye(3) 

    # def apply_tendon_forces(self, t, q):
    #     la_t = q[: self._nq1] + self.la_t_ref(t)
    #     for td, la_t_i in zip(self.tendons, la_t):
    #         td.set_force(lambda t, la=la_t_i: la)

    def control_law(self, t, q, u):
        sys = self.system
        
        q_sys = np.zeros(sys.nq)
        q_sys[self.qDOF] = q
        u_sys = np.zeros(sys.nu)
        u_sys[self.uDOF] = u

        # Build M_inv2
        if self.M_inv2 is None:
            rod = self.rod
            q_rod = q_sys[rod.qDOF]
            B = rod.q_dot_u(t, q_rod).toarray()
            C_2 = self.C_1 @ B
            M = rod.M(t, q_rod).tocsc()
            # M_free = sys.M(t, q_sys)[self.uDOF_free][:, self.uDOF_free]
            self.M_inv2 = splu(M).solve(C_2.T).T

        # Build tendon force directions W_t
        W_t = np.zeros((sys.nu, self.nq))
        for j, td in enumerate(self.tendons):
            np.add.at(W_t[:, j], td.uDOF, -td.W_l(t, q_sys[td.qDOF]))
            # W_t[td.uDOF, j] = -td.W_l(1.0, q_sys[td.qDOF])

        # Current tendon force
        # la_t = q[: self._nq1]

        # Build h consisting of gyroscopic and external forces from system and add compliance forces
        # h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ la_t
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ self._la_t

        # Build y_0_ddot and Jacobian
        y_0_ddot = - self.M_inv2 @ h
        J = self.M_inv2 @ W_t
        # J_inv = np.linalg.inv(J) # Pure inverse (Causes singularities, add damped inverse)
        J_inv = J.T @ np.linalg.solve(J @ J.T + self.inv_damping * np.eye(3), np.eye(3)) # Moore Penrose Pseudo Inverse

        # Error terms
        r_OP_ref = self.r_OP_ref(t)
        v_P_ref = self.v_P_ref(t)
        a_P_ref = np.zeros(3) # CHECK

        r_OP = self.rod._view_nodal_q(q[self._nq1 :])[-1, :3]
        # print(r_OP)
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        
        e = r_OP_ref - r_OP
        e_dot = v_P_ref - v_P

        a = a_P_ref + self.Kd * e_dot + self.Kp * e       # linear PD Control law
        # a = a_P_ref + beta * e_dot + gamma * np.sign(e_dot + beta * e)  # Switching mode control law

        # Input-Output Feedback Linearization
        la_t = J_inv @ (a + y_0_ddot)
        # la_t *= 0
        return la_t
    
    def q_dot(self, t, q, u):
        return self._la_t_dot
    
    def step_callback(self, t, q, u):
        la_t = self.control_law(t, q, u)
        self._la_t = la_t
        for td, la_t_i in zip(self.tendons, la_t):
            td.set_force(lambda t, la=la_t_i: la)
        return q, u

if __name__ == "__main__":
    # model = CommonModel(damping_ratio=1e-2)
    model = CommonModel()
    system = model.system
    rod = model.rod
    tendons = model.tendons
    
    # Parameters
    # Kp = 1000
    # Kd = 80
    Kp = 0
    Kd = 0
    inv_damping = 1e-3

    # Trajectories
    r_OP_ref_fn = lambda t: np.array([0.0, 0.0, 0.192])
    v_P_ref_fn = lambda t: np.zeros(3)
    
    controller = DynamicController(system, r_OP_ref_fn, v_P_ref_fn, Kp, Kd, inv_damping, rod, tendons)
    system.add(controller)
    system.assemble()

    # Solver
    # t_sim = 1
    t_sim = 0.03
    dt = 1e-4
    # solver = BackwardEuler(system, t_sim, dt)
    solver = ScipyDAE(system, t_sim, dt)

    # fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
    # fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
    # solver = RungeKutta(system, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF)
    
    sol = solver.solve()

    p2p_vis_plot(model, sol, r_OP_ref_fn)