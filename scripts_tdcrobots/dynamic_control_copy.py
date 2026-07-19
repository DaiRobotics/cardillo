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


# G_ACCEL = -1
# G_ACCEL = 9.81
G_ACCEL = 0 # Test

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
        t_lag=4e-3,
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
        self.t_lag = t_lag
        self.name = name
        
        self.M_inv2 = None
        self.nq = len(tendons)
        self.q0 = np.zeros(self.nq)
        self._la_t_dot = np.zeros(self.nq)
        self._la_t = np.zeros(self.nq)
        self._la_t_des = np.zeros(self.nq)

        self.la_ts = []
        self.ts = []

    def assembler_callback(self):
        self.qDOF = np.concatenate([self.my_qDOF, self.rod.qDOF])
        self._nq1 = len(self.my_qDOF)
        self.uDOF = self.rod.uDOF
        # self.uDOF_free = np.setdiff1d(self.uDOF, self.uDOF[self.rod.nodalDOF_u[0]]) # Dont work with free DOFs in Controller, instead in solver!
        self.C_1 = np.zeros((3,self.rod.nq))
        self.C_1[:, self.rod.nodalDOF_r[-1]] = np.eye(3) 


    def control_law(self, t, q, u):
        sys = self.system
        rod = self.rod
        q_rod = rod.q

        q_sys = np.zeros(sys.nq)
        q_sys[self.qDOF] = q
        u_sys = np.zeros(sys.nu)
        u_sys[self.uDOF] = u

        # Build M_inv2
        if self.M_inv2 is None:
            q_rod = q_sys[rod.qDOF]
            B = rod.q_dot_u(t, q_rod).toarray()
            C_2 = self.C_1 @ B
            M = rod.M(t, q_rod).toarray()
            # M_free = sys.M(t, q_sys)[self.uDOF_free][:, self.uDOF_free]
            # self.M_inv2 = splu(M).solve(C_2.T).T
            self.M_inv2 = C_2 @ np.linalg.inv(M)
            # self.M_inv2 = .solve(C_2.T).T

        # Build tendon force directions W_t
        W_t = np.zeros((sys.nu, self.nq))
        for j, td in enumerate(self.tendons):
            np.add.at(W_t[:, j], td.uDOF, -td.W_l(t, q_sys[td.qDOF]))
            # W_t[td.uDOF, j] = -td.W_l(t, q_sys[td.qDOF])

        # Current tendon force
        # la_t = q[: self._nq1]

        # Build h consisting of gyroscopic and external forces from system and add compliance forces
        # h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ la_t
        # h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ self._la_t
        h = rod.h(t, q_rod, u) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ self._la_t
        # h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys) - W_t @ q[:self._nq1] ## New test
        # h = sys.h(t, q_sys, u_sys) - W_t @ q[:self._nq1]

        # Build y_0_ddot and Jacobian
        y_0_ddot = - self.M_inv2 @ h
        J = self.M_inv2 @ W_t
        J_inv = np.linalg.inv(J) # Pure inverse (Causes singularities, add damped inverse)
        # J_inv = J.T @ np.linalg.solve(J @ J.T + self.inv_damping * np.eye(3), np.eye(3)) # Moore Penrose Pseudo Inverse

        # Error terms
        r_OP_ref = self.r_OP_ref(t)
        v_P_ref = self.v_P_ref(t)
        a_P_ref = np.zeros(3) # CHECK

        r_OP = self.rod._view_nodal_q(q[self._nq1 :])[-1, :3]
        # print(r_OP)
        # if not np.isfinite(r_OP).all():      # True if any NaN or inf
        #     print(f"non-finite r_OP at t={t}: {r_OP}")
        #     breakpoint()
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        
        e = r_OP_ref - r_OP
        e_dot = v_P_ref - v_P

        a = a_P_ref + self.Kd * e_dot + self.Kp * e       # linear PD Control law
        # a *= 0
        # a = a_P_ref + beta * e_dot + gamma * np.sign(e_dot + beta * e)  # Switching mode control law

        # Input-Output Feedback Linearization
        la_t = J_inv @ (a + y_0_ddot)
        # if not np.isfinite(la_t).all():      # True if any NaN or inf
            # print(f"non-finite la_t at t={t}: {la_t}")
            # breakpoint()
        # la_t *= 0
        return la_t
    
    def q_dot(self, t, q, u):
        return self._la_t_dot
    
    def step_callback(self, t, q, u):
        la_t = self.control_law(t, q, u)
        self._la_t = la_t
        self.la_ts.append(np.array(la_t))
        self.ts.append(t)
        for td, la_t_i in zip(self.tendons, la_t):
            td.set_force(lambda t, la=la_t_i: la)
        return q, u

    ## New Test
    # def apply_tendon_forces(self, t, q):
    #     # la_t = q[: self._nq1] + self.la_t_ref(t)
    #     for td, la_t_i in zip(self.tendons, q[: self._nq1]):
    #         td.set_force(lambda t, la=la_t_i: la)

    # def q_dot(self, t, q, u):
    #     self.apply_tendon_forces(t, q)
    #     return (self._la_t_des - q[: self._nq1]) / self.t_lag
    
    # def step_callback(self, t, q, u):
    #     self._la_t_des = self.control_law(t, q, u)
    #     return q, u

def la_t_plot(controller, model):
    import matplotlib.pyplot as plt
    ts = np.array(controller.ts)
    la_ts = np.array(controller.la_ts)      # shape (n_steps, n_tendons)

    fig, ax = plt.subplots(num="TendonForces", figsize=(8, 4))
    for k in range(model.n_tendons):
        ax.plot(ts, la_ts[:, k], label=f"tendon {k+1}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Tendon Force [N]")
    ax.set_title("Tendon Forces"); ax.legend(); ax.grid(True)
    plt.show()

if __name__ == "__main__":
    damping_ratio = 0.0
    # model = CommonModel(damping_ratio=0.3) 
    model = CommonModel(damping_ratio=damping_ratio) # dt = 1e-4 for damping_ratio = 1e-3
    system = model.system
    rod = model.rod
    tendons = model.tendons
    
    # Parameters
    # Kp = 100
    # Kd = 20
    Kp = 0.1
    Kd = 0.0
    inv_damping = 1e-3

    # Trajectories
    # r_OP_ref_fn = lambda t: np.array([0.0, 0.0, 0.192])
    r_OP_ref_fn = lambda t: SETPOINT_TABLE["A"]
    v_P_ref_fn = lambda t: np.zeros(3)
    
    t_lag =2e-3
    # t_lag = 4e-3
    # t_lag = 1e-2
    controller = DynamicController(system, r_OP_ref_fn, v_P_ref_fn, Kp, Kd, inv_damping, rod, tendons, t_lag=t_lag)
    system.add(controller)
    system.assemble()

    # Solver
    # t_sim = 2
    t_sim = 0.005
    # t_sim = 0.5
    dt = 1e-4
    # solver = BackwardEuler(system, t_sim, dt)
    # solver = ScipyDAE(system, t_sim, dt)

    fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
    fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
    solver = RungeKutta(system, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF)
    
    sol = solver.solve()
    print(f"Kp = {Kp}, Kd = {Kd}, damping ratio = {damping_ratio}, t_sim = {t_sim}, dt = {dt}, t_lag = {t_lag}")

    p2p_vis_plot(model, sol, r_OP_ref_fn)
    la_t_plot(controller, model)