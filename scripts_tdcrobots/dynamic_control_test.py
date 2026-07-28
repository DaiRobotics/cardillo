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
from pathlib import Path
import pandas as pd

from espedal_control_test import p2p_vis_plot
from runge_kutta import *
from dynamic_controller import DynamicControllerPD

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
    def __init__(self, damping_ratio=0):
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

    fig, ax = plt.subplots(num="TendonForces", figsize=(8, 4))
    for k in range(model.n_tendons):
        ax.plot(ts, la_ts[:, k], label=f"tendon {k+1}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Tendon Force [N]")
    ax.set_title("Tendon Forces"); ax.legend(); ax.grid(True)
    plt.show()

def compute_la_ts(controller, sol):
    la_ts = np.array([
        controller.la_tau(t, q[controller.qDOF], u[controller.uDOF])
        for t, q, u in zip(sol.t, sol.q, sol.u)
    ])
    return la_ts

if __name__ == "__main__":

    csv_file = Path(__file__).parent / "p2p_q0_gamma0.csv"
    q0_E = pd.read_csv(csv_file)["q0_E"].to_numpy()

    damping_ratio = 0.1
    model = CommonModel(damping_ratio=damping_ratio) # dt = 1e-4 for damping_ratio = 1e-3
    # model.rod.q0 = q0_E.copy() # Start at E
    system = model.system
    rod = model.rod
    tendons = model.tendons
    
    # Parameters
    Kp = 200.0
    Kd = 20.0
    inv_damping = 1e-3

    # Trajectories
    # r_OP_ref_fn = lambda t: np.array([0.0, 0.0, 0.192])
    r_OP_ref_fn = lambda t: SETPOINT_TABLE["A"]
    
    # P2P Setpoints Trajectory
    # def p2p_sequence(names, t_hold=5.0):
    #     pts = [SETPOINT_TABLE[n] for n in names]
    #     def r_OP_ref_fn(t):
    #         idx = min(int(t // t_hold), len(pts) - 1)
    #         return pts[idx]
    #     return r_OP_ref_fn

    # No smoothing
    # r_OP_ref_fn = p2p_sequence(["A", "B", "C", "D", "E"], t_hold=5.0)
    v_P_ref_fn = lambda t: np.zeros(3)
    a_P_ref_fn = lambda t: np.zeros(3)

    def smooth_p2p_sequence(names, t_hold=5.0, t_move=1.0):
        pts = [SETPOINT_TABLE[name] for name in names]
        n = len(pts)
        t_transition = 0.5 * t_move

        def smoothing(r_OP0, r_OP1, s, T):
            d = r_OP1 - r_OP0
            zd = r_OP0 + d * (10*s**3 - 15*s**4 + 6*s**5)
            zd_dot = d * (30*s**2 - 60*s**3 + 30*s**4) / T
            zd_ddot = d * (60*s - 180*s**2 + 120*s**3) / (T**2)
            return zd, zd_dot, zd_ddot

        def ref_fns(t):
            seg = max(0 ,min(int(t // t_hold), n - 1))
            t_seg0 = seg * t_hold
            t_seg1 = (seg + 1) * t_hold
            # entering segment
            if seg >= 1 and (t - t_seg0) < t_transition:
                return smoothing(pts[seg-1], pts[seg], (t - (t_seg0 - t_transition)) / t_move, t_move)
            if seg <= n - 2 and (t_seg1 - t) < t_transition:         # about to leave seg (to seg+1)
                return smoothing(pts[seg], pts[seg+1], (t - (t_seg1 - t_transition)) / t_move, t_move)
            return pts[seg], np.zeros(3), np.zeros(3) # hold

        return (lambda t: ref_fns(t)[0], lambda t: ref_fns(t)[1], lambda t: ref_fns(t)[2])

    # Smoothing
    # r_OP_ref_fn , v_P_ref_fn, a_P_ref_fn = smooth_p2p_sequence(["A", "B", "C", "D", "E"], t_hold=5.0, t_move=1.0)

    controller = DynamicControllerPD(system, rod, tendons, r_OP_ref_fn, v_P_ref_fn=v_P_ref_fn, a_P_ref_fn=a_P_ref_fn, Kp=Kp, Kd=Kd, inv_damping=inv_damping)
    # system.add(controller)
    system.assemble()

    # Solver
    # t_sim = 25
    # t_sim = 5
    t_sim = 2
    dt = 1e-4
    # solver = BackwardEuler(system, t_sim, dt)
    solver = ScipyDAE(system, t_sim, dt)

    fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
    fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
    # solver = RungeKutta(system, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF) # Doesn't work when damping is added
    # solver = RungeKuttaAdaptive(system, t_sim, dt=1e-3, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF, rtol=1e-4, atol=1e-7)
    # solver = RungeKutta45(system, t_sim, dt=1e-3, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF, rtol=1e-4, atol=1e-7)

    # Test to see the solution exploding due to la_tau not being updated properly
    # solver = ProbeRK(system, controller, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF)
    
    sol = solver.solve()
    print(f"Kp = {Kp}, Kd = {Kd}, damping ratio = {damping_ratio}, t_sim = {t_sim}, dt = {dt}")

    p2p_vis_plot(model, sol, r_OP_ref_fn)
    plt.show()
    # la_ts = compute_la_ts(controller, sol)
    # la_t_plot(model, la_ts, sol)
    # probe_plot(solver)
