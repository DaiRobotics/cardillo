from abc import ABC

from cardillo.math import A_IB_basic
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.forces import Force
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986, DiscreteRod, RodTendonForce

from cardillo.solver import ScipyDAE, BackwardEuler, Newton, SolverOptions, Solution
from cardillo.system import System

from cardillo.interactions import nPointInteraction

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import splu


def solve_ref_config(r_OP_ref, lambda_t0, tol=5e-4, damping=1e-4, force_steps = 10):
    static_model = StaticModel()
    
    lambda_t = np.clip(np.array(lambda_t0, float), lambda_t_min, lambda_t_max)
    e_n_prev = np.inf
    stall = 0
    k = 0
    # iteratively solve the reference configuration based on optimization
    while True:
        # ======= solve static equilibrium =======
        # print("====force")
        # lambda_t = np.asarray([2.739, 2.524, 0.173, 1.647]) * 1
        sol, x, solver = static_model.apply_forces(lambda_t, verbose=False, force_steps=force_steps, warm_start=False)
        
        # value evaluation
        rod = static_model.rod
        system = static_model.system
        rod = static_model.rod
        tendons = static_model.tendons

        q_guess = sol.q[-1]

        # tip position
        r_OP = rod._view_nodal_q(q_guess[rod.qDOF])[-1, :3]

        x = x.flatten()
        J = solver.jac(x, 1.0)

        nu = system.nu
        n_tendons = static_model.n_tendons
        W_t = np.zeros((nu, n_tendons))
        for j, td in enumerate(tendons):
            # W_t[td.uDOF, j] = -td.W_l(1.0, q_eq[td.qDOF])
            np.add.at(W_t[:, j], td.uDOF, -td.W_l(1.0, q_guess[td.qDOF]))
        rhs = np.zeros((solver.nx, n_tendons))
        rhs[:nu, :] = W_t

        dx = splu(J).solve(-rhs)   # dx_dT, shape (nx, n_tendons)
        pos_idx = rod.qDOF[rod.nodalDOF_r[rod.nnode - 1]]
        Gamma = dx[pos_idx, :]

        # error
        e = r_OP_ref - r_OP
        e_n = np.linalg.norm(e)

        # print(f"  inv-statics it {k:2d}: |tip-target|={e_n*1e3:7.3f} mm, "
        #       f"lambda_t={np.round(lambda_t, 3)}, cond(Gamma)={np.linalg.cond(Gamma):.2e}")
        if e_n < tol:
            break
        # if e_n_prev - e_n < 1e-7:  # converged to the best reachable point
            
        # # if e_n_prev - e_n < 1e-4:  # converged to the best reachable point

        #     stall += 1
        #     if stall >= 5:
        #         break
        # else:
        #     stall = 0
        # e_n_prev = e_n
        # damped (Levenberg-style) least-squares step, with a per-step limiter
        # TODO: check the implementation of Tianxiang Multibody Paper
        dlambda_t = Gamma.T @ np.linalg.solve(Gamma @ Gamma.T + damping * np.eye(3), e)
        dlambda_t = np.clip(dlambda_t, -0.5, 0.5)  # small steps: stay in the uncrushed workspace
        lambda_t = np.clip(lambda_t + dlambda_t, lambda_t_min, lambda_t_max)
        print(f"  inv-statics it {k:2d}: |tip-target|={e_n*1e3:7.3f} mm, "
        f"lambda_t={np.round(lambda_t, 3)}, cond(Gamma)={np.linalg.cond(Gamma):.2e}")
        k += 1
    return lambda_t, q_guess, Gamma

def solve_config(static_model, lambda_t, force_steps=10, q_warm=None):

    if q_warm is not None:
        st_solver = static_model.solver
        nq = st_solver.system.nq
        x0 = st_solver.x0.copy() if st_solver.x0 is not None else np.zeros(st_solver.nx, float)
        x0[:nq] = q_warm
        st_solver.x0 = x0
    # ======= solve static equilibrium =======
    sol, x, solver = static_model.apply_forces(lambda_t, verbose=False, force_steps=force_steps)
    
    # value evaluation
    rod = static_model.rod
    system = static_model.system
    rod = static_model.rod
    tendons = static_model.tendons

    q_guess = sol.q[-1]

    x = x.flatten()
    J = solver.jac(x, 1.0)

    nu = system.nu
    n_tendons = static_model.n_tendons
    W_t = np.zeros((nu, n_tendons))
    for j, td in enumerate(tendons):
        # W_t[td.uDOF, j] = -td.W_l(1.0, q_eq[td.qDOF])
        np.add.at(W_t[:, j], td.uDOF, -td.W_l(1.0, q_guess[td.qDOF]))
    rhs = np.zeros((solver.nx, n_tendons))
    rhs[:nu, :] = W_t

    dx = splu(J).solve(-rhs)   # dx_dT, shape (nx, n_tendons)
    pos_idx = rod.qDOF[rod.nodalDOF_r[rod.nnode - 1]]
    Gamma = dx[pos_idx, :]

    return q_guess, Gamma

class TendonForceControl:
    def __init__(
        self,
        Kp,
        Gamma,
        r_OP_traj,
        la_t_ref,
        rod, 
        tendons:list[RodTendonForce],
        static_model=None,
        gamma_eps=1.0,
        gamma_check_dt = 1.0,
        tau_ff=2.0,
        name="tendon_force_control",
    ) -> None:
        self.Kp = Kp
        self.Gamma = Gamma
        self.Gamma_inv = Gamma.T @ np.linalg.solve(Gamma @ Gamma.T, np.eye(Gamma.shape[0]))
        self.r_OP_traj = r_OP_traj
        self.la_t_ref = la_t_ref
        self.rod = rod
        self.tendons = tendons
        self.name = name

        self.static_model = static_model
        self.gamma_eps = gamma_eps
        self.gamma_check_dt = gamma_check_dt
        self.last_gamma_check_t = -np.inf

        self.tau_ff = tau_ff
        self.la_t_ff = None
        self.t_prev = None

        self.nq = len(tendons)
        self.q0 = np.zeros(self.nq)
        self._la_t_dot = np.zeros(self.nq)

    def assembler_callback(self):
        self.qDOF = np.concatenate([self.my_qDOF, self.rod.qDOF])
        self._nq1 = len(self.my_qDOF)
        self.uDOF = self.rod.uDOF

    def step_callback(self, t, q, u):
        # Gamma = self.Gamma(t)
        # Ramp on feed forward (the bigger tau.ff, the slower the ramp)
        la_t_ref_target = self.la_t_ref(t)
        if self.la_t_ff is None:
            self.la_t_ff = np.array(la_t_ref_target, dtype=float)
        else:
            dt = t - self.t_prev
            # alpha = min(dt / self.tau_ff, 1.0) if dt > 0 else 0.0
            alpha = 0
            self.la_t_ff = self.la_t_ff + alpha * (la_t_ref_target - self.la_t_ff)
        self.t_prev = t
        r_OP_ref = self.r_OP_traj(t)

        # Recompute gamma
        if (self.static_model is not None
                and t - self.last_gamma_check_t >= self.gamma_check_dt):
            self.last_gamma_check_t = t
            lambda_cur = np.clip(np.asarray(q[:self._nq1], dtype=float),
                              lambda_t_min, lambda_t_max)
            try:
                q_rod_cur = np.asarray(q[self._nq1:], dtype=float)
                _, Gamma_cur = solve_config(self.static_model, lambda_cur, q_warm=q_rod_cur)
                dGamma = Gamma_cur - self.Gamma
                if np.linalg.norm(dGamma @ np.linalg.pinv(self.Gamma), 2) >= self.gamma_eps:        # Gamma0 no longer valid -> refresh
                    self.Gamma = Gamma_cur
            except Exception:
                pass

        r_OP = self.rod._view_nodal_q(q[self._nq1:])[-1, :3]
        delta_r_OP = r_OP_ref - r_OP
        self._la_t_dot = self.Kp * self.Gamma_inv @ delta_r_OP
        # self._la_t_dot = self.Kp * pinv(self.Gamma) @ delta_r_OP 
        # self._la_t_dot = self.Kp * self.Gamma.T @ np.linalg.solve(self.Gamma @ self.Gamma.T, delta_r_OP)
        for i, (td, delta_la) in enumerate(zip(self.tendons, q[:self._nq1])):
            # td.set_force(lambda t, la=delta_la + la_t_ref[i]: la)
            td.set_force(lambda t, la=delta_la + self.la_t_ff[i]: la)
            # td.set_force(lambda t, la=delta_la + la_t_ref_table[-1][i]: la)
            # td.set_force(lambda t, la=delta_la: la)
        return q, u

    def q_dot(self, t, q, u):
        # for td, la in zip(self.tendons, q[:self._nq1]):
        #     td.set_force(lambda t, la=la: la)
        return self._la_t_dot


class StaticSolver(Newton):
    def __init__(
        self, system, n_load_steps=1, verbose=True
    ):
        super().__init__(
            system, n_load_steps, verbose
        )
        self.n_load_steps = self.nt - 1
        self.x0 = None

    def set_load_steps(self, n_load_steps):
        if n_load_steps == self.n_load_steps:
            return
        self.n_load_steps = n_load_steps
        self.load_steps = np.linspace(0, 1, n_load_steps + 1)
        self.nt = n_load_steps + 1
        self.len_t = len(str(self.nt))
        x0 = self.x[0]
        self.x = np.zeros((self.nt, len(self.x[0])), dtype=np.float64)
        self.x[0] = x0

    def renew_initial_state(self):
        system = self.system
        self.x0 = np.concatenate((system.q0, system.la_g0))

        
    def solve(self, warm_start=True):
        if warm_start and self.x0 is not None:
            self.x[0] = self.x0
        res = super().solve()
        if warm_start:
            self.x0 = self.x[-1]
        return res

def interp1d(x, y, xi):
    """
    linear interpolation, support multidimensional y
    """

    idx = np.searchsorted(x, xi, side="right") - 1
    if idx == len(y) - 1:
        return y[-1]
    else:
        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        t = (xi - x0) / (x1 - x0)
        yi = y0 + (y1 - y0) * t
        return yi

class CommonModel(ABC):
    def __init__(self):
        super().__init__()
        # ---- pysical parameters ----
        rod_nelement = 10 # 1000
        rod_l0 = 0.192 # [m] length of rod
        rod_r0_base = 1.4e-2 # [m] radius at bottom of rod
        rod_r0_tip = 8.5e-3 # [m] radius at tip of rod
        self.rod_density = 1.41e3 # density of material
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
            return np.array([xi*rod_l0, 0, 0], dtype=np.float64)

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
            cross_section_inertias=CrossSectionInertias(self.rod_density, self.cross_section),
        )

        # ---- rigid connections ----
        rc = RigidConnection(self.rod, self.system.origin, xi1=0)

        # ---- tendons ----
        self.n_tendons = 4
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
                xis=[i / (n - 1) for i in range(n)],
                B_r_CPs=B_r_CP_list,
            )
            self.tendons.append(tendon)

        self.system.add(self.rod, rc, *self.tendons)

class StaticModel(CommonModel):
    def __init__(self):
        super().__init__()
        g_acc = 9.81
        # ---- external forces ----
        gravity = Force_line_distributed(
            lambda t, xi: self.rod_density * self.cross_section.area(xi) * g_acc * np.array([0, -1.0, 0], dtype=np.float64) * t,
            self.rod,
        )
        self.system.add(gravity)
        self.system.assemble()

        self.solver = StaticSolver(
            self.system,
            n_load_steps=1,
            verbose=False,
            # options=SolverOptions(continue_with_unconverged=False),
        )
        self.force_init = np.array([td.la(0) for td in self.tendons])
        self.nt = -1

    def apply_forces(
        self,
        forces: np.ndarray,
        verbose=True,
        force_steps=1,
        ret_all_steps=False,
        warm_start=True,
    ):
        forces = np.atleast_2d(forces)

        ts = np.linspace(0, 1, forces.shape[0] + 1)
        # -----------
        #   tendons
        # -----------
        _forces = np.vstack((self.force_init, forces))
        for i, tendon in enumerate(self.tendons):
            tendon.set_force(lambda t, i=i: interp1d(ts, _forces[:, i], t))
        # ------------
        #   Solve
        # ------------
        self.solver.set_load_steps(forces.shape[0] * force_steps)
        self.solver.verbose = verbose
        sol = self.solver.solve(warm_start=warm_start)
        # ------------------------
        #   Solution Evaluation
        # ------------------------
        if ret_all_steps:
            t, q, la_g, x = (
                sol.t[1:],
                sol.q[1:],
                sol.la_g[1:],
                self.solver.x[1:],
            )
        else:
            t, q, la_g, x = (
                sol.t[force_steps::force_steps],
                sol.q[force_steps::force_steps],
                sol.la_g[force_steps::force_steps],
                self.solver.x[force_steps::force_steps],
            )
        if warm_start:
            self.force_init = forces[-1]
        return Solution(self.solver.system, t, q, la_g=la_g), x, self.solver


class DynamicModel(CommonModel):
    def __init__(self, t_sim, Kp, Gamma, la_t0, r_OP_traj, la_t_ref, q0=None):
        super().__init__()
        g_acc = 9.81
        # ---- external forces ----
        gravity = Force_line_distributed(
            lambda t, xi: self.rod_density * self.cross_section.area(xi) * g_acc * np.array([0, -1.0, 0], dtype=np.float64),
            self.rod,
        )
        static_model = StaticModel()
        self.controller = TendonForceControl(Kp, Gamma, r_OP_traj, la_t_ref, self.rod, self.tendons, static_model=None, gamma_eps=1.0, gamma_check_dt = 1.0)
        for td, la in zip(self.tendons, la_t0):
            td.set_force(lambda t, la=la: la)

        self.system.add(gravity)
        self.system.add(self.controller)
        self.system.assemble()

        # set initial state of the system
        if q0 is not None:
            self.system.set_new_initial_state(np.concatenate((q0, la_t0*0)), np.zeros(self.system.nu))
        self.solver = BackwardEuler(
            self.system,
            t1=t_sim,
            dt=1e-2,
            options=SolverOptions(compute_consistent_initial_conditions=True)
        )



# sol = static_model.apply_forces([1, 0, 0, 0], force_steps=30)

# sol = static_model.apply_forces([10, 0, 0, 0], eval_keys=["sol"], force_steps=10)

# ----- controller parameters -----
lambda_t_min = 0.0
lambda_t_max = 50.0
la_t0 = np.array([1, 1, 1, 1]) * 0.0

# ---- reference trajectories ----

traj_mode = "p2p" # "p2p", "circle_zy", "circle_xy", "star_yz"
t_scale = 1.0 # time scaling if needed

def paper_to_cardillo(u):
    X, Y, Z = u
    return np.array([Y, Z, X])

SETPOINT_TABLE = {
    "A": np.array([15.438e-2, 4.335e-2, 3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2, 9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2, 0.000e-2, -9.842e-2]),
    "E2": np.array([13.951e-2, 0.000e-2, -9.842e-2])*1.2,
}
SETPOINT_TABLE = {k: paper_to_cardillo(u) for k, u in SETPOINT_TABLE.items()}

def make_circle(x_fn, y_fn, z_fn, t_period):
    def ref(t):
        th = 2.0 * np.pi * t / t_period - np.pi / 2  # start at circle bottom (near E)
        return paper_to_cardillo(np.array([x_fn(th), y_fn(th), z_fn(th)]))
    return ref

def make_star_yz(y_c=0.0, z_c=-3.0e-2, R=6.5e-2, x_const=13.5e-2, t_total=70.0):
    angles = np.deg2rad(90.0 + 72.0 * np.arange(5))  # vertex 0 at top
    verts = [np.array([y_c + R * np.cos(a), z_c + R * np.sin(a)]) for a in angles]
    order = [2, 4, 1, 3, 0, 2]  # pentagram (every 2nd), starting at vertex 2 (bottom-left)
    pts = [verts[i] for i in order]  # 6 points -> 5 edges
    def ref(t):
        s = np.clip(t / t_total, 0.0, 1.0) * 5.0
        i = min(int(s), 4)
        a = s - i
        y, z = (1.0 - a) * pts[i] + a * pts[i + 1]
        return paper_to_cardillo(np.array([x_const, y, z]))
    return ref

if traj_mode == "p2p":
    t_end = 50
    sequence = ["A", "B", "C", "D", "E"]
    hold_t = t_end / (len(sequence))

    def r_OP_ref_fn(t):
        k = min(int(t / hold_t), len(sequence) - 1)
        # return SETPOINT_TABLE["E2"]
        return SETPOINT_TABLE[sequence[k]]
    
    la_t_ref_table = []
    q0_table = []
    Gamma0_table = []
    
    for name in sequence:
        r_OP_ref = SETPOINT_TABLE[name]
        # la_ref, _, _ = solve_ref_config(r_OP_ref, la_t0, tol = 3e-4, force_steps=20)
        la_t_ref, q0, Gamma0 = solve_ref_config(r_OP_ref, la_t0, tol = 3e-4, force_steps=20)
        la_t_ref_table.append(la_t_ref)
        q0_table.append(q0)
        Gamma0_table.append(Gamma0)
        print(f"{name}")
    def la_t_ref_fn(t):
        if t == 0.0:
            return la_t_ref_table[-1]
        k = min(int(t / hold_t), len(sequence) - 1)
        return la_t_ref_table[k]

        


elif traj_mode == "circle_zy":
    x_c, z_c, rad = 10.3e-2, -1.75e-2, 3.0e-2
    t_period = 40.0 * t_scale  # [s] per lap (paper: 2 laps in ~80 s)
    t_end = 2 * t_period
    r_OP_ref_fn = make_circle(
        lambda th: x_c + rad * np.sin(th + np.pi),  
        lambda th: 0.0,
        lambda th: z_c + rad * np.cos(th),
        t_period,
    )

elif traj_mode == "circle_xy":
    x_const, y_c, z_c, rad = 13.5e-2, 0.0, -3.5e-2, 5.5e-2
    t_period = 40.0 * t_scale
    t_end = 2 * t_period
    r_OP_ref_fn = make_circle(
        lambda th: x_const,
        lambda th: y_c + rad * np.cos(th),
        lambda th: z_c + rad * np.sin(th),
        t_period,
    )

elif traj_mode == "star_yz":
    t_star_total = 70.0 * t_scale
    t_end = t_star_total
    r_OP_ref_fn = make_star_yz(t_total=t_star_total)

else:
    raise ValueError(f"unknown tdcm_traj mode: {traj_mode!r}")

# ----- reference Gamma -----


# ---- build controller ----
# if traj_mode == "p2p":
#     gamma_table = {}
#     la_table = {}
#     q_table = {}
#     lambda_t0 = la_t0
#     for name in sequence:
#         r_OP_ref = SETPOINT_TABLE[name]
#         # r_OP_ref = r_OP_ref_fn(0.0)  
#         la_t0, q0, Gamma0 = solve_ref_config(r_OP_ref, tol=1e-7, lambda_t0=lambda_t0, force_steps=3)
#         gamma_table[name] = Gamma0
#         la_table[name] = la_t0
#         q_table[name] = q0
#         lambda_t0 = la_t0

#     def gamma_fn(t):
#         k = min(int(t / hold_t), len(sequence) - 1)
#         return gamma_table[sequence[k]]
    
#     Gamma = gamma_fn
#     la_t0 = la_table[sequence[0]]
#     q0 = q_table[sequence[0]]

# else:
#     setpoint = "E"
#     r_OP_ref = SETPOINT_TABLE[setpoint]
#     la_t0, q0, Gamma0 = solve_ref_config(r_OP_ref, tol=1e-7, lambda_t0=la_t0, force_steps=3)

# setpoint = "E"
# r_OP_ref = SETPOINT_TABLE[setpoint]
# # # r_OP_ref = r_OP_ref_fn(0.0)  
# la_t0, q0, Gamma0 = solve_ref_config(r_OP_ref, tol=3e-4, lambda_t0=la_t0, force_steps=10)

q0 = q0_table[-1]
Gamma0 = Gamma0_table[-1]

# Kp = 0.5
# Kp = 0.2
Kp = 0.1
t_sim = t_end
dynamic_model = DynamicModel(t_sim, Kp, Gamma0, la_t0, r_OP_ref_fn, la_t_ref_fn, q0)



# ---- visualization ----
rod = dynamic_model.rod
tendons = dynamic_model.tendons
system = dynamic_model.system

from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
for tendon in tendons:
    VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))

window_size = (960, 540)
plotter = Plotter(system, window_size)
plotter.add_ground(-0.2, 0.2, -0.2, 0.2, 10, 10)
r_OF = np.array([0, -0.05, 0.10], float)
r_OC = r_OF + np.array([0, 0, 0.45], float)
e_x_cam = np.array([1, 0, 0], float)
e_z_cam = r_OF - r_OC
e_z_cam /= np.linalg.norm(e_z_cam)
e_y_cam = np.cross(e_z_cam, e_x_cam)
fx = 2635.5177
px, py = 3840, 2160
cam = plotter.camera
cam.view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)
cam.parallel_projection = False
cam.position = r_OC
cam.focal_point = r_OF
cam.view_up = -e_y_cam
cam.clipping_range = (0.01, 2)
cam.Zoom(1)


# plotter.live_render()
sol = dynamic_model.solver.solve()
plotter.render_solution(sol, True, play_speed_up=1)

from matplotlib import pyplot as plt

t = sol.t
q = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
r_OP_traj = np.array([r_OP_ref_fn(ti) for ti in t])

# ---- X-Y plane plots ----
# fig = plt.figure(figsize=(10, 4))
# gs  = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])

# axy = fig.add_subplot(gs[:, 0])
# axy.plot(r_OP_traj[:, 0], r_OP_traj[:, 1], "b--", label="desired")
# axy.plot(q[:, -1, 0], q[:, -1, 1], "r", label="actual")
# axy.set_xlabel("X [m]")
# axy.set_ylabel("Y [m]")
# axy.legend()
# axy.grid(True)

# atx = fig.add_subplot(gs[0, 1])
# atx.plot(t, r_OP_traj[:, 0], "b--", label="desired")
# atx.plot(t, q[:, -1, 0], "r", label="actual")
# atx.set_xlabel("time [s]")
# atx.set_ylabel("X [m]")
# atx.legend()
# atx.grid(True)

# aty = fig.add_subplot(gs[1, 1])
# aty.plot(t, r_OP_traj[:, 1], "b--", label="desired")
# aty.plot(t, q[:, -1, 1], "r", label="actual")
# aty.set_xlabel("time [s]")
# aty.set_ylabel("Y [m]")
# aty.legend()
# aty.grid(True)

# fig.suptitle(f"Trajectory tracking in X-Y plane")
# fig.tight_layout()

# ---- Z-Y plane plots ----
# fig = plt.figure(figsize=(10, 4))
# gs  = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])

# azy = fig.add_subplot(gs[:, 0])
# azy.plot(r_OP_traj[:, 2], r_OP_traj[:, 1], "b--", label="desired")
# azy.plot(q[:, -1, 2], q[:, -1, 1], "r", label="actual")
# azy.set_xlabel("Z [m]")
# azy.set_ylabel("Y [m]")
# azy.legend()
# azy.grid(True)

# atz = fig.add_subplot(gs[0, 1])
# atz = fig.add_subplot(gs[1, 1])
# atz.plot(t, r_OP_traj[:, 2], "b--", label="desired")
# atz.plot(t, q[:, -1, 2], "r", label="actual")
# atz.set_xlabel("time [s]")
# atz.set_ylabel("X [m]")
# atz.legend()
# atz.grid(True)

# aty = fig.add_subplot(gs[1, 1])
# aty.plot(t, r_OP_traj[:, 1], "b--", label="desired")
# aty.plot(t, q[:, -1, 1], "r", label="actual")
# aty.set_xlabel("time [s]")
# aty.set_ylabel("Y [m]")
# aty.legend()
# aty.grid(True)

# fig.suptitle(f"Trajectory tracking in Z-Y plane")
# fig.tight_layout()

# ---- Point to Point plots ----
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 1)

atx = fig.add_subplot(gs[0, 0])
atx.plot(t, q[:, -1, 0] * 100, "r", label="actual")
atx.plot(t, r_OP_traj[:, 0] * 100, "b--", label="desired")
atx.set_xlabel("Time [s]")
atx.set_xlim(0, 50)
atx.set_xticks(np.arange(0, 50.1, 5))
atx.set_ylabel("X [cm]")
atx.set_ylim(-5.2, 10)
atx.set_yticks(np.array([-5, 0, 5, 10]))
atx.legend()
atx.grid(True)

aty = fig.add_subplot(gs[1, 0])
aty.plot(t, q[:, -1, 1] * 100, "r", label="actual")
aty.plot(t, r_OP_traj[:, 1] * 100, "b--", label="desired")
aty.set_xlabel("Time [s]")
aty.set_xlim(0, 50)
aty.set_xticks(np.arange(0, 50.1, 5))
aty.set_ylabel("Y [cm]")
aty.set_ylim(-10, 5)
aty.set_yticks(np.array([-10, -5, 0, 5]))
aty.legend()
aty.grid(True)

atz = fig.add_subplot(gs[2, 0])
atz.plot(t, q[:, -1, 2] * 100, "r", label="actual")
atz.plot(t, r_OP_traj[:, 2] * 100, "b--", label="desired")
atz.set_xlabel("Time [s]")
atz.set_xlim(0, 50)
atz.set_xticks(np.arange(0, 50.1, 5))
atz.set_ylabel("Z [cm]")
atz.set_ylim(10, 18)
atz.set_yticks(np.arange(10, 18.1, 2))
atz.legend()
atz.grid(True)

fig.suptitle(f"Trajectory tracking (point-to-point)")
fig.tight_layout()



plt.show()