from cardillo.math import A_IB_basic
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, TendonForce
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.discreteRod import DiscreteRod

from cardillo.solver import ScipyDAE, BackwardEuler, Newton, SolverOptions
from cardillo.system import System

from cardillo.interactions import nPointInteraction

import numpy as np
from scipy.linalg import pinv
from scipy.sparse.linalg import splu
import os



# ---- parameters ----
rod_nelement = 50 # 1000
rod_l0 = 0.192 # [m] length of rod
rod_r0_base = 1.4e-2 # [m] radius at bottom of rod
rod_r0_tip = 8.5e-3 # [m] radius at tip of rod
density = 1.41e3 # density of material
rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1

scaling_factor = 1.0
rod_l0_new = rod_l0 * scaling_factor
rod_r0_base_new = rod_r0_base * scaling_factor
rod_r0_tip_new = rod_r0_tip * scaling_factor

lambda_gain = 200.0*1.0 # control gain lambda for Gamma0^T. 
lambda_t_min = 0.0
lambda_t_max = 8.0
lambda_t_star = np.array([1, 1, 1, 1]) * 0.5
delta_bound = 0.05 # [N] per-step tension change limit. With lambda_gain=200 this is loose (typical per-step Delta < 0.05), so it acts as a safety cap only; the controller integrates freely most of the time.
# t_end = 50.0 # [s] total horizon -> HOLD_T = 10 s
dt = 1e-2

SHOW = os.environ.get("TDCM_SHOW", "1") == "1"

# ---- rod ----
radius = lambda xi: rod_r0_base_new * (1 - xi) + rod_r0_tip_new * xi
cross_section = CircularCrossSection(radius)
E, G = 2.563e5, 8.543e4 
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
Rod = DiscreteRod

def r_OP(xi):
    return np.array([xi*rod_l0_new, 0, 0], dtype=np.float64)

A_IB = lambda xi: np.eye(3, dtype=np.float64)
q0 = Rod.pose_configuration(
    rod_nelement,
    r_OP,
    A_IB,
    A_IB0=rod_A_IB0,
)
Q = q0.copy()

rod = Rod(
    cross_section,
    material_model,
    rod_nelement,
    Q=Q,
    q0=q0,
    cross_section_inertias=CrossSectionInertias(density, cross_section),
)

# ---- external forces ----
g_acc = 9.81
gravity = Force_line_distributed(
    lambda t, xi: -density * cross_section.area(xi) * g_acc * np.array([0, 1.0, 0], dtype=np.float64),
    rod,
)
# ---- rigid connections ----
rc = RigidConnection(rod, system.origin, xi1=0)

# ---- tendons ----
n_tendons = 4
tendons = []
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
    for phi in np.linspace(0, 2 * np.pi, n_tendons, endpoint=False)
]
for B_r_CP_list in B_r_CP_lists:
    n = len(B_r_CP_list)
    tendon = TendonForce(
        subsystem_list=[rod.get_marker(i / (n - 1)) for i in range(n)],
        connectivity=[(i, i + 1) for i in range(n - 1)],
        xi_list=[i / (n - 1) for i in range(n)],
        B_r_CP_list=B_r_CP_list,
    )
    tendons.append(tendon)

system.add(rod, rc, *tendons, gravity)
system.assemble()

# ---- reference ----

traj_mode = os.environ.get("TDCM_TRAJ", "circle_xz")  # "p2p", "circle_xz", "circle_yz", "star_yz"
t_scale = float(os.environ.get("TDCM_TSCALE", 1.0)) # time scaling if needed

def paper_to_cardillo(u):
    X, Y, Z = u
    return np.array([Y, Z, X])

TABLE_II = {
    "A": np.array([15.438e-2, 4.335e-2, 3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2, 9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2, 0.000e-2, -9.842e-2]),
}
TABLE_II = {k: paper_to_cardillo(u) for k, u in TABLE_II.items()}

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
    t_end = 50.0
    sequence = ["A", "B", "C", "D", "E"]
    hold_t = t_end / (len(sequence))

    def r_OP_ref_fn(t):
        k = min(int(t / hold_t), len(sequence) - 1)
        return TABLE_II[sequence[k]]

elif traj_mode == "circle_xz":
    x_c, z_c, rad = 11.0e-2, -1.75e-2, 3.0e-2
    t_period = 40.0 * t_scale  # [s] per lap (paper: 2 laps in ~80 s)
    t_end = 2 * t_period
    r_OP_ref_fn = make_circle(
        lambda th: x_c + rad * np.cos(th),
        lambda th: 0.0,
        lambda th: z_c + rad * np.sin(th),
        t_period,
    )

elif traj_mode == "circle_yz":
    x_const, y_c, z_c, rad = 13.5e-2, 0.0, -2.0e-2, 5.5e-2
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

# t_end = float(os.environ.get("TDCM_TEND", t_end))

# ---- controller ----
def tip_position(q_full):
    pos_idx = rod.qDOF[rod.nodalDOF_r[rod.nnode - 1]]
    return np.asarray(q_full[pos_idx], dtype=float).copy()

def set_tendon_forces(lambda_t, ramp):
    for i, td in enumerate(tendons):
        c = float(lambda_t[i])
        td.set_force((lambda t, c=c: c * t) if ramp else c)
        
def static_newton(q_guess, n_load_steps):
    if q_guess is not None:
        system.q0 = q_guess.copy()
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        verbose=False,
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-8, newton_max_iter=100),
    )
    sol = solver.solve()
    return solver, sol

def forward_statics(lambda_t, q_guess=None, ramp=False, n_load_steps=1):
    set_tendon_forces(lambda_t, ramp)
    solver, sol = static_newton(q_guess, n_load_steps)
    q_eq = sol.q[-1]
    return q_eq, tip_position(q_eq), sol.la_c[-1], solver

def compute_Gamma(lambda_t, q_guess=None):
    ramp = q_guess is None
    n_load_steps = 20 if ramp else 1
    q_eq, r_OP, la_c_eq, solver = forward_statics(lambda_t, q_guess, ramp, n_load_steps)

    x = solver.x[-1]
    solver.fun(x, 1.0)
    J = solver.jac(x, 1.0).tocsc()

    nu = system.nu
    W_t = np.zeros((nu, n_tendons))
    for j, td in enumerate(tendons):
        # W_t[td.uDOF, j] = -td.W_l(1.0, q_eq[td.qDOF])
        np.add.at(W_t[:, j], td.uDOF, -td.W_l(1.0, q_eq[td.qDOF]))
    rhs = np.zeros((solver.nx, n_tendons))
    rhs[:nu, :] = W_t

    dx = splu(J).solve(-rhs)   # dx_dT, shape (nx, n_tendons)
    pos_idx = rod.qDOF[rod.nodalDOF_r[rod.nnode - 1]]
    Gamma = dx[pos_idx, :]
    return Gamma, q_eq, r_OP, solver

def solve_lambda_t_star(r_OP_ref, lambda_t0, n_iter=40, tol=5e-4, damping=1e-4):
    lambda_t = np.clip(np.array(lambda_t0, float), lambda_t_min, lambda_t_max)
    q_guess = J = None
    e_n_prev = np.inf
    stall = 0
    for k in range(n_iter):
        Gamma, q_guess, r_OP, _solver = compute_Gamma(lambda_t, q_guess=q_guess)
        e = r_OP - r_OP_ref
        e_n = np.linalg.norm(e)
        print(f"  inv-statics it {k:2d}: |tip-target|={e_n*1e3:7.3f} mm, "
              f"lambda_t={np.round(lambda_t, 3)}, cond(Gamma)={np.linalg.cond(Gamma):.2e}")
        if e_n < tol:
            break
        if e_n_prev - e_n < 1e-7:  # converged to the best reachable point
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        e_n_prev = e_n
        # damped (Levenberg-style) least-squares step, with a per-step limiter
        dlambda_t = -Gamma.T @ np.linalg.solve(Gamma @ Gamma.T + damping * np.eye(3), e)
        dlambda_t = np.clip(dlambda_t, -0.5, 0.5)  # small steps: stay in the uncrushed workspace
        lambda_t = np.clip(lambda_t + dlambda_t, lambda_t_min, lambda_t_max)
    return lambda_t, q_guess, Gamma

class TendonController:
    def __init__(self, tendons, r_OP_ref_fn, Gamma0_inv, lambda_t0, dt, kp=0.0, ki = lambda_gain, kd=0.0,
                 lambda_gain=lambda_gain, lambda_t_min=lambda_t_min, lambda_t_max=lambda_t_max,
                 delta_bound=delta_bound, verbose=True):
        self.tendons = tendons
        self.r_OP_ref_fn = r_OP_ref_fn
        self.Gamma0_inv = np.asarray(Gamma0_inv, dtype=float)
        self.lambda_t = np.array(lambda_t0, dtype=float)
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.lambda_gain = float(lambda_gain)
        self.lambda_t_min = float(lambda_t_min)
        self.lambda_t_max = float(lambda_t_max)
        self.delta_bound = float(delta_bound)
        self.verbose = verbose
        self.e_int = np.zeros(3)        # integral error state
        self.saturated = False # antiwindup flag
        self.history = {"t": [], "lambda_t": [], "r_OP": [], "r_OP_ref": [], "error": []}

        # wire each tendon to read its scalar tension from this controller's state
        for i, td in enumerate(self.tendons):
            td.set_force(self._make_la_fn(i))

    def _make_la_fn(self, i):
        return lambda t, i=i: float(self.lambda_t[i])

    def update(self, t, q_full, u_full):
        """Called once per accepted dynamic step with the full state."""
        r_OP = tip_position(q_full)
        r_OP_r = self.r_OP_ref_fn(t)
        e = r_OP - r_OP_r
        v_P = u_full[rod.qDOF[rod.nodalDOF_r_u[rod.nnode - 1]]]  # tip velocity

        if not self.saturated:
            self.e_int += e * self.dt

        # T_

        # one forward-Euler step of  T_dot = -lambda * Gamma0_inv * e
        # (with Gamma0_inv = Gamma0^T per paper Remark 4).
        # Mirrors test_tdcm_li2023.py: compute delta, clip its magnitude per-step
        # to suppress overshoots from Gamma0^T's directional inaccuracies, then
        # apply with the [t_min, t_max] tendon bounds.
        delta = -self.lambda_gain * self.dt * (self.Gamma0_inv @ e)
        delta = np.clip(delta, -self.delta_bound, self.delta_bound)
        self.lambda_t = np.clip(self.lambda_t + delta, self.lambda_t_min, self.lambda_t_max)

        err = float(np.linalg.norm(e))
        self.history["t"].append(float(t))
        self.history["lambda_t"].append(self.lambda_t.copy())
        self.history["r_OP"].append(r_OP.copy())
        self.history["r_OP_ref"].append(r_OP_r.copy())
        self.history["error"].append(err)
        self._n_calls = getattr(self, "_n_calls", 0) + 1
        if self.verbose and self._n_calls % 25 == 0:  # throttle console output
            lt = ", ".join(f"{v:5.2f}" for v in self.lambda_t)
            print(f"  t={t:6.3f}s | |e|={err*1e3:7.3f} mm | T=[{lt}] N")

# ---- build controller ----
ref_target = TABLE_II[os.environ.get("TDCM_REF", "E")]
lambda_t_star, q_star, Gamma0 = solve_lambda_t_star(ref_target, lambda_t0=lambda_t_star)

q_star, _, la_c_star, __ = forward_statics(lambda_t_star, q_guess=q_star, ramp=False, n_load_steps=1) # for good first step guess

controller = TendonController(
    tendons, r_OP_ref_fn=r_OP_ref_fn, Gamma0_inv=Gamma0.T, lambda_t0=lambda_t_star, dt=dt, verbose=True
)

system.t0 = 0.0
system.q0 = q_star.copy()
system.u0 = np.zeros(system.nu)
system.la_c0 = la_c_star.copy()

_orig_step_callback = system.step_callback # connect controller to each dynamic step

def _controlled_step_callback(t, q, u):
    q, u = _orig_step_callback(t, q, u)  # keeps rod quaternion normalisation
    controller.update(t, q, u)           # updates tensions for the next step
    return q, u

system.step_callback = _controlled_step_callback

# ---- solver ----
solver = BackwardEuler(system, t1=t_end, dt=dt, options=SolverOptions(newton_atol=1e-8, newton_rtol=1e-6, newton_max_iter=10))
sol = solver.solve()

# ---- plots ----
if controller.history["error"]:
    print(f"final |e| = {controller.history['error'][-1]*1e3:.3f} mm "
          f"(tip {tip_position(sol.q[-1])*1e2} cm, target {r_OP_ref_fn(sol.t[-1])*1e2} cm)")

if not SHOW:
    raise SystemExit(0)

from matplotlib import pyplot as plt

hist = controller.history
t = np.asarray(hist["t"])
r_OP = np.asarray(hist["r_OP"]) * 1e2     # cm
r_OP_ref = np.asarray(hist["r_OP_ref"]) * 1e2
abs_err = np.abs(r_OP - r_OP_ref)

fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
for i, label in enumerate(["x", "y", "z"]):
    ax[i].plot(t, r_OP[:, i], label="actual")
    ax[i].plot(t, r_OP_ref[:, i], "--", label="desired")
    ax[i].set_ylabel(f"{label} [cm]")
    ax[i].grid(True)
    ax[i].legend()
ax[2].set_xlabel("time [s]")
fig.suptitle("Static-model controller, dynamic plant (BackwardEuler)")
plt.tight_layout()

if traj_mode != "p2p":
    # plane plots
    h_idx, h_lab = (0, "X") if traj_mode == "circle_xz" else (1, "Y")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.plot(r_OP[:, h_idx], r_OP[:, 2], "r", label="actual")
    ax2.plot(r_OP_ref[:, h_idx], r_OP_ref[:, 2], "b--", label="desired")
    ax2.set_xlabel(f"{h_lab} [cm]")
    ax2.set_ylabel("Z [cm]")
    ax2.set_aspect("equal")
    ax2.grid(True)
    ax2.legend()
    fig2.suptitle(f"Trajectory tracking ({traj_mode})")
    plt.tight_layout()

    # absolute error per axis
    fig3, ax3 = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    for i, label in enumerate(["X", "Y", "Z"]):
        ax3[i].plot(t, abs_err[:, i])
        ax3[i].set_ylabel(f"|{label} err| [cm]")
        ax3[i].grid(True)
    ax3[2].set_xlabel("time [s]")
    fig3.suptitle("Absolute tracking error (paper axes)")
    plt.tight_layout()

plt.show()

# ---- visualization ----
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

plotter.render_solution(sol, True, play_speed_up=2.0)