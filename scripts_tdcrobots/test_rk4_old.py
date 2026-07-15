
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
from scipy.linalg import pinv
from scipy.sparse.linalg import splu
from tdcm_li2023 import interp1d, interp1d_poly, smoothing_polynomial, StaticSolver, visualization_p2p
from runge_kutta import RungeKutta


# ---- physical parameters ----
rod_nelement = 10  # 1000
rod_l0 = 0.192  # [m] length of rod
rod_r0_base = 1.4e-2  # [m] radius at bottom of rod
rod_r0_tip = 8.5e-3  # [m] radius at tip of rod
rod_density = 1.41e3  # density of material
rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
E, G = 2.563e5, 8.543e4

G_ACCEL = 9.81

# ---- rod ----
radius = lambda xi: rod_r0_base * (1 - xi) + rod_r0_tip * xi
cross_section = CircularCrossSection(radius)
EA = lambda xi: E * cross_section.area(xi)
EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
GA = lambda xi: G * cross_section.area(xi)
GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
material_model = Simo1986(
    lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
    lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
)
damping_ratio = 0

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
    A_IB0=rod_A_IB0,
)
Q = q0.copy()

rod = DiscreteRod(
    cross_section,
    material_model,
    rod_nelement,
    Q=Q,
    q0=q0,
    cross_section_inertias=CrossSectionInertias(
        rod_density, cross_section
    ),
    damping_ratio=damping_ratio,
)

# ---- rigid connections ----
# Redundant when using RK4 due to node 0 not contributing any DOF
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
    tendon = RodTendonForce(
        rod,
        [i / (n - 1) for i in range(n)],
        B_r_CPs=B_r_CP_list,
    )
    tendons.append(tendon)

system.add(rod, rc, *tendons)

# ---- external forces ----
gravity = Force_line_distributed(
    lambda t, xi: rod_density
    * cross_section.area(xi)
    * G_ACCEL
    * np.array([0, -1.0, 0], dtype=np.float64),
    rod,
)
# gravity = Force_line_distributed(
#     lambda t, xi: rod_density
#     * cross_section.area(xi)
#     * G_ACCEL
#     * t
#     * np.array([0, -1.0, 0], dtype=np.float64),
#     rod,
# )
system.add(gravity)
system.assemble()

force_init = np.array([td.la(0) for td in tendons])

# solver = Newton(system, n_load_steps=100, verbose=True)

t1 = 5
dt = 1e-3
# solver = BackwardEuler(system, t1, dt)

fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
solver = RungeKutta(system, t1, dt, fixed_qDOF = fixed_qDOF, fixed_uDOF=fixed_uDOF)
sol = solver.solve()
solver2 = ScipyDAE(system,t1,dt)
sol2 = solver2.solve()

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

# plotter.live_render()
plotter.render_solution(sol, True, play_speed_up=1)

from matplotlib import pyplot as plt

t = sol.t
q = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
r_OP = q[:,-1, 0:3]

t2 = sol2.t
q2 = sol2.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
r_OP2 = q[:,-1, 0:3]

# ---- Point to Point plots ----
fig = plt.figure(figsize=(8,6))
gs = fig.add_gridspec(3, 1)

atx = fig.add_subplot(gs[0, 0])
atx.plot(t, q[:, -1, 0], "r", label="actual")
atx.plot(t2, q2[:, -1, 0], "--b", label="ScipyDAE")
atx.set_xlabel("Time [s]")
atx.set_ylabel("X [m]")
atx.legend()
atx.grid(True)

aty = fig.add_subplot(gs[1, 0])
aty.plot(t, q[:, -1, 1], "r", label="actual")
aty.plot(t2, q2[:, -1, 1], "--b", label="ScipyDAE")
aty.set_xlabel("Time [s]")
aty.set_ylabel("Y [m]")
aty.legend()
aty.grid(True)

atz = fig.add_subplot(gs[2, 0])
atz.plot(t, q[:, -1, 2], "r", label="actual")
atz.plot(t2, q2[:, -1, 2], "--b", label="ScipyDAE")
atz.set_xlabel("Time [s]")
atz.set_ylabel("Z [m]")
atz.legend()
atz.grid(True)


fig.suptitle(f"Falling Test")
fig.tight_layout()

plt.show()