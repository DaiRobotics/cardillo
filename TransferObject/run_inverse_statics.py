import numpy as np
import sys
from pathlib import Path
from scipy.optimize import minimize, least_squares

from cardillo import solver
from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce
from cardillo.math import Exp_SO3_quat
from cardillo.rods import (
    CircularCrossSection,
    Simo1986,
    DiscreteRod,
    CrossSectionInertias,
)

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System


def get_model(rod_nelement=10):
    ###### Parameters ######
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = 1  # B_ey = +e_x
    rod_A_IB0[1, 2] = -1  # B_ez = -e_y (right-handed)
    rod_A_IB0[2, 0] = -1  # B_ex = -e_z (rod hangs down)

    rod_l = 1.0
    rod_r = 15e-3  # [m] rod radius (not sure if this is the right radius)
    L1 = 0.5
    xi_end_short = L1 / rod_l
    ds = 10e-3  # [m] short tendon distance from centerline
    dl = 5e-3  # [m] long tendon distance from centerline
    E, G = 5.93e5, 1.977e5

    ###### System ######
    system = System()
    density = 1.41e3  # [kg/m^3] (not sure if this is the right density; unused without gravity)
    radius = lambda xi: rod_r
    EA = lambda xi: E * cross_section.area(xi)
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
    GA = lambda xi: G * cross_section.area(xi)
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
    material_model = Simo1986(
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
    )
    cross_section = CircularCrossSection(radius=radius)

    # generate initial configuration
    def r_OP(xi):
        return np.array([xi * rod_l, 0, 0], dtype=np.float64)

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
        cross_section_inertias=CrossSectionInertias(density, cross_section),
    )
    # ---- rigid connections ----
    rc = RigidConnection(rod, system.origin, xi1=0)  # clamp the base at the origin

    ###### Tendons (4 straight tendons) ######
    B_r_tendon_parameters = [
        (xi_end_short, ds, 0.0),  # [0] short, right
        (xi_end_short, -ds, 0.0),  # [1] short, left
        (1.0, dl, 0.0),  # [2] long, right
        (1.0, -dl, 0.0),  # [3] long, left
    ]
    tendons = []
    for xi_end, x_w, y_w in B_r_tendon_parameters:
        n_path_points = int(np.ceil(xi_end * rod_nelement)) + 1  # one point per node
        B_r_CP_list = [
            rod_A_IB0.T
            @ np.array(
                [
                    x_w,
                    y_w,
                    0,
                ]
            )  # world offset -> body frame B
            for xi in np.linspace(0, xi_end, n_path_points)
        ]
        n = len(B_r_CP_list)
        tendon = RodTendonForce(
            rod,
            xis=[i * xi_end / (n - 1) for i in range(n)],
            B_r_CPs=B_r_CP_list,
        )
        tendons.append(tendon)

    # ---- add to system ----
    system.add(rod)
    system.add(*tendons)
    system.add(rc)
    system.assemble()
    return system, rod, tendons


def test_forward_statics(la_t, rod_nelement=10, n_load_steps=1, export_VTK=False):
    system, rod, tendons = get_model(rod_nelement=rod_nelement)
    system.set_tau(lambda t: t * la_t)

    newton = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
    )
    sol = newton.solve(debug=False)

    ###### VTK export ######
    if export_VTK:
        dir_name = Path(sys.argv[0]).parent
        print("exporting VTK")
        system.export(dir_name, f"vtk/transfer_forward_{rod_nelement}", sol, fps=50)
        print("finished")
    return sol


def test_inverse_statics(r_OP_des, export_VTK=True):
    system, rod, tendons = get_model(rod_nelement=10)

    newton = Newton(
        system,
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
        verbose=False,
    )

    sol = None

    def forward_statics(la_t):
        nonlocal sol, la0
        # warm start from the last solution if available
        if sol is not None:
            # last solution
            la0 = system.tau(1.0)
            x0 = newton.x[-1]
            f0 = newton.f[-1]
            newton.reset(x0=x0, f0=f0)
        else:
            la0 = np.zeros(4)

        # set tendon force
        system.set_tau(
            lambda t, la0=la0.copy(), la_t=la_t.copy(): la0 + t * (la_t - la0)
        )

        # compute statics wrt. the tendon force
        sol = newton.solve()
        q = sol.q[-1]  # final configuration

        # evaluate tip position and orientation
        r_OP = q[-7:-4]
        p = q[-4:]
        A_IB = Exp_SO3_quat(p)

        return r_OP, A_IB, sol

    def constraints_eq(x):
        la_t = x * lambda_scale
        r_OP, A_IB, _ = forward_statics(la_t)

        dr = (r_OP - r_OP_des)[[0, 2]]
        da = A_IB[0, 0]
        c = np.concatenate([dr, [da]])
        print(
            f"la_t = {la_t}, r={np.round(r_OP,2)}, dr = {dr}, da = {da}, c = {np.round(c, 6)}"
        )
        assert sol.success, "Newton solver failed to converge"
        return c

    cons = [{"type": "eq", "fun": constraints_eq}]

    # initial guess for tendon forces
    la0 = np.array([38, 15, 41, 64]) * 0.0  # [rs, ls, rl, ll]
    la0 = np.array([0.0] * 4)  # [rs, ls, rl, ll]
    lambda_scale = 30
    lambda_min = 0.0
    lambda_max = np.inf
    res = minimize(
        lambda x: x @ x,  # objective ||lambda||^2 / lambda_scale^2
        x0=la0 / lambda_scale,
        jac=lambda x: 2 * x,  # gradient of the objective
        method="SLSQP",
        bounds=[(lambda_min / lambda_scale, lambda_max / lambda_scale)] * 4,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 200, "eps": 1e-5},  # 200 iterations
    )
    print(res)
    la_t = res.x * lambda_scale
    return la_t


H = 1.0  # [m] base height above ground
x_right = 0.5
z_right = 0.8

r_OP_des = np.array([x_right, 0.0, z_right - H])  # desired tip position

la_t = test_inverse_statics(r_OP_des)

test_forward_statics(la_t, n_load_steps=1, export_VTK=True)

# ###### Visualization ######
# if visualize and sol is not None:
#     from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

#     VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
#     for tendon in tendons:
#         VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))
#     plotter = Plotter(system, (960, 540))
#     # camera looks along +y at the x-z task plane
#     cam = plotter.camera
#     cam.parallel_projection = False
#     cam.position = np.array([0, -3.0, -0.5], float)
#     cam.focal_point = np.array([0, 0, -0.5], float)
#     cam.view_up = np.array([0, 0, 1], float)
#     cam.clipping_range = (0.01, 100)
#     plotter.render_solution(sol, True, play_speed_up=0.5)
