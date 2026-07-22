import numpy as np
import sys
from pathlib import Path

from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System


from matplotlib import pyplot as plt

if __name__ == "__main__":
    rod_nelement = 58  # number of elements for the rod discretization
    VTK_export = True
    # ---- parameters ----
    rod_r0 = 30e-3  # [m] rod radius
    rod_l0 = 95e-3  # [m] length of the rod
    rod_r_ratio = (
        0.533  # radius ratio of the rod along its length (tip radius / base radius)
    )
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
    rod_l_new = 0.58  # [m] new length of the rod
    rod_r_new = 15e-3  # [m] rod radius
    d1=10e-3
    d2=5e-3
    xi_end_base=0.20/rod_l_new
    p1=0.2
    p2=0.29
    ##################
    ## build system ##
    ##################

    # ---- system ----
    system = System()

    # ---- rod ----
    density=1.41e3
    radius = lambda xi: rod_r_new * (1 - xi * (1 - rod_r_ratio))
    cross_section = CircularCrossSection(radius)
    EA = lambda xi: E * cross_section.area(xi)
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
    GA = lambda xi: G * cross_section.area(xi)
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
    material_model = Simo1986(
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
    )
    cross_section = CircularCrossSection(radius=radius)
    E, G = 5.93e5,1.977e5

    # generate initial configuration
    def r_OP(xi):
        return np.array([xi * rod_l_new, 0, 0], dtype=np.float64)

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
    rc = RigidConnection(rod, system.origin, xi1=0)

    # ---- tendons ( 4 Spiral tendon) ----
    B_r_tendon_parameters = [ 
    (xi_end_base, lambda X: d1*np.cos(2*np.pi*X/p1), lambda X: d1*np.sin(2*np.pi*X/p1)),
    (xi_end_base, lambda X: d1*np.cos(2*np.pi*X/p1), lambda X: -d1*np.sin(2*np.pi*X/p1)), 
    (1.0, lambda X: d2*np.cos(2*np.pi*X/p2), lambda X: d2*np.sin(2*np.pi*X/p2) ),
    (1.0, lambda X: d2*np.cos(2*np.pi*X/p2), lambda X: -d2*np.sin(2*np.pi*X/p2)),
    ]

    tendons = [ ]
    for xi_end, y, z in B_r_tendon_parameters:
        n_path_points = int(np.ceil(xi_end * rod_nelement)) + 1 
        B_r_CP_list=[
        rod_A_IB0.T@ np.array(
            [
                y(xi*rod_l_new),
                z(xi*rod_l_new),
                0,
            ]
        )
        for xi in np.linspace(0, xi_end, n_path_points)
        ]
        n = len(B_r_CP_list)  
        tendon = RodTendonForce(
        rod, 
        xis=[i * xi_end / (n - 1) for i in range(n)],#或者np.linspace(0.0, xi_end ,n_path_points)
        B_r_CPs=B_r_CP_list,
        )
        tendons.append(tendon)

    # tendons[1].la = lambda t: 50 * (1 + np.sin(2 * np.pi * t / T + np.pi)) / 2
    # tendons[1].la = lambda t: t * 1.5

    # ---- add to system ----
    system.add(rod)
    system.add(*tendons)
    system.add(rc)
    system.assemble()

    ############
    ## solver ##
    ############
    F0 = 4
    tendons[0].la = lambda t: F0 * t
    solver = Newton(
        system,
        n_load_steps=8,
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
    )

    sol = solver.solve()

    ############
    # VTK export
    ############
    if VTK_export:
        dir_name = Path(sys.argv[0]).parent
        print("exporting VTK")
        # fake second bob for export
        system.export(dir_name, f"vtk/tendon_robot_{rod_nelement}", sol, fps=50)
        print("finished")

    #################
    # visualization #
    #################
    # ---- visual objects ----
    from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

    VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
    for tendon in tendons:
        VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))  # (130, 130, 130),
    # VisualCoordSystem(system.origin, 0.05)
    # ---- plotter ----
    window_size = (960, 540)
    plotter = Plotter(system, window_size)
    x0, x1 = -0.2, 0.2
    y0, y1 = -0.2, 0.2
    res_x = res_y = 10
    # plotter.add_ground(x0, x1, y0, y1, res_x, res_y)
    # ---- camera pose ----
    r_OC = np.array([0, -0.35, 0.1], float)
    # r_OC = np.array([0, -0.35, 0.15], float)
    r_OF = np.array([0, 0, 0.06], float)  # camera focal point
    e_x_cam = np.array([1, 0, 0], float)
    e_z_cam = r_OF - r_OC
    e_z_cam /= np.linalg.norm(e_z_cam)
    e_y_cam = np.cross(e_z_cam, e_x_cam)
    zoom = 1
    # zoom = 1.5
    fx = fy = 2635.5177
    px, py = 3840, 2160  # camera 4k resolution
    cam_view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)
    cam = plotter.camera
    cam.view_angle = cam_view_angle
    cam.parallel_projection = False
    cam.position = r_OC
    cam.focal_point = r_OF
    cam.view_up = -e_y_cam
    cam.clipping_range = (0.01, 1)
    cam.Zoom(zoom)

    # plotter.live_render()

    plotter.render_solution(sol, True, play_speed_up=0.5)

