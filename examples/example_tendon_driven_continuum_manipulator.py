import numpy as np
import sys
from pathlib import Path

from cardillo.rigid_connection import RigidConnection
from cardillo.forces import TendonForce

from cardillo.rods import CircularCrossSection
from cardillo.rods.discreteRod import DiscreteRod

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System


from matplotlib import pyplot as plt

if __name__ == "__main__":
    rod_nelement = 100  # number of elements for the rod discretization
    VTK_export = False
    # ---- parameters ----
    rod_r0 = 30e-3  # [m] rod radius
    rod_l0 = 95e-3  # [m] length of the rod
    rod_r_ratio = (
        0.4  # radius ratio of the rod along its length (tip radius / base radius)
    )
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
    rod_l_new = 0.2  # [m] new length of the rod
    rod_r_new = rod_l_new * 0.05  # [m] rod radius

    ##################
    ## build system ##
    ##################

    # ---- system ----
    system = System()

    # ---- rod ----
    radius = lambda xi: rod_r_new * (1 - xi * (1 - rod_r_ratio))
    cross_section = CircularCrossSection(radius=radius)
    E, G = 7e5, 2e5

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
        E,
        G,
        rod_nelement,
        Q=Q,
        q0=q0,
    )

    # ---- rigid connections ----
    rc = RigidConnection(rod, system.origin, xi1=0)

    # ---- tendons ----
    B_r_CP_list = [
        rod_A_IB0.T
        @ np.array(
            [
                radius(xi) * np.cos(0),
                radius(xi) * np.sin(0),
                0,
            ]
        )
        for xi in np.linspace(0, 1, rod_nelement + 1)
    ]
    tendons = []
    n = len(B_r_CP_list)
    tendon = TendonForce(
        subsystem_list=[rod for _ in range(n)],
        connectivity=[(i, i + 1) for i in range(n - 1)],
        xi_list=[i / (n - 1) for i in range(n)],
        B_r_CP_list=B_r_CP_list,
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
    ########
    # plot #
    ########
    t = sol.t
    q_nodes = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
    plt.plot(q_nodes[:, -1, 0], q_nodes[:, -1, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.grid()

    _, B_gamma, B_kappa = rod._eval_els(sol.q[-1, rod.qDOF])
    element_indices = np.arange(1, rod_nelement + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # axial strain gamma_1
    axes[0].plot(element_indices, B_gamma[:, 0])
    axes[0].set_xlabel("Element index $i$")
    axes[0].set_ylabel(r"$\gamma_{1}$")
    axes[0].set_title("Axial strain")
    axes[0].grid()

    # shear strain gamma_2
    axes[1].plot(element_indices, B_gamma[:, 1])
    axes[1].set_xlabel("Element index $i$")
    axes[1].set_ylabel(r"$\gamma_{2}$")
    axes[1].set_title("Shear strain")
    axes[1].grid()

    # bending curvature kappa_3
    axes[2].plot(element_indices, B_kappa[:, 2])
    axes[2].set_xlabel("Element index $i$")
    axes[2].set_ylabel(r"$\kappa_{3}$  [1/m]")
    axes[2].set_title("Bending curvature")
    axes[2].grid()

    plt.tight_layout()
    plt.show(block=True)

    plotter.render_solution(sol, True, play_speed_up=0.5)
