import numpy as np

from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, TendonForce
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods import DiscreteRod

from cardillo.solver import Newton, ScipyDAE
from cardillo.system import System

if __name__ == "__main__":
    ##################
    ## build system ##
    ##################
    # ---- parameters ----
    rod_nelement = 500  # number of elements for the rod discretization
    rod_m = 0.433 * 2  # [kg] mass of the rod
    rod_r0 = 30e-3  # [m] rod radius
    rod_l0 = 95e-3  # [m] length of the rod
    density = rod_m / (
        np.pi * rod_r0**2 * rod_l0
    )  # [kg/m^3] density of the rod material
    rod_l = 1.0
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
    mk_platform_m = 0.185  # [kg] mass of the marker platform
    mk_platform_h = 14.5e-3  # [m] height of the marker platform
    mk_platform_cut_h = (
        11.5e-3  # [m] height of the cut on the marker platform for the rod to pass into
    )
    gravity_g = 9.81  # [m/s^2] gravitational acceleration
    tendon_hole_r = 65e-3  # [m] radius of the hole on the marker platform for the tendon to pass through

    # ---- system ----
    system = System()

    # ---- rod ----
    radius = lambda xi: rod_r0 * (1 - xi * (1 - 0.2))
    cross_section = CircularCrossSection(radius=radius)
    E, G = 7e5, 2e5
    EA = lambda xi: E * cross_section.area(xi)
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
    GA = lambda xi: G * cross_section.area(xi)
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
    material_model = Simo1986(
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
    )

    # generate initial configuration
    Rod = DiscreteRod
    Q = Rod.straight_configuration(rod_nelement, rod_l, A_IB0=rod_A_IB0)

    def r_OP(xi):
        return np.array([xi * rod_l, 0, 0], dtype=np.float64)

    A_IB = lambda xi: np.eye(3, dtype=np.float64)
    q0 = Rod.pose_configuration(
        rod_nelement,
        r_OP,
        A_IB,
        A_IB0=rod_A_IB0,
    )

    rod = Rod(
        cross_section,
        material_model,
        rod_nelement,
        Q=Q,
        q0=q0,
        cross_section_inertias=CrossSectionInertias(density, cross_section),
    )

    # ---- frame ----
    A = 0.1
    Omg = 2 * np.pi
    r_OP = lambda t: np.array([A * (1 - np.cos(Omg * t)), 0, 0], dtype=float)
    r_OP_t = lambda t: np.array([A * Omg * np.sin(Omg * t), 0, 0], dtype=float)
    r_OP_tt = lambda t: np.array([A * Omg**2 * np.cos(Omg * t), 0, 0], dtype=float)
    frame = Frame(r_OP, r_OP_t, r_OP_tt)

    # ---- rigid connections ----
    rc = RigidConnection(rod, frame, xi1=0)

    # ---- external forces ----

    # ---- add to system ----
    system.add(rod)
    system.add(rc)
    system.add(frame)

    system.assemble()

    ############
    ## solver ##
    ############
    solver = ScipyDAE(
        system,
        t1=1.0,
        dt=0.01,
    )
    sol = solver.solve()

    ########
    # plot #
    ########
    from matplotlib import pyplot as plt

    t = sol.t
    q_nodes = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
    plt.plot(t, q_nodes[:, 0, 0])
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.grid()
    plt.show(block=False)

    #################
    # visualization #
    #################
    # ---- visual objects ----
    from cardillo.visualization import (
        Plotter,
        VisualDiscreteRod,
    )

    VisualDiscreteRod(rod, subdivision=4)
    # VisualCoordSystem(system.origin, 0.05)
    # ---- plotter ----
    window_size = (960, 540)
    plotter = Plotter(system, window_size)
    x0, x1 = -0.2, 0.2
    y0, y1 = -0.2, 0.2
    res_x = res_y = 10
    plotter.add_ground(x0, x1, y0, y1, res_x, res_y)
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

    plotter.show()
    plotter.render_solution(sol, True, play_speed_up=0.1)
