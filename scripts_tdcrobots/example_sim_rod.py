import numpy as np
from time import perf_counter

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

from cardillo.solver import Newton
from cardillo.system import System

if __name__ == "__main__":
    ##################
    ## build system ##
    ##################
    # ---- parameters ----
    rod_nelement = 10  # number of elements for the rod discretization
    rod_m = 0.433  # [kg] mass of the rod
    rod_r = 30e-3  # [m] rod radius
    rod_l = 95e-3  # [m] length of the rod
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
    connector_h = 11.5e-3  # [m] height of the rod foot
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
    cross_section = CircularCrossSection(rod_r)
    E, G = 7e5, 2e5
    EA = E * cross_section.area
    EI = E * cross_section.second_moment[1, 1]
    GA = G * cross_section.area
    GJ = G * cross_section.second_moment[0, 0]
    material_model = Simo1986(
        np.array([EA, GA, GA]),
        np.array([GJ, EI, EI]),
    )

    # generate initial configuration
    r_OP0 = np.array([0, 0, connector_h])
    Rod = DiscreteRod
    rod_l_ref = rod_l / (1 - (mk_platform_m + rod_m / 2) * gravity_g / EA)
    Q = Rod.straight_configuration(
        rod_nelement, rod_l_ref, r_OP0=r_OP0, A_IB0=rod_A_IB0
    )

    def r_OP(xi):
        z = (
            xi
            + rod_m * gravity_g / EA / 2 * (xi**2 - 2 * xi)
            - mk_platform_m * gravity_g / EA * xi
        ) * rod_l_ref
        return np.array([z, 0, 0], dtype=np.float64)

    A_IB = lambda xi: np.eye(3, dtype=np.float64)
    q0 = Rod.pose_configuration(
        rod_nelement,
        r_OP,
        A_IB,
        r_OC0=r_OP0,
        A_IB0=rod_A_IB0,
    )

    density = rod_m / rod_l_ref / cross_section.area
    rod = Rod(
        cross_section,
        material_model,
        rod_nelement,
        Q=Q,
        q0=q0,
        cross_section_inertias=CrossSectionInertias(density, cross_section),
    )

    # ---- rigid connections ----
    rc1 = RigidConnection(rod, system.origin, xi1=0)

    # ---- external forces ----
    gravity_rod = Force_line_distributed(
        np.array([0, 0, -rod_m * gravity_g / rod_l_ref]),
        rod,
    )
    gravity_marker_platform = Force(
        np.array([0, 0, -mk_platform_m * gravity_g]),
        rod,
        xi=1,
        B_r_CP=rod_A_IB0.T
        @ np.array([0, 0, connector_h + mk_platform_h / 2 - mk_platform_cut_h]),
        name="gravity_marker_platform",
    )

    # ---- tendons ----
    B_r_CP_lists = [
        [
            rod_A_IB0.T
            @ np.array([tendon_hole_r * np.cos(phi), tendon_hole_r * np.sin(phi), 0]),
            rod_A_IB0.T
            @ np.array(
                [
                    tendon_hole_r * np.cos(phi),
                    tendon_hole_r * np.sin(phi),
                    connector_h - mk_platform_cut_h,
                ]
            ),
        ]
        for phi in np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ]
    tendons = []
    for B_r_CP_list in B_r_CP_lists:
        tendon = RodTendonForce(
            rod,
            [0, 1],
            B_r_CPs=B_r_CP_list,
        )
        tendons.append(tendon)
    tendons[0].la = lambda t: t * 50
    system.add(*tendons)
    # ---- add to system ----
    system.add(rod)
    system.add(gravity_marker_platform)
    system.add(gravity_rod)
    system.add(rc1)

    system.assemble()
    force_init = np.array([td.la(0) for td in tendons])

    ############
    ## solver ##
    ############
    solver = Newton(
        system,
        n_load_steps=100,
        verbose=True,
    )

    solver.fun(solver.x[0], system.t0)
    solver.jac(solver.x[0], system.t0)

    # from cProfile import Profile
    # prof = Profile()
    # prof.enable()

    t0 = perf_counter()
    sol = solver.solve()
    print("time:", perf_counter() - t0)

    # prof.disable()
    # prof.dump_stats("prof.prof")
    # exit()
    #################
    # visualization #
    #################
    # ---- visual objects ----
    from cardillo.visualization import (
        Plotter,
        VisualDiscreteRod,
        VisualSTL,
        # VisualCoordSystem,
        VisualTendon,
        VisualArUco,
    )

    VisualArUco(
        rod,
        xi=1,
        mk_size=0.04,
        mk_dis=0.05,
        A_BM=rod_A_IB0.T,
        B_r_CP=rod_A_IB0.T
        @ np.array([0, 0, connector_h + mk_platform_h - mk_platform_cut_h]),
    )

    VisualSTL(
        rod,
        "scripts_tdcrobots/stl/Segment_Foot_V2.stl",
        xi=1,
        A_BM=rod_A_IB0.T,
        B_r_CP=rod_A_IB0.T @ np.array([0, 0, connector_h / 2]),
        scale=1e-3,
        color=(160, 160, 160),
    )
    VisualSTL(
        rod,
        "scripts_tdcrobots/stl/Marker_Platform_Target_V2.stl",
        xi=1,
        B_r_CP=rod_A_IB0.T @ np.array([0, 0, connector_h - mk_platform_cut_h]),
        A_BM=rod_A_IB0.T,
        scale=1e-3,
        color=(255, 250, 240),
    )
    VisualSTL(
        system.origin,
        "scripts_tdcrobots/stl/Segment_Foot_V2.stl",
        B_r_CP=np.array([0, 0, connector_h - mk_platform_cut_h / 2]),
        scale=1e-3,
        color=(160, 160, 160),
    )
    VisualDiscreteRod(rod, subdivision=4)
    for tendon in tendons:
        VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))  # (130, 130, 130),
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

    plotter.render_solution(sol, True)
