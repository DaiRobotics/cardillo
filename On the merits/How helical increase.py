import os

import numpy as np
import sys
import csv
from datetime import datetime
from os import path
from pathlib import Path

from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System


from matplotlib import pyplot as plt

if __name__ == "__main__":
    rod_nelement = 58  # number of elements for the rod discretization
    VTK_export = False
    # ---- parameters ----
    rod_r_ratio = (
        1.0  # radius ratio of the rod along its length (tip radius / base radius)
    )
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1
    rod_l_new = 0.1 # [m] new length of the rod
    rod_o = 0.4e-3  # [m] rod outer radius
    rod_i=0.27e-3 # [m] rod inner radius

    d=5e-3
    beta = np.deg2rad(180)     # sets the helical tendon path
    p = 2*np.pi * rod_l_new / beta
                    
    ##################
    ## build system ##
    ##################

    # ---- system ----
    system = System()

    # ---- rod ----
    density=6450
    radius = rod_o
    cross_section = CircularCrossSection(radius)
    A = np.pi * (rod_o**2 - rod_i**2)
    I = np.pi/4 * (rod_o**4 - rod_i**4)
    J = 2 * I

    cross_section._area = A
    cross_section._second_moment = np.diag([J, I, I])
           
    E=70e9
    nu=0.3
    G = E / (2 * (1 + nu))

    EA = E * cross_section.area
    EI = E * cross_section.second_moment[1, 1]
    GA = G * cross_section.area
    GJ = G * cross_section.second_moment[0, 0]

    material_model = Simo1986(
        np.array([EA, GA, GA]),
        np.array([GJ, EI, EI]),
    )
  
    
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
    cross_section_inertias=CrossSectionInertias(density, cross_section)

    rod = DiscreteRod(
        cross_section,
        material_model,
        rod_nelement,
        Q=Q,
        q0=q0,
        cross_section_inertias=cross_section_inertias,
    )
    # ---- rigid connections ----
    rc = RigidConnection(rod, system.origin, xi1=0)

    # ---- tendons ( basic: 3 Straight tendon only, 1 helical tendon ) ----
    B_r_tendon_parameters = [ 
    (1.0, lambda X: 1/2*d, lambda X: np.sqrt(3)/2*d),
    (1.0, lambda X: -d, lambda X: 0), 
    (1.0, lambda X: 1/2*d, lambda X: -np.sqrt(3)/2*d ),
    (1.0, lambda X: d*np.cos(2*np.pi*X/p), lambda X: d*np.sin(2*np.pi*X/p)),
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
        xis=[i * xi_end / (n - 1) for i in range(n)],
        B_r_CPs=B_r_CP_list,
        )
        tendons.append(tendon)

    # ---- add to system ----
    system.add(rod)
    system.add(*tendons)
    system.add(rc)
    system.assemble()

    ##################################
    ###tendon force interpolation#####
    ##################################
    # generate desired forces 
    # basic：only 2 straight tendons are driven, the 3rd straight tendon and the helical tendon are always 0
    f_min = 0
    f_max = 5.8
    step = 200

    reverse = np.array([False] * 2)   # basic:Case is 2D，reverse level is 2, so reverse = [False, False] for the 2D case

    def gen_force(min_val, max_val, step, deep):
        if deep == 0:
            if reverse[deep]:
                return np.expand_dims(np.linspace(max_val, min_val, step), 0).T
            else:
                return np.expand_dims(np.linspace(min_val, max_val, step), 0).T
        else:
            if reverse[deep]:
                val_iter = np.linspace(max_val, min_val, step)
            else:
                val_iter = np.linspace(min_val, max_val, step)
            ret = None
            for val in val_iter:
                r = gen_force(min_val, max_val, step, deep - 1)
                r = np.hstack((np.ones((r.shape[0], 1)) * val, r))
                if ret is None:
                    ret = r
                else:
                    ret = np.vstack((ret, r))
                reverse[deep - 1] = False if reverse[deep - 1] else True
            return ret



    forces = gen_force(f_min, f_max, step, 2 - 1)   # basic: 2D not 3D, so deep=2-1=1
    assert np.all(np.sum(forces[:-1] != forces[1:], axis=1) == 1)

    ts = np.linspace(0, 1, forces.shape[0])


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


    plt.figure()
    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.plot(ts, forces[:, i])
        plt.xlabel('Time')
        plt.ylabel('Force')
    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), 'tendon_force_profiles_basic_3case.png'), dpi=300, bbox_inches='tight')
    plt.show()

    ############################################################
    ## Mechanism experiment
    ############################################################
    basic_npy = r"C:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\On the merits\workspace_samples_basic_0_5.8_step_200_3case.npy"

    F_s = 3.0          # N, straight force on tendons[1] after the first phase
    F_h_max = 5.8      # N, helical force at the end of run A
    n_steps = 100
    t_split = 0.3      # t < t_split: straight tendon ramps 0 -> F_s; t >= t_split: second phase

    def phase2(t):
        return max(0.0, (t - t_split) / (1.0 - t_split))

    def solve_sweep(F_straight, F_helical):
        tendons[0].la = lambda t: 0.0
        tendons[2].la = lambda t: 0.0
        tendons[1].la = F_straight
        tendons[3].la = F_helical
        solver = Newton(
            system,
            n_load_steps=n_steps,
            options=SolverOptions(newton_atol=1e-8, newton_rtol=1e-5),
        )
        return solver.solve()

    print("\n==== run A: helical force ramp ====")
    sol_A = solve_sweep(
        lambda t: F_s * min(t / t_split, 1.0),
        lambda t: F_h_max * phase2(t),
    )
    print("\n==== run B: straight force only (control) ====")
    sol_B = solve_sweep(
        lambda t: F_s * min(t / t_split, 1.0) + (F_h_max - F_s) * phase2(t),
        lambda t: 0.0,
    )

    def centerline(q, n=40):
        return np.array([rod.r_OP(0.0, q[rod.local_qDOF_P(xi)], xi) for xi in np.linspace(0, 1, n)])

    def metrics(P):
        dh = np.diff(P, axis=0)[:, :2]              # horizontal part of each rod piece
        seg = np.linalg.norm(dh, axis=1)
        # 1 = all pieces push the tip the same way, <1 = pieces partly cancel
        coherence = np.linalg.norm(dh.sum(axis=0)) / seg.sum() if seg.sum() > 0 else np.nan
        c = P - P.mean(axis=0)
        oop = np.sqrt(np.mean((c @ np.linalg.svd(c, full_matrices=False)[2][2]) ** 2))
        m = seg > 0.05 * seg.max()
        az = np.unwrap(np.arctan2(dh[m, 1], dh[m, 0]))
        turn = np.rad2deg(az.max() - az.min()) if m.sum() > 1 else 0.0
        return coherence, oop * 1000, turn          # oop in mm, turn in deg

    def analyze(sol):
        n_sol = len(sol.q)
        rows, shapes = [], []
        for j in range(int(t_split * (n_sol - 1)), n_sol):
            u = phase2(j / (n_sol - 1))             # progress of the second phase, 0 -> 1
            P = centerline(sol.q[j])
            coh, oop, turn = metrics(P)
            rows.append([u, np.hypot(P[-1, 0], P[-1, 1]) * 1000, P[-1, 2] * 1000, coh, oop, turn])
            shapes.append(P)
        return np.array(rows), shapes

    rows_A, shapes_A = analyze(sol_A)
    rows_B, shapes_B = analyze(sol_B)
    hdr = "u,r_tip[mm],z_tip[mm],coherence,out_of_plane[mm],direction_turn[deg]"
    np.savetxt(path.join(path.dirname(__file__), "mechanism_helical.csv"), rows_A, delimiter=",", header=hdr)
    np.savetxt(path.join(path.dirname(__file__), "mechanism_control.csv"), rows_B, delimiter=",", header=hdr)

    basic = np.load(basic_npy)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    ax[0].scatter(np.hypot(basic[:, 0], basic[:, 1]) * 1000, basic[:, 2] * 1000, s=1, alpha=0.08, c="gray")
    ax[0].plot(rows_B[:, 1], rows_B[:, 2], "k-", lw=1.5, label="control: straight force only")
    sc = ax[0].scatter(rows_A[:, 1], rows_A[:, 2], c=rows_A[:, 0] * F_h_max, cmap="viridis", s=16,
                       label="helical force ramp")
    plt.colorbar(sc, ax=ax[0], label="helical force F_h (N)")
    ax[0].set_xlabel("r_tip in mm"); ax[0].set_ylabel("z_tip in mm")
    ax[0].set_title("tip path vs basic shell (grey)"); ax[0].legend()

    for k in np.linspace(0, len(shapes_A) - 1, 4).astype(int):
        P = shapes_A[k]
        ax[1].plot(P[:, 0] * 1000, P[:, 1] * 1000, label=f"F_h = {rows_A[k, 0] * F_h_max:.1f} N")
        ax[1].plot(P[-1, 0] * 1000, P[-1, 1] * 1000, "o", c=ax[1].lines[-1].get_color())
    P = shapes_B[-1]
    ax[1].plot(P[:, 0] * 1000, P[:, 1] * 1000, "k--", label="control, end")
    ax[1].set_aspect("equal"); ax[1].legend()
    ax[1].set_xlabel("x in mm"); ax[1].set_ylabel("y in mm")
    ax[1].set_title("top view of the rod centerline")

    ax[2].plot(rows_A[:, 0], rows_A[:, 3], "C0-", label="coherence, helical")
    ax[2].plot(rows_B[:, 0], rows_B[:, 3], "C0--", label="coherence, control")
    ax[2].set_xlabel("progress of the second phase u"); ax[2].set_ylabel("coherence", color="C0")
    ax2 = ax[2].twinx()
    ax2.plot(rows_A[:, 0], rows_A[:, 5], "C3-", label="direction turn, helical")
    ax2.plot(rows_B[:, 0], rows_B[:, 5], "C3--", label="direction turn, control")
    ax2.set_ylabel("direction turn (deg)", color="C3")
    ax[2].set_title("do the pieces still push the same way?")

    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), "mechanism_sweep_basic.png"), dpi=300, bbox_inches="tight")
    plt.show()

    sol = sol_A     # the final animation below shows run A

    ############
    # VTK export
    ############
    if VTK_export:
        dir_name = Path(sys.argv[0]).parent
        print("exporting VTK")
        system.export(dir_name, f"vtk/tendon_robot_{rod_nelement}", sol, fps=50)
        print("finished")

    #################
    # visualization #
    #################
    from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

    VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
    for tendon in tendons:
        VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))
    window_size = (960, 540)
    plotter = Plotter(system, window_size)
    x0, x1 = -0.2, 0.2
    y0, y1 = -0.2, 0.2
    res_x = res_y = 10
    r_OC = np.array([0, -0.35, 0.1], float)
    r_OF = np.array([0, 0, 0.06], float)
    e_x_cam = np.array([1, 0, 0], float)
    e_z_cam = r_OF - r_OC
    e_z_cam /= np.linalg.norm(e_z_cam)
    e_y_cam = np.cross(e_z_cam, e_x_cam)
    zoom = 1
    fx = fy = 2635.5177
    px, py = 3840, 2160
    cam_view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)
    cam = plotter.camera
    cam.view_angle = cam_view_angle
    cam.parallel_projection = False
    cam.position = r_OC
    cam.focal_point = r_OF
    cam.view_up = -e_y_cam
    cam.clipping_range = (0.01, 1)
    cam.Zoom(zoom)

    plotter.render_solution(sol, True, play_speed_up=0.5)