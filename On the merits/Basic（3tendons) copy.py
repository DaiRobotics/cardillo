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
    beta = np.deg2rad(180)    
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

    # ---- tendons ( control: 3 straight tendons driven together, helical tendon not driven ) ----
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
    
    f_min = 0
    f_max = 5.8
    step = 53      # 53^3 = 148,877 force combinations, about the same as 3 x 37^3 of the Type I union
    file_path = path.join(
        path.dirname(__file__), f"workspace_samples_straight3driven_{f_min}_{f_max}_step_{step}.npy"
    )


    reverse = np.array([False] * 3)   # three driven straight tendons -> 3D force grid

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



    forces = gen_force(f_min, f_max, step, 3 - 1)   # 3 channels, deep = 3 - 1
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
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(ts, forces[:, i])
        plt.xlabel('Time')
        plt.ylabel('Force')
    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), 'tendon_force_profiles_straight3.png'), dpi=300, bbox_inches='tight')
    plt.show()

    ############
    ## solver ##
    ############
    tendons[0].la = lambda t: interp1d(ts, forces[:, 0], t)
    tendons[1].la = lambda t: interp1d(ts, forces[:, 1], t)
    tendons[2].la = lambda t: interp1d(ts, forces[:, 2], t)   # third straight tendon driven too
    tendons[3].la = lambda t: 0.0        # helical tendon not driven

    solver = Newton(
        system,
        n_load_steps=forces.shape[0]-1,
        options=SolverOptions(newton_atol=1e-8, newton_rtol=1e-5),
    )

    #sol = solver.solve()

    # ############################
    # ## workspace ##
    # ############################
    # el_dof = rod.local_qDOF_P(1.0)

    # tip_points = []
    # for q in sol.q:
    #     qe = q[el_dof]
    #     tip = rod.r_OP(0.0, qe, 1.0)
    #     if tip[2] > 0:
    #         tip_points.append(tip)
    # tip_points = np.array(tip_points)
    # print(f"totalload step: {len(sol.q)}, z>0的tip points: {len(tip_points)}")
    # print("tip workspace (bounding box):")
    # print(f"x: {tip_points[:,0].min()*1000:.2f} ~ {tip_points[:,0].max()*1000:.2f} mm")
    # print(f"y: {tip_points[:,1].min()*1000:.2f} ~ {tip_points[:,1].max()*1000:.2f} mm")
    # print(f"z: {tip_points[:,2].min()*1000:.2f} ~ {tip_points[:,2].max()*1000:.2f} mm")

    # def voxel_volume(points, voxel_size=1e-3):
    #     voxel_idx = np.floor(points / voxel_size).astype(int)
    #     unique_voxels = np.unique(voxel_idx, axis=0)
    #     return len(unique_voxels) * voxel_size**3

    # if len(tip_points) > 0:
    #     volume_cm3 = voxel_volume(tip_points, voxel_size=1e-3) * 1e6
    #     print(f"3 straight tendons driven together, workspace volume: {volume_cm3:.2f} cm^3")
    # else:
    #     print("Warning: No tip points in z>0 space, workspace is 0")
    #     volume_cm3 = 0.0

    # np.save(file_path, tip_points)
    # print(f"tip point cloud saved to: {file_path}")

    #########Workspace volume visualization#################

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        tip_points[:, 0]*1000, tip_points[:, 1]*1000, tip_points[:, 2]*1000,
        s=1, alpha=0.3, c='gray'
    )
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.set_title(f'3 straight tendons driven together, V={volume_cm3:.1f} cm^3')
    plt.tight_layout()
    plt.savefig(path.join(path.dirname(__file__), 'scatter_straight3driven.png'), dpi=300, bbox_inches='tight')
    plt.close()
    ################### ----Print out Summary----###############################
    summary_file = path.join(path.dirname(__file__), "workspace_summary.csv")
    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:   
            writer.writerow(["Time", "routing", "beta(deg)", "step", "Total Points", "z>0 points", "Volume(cm^3)",
                            "x_min(mm)", "x_max(mm)", "y_min(mm)", "y_max(mm)", "z_min(mm)", "z_max(mm)",
                            "npy file"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            "3 straight driven (control)",
            "N/A",                              
            step,
            len(sol.q),
            len(tip_points),
            f"{volume_cm3:.2f}",
            f"{tip_points[:,0].min()*1000:.2f}",
            f"{tip_points[:,0].max()*1000:.2f}",
            f"{tip_points[:,1].min()*1000:.2f}",
            f"{tip_points[:,1].max()*1000:.2f}",
            f"{tip_points[:,2].min()*1000:.2f}",
            f"{tip_points[:,2].max()*1000:.2f}",
            path.basename(file_path),
        ])
    print(f"Summary saved to: {summary_file}")

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