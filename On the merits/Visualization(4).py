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



if __name__ == "__main__":
    rod_nelement = 58  # number of elements for the rod discretization
    ROUTING = "I"          # "basic" | "I" | "II" | "III"
    ACTIVE = [1,2,3]            #[1,2]and [1,2,3]
    N_LOAD_STEPS = 150       # load steps of the 0 -> 5.8 N ramp
    FRAME_STRIDE = 3         # export every 3rd load step as one frame
    F_MAX = 5.8              # [N]
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

    # ---- tendon routing (same formulas as the workspace scripts) ----
    half_l = rod_l_new / 2
    p_half = 2 * np.pi * half_l / beta

    def y_I(X):   return d * np.cos(beta * X / rod_l_new)
    def z_I(X):   return d * np.sin(beta * X / rod_l_new)

    def y_II(X):  return d if X < half_l else d * np.cos(2 * np.pi * (X - half_l) / p_half)
    def z_II(X):  return 0.0 if X < half_l else d * np.sin(2 * np.pi * (X - half_l) / p_half)

    def y_III(X): return d * np.cos(2 * np.pi * X / p_half) if X < half_l else d * np.cos(beta - 2 * np.pi * (X - half_l) / p_half)
    def z_III(X): return d * np.sin(2 * np.pi * X / p_half) if X < half_l else d * np.sin(beta - 2 * np.pi * (X - half_l) / p_half)

    helical = {"I": (y_I, z_I), "II": (y_II, z_II), "III": (y_III, z_III)}.get(ROUTING)#choose the routing type, if not basic

    B_r_tendon_parameters = [
        (1.0, lambda X: 1/2*d, lambda X: np.sqrt(3)/2*d),
        (1.0, lambda X: -d,    lambda X: 0),
        (1.0, lambda X: 1/2*d, lambda X: -np.sqrt(3)/2*d),
    ]
    if helical is not None:
        B_r_tendon_parameters.append((1.0, helical[0], helical[1]))

    tendons, tendon_B_r_CPs = [], []
    n_path_points = rod_nelement + 1
    xis_path = np.linspace(0.0, 1.0, n_path_points)
    for xi_end, y, z in B_r_tendon_parameters:
        B_r_CP_list = [rod_A_IB0.T @ np.array([y(xi * rod_l_new), z(xi * rod_l_new), 0]) for xi in xis_path]#Transform the path point offsets into the local coordinate system.
        tendons.append(RodTendonForce(rod, xis=list(xis_path), B_r_CPs=B_r_CP_list))
        tendon_B_r_CPs.append(B_r_CP_list)

    system.add(rod)
    system.add(*tendons)
    system.add(rc)
    system.assemble()

    # ---- load case: selected tendons ramp 0 -> F_MAX together, the others stay at 0 ----
    if ACTIVE is None:
        ACTIVE = [1, 2, 3] if helical is not None else [1, 2]    # Type: 2 straight + helical, basic: 2 straight
    print("routing:", ROUTING, "| active tendons:", ACTIVE)
    for k, tendon in enumerate(tendons):
        if k in ACTIVE:
            tendon.la = lambda t: F_MAX * t
        else:
            tendon.la = lambda t: 0.0

    solver = Newton(
        system,
        n_load_steps=N_LOAD_STEPS,
        options=SolverOptions(newton_atol=1e-8, newton_rtol=1e-5),
    )
    sol = solver.solve()
    print("solved, frames in solution:", len(sol.q))

    # ---- export deformed backbone + tendons as VTK time series (unit: mm) ----
    out_dir = path.join(path.dirname(path.abspath(__file__)),
                    f"vtk_ramp_{ROUTING}_{''.join(map(str, ACTIVE))}")
    os.makedirs(out_dir, exist_ok=True)

    def point_on_rod(q, xi, B_r_CP):
        qe = q[rod.local_qDOF_P(xi)]
        try: #first try to use the B_r_CP argument, if it is not provided, fall back to the default behavior
            return np.asarray(rod.r_OP(0.0, qe, xi, B_r_CP), float)       # position of the tendon point in the world frame
        except TypeError:
            return np.asarray(rod.r_OP(0.0, qe, xi), float) + np.asarray(rod.A_IB(0.0, qe, xi), float) @ B_r_CP

    def write_frame(fname, polylines, ids):#write a single frame to a VTK file
        pts = np.vstack(polylines) * 1000.0
        with open(fname, "w") as f:
            f.write("# vtk DataFile Version 3.0\ndeformed tendons\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(pts)} float\n")
            for pt in pts:
                f.write(f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}\n")
            f.write(f"LINES {len(polylines)} {sum(len(pl) + 1 for pl in polylines)}\n")
            off = 0
            for pl in polylines:
                f.write(str(len(pl)) + " " + " ".join(str(off + i) for i in range(len(pl))) + "\n")
                off += len(pl)
            f.write(f"CELL_DATA {len(polylines)}\nSCALARS tendon_id int 1\nLOOKUP_TABLE default\n")
            for i in ids:
                f.write(f"{i}\n")

    frame_ids = list(range(0, len(sol.q), FRAME_STRIDE))
    if frame_ids[-1] != len(sol.q) - 1:
        frame_ids.append(len(sol.q) - 1)                    # always include the final, fully loaded frame
    for n_frame, k in enumerate(frame_ids):
        q = sol.q[k]
        pls = [np.array([point_on_rod(q, xi, np.zeros(3)) for xi in xis_path])]      # backbone, id 0
        ids = [0]
        for j, B_list in enumerate(tendon_B_r_CPs, start=1):
            pls.append(np.array([point_on_rod(q, xi, B) for xi, B in zip(xis_path, B_list)]))
            ids.append(j)                                   # 1.2.3 straight, 4 helical
        write_frame(path.join(out_dir, f"frame_{n_frame:04d}.vtk"), pls, ids)
    print(f"wrote {len(frame_ids)} frames to {out_dir}")