from tdcm_li2023_main import solve_config, solve_ref_config, paper_to_cardillo, DynamicModel, StaticModel, StaticSolver, TendonForceControl, CommonModel
import numpy as np
from matplotlib import pyplot as plt

# ---- Reference Trajectories ----
la_t0 = np.zeros(4)
la_t_ref_table = []
q0_table = []
Gamma0_table = []

SETPOINT_TABLE = {
    "A": np.array([15.438e-2, 4.335e-2, 3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2, 9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2, 0.000e-2, -9.842e-2]),
    "E2": np.array([13.951e-2, 0.000e-2, -9.842e-2])*1.2,
}
SETPOINT_TABLE = {k: paper_to_cardillo(u) for k, u in SETPOINT_TABLE.items()}

t_end = 50
sequence = ["A", "B", "C", "D", "E"]
hold_t = t_end / (len(sequence))

def r_OP_ref_fn(t):
    k = min(int(t / hold_t), len(sequence) - 1)
    # return SETPOINT_TABLE["E2"]
    return SETPOINT_TABLE["A"]
# la_tA_table = []
# la_tB_table = []
# la_tC_table = []
# la_tD_table = []
# la_tE_table = []
for name in sequence:
    r_OP_ref = SETPOINT_TABLE[name]
    # la_ref, _, _ = solve_ref_config(r_OP_ref, la_t0, tol = 3e-4, force_steps=20)
    la_t_ref, q0, Gamma0, lambda_t_table = solve_ref_config(r_OP_ref, la_t0, force_steps=20)
    if name == "A":
        la_tA_table = lambda_t_table
    elif name == "B":
        la_tB_table = lambda_t_table
    elif name == "C":
        la_tC_table = lambda_t_table
    elif name == "D":
        la_tD_table = lambda_t_table
    else:
        la_tE_table = lambda_t_table
    la_t_ref_table.append(la_t_ref)
    q0_table.append(q0)
    Gamma0_table.append(Gamma0)
    print(f"{name}")
# def la_t_ref_fn(t):
#     if t == 0.0:
#         return la_t_ref_table[-1]
#     k = min(int(t / hold_t), len(sequence) - 1)
#     return la_t_ref_table[k]

def la_t_ref_fn(t):
    return la_t_ref_table

# # CSV 
import csv
from itertools import zip_longest

# la_t tables per point
# csv_path = r"C:\Users\tongd\OneDrive\Documents\Uni\HIWI_INM\cardillo\BA_Repo\scripts_tdcrobots\p2p_la_t.csv"
# with open(csv_path, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["la_tA", "la_tB", "la_tC", "la_tD", "la_tE"])
#     for row in zip_longest(la_tA_table, la_tB_table, la_tC_table, la_tD_table, la_tE_table, fillvalue=""):
#         writer.writerow(row)

# q0 and Gamma0 and each point
csv_path = r"C:\Users\tongd\OneDrive\Documents\Uni\HIWI_INM\cardillo\BA_Repo\scripts_tdcrobots\p2p_q0_gamma0.csv"
columns = [np.asarray(q0).ravel() for q0 in q0_table] \
        + [np.asarray(G).ravel()  for G  in Gamma0_table]
header  = [f"q0_{n}" for n in sequence] + [f"Gamma0_{n}" for n in sequence]
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in zip_longest(*columns, fillvalue=""):
        writer.writerow(row)
# exit()

## Run Simulation

# setpoint = "E"
# r_OP_ref = SETPOINT_TABLE[setpoint]
# # # r_OP_ref = r_OP_ref_fn(0.0)  
# la_t0, q0, Gamma0 = solve_ref_config(r_OP_ref, tol=3e-4, lambda_t0=la_t0, force_steps=10)
q0 = q0_table[-1]
Gamma0 = Gamma0_table[-1]

# Kp = 0.5
# Kp = 0.2
Kp = 0.1
t_sim = t_end
dynamic_model = DynamicModel(t_sim, Kp, Gamma0, la_t0, r_OP_ref_fn, la_t_ref_fn, q0)
exit()
# ---- visualization ----
rod = dynamic_model.rod
tendons = dynamic_model.tendons
system = dynamic_model.system

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
sol = dynamic_model.solver.solve()
plotter.render_solution(sol, True, play_speed_up=1)

# ---- Point to Point plots ----
t = sol.t
q = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
r_OP_traj = np.array([r_OP_ref_fn(ti) for ti in t])
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 1)

atx = fig.add_subplot(gs[0, 0])
atx.plot(t, q[:, -1, 0] * 100, "r", label="actual")
atx.plot(t, r_OP_traj[:, 0] * 100, "b--", label="desired")
atx.set_xlabel("Time [s]")
atx.set_xlim(0, 50)
atx.set_xticks(np.arrange(0, 50.1, 5))
atx.set_ylabel("X [cm]")
atx.set_ylim(-5.2, 10)
atx.set_yticks(np.array([-5, 0, 5, 10]))
atx.legend()
atx.grid(True)

aty = fig.add_subplot(gs[1, 0])
aty.plot(t, q[:, -1, 1] * 100, "r", label="actual")
aty.plot(t, r_OP_traj[:, 1] * 100, "b--", label="desired")
aty.set_xlabel("Time [s]")
aty.set_xlim(0, 50)
aty.set_xticks(np.arrange(0, 50.1, 5))
aty.set_ylabel("Y [cm]")
aty.set_ylim(-10, 5)
aty.set_yticks(np.array([-10, -5, 0, 5]))
aty.legend()
aty.grid(True)

atz = fig.add_subplot(gs[2, 0])
atz.plot(t, q[:, -1, 2] * 100, "r", label="actual")
atz.plot(t, r_OP_traj[:, 2] * 100, "b--", label="desired")
atz.set_xlabel("Time [s]")
atz.set_xlim(0, 50)
atz.set_xticks(np.arrange(0, 50.1, 5))
atz.set_ylabel("Z [cm]")
atz.set_ylim(10, 18)
atz.set_yticks(np.arrange(10, 18.1, 2))
atz.legend()
atz.grid(True)

fig.suptitle(f"Trajectory tracking (point-to-point)")
fig.tight_layout()



plt.show()