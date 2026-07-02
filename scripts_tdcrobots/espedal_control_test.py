from tdcm_li2023 import *
import pandas as pd
import ast
from pathlib import Path

SETPOINT_TABLE = {
    "A": np.array([15.438e-2, 4.335e-2, 3.399e-2]),
    "B": np.array([15.272e-2, -5.114e-2, -0.463e-2]),
    "C": np.array([10.888e-2, 9.106e-2, -5.492e-2]),
    "D": np.array([14.615e-2, -4.486e-2, -6.375e-2]),
    "E": np.array([13.951e-2, 0.000e-2, -9.842e-2]),
}


def paper_to_cardillo(u):
    X, Y, Z = u
    return np.array([Y, Z, X])


SETPOINT_TABLE = {k: paper_to_cardillo(u) for k, u in SETPOINT_TABLE.items()}

la_t_A = np.array([0.73792114, 3.17753766, 0.0, 0.0])
la_t_B = np.array([2.28280023, 3.48103006, 3.38276666, 1.28709287])
la_t_C = np.array([3.02999893, 2.79040527, 0.469079, 1.96589514])
la_t_D = np.array([2.01397515, 3.03451165, 3.10051456, 2.36642629])
la_t_E = np.array([1.9244945,  1.78542759, 1.9244945,  2.05885467])


def make_zy_circle(t, center = np.array([9.7, 0, -1.8]) * 1e-2, radius=3*1e-2, t_period=40.0):
    # in X-Z plane
    x_c, y_c, z_c = center # X = 10.2*1e-2 , Y = 0.0, Z = -4.6*1e-2 in Paper
    phi = 2 * np.pi * t / t_period - np.pi / 2

    x = x_c - radius * np.sin(phi)
    y = y_c
    z = z_c + radius * np.cos(phi)
    return paper_to_cardillo(np.array([x, y, z]))

def create_zy_circle_csv(N=50):
    t_sim = 40
    N = 50
    r_OP_refs = discrete_path(make_zy_circle, t_sim, N)
    
    la_t0 = np.array([0.5, 0.5, 0.5, 0.5])
    la_ts, qs, Gammas = inverse_statics(r_OP_refs, la_t0,force_steps=3)

    csv_file = Path(__file__).parent / "inverse_statics_results.csv"
    df = pd.DataFrame(
        {i: [la_ts[i].tolist(), qs[i].tolist(), Gammas[i].tolist()] for i in range(len(la_ts))},
        index=["la_t", "q", "Gamma"]
    )
    df.to_csv(csv_file)
    

def zy_circle_trajectory():

    ## Load csv
    csv_file = Path(__file__).parent / "inverse_statics_results.csv"

    df = pd.read_csv(csv_file, index_col=0)
    df.columns = df.columns.astype(int)

    la_ts = np.array([np.array(ast.literal_eval(df.loc["la_t", i])) for i in df.columns])
    qs = np.array([np.array(ast.literal_eval(df.loc["q", i])) for i in df.columns])
    Gammas = np.array([np.array(ast.literal_eval(df.loc["Gamma", i])) for i in df.columns])

    t_sim = 40
    q0 = qs[0]
    la_t_fb0 = la_ts[0]
    Gamma0 = Gammas[0]

    ts_ff = np.linspace(0, t_sim, len(la_ts))

    def la_t_ref_fn(t):
        return np.array([interp1d(ts_ff, la_ts[:, i], t) for i in range(4)])

    # print("dynamic control:")
    Kp = 0
    Kd = 0.05
    dynamic_model = DynamicModel(
        t_sim, Kp, Kd, Gamma0, la_t_fb0, 
        make_zy_circle, la_t_ref_fn, q0, damping_ratio=0.3
    )
    sol = dynamic_model.solver.solve()
    print(Kp, Kd)

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
    r_OF = np.array([0, -0.02, 0.10], float)
    r_OC = r_OF + np.array([0.45, 0, 0], float)
    e_x_cam = np.array([0, 0, 1], float)
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
    r_OP_ref = np.array([make_zy_circle(ti) for ti in t])
    r_OP = q[:,-1, 0:3]
    e = r_OP_ref - r_OP
    e_n = np.array([np.linalg.norm(e[i]) for i in range(len(e))])

    # ---- Point to Point plots ----
    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(2, 1)

    aty = fig.add_subplot(gs[0, 0])
    aty.plot(t, q[:, -1, 1], "r", label="actual")
    aty.plot(t, r_OP_ref[:, 1], "b--", label="desired")
    aty.set_xlabel("Time [s]")
    aty.set_ylabel("Y [m]")
    aty.legend()
    aty.grid(True)

    atz = fig.add_subplot(gs[1, 0])
    atz.plot(t, q[:, -1, 2], "r", label="actual")
    atz.plot(t, r_OP_ref[:, 2], "b--", label="desired")
    atz.set_xlabel("Time [s]")
    atz.set_ylabel("Z [m]")
    atz.legend()
    atz.grid(True)


    fig.suptitle(f"Tip Trajectory Tracking (ZY Plane Circle)")
    fig.tight_layout()


    fig2, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, lbl in enumerate(("e_y", "e_z")):
        ax[i].plot(t, e[:, i], "r")
        ax[i].set_ylabel(rf"${lbl}$ [m]")
        ax[i].grid(True)

    ax[2].plot(t, e_n, "k")              
    ax[2].set_ylabel(r"$e_{norm}$ [m]")
    ax[2].grid(True)

    ax[-1].set_xlabel("Time [s]")          
    fig2.suptitle("Tracking Error per Direction (ZY Plane Circle)")
    fig2.tight_layout()

    fig3, zy = plt.subplots(figsize=(6, 6))
    zy.plot(r_OP_ref[:, 2], r_OP_ref[:, 1], "b--", label="desired")
    zy.plot(r_OP[:, 2], r_OP[:, 1], "r", label="actual")
    zy.set_xlabel("Z [m]")
    zy.set_ylabel("Y [m]")
    zy.set_title("Circle Reference Trajectory (ZY Plane)")
    zy.legend()
    zy.grid(True)
    fig3.tight_layout()

    plt.show()    

def setpoint_table_trajectory():
    t_sim = 50

    la_t_A = np.array([0.73792114, 3.17753766, 0.0, 0.0])
    la_t_B = np.array([2.28280023, 3.48103006, 3.38276666, 1.28709287])
    la_t_C = np.array([3.02999893, 2.79040527, 0.469079, 1.96589514])
    la_t_D = np.array([2.01397515, 3.03451165, 3.10051456, 2.36642629])
    la_t_E = np.array([1.9244945,  1.78542759, 1.9244945,  2.05885467])

    la_t_ref = np.concatenate((la_t_A[None, :], la_t_B[None, :], la_t_C[None, :], la_t_D[None, :], la_t_E[None, :]))
    
    static_model = StaticModel()

    print("calc E")
    Gamma0, r_OP_E, q0 = eval_gamma(static_model, la_t_E)
    # print("position E:", r_OP_E)

    print("calc E to A")
    # TODO interpolate the force manually, and set ret_all_steps=False
    sol, x, solver = static_model.apply_forces(
        la_t_ref, force_steps=50, ret_all_steps=True, verbose=True
    )
    r_OP_ref = sol.q[:, -7:-4]
    r_OP_ref = np.concatenate((r_OP_E[None, :], r_OP_ref))
    # print("position A:", r_OP_ref[-1])

    # def ref traj
    ts = np.linspace(0, t_sim, len(r_OP_ref))

    def r_OP_ref_fn(t):
        return np.array([interp1d(ts, r_OP_ref[:, i], t) for i in range(3)])

    la_t_ref = np.concatenate((la_t_E[None, :], la_t_ref))
    ts2 = np.linspace(0, t_sim, len(la_t_ref))

    def la_t_ref_fn(t):
        return np.array([interp1d(ts2, la_t_ref[:, i], t) for i in range(4)])

    la_t_fb0 = la_t_E

    print("dynamic control:")
    Kp = 1.0
    Kd = 0.0
    dynamic_model = DynamicModel(
        t_sim, Kp, Kd, Gamma0, la_t_fb0, r_OP_ref_fn, la_t_ref_fn, q0, damping_ratio=0.0
    )
    sol = dynamic_model.solver.solve()
    print(Kp, Kd)


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
    plotter.render_solution(sol, True, play_speed_up=1)

    from matplotlib import pyplot as plt

    t = sol.t
    q = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))
    r_OP_ref = np.array([r_OP_ref_fn(ti) for ti in t])
    r_OP = q[:,-1, 0:3]
    e = r_OP_ref - r_OP
    e_n = np.array([np.linalg.norm(e[i]) for i in range(len(e))])

   # ---- Point to Point plots ----
    fig = plt.figure(figsize=(8,6))
    gs = fig.add_gridspec(3, 1)

    atx = fig.add_subplot(gs[0, 0])
    atx.plot(t, q[:, -1, 0], "r", label="actual")
    atx.plot(t, r_OP_ref[:, 0], "b--", label="desired")
    atx.set_xlabel("Time [s]")
    atx.set_ylabel("X [m]")
    atx.legend()
    atx.grid(True)

    aty = fig.add_subplot(gs[1, 0])
    aty.plot(t, q[:, -1, 1], "r", label="actual")
    aty.plot(t, r_OP_ref[:, 1], "b--", label="desired")
    aty.set_xlabel("Time [s]")
    aty.set_ylabel("Y [m]")
    aty.legend()
    aty.grid(True)

    atz = fig.add_subplot(gs[2, 0])
    atz.plot(t, q[:, -1, 2], "r", label="actual")
    atz.plot(t, r_OP_ref[:, 2], "b--", label="desired")
    atz.set_xlabel("Time [s]")
    atz.set_ylabel("Z [m]")
    atz.legend()
    atz.grid(True)


    fig.suptitle(f"Tip Trajectory Tracking (E to A)")
    fig.tight_layout()


    fig2, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    for i, lbl in enumerate(("e_x", "e_y", "e_z")):
        ax[i].plot(t, e[:, i], "r")
        ax[i].set_ylabel(rf"${lbl}$ [m]")
        ax[i].grid(True)

    ax[3].plot(t, e_n, "k")              
    ax[3].set_ylabel(r"$e_{norm}$ [m]")
    ax[3].grid(True)

    ax[-1].set_xlabel("Time [s]")          
    fig2.suptitle("Tracking Error per Direction (E to A)")
    fig2.tight_layout()

    plt.show()



if __name__ == "__main__":
    # create_zy_circle_csv(N=50)
    # zy_circle_trajectory()
    setpoint_table_trajectory()