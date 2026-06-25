from tdcm_li2023_main import *



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


def test_warm_start():
    print("Test warm start:")
    static_model = StaticModel()
    static_model.apply_forces([1]*4, verbose=True, force_steps=10)
    static_model.apply_forces([1]*4, verbose=True, force_steps=1)


def test_one_setpoint_convergence():
    t_sim = 10

    static_model = StaticModel()
    
    print("calc E")
    la_t_E = np.array([1.9, 1.8, 1.9, 2.1])
    sol, x, solver = static_model.apply_forces(la_t_E, force_steps=10, ret_all_steps=False, verbose=True)
    r_OP_E = sol.q[-1, -7:-4]
    print("position E:", r_OP_E)


    print("calc E to A")
    la_t_A = np.array([0.7, 3.2, 0, 0])
    la_t_EA = np.array((la_t_E, la_t_A))
    sol, x, solver = static_model.apply_forces(la_t_EA, force_steps=40, ret_all_steps=True, verbose=True)
    r_OP_ref = sol.q[:, -7:-4]
    print("position A:", r_OP_ref[-1])

    print("calc ref jacobians")
    # TODO: fix bug
    static_model2 = StaticModel()
    sol2, x, solver = static_model2.apply_forces(la_t_E, force_steps=10, ret_all_steps=False, verbose=True)
    la_t_fb0, q0, Gamma0 = solve_ref_config(static_model2, r_OP_E, tol=1e-7, lambda_t0=la_t_E, force_steps=10)
    q0[-7:-4] - r_OP_E
    
    q0 = sol.q[-1]
    la_t_fb0 = la_t_E

    # def ref traj
    ts = np.linspace(0, t_sim, len(r_OP_ref))
    def r_OP_ref_fn(t):
        return np.array([interp1d(ts, r_OP_ref[:, i], t) for i in range(3)])

    Kp = 0
    dynamic_model = DynamicModel(t_sim, Kp, Gamma0, la_t_fb0, r_OP_ref_fn, lambda t: la_t_fb0, q0)    
    sol = dynamic_model.solver.solve()



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
    r_OP_traj = np.array([r_OP_ref_fn(ti) for ti in t])

    # ---- Point to Point plots ----
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 1)

    atx = fig.add_subplot(gs[0, 0])
    atx.plot(t, q[:, -1, 0], "r", label="actual")
    atx.plot(t, r_OP_traj[:, 0], "b--", label="desired")
    atx.set_xlabel("time [s]")
    atx.set_ylabel("X [m]")
    atx.legend()
    atx.grid(True)

    aty = fig.add_subplot(gs[1, 0])
    aty.plot(t, q[:, -1, 1], "r", label="actual")
    aty.plot(t, r_OP_traj[:, 1], "b--", label="desired")
    aty.set_xlabel("time [s]")
    aty.set_ylabel("Y [m]")
    aty.legend()
    aty.grid(True)

    atz = fig.add_subplot(gs[2, 0])
    atz.plot(t, q[:, -1, 2], "r", label="actual")
    atz.plot(t, r_OP_traj[:, 2], "b--", label="desired")
    atz.set_xlabel("time [s]")
    atz.set_ylabel("Z [m]")
    atz.legend()
    atz.grid(True)

    fig.suptitle(f"Trajectory tracking (point-to-point)")
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    # test_warm_start()
    test_one_setpoint_convergence()