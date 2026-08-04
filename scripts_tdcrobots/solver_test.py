from espedal_control_test import *

" 1 for full setpoint table trajectory"
" 2 for trajectory from point E to A"
" Anything else for a falling test"
" When no Kp or Kd is given, we do a feedforward test"


def compute_la_ts(dynamic_model, sol):
    """Total tendon forces of the static (TendonForceControl) controller over time.

    la_t(t) = feedback (controller DOFs, last n_tendons columns of sol.q)
              + feedforward (static inverse-statics reference la_t_ref(t))
    """
    controller = dynamic_model.controller
    ntendons = dynamic_model.n_tendons
    la_t_fb = sol.q[:, -ntendons:]                                   # feedback part
    la_t_ff = np.array([controller.la_t_ref(t) for t in sol.t])      # feedforward part
    return la_t_fb + la_t_ff, la_t_ff


def la_t_plot(dynamic_model, la_ts, sol, la_t_ff=None):
    import matplotlib.pyplot as plt

    ts = sol.t
    colors = ["r", "b", "g", "m", "c", "y"]
    fig, ax = plt.subplots(num="TendonForces", figsize=(8, 4))
    for k in range(dynamic_model.n_tendons):
        c = colors[k % len(colors)]
        ax.plot(ts, la_ts[:, k], c, label=f"tendon {k+1}")
        if la_t_ff is not None:
            ax.plot(ts, la_t_ff[:, k], c + "--", alpha=0.5)  # feedforward reference
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tendon Force [N]")
    ax.set_title("Tendon Forces (static feedforward + PD)")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    # Parameters
    Kp = 0.5
    Kd = 0.1
    t_move = 5
    t_hold = 0
    # dt = 1e-2
    dt = 1e-4

    # ---- Full Setpoint Table Trajectory ----
    # dynamic_model, t_sim, r_OP_ref_fn = setpoint_trajectory_as(t_move, t_hold, Kp, Kd)
    # dynamic_model, t_sim, r_OP_ref_fn = setpoint_trajectory_as(t_move, t_hold) # Feedforward test

    # ---- Point E to A ----
    dynamic_model, t_sim, r_OP_ref_fn = e2a_as(t_move, t_hold, Kp, Kd, save_ref="e2a_ref_points.csv")
    # dynamic_model, t_sim, r_OP_ref_fn = e2a_as(t_move, t_hold) # Feedforward Test

    # dynamic_model, t_sim = falling_test()

    # dynamic_model, t_sim, r_OP_ref_fn = traj_test(1, t_move, t_hold)

    # rod = dynamic_model.rod
    # fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
    # fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
    # solver = RungeKutta(dynamic_model.system, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF)

    solver = ScipyDAE(dynamic_model.system, t_sim, dt)
    
    # solver = BackwardEuler(dynamic_model.system, t_sim, dt)
    
    sol = solver.solve()
    p2p_vis_plot(dynamic_model, sol, r_OP_ref_fn)
    plt.show()

    la_ts, la_t_ff = compute_la_ts(dynamic_model, sol)
    la_t_plot(dynamic_model, la_ts, sol, la_t_ff=la_t_ff)
    # falling_vis_plot(dynamic_model,sol)