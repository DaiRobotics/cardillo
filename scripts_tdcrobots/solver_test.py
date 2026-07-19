from espedal_control_test import *

" 1 for full setpoint table trajectory"
" 2 for trajectory from point E to A"
" Anything else for a falling test"
" When no Kp or Kd is given, we do a feedforward test"

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
    # dynamic_model, t_sim, r_OP_ref_fn = e2a_as(t_move, t_hold, Kp, Kd)
    # dynamic_model, t_sim, r_OP_ref_fn = e2a_as(t_move, t_hold) # Feedforward Test

    # dynamic_model, t_sim = falling_test()

    dynamic_model, t_sim, r_OP_ref_fn = traj_test(1, t_move, t_hold)

    # rod = dynamic_model.rod
    # fixed_qDOF = rod.qDOF[rod.nodalDOF[0]]
    # fixed_uDOF = rod.uDOF[rod.nodalDOF_u[0]]
    # solver = RungeKutta(dynamic_model.system, t_sim, dt, fixed_qDOF=fixed_qDOF, fixed_uDOF=fixed_uDOF)

    solver = ScipyDAE(dynamic_model.system, t_sim, dt)
    
    # solver = BackwardEuler(dynamic_model.system, t_sim, dt)
    
    sol = solver.solve()
    p2p_vis_plot(dynamic_model, sol, r_OP_ref_fn)
    plt.show()
    # falling_vis_plot(dynamic_model,sol)