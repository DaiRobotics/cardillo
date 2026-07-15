import numpy as np
from matplotlib import pyplot as plt

import scipy
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.solver import Solution, SolverOptions, SolverSummary

## Simple Runge-Kutta Test
def rk_step(f, t_k, x_k, h):
    s1 = f(x_k, t_k)
    s2 = f(x_k + h / 2 * s1, t_k + h / 2)
    s3 = f(x_k + h / 2 * s2, t_k + h / 2)
    s4 = f(x_k + h * s3, t_k + h)
    x_k1 = x_k + h / 6 * (s1 + 2 * s2 + 2 * s3 + s4)
    return x_k1

def rk_integrate(f, x0, t0, t_end, h):
    t = np.arange(t0, t_end, h)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = rk_step(f, t[i], x[i], h)
    return t, x

## Test
def cos_test():
    f = lambda x, t: np.cos(t)
    x0 = 0
    t0 = 0
    t_end = 10
    h = 1e-2
    t, x = rk_integrate(f, x0, t0, t_end, h)

    plt.plot(t, x, "b", label="Runge-Kutta 4")
    plt.plot(t, np.sin(t), "r--", label="Desired")

    plt.xlabel("Time [s]")
    plt.ylabel("X")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    cos_test()


## Runge-Kutta Solver
class RungeKutta:
    def __init__(
            self,
            system,
            t1,
            dt,
            options=SolverOptions(),
            fixed_qDOF=None,
            fixed_uDOF=None,
    ):
        self.system = system
        self.options = options
        self.dt = dt

        self.nq = system.nq
        self.nu = system.nu

        # Split fixed node 0 from the rest

        self.fixed_qDOF = np.asarray([] if fixed_qDOF is None else fixed_qDOF, dtype=int)
        self.fixed_uDOF = np.asarray([] if fixed_uDOF is None else fixed_uDOF, dtype=int)
        self.new_qDOF = np.setdiff1d(np.arange(self.nq), self.fixed_qDOF)
        self.new_uDOF = np.setdiff1d(np.arange(self.nu), self.fixed_uDOF)

        self.nq_red = len(self.new_qDOF)
        self.nu_red = len(self.new_uDOF)

        # fixed values at node 0
        self.q_fixed = system.q0[self.fixed_qDOF].copy()
        self.u_fixed = system.u0[self.fixed_uDOF].copy()

        # reduced initial state
        self.x0 = np.concatenate([system.q0[self.new_qDOF], system.u0[self.new_uDOF]])

        # integration time
        t0 = system.t0
        if t1 <= t0:
            raise ValueError("t1 must be larger than initial time t0")
        self.t1 = t1
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        M0 = self.system.M(t0, system.q0, format="csc")
        M_ff = csc_array(M0[self.new_uDOF][:, self.new_uDOF])
        self.M_ff_inv = scipy.sparse.linalg.inv(M_ff)

    def get_full_state(self, x_red):
        q = np.zeros(self.nq)
        u = np.zeros(self.nu)
        q[self.new_qDOF] = x_red[: self.nq_red]
        q[self.fixed_qDOF] = self.q_fixed
        u[self.new_uDOF] = x_red[self.nq_red :]
        u[self.fixed_uDOF] = self.u_fixed
        return q, u
    
    def dxdt(self, t, x_red):
        q, u = self.get_full_state(x_red)

        q_dot = self.system.q_dot(t, q, u)

        h = self.system.h(t, q, u)
        W_tau = self.system.W_tau(t, q, format="csr")
        la_tau = self.system.la_tau(t, q, u)
        W_c = self.system.W_c(t, q, format="Coo").tocsr(fix_size=True)
        # W_c = self.system.W_c(t, q, format="csr")
        la_c = self.system.la_c(t, q, u)

        rhs = h + W_tau @ la_tau + W_c @ la_c

        u_dot_free = self.M_ff_inv @ rhs[self.new_uDOF]

        dx = np.zeros(self.nq_red + self.nu_red)
        dx[:self.nq_red] = q_dot[self.new_qDOF]
        dx[self.nq_red:] = u_dot_free
        return dx
    
    def rk4_step(self, tk, xk):
        h = self.dt
        s1 = self.dxdt(tk, xk)
        s2 = self.dxdt(tk + h / 2, xk + h * s1 / 2 )
        s3 = self.dxdt(tk + h / 2, xk + h * s2 / 2)
        s4 = self.dxdt(tk + h, xk + h * s3)
        return xk + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)
    
    def rk_38_step(self, tk, xk):
        h = self.dt
        s1 = self.dxdt(tk, xk)
        s2 = self.dxdt(tk + (h / 3), xk + (h * s1) / 3)
        s3 = self.dxdt(tk + (2 * h) / 3, xk - ((h * s1) / 3) + h * s2)
        s4 = self.dxdt(tk + h, xk + h * s1 - h * s2 + h * s3)
        return xk + (h / 8) * (s1 + 3 * s2 + 3 * s3 + s4)

    def solve(self):
        solver_summary = SolverSummary("Runge-Kutta 4 with fixed node elimination")

        x = self.x0.copy()
        t = self.t_eval

        q0, u0 = self.get_full_state(x)
        q = [q0]
        u = [u0]

        for n in tqdm(range(len(t) - 1)):
            # Move one step
            x = self.rk4_step(t[n], x)
            # x = self.rk_38_step(t[n], x)




            # step_callback norms the quaternion in rod's step_callback
            # controller's step_callback updates tendon forces
            qn1, un1 = self.get_full_state(x)

            # --- diagnostic: which part blew up first? ---
            q_rod   = qn1[:-4]     # rod positions (mechanics)
            la_t_fb = qn1[-4:]     # controller feedback state
            # mech_bad = not (np.isfinite(q_rod).all() and np.isfinite(un1).all())
            # ctrl_bad = not np.isfinite(la_t_fb).all()
            # if mech_bad or ctrl_bad:
            #     print(f"t={t[n+1]:.5f}  mech_bad={mech_bad}  ctrl_bad={ctrl_bad}")
            #     print(f"  la_t_fb   = {la_t_fb}")
            #     print(f"  max|u|    = {np.nanmax(np.abs(un1)):.3e}")   # rod velocities
            #     print(f"  max|q_rod|= {np.nanmax(np.abs(q_rod)):.3e}")
            #     breakpoint()
            # ---------------------------------------------
            
            qn1, un1 = self.system.step_callback(t[n + 1], qn1, un1)

            # back to reduced state
            x[:self.nq_red] = qn1[self.new_qDOF]
            x[self.nq_red:] = un1[self.new_uDOF]

            q.append(qn1)
            u.append(un1)
        
        solver_summary.print()
        return Solution(
            system=self.system,
            t=t,
            q=np.array(q),
            u=np.array(u),
            solver_summary=solver_summary,
        )