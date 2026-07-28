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


## Runge-Kutta Explicit Solver
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
    
class RungeKuttaAdaptive(RungeKutta):
    ## Adaptive step-size RK4 through step doubling (Richardson extrapolation for 5th order error)

    def __init__(
        self,
        system,
        t1,
        dt,
        options=SolverOptions(),
        fixed_qDOF=None,
        fixed_uDOF=None,
        rtol=1e-3,
        atol=1e-6,
        first_step=None,
        min_step=1e-10,
        max_step=np.inf,
        safety=0.9,
        min_factor=0.2,
        max_factor=5.0,
        local_extrapolation=True,
    ):
        super().__init__(
            system,
            t1,
            dt,
            options=options,
            fixed_qDOF=fixed_qDOF,
            fixed_uDOF=fixed_uDOF,
        )
        self.rtol = rtol
        self.atol = atol
        # initial trial step, falls back to the nominal dt when unset
        self.h_init = dt if first_step is None else first_step
        self.min_step = min_step
        self.max_step = max_step
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.local_extrapolation = local_extrapolation
        # RK4 local error ~ O(h^5), so the step exponent is 1 / (order + 1)
        self.error_exponent = 1.0 / 5.0

    def _rk4_step(self, tk, xk, h):
        """One classic RK4 step of arbitrary size h (mirrors self.rk4_step)."""
        s1 = self.dxdt(tk, xk)
        s2 = self.dxdt(tk + h / 2, xk + h * s1 / 2)
        s3 = self.dxdt(tk + h / 2, xk + h * s2 / 2)
        s4 = self.dxdt(tk + h, xk + h * s3)
        return xk + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4)

    def _error_norm(self, err, x_old, x_new):
        """RMS error scaled by atol + rtol * max(|x_old|, |x_new|)."""
        scale = self.atol + self.rtol * np.maximum(np.abs(x_old), np.abs(x_new))
        return np.sqrt(np.mean((err / scale) ** 2))

    def solve(self):
        solver_summary = SolverSummary("Adaptive Runge-Kutta 4 (step doubling)")

        t0 = self.t_eval[0]
        t1 = self.t1
        x = self.x0.copy()

        q0, u0 = self.get_full_state(x)
        t_list = [t0]
        q = [q0]
        u = [u0]

        t = t0
        h = min(self.h_init, self.max_step, t1 - t0)

        n_accept = 0
        n_reject = 0

        pbar = tqdm(total=100, desc="Adaptive RK4", unit="it")
        progress = 0

        while t < t1 - 1e-14:
            # never overshoot the final time
            h = min(h, t1 - t)

            # one full step vs. two half steps
            x_big = self._rk4_step(t, x, h)
            x_half = self._rk4_step(t, x, h / 2)
            x_small = self._rk4_step(t + h / 2, x_half, h / 2)

            err = x_small - x_big
            error_norm = self._error_norm(err, x, x_small)

            # accept, or force acceptance once we can no longer shrink
            if error_norm <= 1.0 or h <= self.min_step:
                t_new = t + h
                # local extrapolation lifts the accepted solution to 5th order
                if self.local_extrapolation:
                    x_new = x_small + err / 15.0
                else:
                    x_new = x_small

                qn1, un1 = self.get_full_state(x_new)
                qn1, un1 = self.system.step_callback(t_new, qn1, un1)
                x_new[: self.nq_red] = qn1[self.new_qDOF]
                x_new[self.nq_red :] = un1[self.new_uDOF]

                t = t_new
                x = x_new
                t_list.append(t)
                q.append(qn1)
                u.append(un1)
                n_accept += 1

                if error_norm == 0.0:
                    factor = self.max_factor
                else:
                    factor = self.safety * error_norm ** (-self.error_exponent)
            else:
                # reject and shrink
                n_reject += 1
                factor = self.safety * error_norm ** (-self.error_exponent)

            # clamp the growth/shrink factor and the resulting step size
            factor = max(self.min_factor, min(self.max_factor, factor))
            h = float(np.clip(h * factor, self.min_step, self.max_step))

            new_progress = int(100 * (t - t0) / (t1 - t0))
            pbar.update(new_progress - progress)
            progress = new_progress

        pbar.close()

        print(
            f"Adaptive RK4: {n_accept} accepted, {n_reject} rejected steps "
            f"({len(t_list)} time points)"
        )
        solver_summary.print()
        return Solution(
            system=self.system,
            t=np.array(t_list),
            q=np.array(q),
            u=np.array(u),
            solver_summary=solver_summary,
        )


class RungeKutta45(RungeKuttaAdaptive):
    # Adaptive Dormand-Prince RK45 (with errors weighted).

    C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])

    A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]

    # 5th-order solution weights
    B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])

    # error weights = 5th-order weights minus embedded 4th-order weights
    B4 = np.array(
        [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    E = B - B4

    def rk45_step(self, tk, xk, h):
        s = len(self.C)
        k = [None] * s
        k[0] = self.dxdt(tk, xk)
        for i in range(1, s):
            xi = xk + h * sum(self.A[i][j] * k[j] for j in range(i))
            k[i] = self.dxdt(tk + self.C[i] * h, xi)

        x5 = xk + h * sum(self.B[i] * k[i] for i in range(s))
        err = h * sum(self.E[i] * k[i] for i in range(s))
        return x5, err

    def solve(self):
        solver_summary = SolverSummary("Adaptive Dormand-Prince RK45")

        t0 = self.t_eval[0]
        t1 = self.t1
        x = self.x0.copy()

        q0, u0 = self.get_full_state(x)
        t_list = [t0]
        q = [q0]
        u = [u0]

        t = t0
        h = min(self.h_init, self.max_step, t1 - t0)

        n_accept = 0
        n_reject = 0

        pbar = tqdm(total=100, desc="Runge-Kutta 45", unit="it")
        progress = 0

        while t < t1 - 1e-14:
            # never overshoot the final time
            h = min(h, t1 - t)

            # embedded 5th/4th-order pair -> solution and error estimate
            x5, err = self.rk45_step(t, x, h)
            error_norm = self._error_norm(err, x, x5)

            # accept, or force acceptance once we can no longer shrink
            if error_norm <= 1.0 or h <= self.min_step:
                t_new = t + h
                x_new = x5

                qn1, un1 = self.get_full_state(x_new)
                qn1, un1 = self.system.step_callback(t_new, qn1, un1)
                x_new[: self.nq_red] = qn1[self.new_qDOF]
                x_new[self.nq_red :] = un1[self.new_uDOF]

                t = t_new
                x = x_new
                t_list.append(t)
                q.append(qn1)
                u.append(un1)
                n_accept += 1

                if error_norm == 0.0:
                    factor = self.max_factor
                else:
                    factor = self.safety * error_norm ** (-self.error_exponent)
            else:
                # reject and shrink
                n_reject += 1
                factor = self.safety * error_norm ** (-self.error_exponent)

            # clamp the growth/shrink factor and the resulting step size
            factor = max(self.min_factor, min(self.max_factor, factor))
            h = float(np.clip(h * factor, self.min_step, self.max_step))

            new_progress = int(100 * (t - t0) / (t1 - t0))
            pbar.update(new_progress - progress)
            progress = new_progress

        pbar.close()

        print(
            f"Runge-Kutta 45: {n_accept} accepted, {n_reject} rejected steps "
            f"({len(t_list)} time points)"
        )
        solver_summary.print()
        return Solution(
            system=self.system,
            t=np.array(t_list),
            q=np.array(q),
            u=np.array(u),
            solver_summary=solver_summary,
        )


class ProbeRK(RungeKutta):

    def __init__(self, system, ctrl, t1, dt, **kwargs):
        super().__init__(system, t1, dt, **kwargs)
        self.ctrl = ctrl
        self.probe_t = []
        self.la_t_mismatch = []
        self.la_t_applied = []

    def dxdt(self, t, x_red):
        q, u = self.get_full_state(x_red)
        ctrl = self.ctrl
        la_fresh = ctrl.control_law(t, q[ctrl.qDOF], u[ctrl.uDOF])
        la_applied = np.array([td.la(t) for td in ctrl.tendons])
        self.probe_t.append(t)
        self.la_t_mismatch.append(np.linalg.norm(la_fresh - la_applied))
        self.la_t_applied.append(np.linalg.norm(la_applied))
        return super().dxdt(t, x_red)