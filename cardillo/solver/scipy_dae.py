import numpy as np
from scipy.sparse import eye_array
from scipy_dae.integrate import solve_dae
from tqdm import tqdm

from cardillo.solver import Solution, SolverSummary
from cardillo.utility.coo_matrix import CooMatrix


# TODO:
# - Add Jacobian of GGl term if convergence problems occur
class ScipyDAE:
    """Wrapper around Radau IIA and BDF methods implementted in `scipy_dae`. 
    A stabilized index 1 formulation is used as proposed by Anantharaman and Hiller.

    References:
    -----------
    scipy_dae: https://github.com/JonasBreuling/scipy_dae \\
    Anantharaman and Hiller.: https://doi.org/10.1002/nme.1620320803
    """

    def __init__(
        self,
        system,
        t1,
        dt,
        method="Radau",
        rtol=1.0e-3,
        atol=1.0e-6,
        **kwargs,
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.kwargs = kwargs

        self.nq = system.nq
        self.nu = system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_c = self.system.nla_c
        self.ny = self.nq + self.nu + 2 * self.nla_g + self.nla_gamma + self.nla_c
        self.split = np.cumsum(
            np.array(
                [
                    self.nq,
                    self.nu,
                    self.nla_g,
                    self.nla_g,
                    self.nla_gamma,
                    self.nla_c,
                ],
                dtype=int,
            )
        )[:-1]
        self.y0 = np.concatenate(
            (
                system.q0,
                system.u0,
                0 * system.la_g0,
                0 * system.la_g0,
                0 * system.la_gamma0,
                0 * system.la_c0,
            )
        )
        self.y_dot0 = np.concatenate(
            (
                system.q_dot0,
                system.u_dot0,
                0 * system.la_g0,  # GGL multiplier
                system.la_g0,
                system.la_gamma0,
                system.la_c0,
            )
        )

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t_eval = np.arange(t0, self.t1 + self.dt, self.dt)

        self.frac = (t1 - t0) / 100
        self.pbar = tqdm(total=100, unit="pct")
        self.pbar_i = 0

        # data allocation
        self.F = CooMatrix((1, self.ny), manual_sync=True)
        self.h = CooMatrix((1, system.nu), manual_sync=True)
        self.c = CooMatrix((1, system.nla_c), manual_sync=True)
        self.g_q1 = CooMatrix((system.nla_g, system.nq), manual_sync=True)
        self.g_q1_T = CooMatrix((system.nq, system.nla_g), manual_sync=True)
        self._W_tau = CooMatrix((system.nu, system.nla_tau), manual_sync=True)
        self.W_g1 = CooMatrix((system.nu, system.nla_g), manual_sync=True)
        self.W_gamma1 = CooMatrix((system.nu, system.nla_gamma), manual_sync=True)
        self.W_c1 = CooMatrix((system.nu, system.nla_c), manual_sync=True)
        self.q_dot = CooMatrix((1, system.nq), manual_sync=True)
        self.q_dot_q = CooMatrix((system.nq, system.nq), manual_sync=True)
        self.q_dot_u = CooMatrix((system.nq, system.nu), manual_sync=True)

        self.Mu_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.h_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.h_u = CooMatrix((system.nu, system.nu), manual_sync=True)
        self.Wla_tau_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.Wla_tau_u = CooMatrix((system.nu, system.nu), manual_sync=True)
        self.Wla_g_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.Wla_gamma_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.Wla_c_q = CooMatrix((system.nu, system.nq), manual_sync=True)
        self.g_dot_q = CooMatrix((system.nla_g, system.nq), manual_sync=True)
        self.g_dot_u = CooMatrix((system.nla_g, system.nu), manual_sync=True)
        self.gamma_q = CooMatrix((system.nla_gamma, system.nq), manual_sync=True)
        self.gamma_u = CooMatrix((system.nla_gamma, system.nu), manual_sync=True)
        self.c_q = CooMatrix((system.nla_c, system.nq), manual_sync=True)
        self.c_u = CooMatrix((system.nla_c, system.nu), manual_sync=True)
        self.M1 = CooMatrix((system.nu, system.nu), manual_sync=True)
        self.M2 = CooMatrix((system.nu, system.nu), manual_sync=True)
        self.g_q2 = CooMatrix((system.nla_g, system.nq), manual_sync=True)
        self.W_g2 = CooMatrix((system.nu, system.nla_g), manual_sync=True)
        self.W_gamma2 = CooMatrix((system.nu, system.nla_gamma), manual_sync=True)
        self.W_c2 = CooMatrix((system.nu, system.nla_c), manual_sync=True)

        self.Jy = CooMatrix((self.ny, self.ny), manual_sync=True)
        self.Jyp = CooMatrix((self.ny, self.ny), manual_sync=True)
        eye_q = eye_array(self.nq)
        c_la_c = self.system.c_la_c()
        self.Jyp["eye_q", : self.split[0], : self.split[0]] = eye_q
        self.Jyp["c_la_c", self.split[4] :, self.split[4] :] = c_la_c

    def event(self, t, y, yp):
        q, u = np.array_split(y, self.split)[:2]
        q, u = self.system.step_callback(t, q, u)
        return 1

    def fun(self, t, y, yp):
        t = float(t)
        # update progress bar
        pbar_i = int(np.floor((t + self.frac / 2) / self.frac))
        self.pbar.update(pbar_i - self.pbar_i)
        self.pbar.set_description(f"{self.method}: t {t:0.2e}s < {self.t1:0.2e}s")
        self.pbar_i = pbar_i

        # unpack vectors
        s1, s2, s3, s4, s5 = self.split
        q, u = y[:s1], y[s1:s2]
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = (
            yp[:s1],
            yp[s1:s2],
            yp[s2:s3],
            yp[s3:s4],
            yp[s4:s5],
            yp[s5:],
        )

        # residual
        F = self.F
        sys = self.system
        h = self.h = self.system.h(t, q, u, format="Coo", coo=self.h)
        c = self.c = self.system.c(t, q, u, la_c, format="Coo", coo=self.c)
        q_dot2 = self.q_dot = self.system.q_dot(t, q, u, format="Coo", coo=self.q_dot)
        if self.nla_g:
            g_q = self.g_q1 = self.system.g_q(t, q, format="Coo", coo=self.g_q1)
            g_q_T = self.g_q1_T = g_q.transpose(copy=False, coo=self.g_q1_T)
        if sys.nla_tau:
            W_tau = self._W_tau = self.system.W_tau(t, q, format="Coo", coo=self._W_tau)
        if sys.nla_g:
            W_g = self.W_g1 = self.system.W_g(t, q, format="Coo", coo=self.W_g1)
        if sys.nla_gamma:
            W_gamma = self.W_gamma1 = self.system.W_gamma(
                t, q, format="Coo", coo=self.W_gamma1
            )
        if sys.nla_c:
            W_c = self.W_c1 = self.system.W_c(t, q, format="Coo", coo=self.W_c1)
        ####################
        # kinematic equation
        ####################
        F["q_dot", 0, :s1] = q_dot
        F["q_dot2", 0, :s1, True] = q_dot2
        if self.nla_g:
            g_q.manual_sync()
            F["g_q_T_mu_g", 0, :s1, True] = g_q_T.tocsr(fix_size=True) @ mu_g

        ####################
        # equations of motion
        ####################
        M = self.M2 = self.system.M(t, q, format="Coo", coo=self.M2)
        F["Mu", 0, s1:s2] = M.tocsr(fix_size=True) @ u_dot
        F["h", 0, s1:s2, True] = h
        if sys.nla_tau:
            W_tau.manual_sync()
            F["Wla_tau", 0, s1:s2, True] = W_tau.tocsr(
                fix_size=True
            ) @ self.system.la_tau(t, q, u)
        if sys.nla_g:
            W_g.manual_sync()
            F["Wla_g", 0, s1:s2, True] = W_g.tocsr(fix_size=True) @ la_g
        if sys.nla_gamma:
            F["Wla_gamma", 0, s1:s2, True] = W_gamma.tocsr(fix_size=True) @ la_gamma
        if sys.nla_c:
            W_c.manual_sync()
            F["Wla_c", 0, s1:s2, True] = W_c.tocsr(fix_size=True) @ la_c

        #######################
        # bilateral constraints
        #######################
        if sys.nla_g:
            F["g", 0, s2:s3] = self.system.g(t, q)
            F["g_dot", 0, s3:s4] = self.system.g_dot(t, q, u)

        if sys.nla_gamma:
            F["gamma", 0, s4:s5] = self.system.gamma(t, q, u)

        ############
        # compliance
        ############
        if sys.nla_c:
            F["c", 0, s5:] = c
        F.manual_sync()
        return F.toarray(fix_size=True).ravel()

    def jac(self, t, y, yp):
        t = float(t)
        sys = self.system

        # unpack vectors
        s0, s1, s2, s3, s4 = self.split
        q, u = y[:s0], y[s0:s1]
        u_dot = yp[s0:s1]
        la_g = yp[s2:s3]
        la_gamma = yp[s3:s4]
        la_c = yp[s4:]

        # evaluate used quantities
        q_dot_q = self.q_dot_q = self.system.q_dot_q(
            t, q, u, format="Coo", coo=self.q_dot_q
        )
        q_dot_u = self.q_dot_u = self.system.q_dot_u(
            t, q, format="Coo", coo=self.q_dot_u
        )
        if sys.nla_g:
            Wla_g_q = self.Wla_g_q = self.system.Wla_g_q(
                t, q, la_g, format="Coo", coo=self.Wla_g_q
            )
            g_q = self.g_q2 = self.system.g_q(t, q, format="Coo", coo=self.g_q2)
            g_dot_q = self.g_dot_q = self.system.g_dot_q(
                t, q, u, format="Coo", coo=self.g_dot_q
            )
            g_dot_u = self.g_dot_u = self.system.g_dot_u(
                t, q, format="Coo", coo=self.g_dot_u
            )

        if sys.nla_c:
            Wla_c_q = self.Wla_c_q = self.system.Wla_c_q(
                t, q, la_c, format="Coo", coo=self.Wla_c_q
            )
            c_q = self.c_q = self.system.c_q(t, q, u, la_c, format="Coo", coo=self.c_q)
            c_u = self.c_u = self.system.c_u(t, q, u, la_c, format="Coo", coo=self.c_u)

        if sys.nla_g:
            W_g = self.W_g2 = self.system.W_g(t, q, format="Coo", coo=self.W_g2)

        if sys.nla_gamma:
            W_gamma = self.W_gamma2 = self.system.W_gamma(
                t, q, format="Coo", coo=self.W_gamma2
            )

        if sys.nla_c:
            W_c = self.W_c2 = self.system.W_c(t, q, format="Coo", coo=self.W_c2)

        if sys.nla_tau:
            Wla_tau_q = self.Wla_tau_q = self.system.Wla_tau_q(
                t, q, u, format="Coo", coo=self.Wla_tau_q
            )
            Wla_tau_u = self.Wla_tau_u = self.system.Wla_tau_u(
                t, q, u, format="Coo", coo=self.Wla_tau_u
            )

        if sys.nla_gamma:
            Wla_gamma_q = self.Wla_gamma_q = self.system.Wla_gamma_q(
                t, q, la_gamma, format="Coo", coo=self.Wla_gamma_q
            )
            gamma_q = self.gamma_q = self.system.gamma_q(
                t, q, u, format="Coo", coo=self.gamma_q
            )
            gamma_u = self.gamma_u = self.system.gamma_u(
                t, q, format="Coo", coo=self.gamma_u
            )

        # first Jacobian w.r.t. y
        Jy = self.Jy

        Mu_q = self.Mu_q = self.system.Mu_q(t, q, u_dot, format="Coo", coo=self.Mu_q)
        h_q = self.h_q = self.system.h_q(t, q, u, format="Coo", coo=self.h_q)
        h_u = self.h_u = self.system.h_u(t, q, u, format="Coo", coo=self.h_u)

        Jy["q_dot_q", :s0, :s0, True] = q_dot_q
        Jy["q_dot_u", :s0, s0:s1, True] = q_dot_u

        # note: Here we ignore the derivative d((dg/dq)^T mu) / dq since
        # `solve_dae` already performs an inexact Newton method.
        # Jy[:s1, s2:s3] = g_q_T_mu_q

        Jy["Mu_q", s0:s1, :s0] = Mu_q
        Jy["h_q", s0:s1, :s0, True] = h_q
        Jy["h_u", s0:s1, s0:s1, True] = h_u

        if sys.nla_tau:
            Jy["Wla_tau_q", s0:s1, :s0, True] = Wla_tau_q
            Jy["Wla_tau_u", s0:s1, s0:s1, True] = Wla_tau_u

        if sys.nla_gamma:
            Jy["Wla_gamma_q", s0:s1, :s0, True] = Wla_gamma_q
            Jy["gamma_q", s3:s4, :s0] = gamma_q
            Jy["gamma_u", s3:s4, s0:s1] = gamma_u

        if sys.nla_g:
            Jy["Wla_g_q", s0:s1, :s0, True] = Wla_g_q
            # TODO: remove manual_sync, currently not possible due to g_q.T
            g_q.manual_sync()
            Jy["g_q", s1:s2, :s0] = g_q
            Jy["g_dot_q", s2:s3, :s0] = g_dot_q
            Jy["g_dot_u", s2:s3, s0:s1] = g_dot_u

        if sys.nla_c:
            Jy["Wla_c_q", s0:s1, :s0, True] = Wla_c_q
            Jy["c_q", s4:, :s0] = c_q
            Jy["c_u", s4:, s0:s1] = c_u

        # second Jacobian w.r.t. yp
        Jyp = self.Jyp

        M = self.M1 = self.system.M(t, q, format="Coo", coo=self.M1)

        Jyp["M", s0:s1, s0:s1] = M
        if sys.nla_g:
            Jyp["g_q_T", :s0, s1:s2, True] = g_q.T
            Jyp["W_g", s0:s1, s2:s3, True] = W_g

        if sys.nla_gamma:
            Jyp["W_gamma", s0:s1, s3:s4, True] = W_gamma
        if sys.nla_c:
            Jyp["W_c", s0:s1, s4:, True] = W_c
        Jy.manual_sync()
        Jyp.manual_sync()
        return Jy.tocsc(fix_size=True), Jyp.tocsc(fix_size=True)

        # note: Keep this for debugging the Jacobian

        # from scipy.optimize._numdiff import approx_derivative

        # Jy_num = approx_derivative(lambda y: self.fun(t, y, yp), y, method="2-point")
        # diff_Jy = Jy - Jy_num
        # diff_Jy = diff_Jy[s1:, s1:] # ignore kinematic equations since GGL Jacobian use not implemented
        # error_Jy = np.linalg.norm(diff_Jy)
        # print(f"error_Jy: {error_Jy}")

        # Jyp_num = approx_derivative(lambda yp: self.fun(t, y, yp), yp, method="2-point")
        # diff_Jyp = Jyp - Jyp_num
        # error_Jyp = np.linalg.norm(diff_Jyp)
        # print(f"error_Jyp: {error_Jyp}")

        # return Jy_num, Jyp_num

    def solve(self):
        # solver_summary = SolverSummary(f"Scipy solve_dae with method '{self.method}'")
        sol = solve_dae(
            self.fun,
            self.t_eval[[0, -1]],
            self.y0,
            self.y_dot0,
            t_eval=self.t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            events=[self.event],
            jac=self.jac,
            **self.kwargs,
        )
        self.pbar.close()
        # solver_summary.print()

        # unpack solution
        t = sol.t
        q, u, _, _, _, _ = np.array_split(sol.y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(sol.yp, self.split)

        return Solution(
            system=self.system,
            t=t,
            q=q.T,
            u=u.T,
            q_dot=q_dot.T,
            u_dot=u_dot.T,
            mu_g=mu_g.T,
            la_g=la_g.T,
            la_gamma=la_gamma.T,
            la_c=la_c.T,
            # solver_summary=solver_summary,
            nfev=sol.nfev,
            njev=sol.njev,
            nlu=sol.nlu,
        )
