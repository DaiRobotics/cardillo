import numpy as np
from scipy.sparse import eye_array
from scipy_dae.integrate import solve_dae
from tqdm import tqdm
from jax import jit, numpy as jnp

from cardillo.solver import Solution, SolverSummary
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.jaxproxy import coo_matvec_jit


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
        verbose=True,
        **kwargs,
    ):
        self.system = system
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.verbose = verbose
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

        if verbose:
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

        if system._jaxed:
            s0, s1, s2, s3, s4 = self.split
            self._fun_jit = jit(self._fun_jax)
            self._jac_jit = jit(self._jac_jax)

            if not system.constant_mass_matrix:
                raise NotImplementedError("Variable mass matrix not implemented yet.")

            self.Jy_jax = CooMatrix((self.ny, self.ny))
            self.Jy_jax.row = np.concatenate(
                [
                    system.q_dot_q_jax_row,
                    system.q_dot_u_jax_row,
                    # system._Mu_q_jax_row,
                    system.h_q_jax_row + s0,
                    system.Wla_tau_q_jax_row + s0,
                    # system._Wla_gamma_q_jax_row + s0,
                    system.Wla_g_q_jax_row + s0,
                    system.Wla_c_q_jax_row + s0,
                    system.h_u_jax_row + s0,
                    system.Wla_tau_u_jax_row + s0,
                    system.g_q_jax_row + s1,
                    system.g_dot_q_jax_row + s2,
                    system.g_dot_u_jax_row + s2,
                    # system._gamma_q_jax_row + s3,
                    # system._gamma_u_jax_row + s3,
                    system.c_q_jax_row + s4,
                    system.c_u_jax_row + s4,
                ]
            )
            self.Jy_jax.col = np.concatenate(
                [
                    system.q_dot_q_jax_col,
                    system.q_dot_u_jax_col + s0,
                    # system._Mu_q_jax_col + s1,
                    system.h_q_jax_col,
                    system.Wla_tau_q_jax_col,
                    # system._Wla_gamma_q_jax_col,
                    system.Wla_g_q_jax_col,
                    system.Wla_c_q_jax_col,
                    system.h_u_jax_col + s0,
                    system.Wla_tau_u_jax_col + s0,
                    system.g_q_jax_col,
                    system.g_dot_q_jax_col,
                    system.g_dot_u_jax_col + s0,
                    # system._gamma_q_jax_col,
                    # system._gamma_u_jax_col + s1,
                    system.c_q_jax_col,
                    system.c_u_jax_col + s0,
                ]
            )

            self.Jyp_jax = CooMatrix((self.ny, self.ny))
            self.Jyp_jax.row = np.concatenate(
                [
                    np.arange(system.nq, dtype=int),  # for eye_q
                    system.g_q_jax_col,  # g_q is transposed in Jyp
                    system.M_jax_row + s0,
                    system.W_g_jax_row + s0,
                    # system._W_gamma_jax_row + s0,
                    system.W_c_jax_row + s0,
                    system.c_la_c_jax_row + s4,
                ]
            )
            self.Jyp_jax.col = np.concatenate(
                [
                    np.arange(system.nq, dtype=int),  # for eye_q
                    system.g_q_jax_row + s1,  # g_q is transposed in Jyp
                    system.M_jax_col + s0,
                    system.W_g_jax_col + s2,
                    # system._W_gamma_jax_col + s3,
                    system.W_c_jax_col + s4,
                    system.c_la_c_jax_col + s4,
                ]
            )

    def event(self, t, y, yp):
        q, u = np.array_split(y, self.split)[:2]
        q, u = self.system.step_callback(t, q, u)
        return 1

    def _fun_jax(self, t, y, yp, tau):
        sys = self.system
        # unpack vectors
        s0, s1, s2, s3, s4 = self.split
        q, u = y[:s0], y[s0:s1]
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = (
            yp[:s0],
            yp[s0:s1],
            yp[s1:s2],
            yp[s2:s3],
            yp[s3:s4],
            yp[s4:],
        )

        # residual
        F = jnp.zeros(self.ny)
        ####################
        # kinematic equation
        ####################
        F = F.at[:s0].add(q_dot)
        F = F.at[:s0].subtract(sys.q_dot_jax(t, q, u))
        if self.nla_g:
            g_q = sys.g_q_jax(t, q)
            # g_q.T @ mu_g
            F = F.at[:s0].subtract(
                coo_matvec_jit(sys.g_q_jax_col, sys.g_q_jax_row, g_q, mu_g, sys.nq)
            )

        ####################
        # equations of motion
        ####################
        F = F.at[s0:s1].add(
            coo_matvec_jit(sys.M_jax_row, sys.M_jax_col, sys.M_jax(t, q), u_dot, sys.nu)
        )
        F = F.at[s0:s1].subtract(sys.h_jax(t, q, u))
        if sys.nla_tau:
            la_tau = sys.la_tau_jax(t, q, u, tau)
            F = F.at[s0:s1].subtract(
                coo_matvec_jit(
                    sys.W_tau_jax_row,
                    sys.W_tau_jax_col,
                    sys.W_tau_jax(t, q),
                    la_tau,
                    sys.nu,
                )
            )
        if sys.nla_g:
            F = F.at[s0:s1].subtract(
                coo_matvec_jit(
                    sys.W_g_jax_row, sys.W_g_jax_col, sys.W_g_jax(t, q), la_g, sys.nu
                )
            )
        if sys.nla_gamma:
            # F["Wla_gamma", 0, s1:s2, True] = W_gamma.tocsr(fix_size=True) @ la_gamma
            raise NotImplementedError("gamma constraints not implemented yet")
        if sys.nla_c:
            F = F.at[s0:s1].subtract(
                coo_matvec_jit(
                    sys.W_c_jax_row, sys.W_c_jax_col, sys.W_c_jax(t, q), la_c, sys.nu
                )
            )

        #######################
        # bilateral constraints
        #######################
        if sys.nla_g:
            F = F.at[s1:s2].set(sys.g_jax(t, q))
            F = F.at[s2:s3].set(sys.g_dot_jax(t, q, u))

        if sys.nla_gamma:
            # F["gamma", 0, s4:s5] = self.system.gamma(t, q, u)
            raise NotImplementedError("gamma constraints not implemented yet")

        ############
        # compliance
        ############
        if sys.nla_c:
            F = F.at[s4:].set(sys.c_jax(t, q, u, la_c))
        return F

    def fun(self, t, y, yp):
        t = float(t)
        tau = self.system.tau(t)
        # update progress bar
        if self.verbose:
            pbar_i = int(np.floor((t + self.frac / 2) / self.frac))
            self.pbar.update(pbar_i - self.pbar_i)
            self.pbar.set_description(
                f"{self.method}: t {t:0.2e}s < {self.t1:0.2e}s", refresh=False
            )
            self.pbar_i = pbar_i

        if self.system._jaxed:
            return self._fun_jit(t, y, yp, tau)
        # unpack vectors
        s0, s1, s2, s3, s4 = self.split
        q, u = y[:s0], y[s0:s1]
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = (
            yp[:s0],
            yp[s0:s1],
            yp[s1:s2],
            yp[s2:s3],
            yp[s3:s4],
            yp[s4:],
        )

        # residual
        F = self.F
        sys = self.system
        h = self.h = sys.h(t, q, u, format="Coo", coo=self.h)
        c = self.c = sys.c(t, q, u, la_c, format="Coo", coo=self.c)
        q_dot2 = self.q_dot = sys.q_dot(t, q, u, format="Coo", coo=self.q_dot)
        if self.nla_g:
            g_q = self.g_q1 = sys.g_q(t, q, format="Coo", coo=self.g_q1)
            g_q_T = self.g_q1_T = g_q.transpose(copy=False, coo=self.g_q1_T)
        if sys.nla_tau:
            W_tau = self._W_tau = sys.W_tau(t, q, format="Coo", coo=self._W_tau)
        if sys.nla_g:
            W_g = self.W_g1 = sys.W_g(t, q, format="Coo", coo=self.W_g1)
        if sys.nla_gamma:
            W_gamma = self.W_gamma1 = sys.W_gamma(t, q, format="Coo", coo=self.W_gamma1)
        if sys.nla_c:
            W_c = self.W_c1 = sys.W_c(t, q, format="Coo", coo=self.W_c1)
        ####################
        # kinematic equation
        ####################
        F["q_dot", 0, :s0] = q_dot
        F["q_dot2", 0, :s0, True] = q_dot2
        if self.nla_g:
            g_q.manual_sync()
            F["g_q_T_mu_g", 0, :s0, True] = g_q_T.tocsr(fix_size=True) @ mu_g

        ####################
        # equations of motion
        ####################
        M = self.M2 = sys.M(t, q, format="Coo", coo=self.M2)
        F["Mu", 0, s0:s1] = M.tocsr(fix_size=True) @ u_dot
        F["h", 0, s0:s1, True] = h
        if sys.nla_tau:
            W_tau.manual_sync()
            F["Wla_tau", 0, s0:s1, True] = W_tau.tocsr(fix_size=True) @ sys.la_tau(
                t, q, u, tau
            )
        if sys.nla_g:
            W_g.manual_sync()
            F["Wla_g", 0, s0:s1, True] = W_g.tocsr(fix_size=True) @ la_g
        if sys.nla_gamma:
            F["Wla_gamma", 0, s0:s1, True] = W_gamma.tocsr(fix_size=True) @ la_gamma
        if sys.nla_c:
            W_c.manual_sync()
            F["Wla_c", 0, s0:s1, True] = W_c.tocsr(fix_size=True) @ la_c

        #######################
        # bilateral constraints
        #######################
        if sys.nla_g:
            F["g", 0, s1:s2] = sys.g(t, q)
            F["g_dot", 0, s2:s3] = sys.g_dot(t, q, u)

        if sys.nla_gamma:
            F["gamma", 0, s3:s4] = sys.gamma(t, q, u)

        ############
        # compliance
        ############
        if sys.nla_c:
            F["c", 0, s4:] = c
        F.manual_sync()
        return F.toarray(fix_size=True).ravel()

    def _jac_jax(self, t, y, yp, tau):
        sys = self.system

        # unpack vectors
        s0, s1, s2, s3, s4 = self.split
        q, u = y[:s0], y[s0:s1]
        u_dot = yp[s0:s1]
        la_g = yp[s2:s3]
        la_gamma = yp[s3:s4]
        la_c = yp[s4:]

        q_dot_q = sys.q_dot_q_jax(t, q, u)
        q_dot_u = sys.q_dot_u_jax(t, q, u)

        # Mu_q = sys.Mu_q_jax(t, q, u)
        h_q = sys.h_q_jax(t, q, u)
        h_u = sys.h_u_jax(t, q, u)

        if sys.nla_g:
            g_q = sys.g_q_jax(t, q)
            g_dot_q = sys.g_dot_q_jax(t, q, u)
            g_dot_u = sys.g_dot_u_jax(t, q)
            W_g = sys.W_g_jax(t, q)
            Wla_g_q = sys.Wla_g_q_jax(t, q, la_g)
        else:
            g_q = g_dot_q = g_dot_u = W_g = Wla_g_q = np.array([], dtype=float)

        if sys.nla_c:
            c_q = sys.c_q_jax(t, q, u, la_c)
            c_u = sys.c_u_jax(t, q, u, la_c)

            W_c = sys.W_c_jax(t, q)
            Wla_c_q = sys.Wla_c_q_jax(t, q, la_c)
            c_la_c = sys.c_la_c_jax()
        else:
            c_q = c_u = W_c = Wla_c_q = c_la_c = np.array([], dtype=float)

        if sys.nla_tau:
            Wla_tau_q = sys.Wla_tau_q_jax(t, q, u, tau)
            Wla_tau_u = sys.Wla_tau_u_jax(t, q, u, tau)
        else:
            Wla_tau_q = Wla_tau_u = np.array([], dtype=float)

        if sys.nla_gamma:
            # TODO: implement gamma constraints
            raise NotImplementedError("gamma constraints not implemented yet")

        return jnp.concatenate(
            [
                -q_dot_q,
                -q_dot_u,
                # Mu_q,
                -h_q,
                -Wla_tau_q,
                # -Wla_gamma_q,
                -Wla_g_q,
                -Wla_c_q,
                -h_u,
                -Wla_tau_u,
                g_q,
                g_dot_q,
                g_dot_u,
                # gamma_q,
                # gamma_u,
                c_q,
                c_u,
            ]
        ), jnp.concatenate(
            [
                jnp.ones(sys.nq),  # for eye_q
                -g_q,  # -g_q.T, so row and col are swapped
                sys.M_jax(t, q),
                -W_g,
                # -W_gamma,
                -W_c,
                c_la_c,
            ]
        )

    def jac(self, t, y, yp):
        t = float(t)
        sys = self.system
        tau = sys.tau(t)

        if sys._jaxed:
            Jy = self.Jy_jax
            Jyp = self.Jyp_jax
            Jy.data, Jyp.data = self._jac_jit(t, y, yp, tau)
            return Jy.tocsc(fix_size=True), Jyp.tocsc(fix_size=True)

        # unpack vectors
        q, u, _, _, _, _ = np.array_split(y, self.split)
        q_dot, u_dot, mu_g, la_g, la_gamma, la_c = np.array_split(yp, self.split)

        # evaluate commonly used quantities
        q_dot_q = sys.q_dot_q(t, q, u)
        q_dot_u = sys.q_dot_u(t, q)

        Mu_q = sys.Mu_q(t, q, u_dot)
        h_q = sys.h_q(t, q, u)
        h_u = sys.h_u(t, q, u)
        Wla_tau_q = sys.Wla_tau_q(t, q, u, tau)
        Wla_tau_u = sys.Wla_tau_u(t, q, u, tau)
        Wla_g_q = sys.Wla_g_q(t, q, la_g)
        Wla_gamma_q = sys.Wla_gamma_q(t, q, la_gamma)
        Wla_c_q = sys.Wla_c_q(t, q, la_c)

        g_dot_q = sys.g_dot_q(t, q, u)
        g_dot_u = sys.g_dot_u(t, q)

        gamma_q = sys.gamma_q(t, q, u)
        gamma_u = sys.gamma_u(t, q)

        c_q = sys.c_q(t, q, u, la_c)
        c_u = sys.c_u(t, q, u, la_c)

        eye_q = eye_array(self.nq)
        M = sys.M(t, q)
        g_q = sys.g_q(t, q)
        W_g = sys.W_g(t, q)
        W_gamma = sys.W_gamma(t, q)
        W_c = sys.W_c(t, q)
        c_la_c = sys.c_la_c()

        # first Jacobian w.r.t. y
        Jy = CooMatrix((self.ny, self.ny))

        Jy[: self.split[0], : self.split[0]] = -q_dot_q
        Jy[: self.split[0], self.split[0] : self.split[1]] = -q_dot_u
        # note: Here we ignore the derivative d((dg/dq)^T mu) / dq since
        # `solve_dae` already performs an inexact Newton method.
        # Jy[:self.split[0], self.split[1]:self.split[2]] = g_q_T_mu_q

        Jy[self.split[0] : self.split[1], : self.split[0]] = (
            Mu_q - h_q - Wla_tau_q - Wla_gamma_q - Wla_g_q - Wla_c_q
        )
        Jy[self.split[0] : self.split[1], self.split[0] : self.split[1]] = (
            -h_u - Wla_tau_u
        )

        Jy[self.split[1] : self.split[2], : self.split[0]] = g_q

        Jy[self.split[2] : self.split[3], : self.split[0]] = g_dot_q
        Jy[self.split[2] : self.split[3], self.split[0] : self.split[1]] = g_dot_u

        Jy[self.split[3] : self.split[4], : self.split[0]] = gamma_q
        Jy[self.split[3] : self.split[4], self.split[0] : self.split[1]] = gamma_u

        Jy[self.split[4] :, : self.split[0]] = c_q
        Jy[self.split[4] :, self.split[0] : self.split[1]] = c_u

        # second Jacobian w.r.t. yp
        Jyp = CooMatrix((self.ny, self.ny))

        Jyp[: self.split[0], : self.split[0]] = eye_q
        Jyp[: self.split[0], self.split[1] : self.split[2]] = -g_q.T

        Jyp[self.split[0] : self.split[1], self.split[0] : self.split[1]] = M
        Jyp[self.split[0] : self.split[1], self.split[2] : self.split[3]] = -W_g
        Jyp[self.split[0] : self.split[1], self.split[3] : self.split[4]] = -W_gamma
        Jyp[self.split[0] : self.split[1], self.split[4] :] = -W_c

        Jyp[self.split[4] :, self.split[4] :] = c_la_c
        # assert np.allclose(Jyp.data, self.Jyp_jax.data)
        # assert np.allclose(Jyp.col, self.Jyp_jax.col)
        # assert np.allclose(Jyp.row, self.Jyp_jax.row)

        # assert np.allclose(Jy.row, self.Jy_jax.row)
        # assert np.allclose(Jy.col, self.Jy_jax.col)
        # assert np.allclose(Jy.data, self.Jy_jax.data)

        # assert np.allclose(Jyp.toarray(fix_size=True), self.Jyp_jax.toarray(fix_size=True))
        # assert np.allclose(Jy.toarray(fix_size=True), self.Jy_jax.toarray(fix_size=True))
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
        if self.verbose:
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
