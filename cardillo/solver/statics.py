import numpy as np
from scipy.sparse import lil_array, bmat
from tqdm import tqdm

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.math.fsolve import fsolve
from cardillo.solver.solver_options import SolverOptions
from cardillo.solver.solution import Solution


class Newton:
    """Force and displacement controlled Newton-Raphson method. This solver
    is used to find a static solution for a mechanical system. Forces and
    bilateral constraint functions are incremented in each load step if they
    depend on the time t in [0, 1]. Thus, a force controlled Newton-Raphson method
    is obtained by constructing a time constant constraint function function.
    On the other hand a displacement controlled Newton-Raphson method is
    obtained by passing constant forces and time dependent constraint functions.
    """

    def __init__(
        self,
        system,
        n_load_steps=1,
        verbose=True,
        options=SolverOptions(),
    ):
        self.system = system
        self.options = options
        self.verbose = verbose

        self.len_maxIter = len(str(self.options.newton_max_iter))

        # other dimensions
        self.nq = system.nq
        self.nu = system.nu
        self.nla_N = system.nla_N

        self.split_f = np.cumsum(
            np.array(
                [system.nu, system.nla_g, system.nla_c, system.nla_S],
                dtype=int,
            )
        )
        self.split_x = np.cumsum(
            np.array(
                [system.nq, system.nla_g, system.nla_c],
                dtype=int,
            )
        )

        # initial conditions
        x0 = np.concatenate((system.q0, system.la_g0, system.la_c0, system.la_N0))
        self.nx = len(x0)
        self.u0 = np.zeros(system.nu)  # zero velocities as system is static

        self.n_load_steps = None
        self.reset(x0, n_load_steps)

        # memory allocation
        self._W_g_coo = CooMatrix((system.nu, system.nla_g), manual_sync=True)
        self._W_c_coo = CooMatrix((system.nu, system.nla_c), manual_sync=True)
        self._W_tau_coo = CooMatrix((system.nu, system.nla_tau), manual_sync=True)
        self._W_N_coo = CooMatrix((system.nu, system.nla_N), manual_sync=True)
        self._h_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_g_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_c_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_tau_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._c_q_coo = CooMatrix((system.nla_c, system.nq), manual_sync=True)
        self._g_q_coo = CooMatrix((system.nla_g, system.nq), manual_sync=True)
        self._g_S_q_coo = CooMatrix((system.nla_S, system.nq), manual_sync=True)
        self._Wla_N_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._g_N_q_coo = CooMatrix((system.nla_N, system.nq), manual_sync=True)
        self._c_coo = CooMatrix((1, system.nla_c), manual_sync=True)
        self._h_coo = CooMatrix((1, system.nu), manual_sync=True)
        self._jac_coo = CooMatrix((self.nx, self.nx), manual_sync=True)
        self._F_coo = CooMatrix((1, self.nx), manual_sync=True)

    def reset(self, x0=None, n_load_steps=None):
        if x0 is None:
            x0 = self.x[0]

        if n_load_steps is not None and n_load_steps != self.n_load_steps:
            self.n_load_steps = n_load_steps
            self.load_steps = np.linspace(0, 1, n_load_steps + 1)
            self.nt = len(self.load_steps)
            self.x = np.zeros((self.nt, self.nx), dtype=float)
            self.len_t = len(str(self.nt))
            self.len_t = len(str(self.nt))

        self.x[0] = x0

    def fun(self, x, t):
        t = float(t)
        c0, c1, c2 = self.split_x
        r0, r1, r2, r3 = self.split_f
        # unpack unknowns
        q, la_g, la_c, la_N = x[:c0], x[c0:c1], x[c1:c2], x[c2:]

        # evaluate quantites that are required for computing the residual and
        # the jacobian
        # csr is used for efficient matrix vector multiplication, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        W_g = self._W_g_coo = self.system.W_g(t, q, format="Coo", coo=self._W_g_coo)
        W_c = self._W_c_coo = self.system.W_c(t, q, format="Coo", coo=self._W_c_coo)
        W_tau = self._W_tau_coo = self.system.W_tau(
            t, q, format="Coo", coo=self._W_tau_coo
        )

        c = self._c_coo = self.system.c(
            t, q, self.u0, la_c, format="Coo", coo=self._c_coo
        )
        h = self._h_coo = self.system.h(t, q, self.u0, format="Coo", coo=self._h_coo)

        # static equilibrium
        F = self._F_coo

        F["h", 0, :r0] = h

        W_g.manual_sync()
        F["Wla_g", 0, :r0] = W_g.tocsr(fix_size=True) @ la_g

        W_c.manual_sync()
        la_tau = self.system.la_tau(t, q, self.u0)
        F["Wla_c", 0, :r0] = W_c.tocsr(fix_size=True) @ la_c

        W_tau.manual_sync()
        F["Wla_tau", 0, :r0] = W_tau.tocsr(fix_size=True) @ la_tau

        if self.nla_N:
            W_N = self._W_N_coo = self.system.W_N(t, q, format="Coo", coo=self._W_N_coo)
            W_N.manual_sync()
            F["Wla_N", 0, :r0] = W_N.tocsr(fix_size=True) @ la_N

        F["g", 0, r0:r1] = self.system.g(t, q)
        F["c", 0, r1:r2] = c
        F["g_S", 0, r2:r3] = self.system.g_S(t, q)

        if self.nla_N:
            g_N = self.g_N = self.system.g_N(t, q)
            F["Rla_N", r3:, 0] = np.minimum(la_N, g_N)
        F.manual_sync()
        return F.toarray(fix_size=True).ravel()

    def jac(self, x, t):
        t = float(t)
        c0, c1, c2 = self.split_x
        r0, r1, r2, r3 = self.split_f
        # unpack unknowns
        q, la_g, la_c, la_N = x[:c0], x[c0:c1], x[c1:c2], x[c2:]

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        h_q = self._h_q_coo = self.system.h_q(
            t, q, self.u0, format="Coo", coo=self._h_q_coo
        )
        c_q = self._c_q_coo = self.system.c_q(
            t, q, self.u0, la_c, format="Coo", coo=self._c_q_coo
        )
        Wla_g_q = self._Wla_g_q_coo = self.system.Wla_g_q(
            t, q, la_g, format="Coo", coo=self._Wla_g_q_coo
        )
        Wla_c_q = self._Wla_c_q_coo = self.system.Wla_c_q(
            t, q, la_c, format="Coo", coo=self._Wla_c_q_coo
        )
        Wla_tau_q = self._Wla_tau_q_coo = self.system.Wla_tau_q(
            t, q, self.u0, format="Coo", coo=self._Wla_tau_q_coo
        )
        g_q = self._g_q_coo = self.system.g_q(t, q, format="Coo", coo=self._g_q_coo)
        g_S_q = self._g_S_q_coo = self.system.g_S_q(
            t, q, format="Coo", coo=self._g_S_q_coo
        )
        c_la_c = self.system.c_la_c()

        # note: csr_matrix is best for row slicing, see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
        if self.nla_N:
            self._jac_coo = CooMatrix((self.nx, self.nx), manual_sync=True)
            raise NotImplementedError(
                "CooMatrix allocation not tested yet for contact problem"
            )
        jac = self._jac_coo
        jac["W_g", :r0, c0:c1] = self._W_g_coo
        jac["W_c", :r0, c1:c2] = self._W_c_coo
        jac["c_la_c", r1:r2, c1:c2] = c_la_c
        jac["g_S_q", r2:r3, :c0] = g_S_q
        jac["g_q", r0:r1, :c0] = g_q
        jac["Wla_g_q", :r0, :c0] = Wla_g_q
        jac["Wla_c_q", :r0, :c0] = Wla_c_q
        jac["Wla_tau_q", :r0, :c0] = Wla_tau_q
        jac["h_q", :r0, :c0] = h_q
        jac["c_q", r1:r2, :c0] = c_q

        if self.nla_N:
            self._g_N_q_coo = self.system.g_N_q(t, q, format="Coo", coo=self._g_N_q_coo)
            self._Wla_N_q_coo = self.system.Wla_N_q(
                t, q, la_N, format="Coo", coo=self._Wla_N_q_coo
            )
            g_N_q = self._g_N_q_coo.tocsr(fix_size=True)

            Rla_N_q = lil_array((self.nla_N, self.nq), dtype=float)
            Rla_N_la_N = lil_array((self.nla_N, self.nla_N), dtype=float)
            for i in range(self.nla_N):
                if la_N[i] < self.g_N[i]:
                    Rla_N_la_N[i, i] = 1.0
                else:
                    Rla_N_q[i] = g_N_q[i]
            jac["W_N", :r0, c2:] = self._W_N_coo
            jac["Rla_N_q", r3:, :c0] = Rla_N_q
            jac["Rla_N_la_N", r3:, c2:] = Rla_N_q
            jac["Wla_N_q", :r0, :c0] = self._Wla_N_q_coo

        jac.manual_sync()
        return jac.tocsc(fix_size=True)
        # return bmat([[      K, self.W_g, self.W_c,   self.W_N],
        #              [    g_q,     None,     None,       None],
        #              [    c_q,     None,   c_la_c,       None],
        #              [  g_S_q,     None,     None,       None],], format="csc")

    def __pbar_text(self, force_iter, newton_iter, error):
        return (
            f" force iter {force_iter+1:>{self.len_t}d}/{self.nt};"
            f" Newton steps {newton_iter+1:>{self.len_maxIter}d}/{self.options.newton_max_iter};"
            f" error {error:.4e}"
        )

    def solve(self):
        pbar = range(0, self.nt)
        if self.verbose:
            pbar = tqdm(pbar, leave=True)

        for i in pbar:
            sol = fsolve(
                self.fun,
                self.x[i],
                jac=self.jac,
                fun_args=(self.load_steps[i],),
                jac_args=(self.load_steps[i],),
                options=self.options,
            )
            self.x[i] = sol.x
            if self.verbose:
                pbar.set_description(self.__pbar_text(i, sol.nit, sol.error))

            if not sol.success and not self.options.continue_with_unconverged:
                # return solution up to this iteration
                if self.verbose:
                    pbar.close()
                print(
                    f"Newton-Raphson method not converged, returning solution "
                    f"up to iteration {i+1:>{self.len_t}d}/{self.nt}"
                )
                return Solution(
                    system=self.system,
                    t=self.load_steps[: i + 1],
                    q=self.x[: i + 1, : self.split_x[0]],
                    u=np.zeros((i + 1, self.nu)),
                    la_g=self.x[: i + 1, self.split_x[0] : self.split_x[1]],
                    la_c=self.x[: i + 1, self.split_x[1] : self.split_x[2]],
                    la_N=self.x[: i + 1, self.split_x[2] :],
                    success=False,
                    solver=self,
                )

            # solver step callback
            self.x[i, : self.split_x[0]], _ = self.system.step_callback(
                self.load_steps[i], self.x[i, : self.split_x[0]], self.u0
            )

            # warm start for next step; store solution as new initial guess
            if i < self.nt - 1:
                self.x[i + 1] = self.x[i]

        # return solution object
        if self.verbose:
            pbar.close()
        x = self.x.copy()
        return Solution(
            self.system,
            t=self.load_steps.copy(),
            q=x[: i + 1, : self.split_x[0]],
            u=np.zeros((len(self.load_steps), self.nu)),
            la_g=x[: i + 1, self.split_x[0] : self.split_x[1]],
            la_c=x[: i + 1, self.split_x[1] : self.split_x[2]],
            la_N=x[: i + 1, self.split_x[2] :],
            success=True,
            solver=self,
        )


# read https://doi.org/10.1016/j.engstruct.2020.111755
class Riks:
    """Linear arc-length solver close to Riks method as dervied in Crisfield1991 
    section 9.3.2 p.273. A variable arc-length is chosen as shown by 
    Crisfield1981 or Crisfield 1983. For the first predictor a tangent predictor 
    is used. For all other predictors a simple secant predictor is sufficient. 
    This enables the solver to 'run forward' instead of 'doubling back on its track'.

    References
    ----------
    - stackexchange : https://scicomp.stackexchange.com/a/28140 \\
    - Wempner1971: https://doi.org/10.1016/0020-7683(71)90038-2 \\
    - Riks1972: https://doi.org/10.1115/1.3422829 \\
    - Riks1979: https://doi.org/10.1016/0020-7683(79)90081-7 \\
    - Crsfield1981: https://doi.org/10.1016/0045-7949(81)90108-5 \\
    - Crisfield1991: http://freeit.free.fr/Finite%20Element/Crisfield%20M.A.%20Vol.1.%20Non-Linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Essentials%20(Wiley,19.pdf \\
    - Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    - Neto1999: https://doi.org/10.1016/S0045-7825(99)00042-0
    """

    def __init__(
        self,
        system,
        iter_goal=4,
        la_arc0=1.0e-3,
        la_arc_span=np.array([0, 1], dtype=float),
        scale_exponent=0.5,
        max_load_steps=int(1e4),
        options=SolverOptions(),
        verbose=True,
        compute_init_ds=True,
    ):
        self.system = system
        self.options = options
        self.la_arc_span = la_arc_span
        self.max_load_steps = max_load_steps
        self.verbose = verbose

        self.nla_N = system.nla_N

        # initial arc-length parameter is not required in the first step and
        # will be computed later
        self.ds = 0

        # step size of finite differences
        self.eps = self.options.numerical_jacobian_eps

        # parameter for the step size scaling
        self.iter_goal = iter_goal
        self.MIN_FACTOR = 0.25  # minimal scaling factor
        self.MAX_FACTOR = 1.5  # maximal scaling factor
        self.scale_exponent = scale_exponent

        # split vectors
        self.split_unknowns = np.cumsum(
            np.array(
                [
                    system.nq,
                    system.nla_c,
                    system.nla_g,
                    system.nla_N,
                    1,
                ],
                dtype=int,
            )
        )[:-1]
        self.split_residual = np.cumsum(
            np.array(
                [
                    system.nu,
                    system.nla_c,
                    system.nla_g,
                    system.nla_S,
                    system.nla_N,
                    1,
                ],
                dtype=int,
            )
        )[:-1]

        # initial
        self.u0 = np.zeros(system.nu)  # statics

        # initial values for generalized coordinates, lagrange multipliers and force scaling
        x0 = np.concatenate(
            (system.q0, system.la_c0, system.la_g0, system.la_N0, np.array([0]))
        )
        self.nx = len(x0)

        self.reset(x0=x0, la_arc0=la_arc0, compute_init_ds=compute_init_ds)

        # memory allocation
        self._W_g_coo = CooMatrix((system.nu, system.nla_g), manual_sync=True)
        self._W_g2_coo = CooMatrix((system.nu, system.nla_g), manual_sync=True)
        self._W_c_coo = CooMatrix((system.nu, system.nla_c), manual_sync=True)
        self._W_tau_coo = CooMatrix((system.nu, system.nla_tau), manual_sync=True)
        self._W_tau2_coo = CooMatrix((system.nu, system.nla_tau), manual_sync=True)
        self._W_N_coo = CooMatrix((system.nu, system.nla_N), manual_sync=True)
        self._h_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_g_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_c_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._Wla_tau_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._c_q_coo = CooMatrix((system.nla_c, system.nq), manual_sync=True)
        self._g_q_coo = CooMatrix((system.nla_g, system.nq), manual_sync=True)
        self._g_S_q_coo = CooMatrix((system.nla_S, system.nq), manual_sync=True)
        self._Wla_N_q_coo = CooMatrix((system.nu, system.nq), manual_sync=True)
        self._g_N_q_coo = CooMatrix((system.nla_N, system.nq), manual_sync=True)
        self._c_coo = CooMatrix((1, system.nla_c), manual_sync=True)
        self._h_coo = CooMatrix((1, system.nu), manual_sync=True)
        self._J_coo = CooMatrix((self.nx, self.nx), manual_sync=True)
        self._R_coo = CooMatrix((1, self.nx), manual_sync=True)

    def reset(self, x0=None, la_arc0=None, compute_init_ds=True):
        if x0 is None:
            x0 = self.x0
        else:
            x0[-1] = 0
            self.x0 = x0

        if la_arc0 is None:
            la_arc0 = self.la_arc0
        else:
            self.la_arc0 = la_arc0

        q0, la_c0, la_g0, la_N0, _ = self._split_x(x0)
        self.q0 = q0
        self.la_c0 = la_c0
        self.la_g0 = la_g0
        self.la_N0 = la_N0

        self.xk = np.concatenate((q0, la_c0, la_g0, la_N0, np.array([0])))
        self.x0_bar = np.concatenate((q0, la_c0, la_g0, la_N0, np.array([la_arc0])))

        if not compute_init_ds:
            return

        ####################################################################################################
        # Solve linearized system for fixed external force using Newtons method.
        # From this solution we can extract the initial ds using the arc length equation.
        # All other ds values will be modified according to the number of used Newton steps,
        # see https://scicomp.stackexchange.com/questions/28137/initialize-arc-length-control-in-riks-method
        ####################################################################################################
        if self.verbose:
            print(f"solve equilibrium for given initial la_arc0")

        def fun(x):
            x = np.concatenate((x, [la_arc0]))
            return self.R(x)[:-1]

        def jac(x):
            x = np.concatenate((x, [la_arc0]))
            return self.J(x)[:-1, :-1]

        sol = fsolve(fun, self.x0_bar[:-1], jac=jac, options=self.options)
        assert (
            sol.success
        ), "solving for initial arc-length parameter 'ds' did not converge => chose another 'la_arc0'"

        # compute initial ds from arc-length equation
        self.x0_bar = np.concatenate((sol.x, [la_arc0]))
        self.ds = self.a(self.x0_bar) ** 0.5
        assert self.ds > 0, "initial ds is zero"
        if self.verbose:
            print(f"initial ds: {self.ds:2.4e}")

    def _split_x(self, x):
        c0, c1, c2, c3 = self.split_unknowns
        # extract generalized coordinates, Lagrange multipliers and arc-length parameter
        q, la_c, la_g, la_N, t = x[:c0], x[c0:c1], x[c1:c2], x[c2:c3], x[c3]
        return q, la_c, la_g, la_N, t

    def a(self, x):
        """The most primitive arc-length equation restricts the change of all
        generalized coordinates `qn1` w.r.t. the last converged Newton step `qn`."""
        qn = self._split_x(self.xk)[0]
        qn1 = self._split_x(x)[0]
        dq = qn1 - qn
        return dq @ dq

    def a_q(self, x):
        qn = self._split_x(self.xk)[0]
        qn1 = self._split_x(x)[0]
        dq = qn1 - qn
        return 2 * dq

    def R(self, x):
        # extract generalized coordinates, Lagrange multipliers and arc-length parameter
        q, la_c, la_g, la_N, t = self._split_x(x)
        t = float(t)

        # evaluate all functions with t = la_arc
        # - this requires the external force that should be scaled to be of the form
        #   h(t, q) = W(g) * t
        # - for displacement control, the bilateral constraints can be time-dependent
        #   g = g(t, q)

        # compute quantities required for Jacobian
        W_g = self._W_g_coo = self.system.W_g(t, q, format="Coo", coo=self._W_g_coo)
        W_c = self._W_c_coo = self.system.W_c(t, q, format="Coo", coo=self._W_c_coo)
        W_tau = self._W_tau_coo = self.system.W_tau(
            t, q, format="Coo", coo=self._W_tau_coo
        )
        self.h = self.system.h(t, q, self.u0)
        self.g = self.system.g(t, q)
        la_tau = self.system.la_tau(t, q, self.u0)

        W_g.manual_sync()
        W_c.manual_sync()
        W_tau.manual_sync()
        self.W_g = W_g.tocsr(fix_size=True)
        self.W_c = W_c.tocsr(fix_size=True)
        self.W_tau = W_tau.tocsr(fix_size=True)

        # build residual
        R = np.zeros_like(x)
        R = x.copy()
        R[: self.split_residual[0]] = (
            self.h + self.W_c @ la_c + self.W_g @ la_g + self.W_tau @ la_tau
        )
        R[self.split_residual[0] : self.split_residual[1]] = self.system.c(
            t, q, self.u0, la_c
        )
        R[self.split_residual[1] : self.split_residual[2]] = self.g
        R[self.split_residual[2] : self.split_residual[3]] = self.system.g_S(t, q)

        if self.nla_N:
            self.g_N = self.system.g_N(t, q)
            R[self.split_residual[3] : self.split_residual[4]] = np.minimum(
                la_N, self.g_N
            )

        R[-1] = self.a(x) - self.ds**2

        return R

    def J(self, x):
        c0, c1, c2, c3 = self.split_unknowns
        r0, r1, r2, r3, r4 = self.split_residual
        # extract generalized coordinates, Lagrange multipliers and arc-length parameter
        q, la_c, la_g, la_N, t = self._split_x(x)
        t = float(t)

        # evaluate additionally required quantites for computing the jacobian
        # coo is used for efficient bmat
        h_q = self._h_q_coo = self.system.h_q(
            t, q, self.u0, format="Coo", coo=self._h_q_coo
        )
        c_q = self._c_q_coo = self.system.c_q(
            t, q, self.u0, la_c, format="Coo", coo=self._c_q_coo
        )
        Wla_g_q = self._Wla_g_q_coo = self.system.Wla_g_q(
            t, q, la_g, format="Coo", coo=self._Wla_g_q_coo
        )
        Wla_c_q = self._Wla_c_q_coo = self.system.Wla_c_q(
            t, q, la_c, format="Coo", coo=self._Wla_c_q_coo
        )
        Wla_tau_q = self._Wla_tau_q_coo = self.system.Wla_tau_q(
            t, q, self.u0, format="Coo", coo=self._Wla_tau_q_coo
        )
        g_q = self._g_q_coo = self.system.g_q(t, q, format="Coo", coo=self._g_q_coo)
        g_S_q = self._g_S_q_coo = self.system.g_S_q(
            t, q, format="Coo", coo=self._g_S_q_coo
        )
        c_la_c = self.system.c_la_c()

        if self.nla_N:
            Wla_N_q = self.system.Wla_N_q(t, q, la_N)

            # note: csr_matrix is best for row slicing, see
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array
            g_N_q = self.system.g_N_q(t, q, format="csr")

            Rla_N_q = lil_array((self.system.nla_N, self.system.nq), dtype=float)
            Rla_N_la_N = lil_array((self.system.nla_N, self.system.nla_N), dtype=float)
            for i in range(self.system.nla_N):
                if la_N[i] < self.g_N[i]:
                    Rla_N_la_N[i, i] = 1.0
                else:
                    Rla_N_q[i] = g_N_q[i]

        # note: We use finite differences to compute the derivatives w.r.t.
        # to the arc-length parameter. Hence, we do not have to specify here
        # how the arc-length parameter enters the vector of generalized forces h.
        # For displacement based approaches, we simply add a corresponding
        # bilateral constraint g(t, q).
        eps = self.eps

        W_g2 = self._W_g2_coo = self.system.W_g(
            t + eps, q, format="Coo", coo=self._W_g2_coo
        )
        W_g2.manual_sync()

        W_tau2 = self._W_tau2_coo = self.system.W_tau(
            t + eps, q, format="Coo", coo=self._W_tau2_coo
        )
        W_tau2.manual_sync()

        h_t = (self.system.h(t + eps, q, self.u0) - self.h) / eps
        Wla_g_t = (W_g2.tocsr(fix_size=True) @ la_g - self.W_g @ la_g) / eps
        Wla_tau_t = (
            W_tau2.tocsr(fix_size=True) @ self.system.la_tau(t + eps, q, self.u0)
            - self.W_tau @ self.system.la_tau(t, q, self.u0)
        ) / eps
        g_t = (self.system.g(t + eps, q) - self.g) / eps

        # derivative of the arc length equation
        a_q = self.a_q(x)

        if self.nla_N:
            self._J_coo = CooMatrix((self.nx, self.nx), manual_sync=True)
            raise NotImplementedError(
                "CooMatrix allocation not tested yet for contact problem"
            )

        J = self._J_coo
        J["h_q", :r0, :c0] = h_q
        J["Wla_c_q", :r0, :c0] = Wla_c_q
        J["Wla_g_q", :r0, :c0] = Wla_g_q

        J["Wla_tau_q", :r0, :c0] = Wla_tau_q
        J["W_c", :r0, c0:c1] = self._W_c_coo
        J["W_g", :r0, c1:c2] = self._W_g_coo

        # Ru_t = h_t + Wla_g_t + Wla_tau_t
        J["h_t", :r0, c3:] = h_t
        J["Wla_g_t", :r0, c3:] = Wla_g_t
        J["Wla_tau_t", :r0, c3:] = Wla_tau_t

        J["c_q", r0:r1, :c0] = c_q
        J["c_la_c", r0:r1, c0:c1] = c_la_c

        J["g_q", r1:r2, :c0] = g_q
        J["g_t", r1:r2, c3:] = g_t[:, None]
        J["g_S_q", r2:r3, :c0] = g_S_q

        if self.nla_N:
            J["Wla_N_q", :r0, :c0] = Wla_N_q
            J["W_N", :r0, c2:c3] = self.system.W_N(t, q, format="csr")
            J["Rla_N_q", r3:r4, :c0] = Rla_N_q
            J["Rla_N_la_N", r3:r4, c2:c3] = Rla_N_la_N

        J["a_q", r4:, :c0] = a_q

        J.manual_sync()
        return J.tocsc(fix_size=True)
        # return bmat([[      K, self.W_c, self.W_g,   self.W_N, Ru_t[:, None]],
        #              [    c_q,   c_la_c,     None,       None,          None],
        #              [    g_q,     None,     None,       None,  g_t[:, None]],
        #              [  g_S_q,     None,     None,       None,          None],
        #              [Rla_N_q,     None,     None, Rla_N_la_N,          None],
        #              [    a_q,     None,     None,       None,          None]], format="csc")

    def solve(self):
        # count number of force increments to get first increment with tangential predictor
        i = 0

        # initialize current generalized coordinates, Lagrange multipliers and
        # arc-length parameter
        q = [self.q0]
        la_c = [self.la_c0]
        la_g = [self.la_g0]
        la_N = [self.la_N0]
        la_arc = [self.la_arc0]

        # loop over ranges of force scaling
        xk1 = self.x0_bar.copy()  # initialize such that Jacobian is regular!

        # progress bar
        if self.verbose:
            pbar = tqdm(total=100, leave=True)
            i0 = 0
        load_step = 0
        while (
            xk1[-1] >= self.la_arc_span[0]
            and xk1[-1] <= self.la_arc_span[1]
            and load_step <= self.max_load_steps
        ):
            # increment number of steps
            i += 1
            # load step counter
            load_step += 1

            # use secant predictor for all other force increments than the first one
            if i > 1:
                # secand predictor for all but the first newton iteration
                dx = self.xk - self.x0
                xk1 += dx

            # solve nonlinear system
            sol = fsolve(self.R, xk1, jac=self.J, options=self.options)
            xk1 = sol.x
            assert sol.success, f"internal newton method is not converged"

            # Scale ds such that iter goal is satisfied. Disable scaling if we
            # have halved the ds parameter before or after the first iteration
            # which requires lots of iterations see Crisfield1991, section 9.5
            # (9.40) or (9.41) for the square root scaling.
            if self.scale_exponent is not None and sol.nit > 0:
                fac = (self.iter_goal / sol.nit) ** self.scale_exponent
                self.ds *= max(self.MIN_FACTOR, min(fac, self.MAX_FACTOR))

            # store last converged newton step
            self.x0 = self.xk.copy()

            # store new converged newton step
            self.xk = xk1.copy()

            # append solutions to lists
            q_, la_c_, la_g_, la_N_, la_arc_ = self._split_x(xk1)
            q.append(q_)
            la_c.append(la_c_)
            la_g.append(la_g_)
            la_N.append(la_N_)
            la_arc.append(la_arc_)

            # update progress bar
            i1 = int(
                100
                * (la_arc_ - self.la_arc_span[0])
                / (self.la_arc_span[1] - self.la_arc_span[0])
            )
            if self.verbose:
                pbar.update(min(i1, 100) - i0)
                pbar.set_description(
                    f"la_arc: {self.la_arc_span[0]:0.2e} <= {la_arc_:0.2e} <= {self.la_arc_span[1]:0.2e}; error: {sol.error:0.2e}; iter: {sol.nit}"
                )
                i0 = i1

        if self.verbose:
            pbar.close()

        # return solution object
        return Solution(
            system=self.system,
            t=np.asarray(la_arc),
            q=np.asarray(q),
            u=np.zeros((len(q), len(self.u0))),
            la_c=np.asarray(la_c),
            la_g=np.asarray(la_g),
            la_N=np.asarray(la_N),
            success=sol.success,
            solver=self,
        )
