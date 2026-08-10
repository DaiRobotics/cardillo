
import numpy as np
from scipy.sparse.linalg import splu

from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.solver import Newton, SolverOptions

import io
import warnings
from contextlib import contextmanager, redirect_stdout

@contextmanager
def _quiet():
    """Swallow the warning and print of a Newton solve we are going to retry."""
    with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def smoothing_polynomial(s):
    s = np.clip(s, 0.0, 1.0)
    return 6 * s**5 - 15 * s**4 + 10 * s**3


class StaticModelTwin:
    def __init__(
        self,
        model_factory,
        g_accel=9.81,
        n_load_steps_init=10,
        n_load_steps_retry=10,
        n_retry_levels=3,
        verbose=False,
        options=None,
        static_atol=1e-6,
        **model_kwargs,
    ):
        self.model = model_factory(damping_ratio=0.0, **model_kwargs)
        self.system = self.model.system
        self.rod = self.model.rod
        self.tendons = self.model.tendons
        self.n_tendons = len(self.tendons)
        self.g_accel = g_accel

        self.n_load_steps_init = n_load_steps_init
        self.n_load_steps_retry = n_load_steps_retry
        self.n_retry_levels = n_retry_levels
        self.static_atol = static_atol

        self.set_gravity(ramped=True)

        self.system.assemble()
        self.solver = Newton(
            self.system,
            n_load_steps=n_load_steps_init,
            verbose=verbose,
            options=options if options is not None else SolverOptions(),
        )

        self.initialized = False
        self.x_eq = None
        self.la_t_prev = np.zeros(self.n_tendons)
        self.n_solves = 0
        self.n_retries = 0

    def set_gravity(self, ramped):
        rho, cs, g = self.model.rod_density, self.model.cross_section, self.g_accel
        scale = (lambda t: g * t) if ramped else (lambda t: g)
        self.model.gravity._h_nodes = Force_line_distributed._make_h_nodes(
            lambda t, xi: rho
            * cs.area(xi)
            * scale(t)
            * np.array([0, -1.0, 0], dtype=np.float64)
        )

    def set_load_steps(self, load_steps):
        solver = self.solver
        x0 = solver.x[0].copy()
        solver.load_steps = np.asarray(load_steps, dtype=float)
        solver.nt = len(solver.load_steps)
        solver.len_t = len(str(solver.nt))
        solver.x = np.zeros((solver.nt, solver.nx), dtype=float)
        solver.x[0] = x0

    def set_tendon_forces(self, la_t, la_t_from=None):
        if la_t_from is None:
            for td, la in zip(self.tendons, la_t):
                td.set_force(lambda t, la=la: la)
        else:
            for td, la0, la1 in zip(self.tendons, la_t_from, la_t):
                td.set_force(lambda t, la0=la0, la1=la1: la0 + (la1 - la0) * smoothing_polynomial(t))

    def run(self, la_t):
        self.solver.solve()
        x_eq = self.solver.x[-1].copy()
        result = self.solver.fun(x_eq, 1.0)
        err = np.linalg.norm(result) / result.size**0.5
        if not np.isfinite(err) or err > self.static_atol:
            raise RuntimeError(f"static solve did not converge (residual {err:.3e}) for la_t = {la_t}")
        return x_eq

    def retry(self, la_t):
        n = self.n_load_steps_retry
        for level in range(self.n_retry_levels):
            self.set_tendon_forces(la_t, la_t_from=self.la_t_prev)
            self.set_load_steps(np.linspace(0.0, 1.0, n + 1))
            self.solver.x[0] = self.x_eq
            last = level == self.n_retry_levels - 1
            try:
                if last:  # let the final attempt fail loudly
                    return self.run(la_t)   
                with _quiet():
                    return self.run(la_t)
            except RuntimeError:
                if last:
                    raise
                n *= 4

    def solve(self, la_t):
        la_t.shape == (self.n_tendons,)

        if not self.initialized:
            self.set_tendon_forces(la_t, la_t_from=np.zeros(self.n_tendons))
            self.set_load_steps(np.linspace(0.0, 1.0, self.n_load_steps_init + 1))
            x_eq = self.run(la_t)
            self.set_gravity(ramped=False)
        else:
            self.set_tendon_forces(la_t)
            self.set_load_steps([1.0])
            self.solver.x[0] = self.x_eq
            try:
                with _quiet():
                    x_eq = self.run(la_t)
            except RuntimeError:
                x_eq = self.retry(la_t)
                self.n_retries += 1

        self.x_eq = x_eq
        self.la_t_prev = la_t.copy()
        self.initialized = True
        self.n_solves += 1
        return self.x_eq

    def q_eq(self):
        return self.x_eq[: self.system.nq]
    
    def r_OP_eq(self):
        rod = self.rod
        return rod._view_nodal_q(self.x_eq[: self.system.nq][rod.qDOF])[-1, :3]

    def eval_J_stat(self):
        assert self.x_eq is not None, "call solve(la_t) first to get an equilibrium"
        system, rod, solver = self.system, self.rod, self.solver
        x = self.x_eq.flatten()
        q = x[: system.nq]

        df_dx = solver.jac(x, 1.0)

        W_tau = np.zeros((system.nu, self.n_tendons))
        for j, td in enumerate(self.tendons):
            np.add.at(W_tau[:, j], td.uDOF, -td.W_l(1.0, q[td.qDOF]))
        df_dla_t = np.zeros((solver.nx, self.n_tendons))
        df_dla_t[: system.nu, :] = W_tau

        dx_dla_tau = splu(df_dx).solve(-df_dla_t)
        pos_idx = rod.qDOF[rod.nodalDOF_r[rod.nnode - 1]]
        return dx_dla_tau[pos_idx, :]

    def solve_and_eval_J_stat(self, la_t):
        self.solve(la_t)
        return self.eval_J_stat()


# ----------------------------------------------------------------------
# inverse statics
# ----------------------------------------------------------------------
def inverse_statics(
    static_model_twin,
    r_OP_ref,
    la_t0=None,
    tol=1e-8,
    damping=1e-10,
    max_step=0.5,
    max_iter=50,
    n_backtrack=10,
    verbose=False,
):

    la_t = (np.zeros(static_model_twin.n_tendons) if la_t0 is None else la_t0.copy())

    J_stat = static_model_twin.solve_and_eval_J_stat(la_t)
    e = r_OP_ref - static_model_twin.r_OP_eq()
    e_n = np.linalg.norm(e)

    for k in range(max_iter):
        if verbose:
            print(f"  inverse statics it {k:2d}: |e| = {e_n * 1e3:9.6f} mm")
        if e_n < tol:
            break

        dla_t = J_stat.T @ np.linalg.solve(J_stat @ J_stat.T + damping * np.eye(3), e)
        # At most max_step on the largest tendon
        dla_t_max = np.max(np.abs(dla_t))
        if dla_t_max > max_step:
            dla_t *= max_step / dla_t_max

        for _ in range(n_backtrack):
            try:
                J_stat_try = static_model_twin.solve_and_eval_J_stat(la_t + dla_t)
            except RuntimeError:
                dla_t *= 0.5  # no equilibrium, step less far
                continue
            e_try = r_OP_ref - static_model_twin.r_OP_eq()
            if np.linalg.norm(e_try) < e_n:
                break
            dla_t *= 0.5
        else:
            if verbose:
                print("  inverse statics: backtracking ran out of retries")
            break

        la_t = la_t + dla_t
        J_stat, e, e_n = J_stat_try, e_try, np.linalg.norm(e_try)
    else:
        print(
            f"inverse statics did not reach tol={tol:.1e} for r_OP_ref = "
            f"{r_OP_ref}, |e| = {e_n * 1e3:.6f} mm"
        )

    # leave the twin at the equilibrium of the tension we return
    static_model_twin.solve(la_t)
    return la_t


class InverseStaticsFeedforward:
    def init_feedforward(
        self,
        n_tendons,
        static_model_twin=None,
        model_factory=None,
        la_t_ff=None,
        g_accel=9.81,
        **twin_kwargs,
    ):
        self.n_tendons = n_tendons
        if static_model_twin is None and model_factory is not None:
            static_model_twin = StaticModelTwin(model_factory, g_accel=g_accel, **twin_kwargs)
        self.static_model_twin = static_model_twin
        self.la_t_ff_pts = None
        self.set_feedforward(la_t_ff)

    # ----- feedforward -----
    def la_t_ff(self, t):
        """Feedforward tendon tension at time t, shape (n_tendons,)."""
        return self._la_t_ff(t)

    def set_feedforward(self, la_t_ff):
        if la_t_ff is None:
            self._la_t_ff = lambda t: np.zeros(self.n_tendons)
        elif callable(la_t_ff):
            self._la_t_ff = la_t_ff
        else:
            const = np.asarray(la_t_ff, dtype=float).copy()
            assert const.shape == (self.n_tendons,)
            self._la_t_ff = lambda t: const
        self._on_feedforward_changed()

    def _on_feedforward_changed(self):
        """Hook for hosts that cache something depending on the feedforward."""


    # ----- build feedforward from inverse statics -----
    def inverse_statics(self, r_OP_ref, la_t0=None, **kwargs):
        assert (self.static_model_twin is not None), "no static model twin: pass model_factory or static_model_twin to init_feedforward"
        return inverse_statics(self.static_model_twin, r_OP_ref, la_t0=la_t0, **kwargs)

    def feedforward_from_setpoints(self, points, t_hold, la_t0=None, verbose=True, **kwargs):
        pts = []
        la_t_guess = la_t0
        if verbose:
            print("feedforward tensions (inverse statics, no clipping):")
        for i, r_OP_ref in enumerate(points):
            la_t_guess = self.inverse_statics(r_OP_ref, la_t0=la_t_guess, **kwargs)
            pts.append(la_t_guess.copy())
            if verbose:
                print(f"  {i}: {np.array2string(la_t_guess, precision=5)}")
        pts = np.array(pts)

        n = len(pts)
        self.set_feedforward(lambda t: pts[min(int(t // t_hold), n - 1)])
        self.la_t_ff_pts = pts
        return pts

    def feedforward_from_trajectory(self, r_OP_ref_fn, ts, la_t0=None, **kwargs):
        """Feedforward for a continuous reference, sampled at ``ts`` and interpolated."""
        ts = np.asarray(ts, dtype=float)
        pts = []
        la_t_guess = la_t0
        for ti in ts:
            la_t_guess = self.inverse_statics(r_OP_ref_fn(ti), la_t0=la_t_guess, **kwargs)
            pts.append(la_t_guess.copy())
        pts = np.array(pts)

        self.set_feedforward(
            lambda t: np.array(
                [np.interp(t, ts, pts[:, j]) for j in range(pts.shape[1])]
            )
        )
        self.la_t_ff_pts = pts
        return pts
