r"""Reusable QP positivity projection: take any controller's la_tau, return a nonnegative one.

    min_x  1/2 || W_tau @ x - W_tau @ la_tau ||^2      s.t.  x >= 0

`QPWTauProjection` is a mixin. It needs only four things from its host controller:

    W_tau(t, q)          (nu, n) force-direction matrix
    W_tau_q(t, q)        (nu, n, nq) its configuration derivative
    la_tau / la_tau_q / la_tau_u     the UNCONSTRAINED control law, reached via super()
    nla_tau, _nq, _nu

Both `DynamicControllerPD` and `StaticControllerLi2023` supply all of these, so the same
projection serves both -- which is the point of splitting it out of `qp_wtau_controller.py`.

    QPWTauController  (qp_wtau_controller.py)  = QPWTauProjection + DynamicControllerPD
    QPStaticController        (this file)      = QPWTauProjection + StaticControllerLi2023

THE TWO HOSTS ARE NOT SYMMETRIC
-------------------------------
`DynamicControllerPD` computes la_tau algebraically from the current state, so projecting its
output is a pure post-process with no side effects.

`StaticControllerLi2023` is different in kind: the tendon tensions ARE its generalized
coordinates (`self.nq = len(tendons)`), integrated by the solver through

    q_dot = Gamma_inv @ outer_loop(t, q, u)

and `la_tau` merely reads that state, `q[:nq] + la_t_ff(t)`. Projecting the OUTPUT of an
integrator while the STATE keeps integrating is textbook windup: q[:nq] wanders far negative
while the applied force sits pinned at zero, and when the error reverses the controller spends
a long time unwinding before anything moves. That is exactly the failure the existing
`_free()` anti-windup mask and `step_callback` clipping were written to avoid.

`QPStaticController` therefore also does back-calculation (`back_calculate=True`): after each
accepted step it writes the projected force back into the state, so state and output never
diverge. Without that the projection is not safe on this host.

WHY BOTHER, GIVEN THE STATIC CONTROLLER ALREADY CLIPS?
------------------------------------------------------
`la_t_min=0.0` already forces nonnegativity by clipping each tendon independently -- that is a
Euclidean projection onto the orthant. The QP projects in the W_tau^T W_tau metric instead,
which knows the tendon force directions are 99.8% correlated and therefore trades tension
between tendons rather than truncating each one in isolation. To actually use it, construct
the host with `la_t_min=-np.inf` and let the QP own positivity.
"""

import contextlib

import numpy as np
from scipy.optimize import nnls


class QPWTauProjection:
    """Project the host's la_tau onto {x >= 0} in the W_tau metric.

    Mix in BEFORE the host controller: `class C(QPWTauProjection, HostController)`.
    """

    def __init__(self, *args, qp_tol=1e-8, qp_reg=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.qp_tol = qp_tol  # a tendon is "free" (not clamped) above this force
        self.qp_reg = qp_reg  # Tikhonov anchor, relative to ||W_tau||; see _solve
        self._baseline_mode = False
        self._qp_cache = None

    ## ----- hooks: the host's UNCONSTRAINED control law -----

    def _unconstrained_la_tau(self, t, q, u):
        return super().la_tau(t, q, u)

    def _unconstrained_la_tau_q(self, t, q, u):
        return super().la_tau_q(t, q, u)

    def _unconstrained_la_tau_u(self, t, q, u):
        return super().la_tau_u(t, q, u)

    @contextlib.contextmanager
    def _as_baseline(self):
        """Make `self.la_tau` resolve to the unconstrained law inside the block.

        Only hosts whose la_tau_q calls `self.la_tau(...)` internally need this --
        DynamicControllerPD does (dynamic_controller.py:111), StaticControllerLi2023 does not.
        Harmless either way.
        """
        was, self._baseline_mode = self._baseline_mode, True
        try:
            yield
        finally:
            self._baseline_mode = was

    ## ----- the QP -----

    def _solve(self, t, q, u):
        """Solve the QP at (t, q, u), memoized on the exact argument triple.

        W_tau is NOT always full column rank. At the straight configuration the tendons at
        0/90/180/270 deg satisfy W_tau @ (1,-1,1,-1) = 0 exactly (measured sigma =
        [2.827, 0.0618, 0.0618, 0.0]), so the plain objective is only positive SEMI-definite,
        the minimizer is not unique, and nnls jumps between equivalent solutions under
        infinitesimal perturbation -- the derivative genuinely does not exist. The rod starts
        straight, so this is hit at t = 0 of every run.

        Solving the anchored problem instead,

            min_x ||W_tau (x - la_tau_real)||^2 + eps ||x - la_tau_real||^2   s.t. x >= 0

        restores strict convexity everywhere: the Hessian is W_tau^T W_tau + eps*I >- 0
        regardless of rank. Stacking [W_tau; sqrt(eps) I] against [c; sqrt(eps) la_tau_real]
        turns it back into a plain NNLS. Among tied minimizers this picks the one closest to
        the unconstrained command, which is the tie-break we want anyway.

        Returns (W_tau, A, x, F, c, la_tau_real, sqrt_eps): W_tau is the bare force map (needed
        by la_tau_u), A and c are the augmented pair the derivatives differentiate.
        """
        key = (t, self._baseline_mode, q.tobytes(), u.tobytes())
        if self._qp_cache is not None and self._qp_cache[0] == key:
            return self._qp_cache[1]

        la_tau_real = self._unconstrained_la_tau(t, q, u)
        W_tau = self.W_tau(t, q)

        n = self.nla_tau
        sqrt_eps = np.sqrt(self.qp_reg * max(np.linalg.norm(W_tau, 2), 1e-300))
        A = np.vstack([W_tau, sqrt_eps * np.eye(n)])
        c = np.concatenate([W_tau @ la_tau_real, sqrt_eps * la_tau_real])
        x, _ = nnls(A, c)

        out = (W_tau, A, x, np.where(x > self.qp_tol)[0], c, la_tau_real, sqrt_eps)
        self._qp_cache = (key, out)
        return out

    def _sensitivity(self, t, q, u, dA, dc):
        """Differentiate the active-set solution. `dA` is (nu+n, n, k) or None.

        Both arguments refer to the AUGMENTED pair from `_solve`, not the bare W_tau.
        """
        _, A, x, F, c, _, _ = self._solve(t, q, u)
        out = np.zeros((self.nla_tau, dc.shape[1]))
        if len(F) == 0:
            return out

        A_F = A[:, F]
        A_F_pinv = np.linalg.pinv(A_F)

        if dA is None:
            out[F, :] = A_F_pinv @ dc
            return out

        dA_F = dA[:, F, :]
        r = c - A_F @ x[F]
        resolve = A_F_pinv @ (dc - np.einsum("afk,f->ak", dA_F, x[F]))
        residual = (A_F_pinv @ A_F_pinv.T) @ np.einsum("afk,a->fk", dA_F, r)
        out[F, :] = resolve + residual
        return out

    ## ----- actuator interface -----

    def la_tau(self, t, q, u):
        if self._baseline_mode:
            return super().la_tau(t, q, u)
        return self._solve(t, q, u)[2]

    def la_tau_q(self, t, q, u):
        if self._baseline_mode:
            return super().la_tau_q(t, q, u)
        W_tau, _, _, _, _, la_tau_real, sqrt_eps = self._solve(t, q, u)
        dW_tau = self.W_tau_q(t, q)
        dla = self._unconstrained_la_tau_q(t, q, u)
        # augmented A = [W_tau; sqrt_eps I]: the identity block is constant, so it contributes
        # nothing to dA, but it does contribute sqrt_eps * dla to dc.
        dA = np.concatenate([dW_tau, np.zeros((self.nla_tau,) + dW_tau.shape[1:])], axis=0)
        dc = np.vstack(
            [np.einsum("ijk,j->ik", dW_tau, la_tau_real) + W_tau @ dla, sqrt_eps * dla]
        )
        return self._sensitivity(t, q, u, dA, dc)

    def la_tau_u(self, t, q, u):
        if self._baseline_mode:
            return super().la_tau_u(t, q, u)
        W_tau, _, _, _, _, _, sqrt_eps = self._solve(t, q, u)
        dla = self._unconstrained_la_tau_u(t, q, u)
        # W_tau has no velocity dependence, so dA = 0 and only dc survives.
        return self._sensitivity(t, q, u, None, np.vstack([W_tau @ dla, sqrt_eps * dla]))


def make_static_qp_controller(base_cls):
    """Build a QP-projected static controller on top of `base_cls`.

    A factory rather than a plain class because the static controller exists in two variants
    (`static_controller_li2023.StaticControllerLi2023` and the Newton twin in
    `new static controller/`), and which one you want depends on the study.
    """

    class QPStaticController(QPWTauProjection, base_cls):
        __doc__ = f"""{base_cls.__name__} with its tendon tensions projected onto {{x >= 0}}.

        Construct the host with `la_t_min=-np.inf` so the QP owns positivity instead of the
        per-tendon clipping. `back_calculate=True` keeps the integrator state equal to the
        projected force, which is what stops windup -- see the module docstring.
        """

        def __init__(self, *args, back_calculate=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.back_calculate = back_calculate

        def step_callback(self, t, q, u):
            q, u = super().step_callback(t, q, u)
            if self.back_calculate:
                # The state IS the commanded tension. If the QP moved the applied force, move
                # the state with it, otherwise the integrator winds up against a clamp it
                # cannot see. la_tau = q[:nq] + la_t_ff(t), so invert that.
                x = self._solve(t, q, u)[2]
                q = q.copy()
                q[: self.nq] = x - self.la_t_ff(t)
                self._qp_cache = None  # state changed underneath the memo
            return q, u

    return QPStaticController
