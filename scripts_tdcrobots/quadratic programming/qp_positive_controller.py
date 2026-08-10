"""Repair the PD/feedback-linearization allocation into a positive one with a QP.

THE PROBLEM
-----------
`DynamicControllerPD.la_tau` allocates tendon tensions with a damped right inverse

    J      = M_tilde_inv @ W_tau                              (3, 4)
    J_inv  = J.T @ inv(J @ J.T + inv_damping * eye(3))
    la_tau = J_inv @ (a + y_0_ddot)

which is sign-indifferent, so tendons routinely come out negative -- a tendon pushing.

THE REPAIR
----------
Leave the control law completely alone and add a projection step afterwards: find the
nearest force that the actuators can actually produce.

    min_x  1/2 ||A (x - la_tau)||^2      s.t.  x >= f_min

`A` picks what "nearest" means:

  * metric="W_tau" (default) -- A = W_tau, the (nu, 4) generalized-force map. This is the
    formulation as requested: match W_tau @ x to W_tau @ la_tau. Since W_tau has full column
    rank the map x -> W_tau x is injective, so this is a metric projection of la_tau onto the
    box in the W_tau^T W_tau metric. It tries to preserve the ENTIRE generalized force vector
    and therefore does NOT exploit the 1-D tendon nullspace that leaves tip acceleration
    unchanged. Conservative, and slightly more lossy on the task than it strictly has to be.

  * metric="J" -- A = J, the (3, 4) task map. Only the tip acceleration is preserved, so the
    nullspace direction is free and the projection is strictly less lossy on the task. This is
    essentially the `nnls(J, b)` path commented out at dynamic_controller.py:103, but anchored
    at the PD output instead of re-solving from b.

Both are implemented so the cost of the stricter W_tau formulation is measurable.

SOLUTION METHOD
---------------
Substituting y = x - f_min * 1 >= 0 and c = A @ (la_tau - f_min * 1) turns the box QP into a
plain NNLS, min ||A y - c|| s.t. y >= 0, which scipy solves exactly (active set, finite
termination). No new dependency -- nnls is already what dynamic_controller.py uses.

WHEN NO CONSTRAINT BINDS
------------------------
If the PD output is already >= f_min then y = la_tau - f_min * 1 is feasible and attains
residual zero, so the QP returns x == la_tau exactly and the derivatives below collapse to the
parent's. This controller is bit-identical to the unconstrained baseline whenever the baseline
was already admissible; it only acts where the baseline was infeasible.
"""

import contextlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

# dynamic_controller.py and its dependencies live one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_controller import DynamicControllerPD, _JbMixin


METRICS = ("W_tau", "J", "whitened", "axial")


def metric_matrix(W_tau, J, metric, damping=1e-3):
    """The matrix A defining what "nearest" means, and which derivative rule it needs.

    The two reweighted metrics exist because W_tau and J are near rank 1 for this robot (column
    correlations > 0.998, sigma = [650.7, 13.2, 13.2]). Under those metrics the QP spends its
    whole budget on the axial direction -- the one direction positivity has already made
    unreachable -- and discards the bending information. De-weighting u_ax fixes that:

      "whitened" : all three task directions weighted equally. The natural writing is
                   A = diag(1/sigma) U^T J = V^T, but sigma_2 == sigma_3 exactly here, so U's
                   lateral columns are not unique and neither is that A. The objective is
                   though: A^T A = J^T (J J^T)^-1 J =: P, the orthogonal projector onto
                   row(J). P is symmetric idempotent, so P^T P = P and A = P reproduces the
                   same QP while being SVD-free -- and therefore exactly differentiable.

                   J does go rank deficient in closed loop (sigma_3 -> 0 along the trajectory,
                   an outright LinAlgError on the undamped form), so S carries the same
                   Tikhonov term the controller already uses for its damped right inverse.
                   That costs idempotence -- the objective becomes ||P_d (x - la_ref)||^2,
                   weighting each row direction by (sigma^2/(sigma^2+d))^2 -- which is if
                   anything the more sensible metric: it fades out directions the robot has
                   nearly lost authority over instead of amplifying them.
      "axial"    : A = (I - u_ax u_ax^T) J, the axial direction dropped outright. This is
                   ClarkeShiftController's projection expressed as a QP. u_ax is the top left
                   singular vector; sigma_1 is well separated (650.7 vs 13.2) so u_ax itself
                   is unique, but its derivative is not implemented -- ANALYSIS ONLY, see
                   _dA_dq.

    Returns (A, kind) where kind tells _dA_dq which derivative rule applies.
    """
    if metric == "W_tau":
        return W_tau, "W_tau"
    if metric == "J":
        return J, "J"
    if metric == "whitened":
        S = J @ J.T + damping * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(S, J), "rowproj"
    u_ax = np.linalg.svd(J)[0][:, 0]
    return J - np.outer(u_ax, u_ax) @ J, "frozen"


def d_rowproj(J, J_qk, damping=1e-3):
    """Exact (4, 4, nq) derivative of P = J^T S^-1 J with S = J J^T + damping * I.

    With M = S^-1 J and D = dJ/dq_k, and using that the damping term is constant,

        dS = D J^T + J D^T,     dP = D^T M + M^T D - M^T (dS) M
    """
    M = np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), J)
    DJT = np.einsum("ajk,bj->abk", J_qk, J)
    dS = DJT + DJT.transpose(1, 0, 2)
    return (
        np.einsum("ajk,al->jlk", J_qk, M)
        + np.einsum("aj,alk->jlk", M, J_qk)
        - np.einsum("aj,abk,bl->jlk", M, dS, M)
    )


def solve_positive_qp(A, la_ref, f_min=0.0, tol=1e-8):
    """Solve  min ||A (x - la_ref)||  s.t.  x >= f_min  via the shifted NNLS.

    Returns
    -------
    x : (n,)   the projected, feasible force
    F : (k,)   index array of the free set, i.e. components strictly above the bound. These are
               the components the derivatives are allowed to move; the rest are clamped.
    c : (m,)   the shifted right-hand side A @ (la_ref - f_min), needed by the derivatives.
    """
    c = A @ (la_ref - f_min)
    y, _ = nnls(A, c)
    return y + f_min, np.where(y > tol)[0], c


class QPPositiveController(_JbMixin, DynamicControllerPD):
    """DynamicControllerPD with its output projected onto {x >= f_min} by a QP.

    Everything about the control law -- W_tau, W_tau_q, M_tilde_inv, the feedback
    linearization, the PD outer loop -- is inherited unchanged from DynamicControllerPD. Only
    the final allocation is post-processed.
    """

    def __init__(self, *args, metric="W_tau", f_min=0.0, qp_tol=1e-8, bypass=False, **kwargs):
        super().__init__(*args, **kwargs)
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.metric = metric
        self.f_min = f_min
        self.qp_tol = qp_tol
        # bypass=True makes this class behave exactly as DynamicControllerPD, so a baseline run
        # and a QP run can share one model. Used by the replay analysis: simulate with the QP
        # off, then project the resulting trajectory offline.
        self.bypass = bypass
        self._qp_cache = None

    ## ----- QP solve, cached -----

    @contextlib.contextmanager
    def _as_baseline(self):
        """Make `self.la_tau` resolve to the unconstrained law for the duration of the block.

        `DynamicControllerPD.la_tau_q` calls `self.la_tau(t, q, u)` internally
        (dynamic_controller.py:111) and uses the result in the derivative of the damped
        pseudo-inverse. Under normal method resolution that call would land on the OVERRIDE
        below and feed the QP-projected force into a formula that is only valid for the
        unconstrained one -- a silent inconsistency worth ~15% of la_tau_q. Suppressing the
        override while the parent runs keeps the parent's formula self-consistent.
        """
        was, self.bypass = self.bypass, True
        try:
            yield
        finally:
            self.bypass = was

    def _base_la_tau_q(self, t, q, u):
        with self._as_baseline():
            return super().la_tau_q(t, q, u)

    def _base_la_tau_u(self, t, q, u):
        with self._as_baseline():
            return super().la_tau_u(t, q, u)

    def _solve(self, t, q, u):
        """Solve the QP at (t, q, u), memoized.

        ScipyDAE calls la_tau, la_tau_q and la_tau_u with identical arguments within one
        residual evaluation, and each of those needs the unconstrained anchor -- which costs a
        full sys.h. DynamicControllerPD has no cache of its own (DynamicControllerPID does),
        so cache here on the exact argument triple.
        """
        key = (t, self.bypass, self.f_min, q.tobytes(), u.tobytes())
        if self._qp_cache is not None and self._qp_cache[0] == key:
            return self._qp_cache[1]

        la_ref = super().la_tau(t, q, u)  # the unconstrained PD allocation
        W_tau = self.W_tau(t, q)
        A, T = metric_matrix(
            W_tau, self.M_tilde_inv @ W_tau, self.metric, self.inv_damping
        )
        x, F, c = solve_positive_qp(A, la_ref, self.f_min, self.qp_tol)

        out = (A, T, x, F, c, la_ref)
        self._qp_cache = (key, out)
        return out

    def _dA_dq(self, t, q, kind):
        """(m, nla_tau, nq) derivative of the metric matrix w.r.t. q.

        M_tilde_inv is frozen after the first call (dynamic_controller.py:38-44), so it
        contributes no term -- the same convention the parent's la_tau_q already uses.

        "frozen" (metric="axial") holds u_ax constant, which is NOT a valid derivative: u_ax
        rotates with q and --check measures relative errors of 0.3-60 for it. That metric is
        for --replay and --check only; do not put it in a closed loop.
        """
        if kind == "W_tau":
            return self.W_tau_q(t, q)
        J_qk = self._J_qk(t, q)
        if kind == "rowproj":
            return d_rowproj(
                self.M_tilde_inv @ self.W_tau(t, q), J_qk, self.inv_damping
            )
        return J_qk

    ## ----- Forces and control law -----

    def la_tau(self, t, q, u):
        if self.bypass:
            return super().la_tau(t, q, u)
        return self._solve(t, q, u)[2]

    def _sensitivity(self, t, q, u, dA, dc):
        """Differentiate the active-set solution of the QP.

        Holding the active set fixed (valid almost everywhere), the free block solves the
        unconstrained normal equations  A_F^T A_F y_F = A_F^T c, so with M = A_F^T A_F and
        residual r = c - A_F y_F,

            dy_F = M^-1 [ (dA_F)^T r + A_F^T (dc - (dA_F) y_F) ]

        Clamped components sit exactly at f_min and have identically zero derivative.

        `dA` is (m, nla_tau, k) or None when A does not depend on the variable; `dc` is (m, k).
        """
        A, _, x, F, c, _ = self._solve(t, q, u)
        out = np.zeros((self.nla_tau, dc.shape[1]))
        if len(F) == 0:
            return out

        A_F = A[:, F]
        y_F = x[F] - self.f_min
        # A_F^+ rather than inv(A_F^T A_F) A_F^T: with metric="J" the Gram matrix is singular
        # whenever the free set outgrows rank(J) = 3. NNLS generically returns at most 3
        # nonzeros for a 3-row system, but ties make that "generically" and not "always".
        # For full column rank the two agree exactly, as does A_F^+ (A_F^+)^T = M^-1 below.
        A_F_pinv = np.linalg.pinv(A_F)

        if dA is None:
            out[F, :] = A_F_pinv @ dc
            return out

        dA_F = dA[:, F, :]
        r = c - A_F @ y_F
        out[F, :] = A_F_pinv @ (dc - np.einsum("afk,f->ak", dA_F, y_F)) + (
            A_F_pinv @ A_F_pinv.T
        ) @ np.einsum("afk,a->fk", dA_F, r)
        return out

    def la_tau_q(self, t, q, u):
        if self.bypass:
            return super().la_tau_q(t, q, u)
        A, T, _, _, _, la_ref = self._solve(t, q, u)
        dA = self._dA_dq(t, q, T)
        # c = A(q) @ (la_ref(q, u) - f_min)
        dc = np.einsum("ijk,j->ik", dA, la_ref - self.f_min) + A @ self._base_la_tau_q(t, q, u)
        return self._sensitivity(t, q, u, dA, dc)

    def la_tau_u(self, t, q, u):
        if self.bypass:
            return super().la_tau_u(t, q, u)
        A = self._solve(t, q, u)[0]
        # A depends on q only, so the only u-dependence enters through the anchor la_ref.
        return self._sensitivity(t, q, u, None, A @ self._base_la_tau_u(t, q, u))


def replay_qp(controller, sol, metric=None):
    """Per-sample QP projection over an existing solution, with the induced errors.

    Post-processes a baseline trajectory without re-simulating: what the unconstrained law
    asked for, what the QP would have delivered, and what that costs. The task error is split
    into axial and lateral parts along the dominant left-singular direction of J, matching
    `minimal_axial_controller.replay_shift` so the numbers are directly comparable.
    """
    qDOF, uDOF = controller.qDOF, controller.uDOF
    metric = metric or controller.metric
    keys = ("t", "la_ref", "la_qp", "e_gen", "e_task", "e_ax", "e_lat", "n_clamped")
    out = {k: [] for k in keys}

    for t, q_sys, u_sys in zip(sol.t, sol.q, sol.u):
        q, u = q_sys[qDOF], u_sys[uDOF]
        la_ref = DynamicControllerPD.la_tau(controller, t, q, u)
        W_tau = controller.W_tau(t, q)
        J = controller.M_tilde_inv @ W_tau
        A, _ = metric_matrix(W_tau, J, metric)
        x, F, _ = solve_positive_qp(A, la_ref, controller.f_min, controller.qp_tol)

        d = x - la_ref
        err = J @ d
        u_ax = np.linalg.svd(J)[0][:, 0]
        e_ax = float(err @ u_ax)

        out["t"].append(t)
        out["la_ref"].append(la_ref)
        out["la_qp"].append(x)
        out["e_gen"].append(np.linalg.norm(W_tau @ d))
        out["e_task"].append(np.linalg.norm(err))
        out["e_ax"].append(abs(e_ax))
        out["e_lat"].append(np.linalg.norm(err - e_ax * u_ax))
        out["n_clamped"].append(controller.nla_tau - len(F))

    return {k: np.asarray(v) for k, v in out.items()}
