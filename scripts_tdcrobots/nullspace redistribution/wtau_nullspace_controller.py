"""Positive tendon forces by redistribution along the nullspace of W_tau itself.

    la_tau = la_tau_pos + B @ alpha,    W_tau @ B = 0
    =>  W_tau @ la_tau = W_tau @ la_tau_pos   for every alpha

so alpha is free to be chosen for positivity while the *generalized force fed to the system*
is untouched. This is a stronger invariance than the one `nullspace_controller.py` enforces:
that module uses B = null(J) with J = M_tilde_inv @ W_tau, which preserves only the commanded
tip acceleration (3 numbers), not the full nu-dimensional wrench.

WHY null(W_tau) IS NOT EMPTY
---------------------------
W_tau is (nu, n_tendons) = (66, 4), so full column rank is the generic expectation. This robot
is not generic. With tendons at phi = 0, 90, 180, 270 deg the straight configuration satisfies

    W_tau @ (1, -1, 1, -1) = 0     to machine precision (measured: 1.3e-18)

Tightening the 0/180 pair and slackening the 90/270 pair cancels moments *within* each pair and
cancels axial compression *between* the pairs. Measured singular values at q0:

    sigma(W_tau) = [2.827, 0.0618, 0.0618, 2.5e-17]

THE CATCH, AND HOW IT IS HANDLED
--------------------------------
The kernel is exact only at the straight configuration. Once the rod bends the tendon paths and
moment arms stop matching and sigma_4 lifts off zero:

    bent equilibrium (E) : sigma_4 = 2.9e-3     sigma_4/sigma_1 = 1.0e-3
    settled at B         : sigma_4 = 7.2e-3     sigma_4/sigma_1 = 2.4e-3

so strictly null(W_tau) = {0} there. B is therefore defined as the smallest right singular
vector of W_tau -- the *numerical* nullspace -- and the residual it leaks,

    ||W_tau @ (F - F_p)|| = |alpha| * sigma_4

is measured and reported rather than assumed away. Note that against sigma_1 the leak is 0.2%,
but against the lateral singular values sigma_2, sigma_3 ~ 0.08 it is ~9%, and lateral is the
bending authority that actually steers the tip. `wtau_nullspace_test.py` plots both.

WHAT THIS CAN AND CANNOT BUY
----------------------------
sum(v) ~ 1e-4 ~ 0 at every configuration: the null direction is an antagonistic pair swap, two
entries up and two down. So alpha raises two tendons only by lowering the other two, and cannot
lift all four off zero. Positivity is achieved where the deficit happens to sit in the tendons v
raises, and is unreachable otherwise. Nothing here clips: when the feasible set is empty the
allocation keeps W_tau @ F exact and takes the point maximising the smallest tendon force.

DERIVATIVES
-----------
ScipyDAE consumes la_tau_q / la_tau_u, so dv/dq is needed. v is the eigenvector of
G = W_tau^T W_tau for its smallest eigenvalue mu_n; standard first-order perturbation gives

    dv = sum_{k != n}  v_k (v_k . dG v) / (mu_n - mu_k)

which is exact as long as mu_n is simple. It is: the measured gap mu_{n-1}/mu_n is 534x at a
bent configuration. The near-degenerate pair mu_2 ~ mu_3 higher up is harmless because the sum
runs over the whole complement of v, and a sum over a degenerate subspace is basis-independent.
Validated against central differences to 2.3e-8.
"""

import sys
from pathlib import Path

import numpy as np
from typing import NamedTuple

# This module lives in a subfolder; dynamic_controller.py is one level up in scripts_tdcrobots.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_controller import DynamicControllerPD, _JbMixin


def _max_min_lam(F_p, v, tol=1e-12):
    """argmax_lam min_i (F_p + lam*v)_i.

    g(lam) = min_i(...) is concave and piecewise linear, so its maximum sits at a breakpoint of
    the lower envelope, i.e. where two of the lines cross. Enumerating all pairs is at most 6
    candidates for 4 tendons. Returns (lam, active) with active the crossing pair, or ().

    Kept local rather than imported from nullspace_controller.py so this folder stands alone;
    that file has moved between directories and the cross-folder import kept breaking.
    """
    n = len(v)
    best_lam, best_active = 0.0, ()
    best_g = np.min(F_p)
    for i in range(n):
        for j in range(i + 1, n):
            dv = v[i] - v[j]
            if abs(dv) < tol:
                continue
            lam = (F_p[j] - F_p[i]) / dv
            g = np.min(F_p + lam * v)
            if g > best_g:
                best_g, best_lam, best_active = g, lam, (i, j)
    return float(best_lam), best_active


class WAlloc(NamedTuple):
    """Everything la_tau, la_tau_q and la_tau_u need from one allocation."""

    F: np.ndarray  # (n,) applied tendon forces, = F_p + alpha * v
    F_p: np.ndarray  # (n,) un-redistributed damped-pinv solution
    v: np.ndarray  # (n,) unit nullspace direction of W_tau (zeros if degenerate)
    alpha: float
    a_lo: float
    a_hi: float
    active: tuple  # (), (k,) or (i, j): the constraints that pin alpha
    feasible: bool  # F.min() >= F_min actually holds
    degenerate: bool  # smallest eigenvalue of W^T W was not simple -> no shift
    sigma: np.ndarray  # (n,) singular values of W_tau
    V: np.ndarray  # (n, n) right singular vectors, columns
    leak_W: float  # ||W_tau @ (F - F_p)||, the wrench this redistribution is not free of
    leak_J: float  # ||J @ (F - F_p)||, its share visible in the tip acceleration
    W: np.ndarray  # (nu, n)
    J: np.ndarray  # (m, n)
    b: np.ndarray  # (m,)
    J_pinv: np.ndarray  # (n, m) damped right inverse, used for F_p
    S_inv: np.ndarray  # (m, m) inv(J J^T + inv_damping I)


def wtau_null_dir(W, gap_tol=1e-9):
    """Smallest right singular vector of W, plus the data needed to differentiate it.

    Returns (v, sigma, V, simple). `simple` is False when the smallest eigenvalue of W^T W is
    not separated from the next one, in which case v is not differentiable and the caller must
    not shift along it.

    The sign of v is pinned so diagnostics are continuous. It does not affect the output: under
    v -> -v the selected alpha flips too, leaving F, la_tau_q and la_tau_u bit-identical.
    """
    U, sigma, Vt = np.linalg.svd(W)
    V = Vt.T
    v = V[:, -1].copy()

    k_sign = 0 if abs(v[0]) > 0.1 else int(np.argmax(np.abs(v)))
    if v[k_sign] < 0.0:
        v = -v

    mu = sigma**2
    simple = bool((mu[-2] - mu[-1]) > gap_tol * max(mu[0], 1e-300))
    return v, sigma, V, simple


def _dv_dq(W, W_q, v, sigma, V):
    """dv/dq by first-order eigenvector perturbation of G = W^T W.

    Contracts with v before forming G_q, so the (n, n, nq) tensor is never built.
    """
    mu = sigma**2
    Wv = W @ v  # (nu,)
    Wqv = np.einsum("ajk,j->ak", W_q, v)  # (nu, nq)
    # dG v = (dW^T W + W^T dW) v, contracted with v on the right
    Gqv = np.einsum("aik,a->ik", W_q, Wv) + W.T @ Wqv  # (n, nq)

    # sum over the complement of v; signs of the V columns cancel (they appear quadratically)
    Vc = V[:, :-1]  # (n, n-1)
    denom = mu[-1] - mu[:-1]  # (n-1,), strictly negative
    return Vc @ ((Vc.T @ Gqv) / denom[:, None])


def wtau_alloc(J, b, W, F_min=0.0, inv_damping=1e-3, tol=1e-12, feas_tol=1e-9, gap_tol=1e-9):
    """Resolve the allocation with F >= F_min by shifting along the nullspace of W.

    J is (m, n), W is (nu, n). `F_min` is a scalar or a per-tendon (n,) floor -- the latter is
    what a non-uniform physical pretension needs, since the tension that must stay non-negative
    is la_pre + la_tau, so the floor on la_tau is -la_pre and differs per tendon.

    Returns a `WAlloc`.
    """
    m, n = J.shape
    F_min = np.broadcast_to(np.asarray(F_min, dtype=float), (n,))

    # --- particular solution: the damped right inverse used throughout dynamic_controller.py,
    # so F_p is exactly what the stock DynamicControllerPD would command ---
    S_inv = np.linalg.inv(J @ J.T + inv_damping * np.eye(m))
    J_pinv = J.T @ S_inv
    F_p = J_pinv @ b

    v, sigma, V, simple = wtau_null_dir(W, gap_tol=gap_tol)

    if not simple:
        return WAlloc(
            F=F_p.copy(), F_p=F_p, v=np.zeros(n), alpha=0.0, a_lo=-np.inf, a_hi=np.inf,
            active=(), feasible=bool((F_p >= F_min - feas_tol).all()), degenerate=True,
            sigma=sigma, V=V, leak_W=0.0, leak_J=0.0,
            W=W, J=J, b=b, J_pinv=J_pinv, S_inv=S_inv,
        )

    # --- feasible interval for F_p + alpha*v >= F_min ---
    idx = np.arange(n)
    big = np.abs(v) > tol
    pos, neg = v > tol, v < -tol
    ratio = np.where(big, (F_min - F_p) / np.where(big, v, 1.0), 0.0)

    if pos.any():
        k_lo = int(idx[pos][np.argmax(ratio[pos])])
        a_lo = float(ratio[k_lo])
    else:
        k_lo, a_lo = -1, -np.inf
    if neg.any():
        k_hi = int(idx[neg][np.argmin(ratio[neg])])
        a_hi = float(ratio[k_hi])
    else:
        k_hi, a_hi = -1, np.inf

    # --- pick alpha ---
    if a_lo <= a_hi:
        # Non-empty segment. Branch on a_lo <= a_hi, never on np.clip: clip silently returns the
        # upper bound when the interval is inverted.
        if a_lo > 0.0:
            alpha, active = a_lo, (k_lo,)
        elif a_hi < 0.0:
            alpha, active = a_hi, (k_hi,)
        else:
            alpha, active = 0.0, ()
    else:
        # Empty segment: keep W @ F as intended (no clipping) and take the least-negative point.
        # Maximise the smallest *margin* F - F_min, which for a scalar F_min is the same argmax
        # as maximising min F, but is the correct objective for a per-tendon floor.
        alpha, active = _max_min_lam(F_p - F_min, v, tol)

    F = F_p + alpha * v
    # Report feasibility from F itself: components with v_i ~ 0 are untouched by any alpha, so a
    # non-empty segment does not by itself imply F >= F_min.
    feasible = bool((F >= F_min - feas_tol).all())

    dF = F - F_p
    return WAlloc(
        F=F, F_p=F_p, v=v, alpha=float(alpha), a_lo=a_lo, a_hi=a_hi, active=active,
        feasible=feasible, degenerate=False, sigma=sigma, V=V,
        leak_W=float(np.linalg.norm(W @ dF)), leak_J=float(np.linalg.norm(J @ dF)),
        W=W, J=J, b=b, J_pinv=J_pinv, S_inv=S_inv,
    )


def achievable_floor(al, tol=1e-12):
    """max_alpha min_i (F_p + alpha*v)_i -- the largest uniform floor this configuration supports.

    Independent of F_min, so it is the ceiling on what W_tau-nullspace redistribution can deliver
    here. Negative means the 1-D redundancy is exhausted and no alpha makes all tendons positive.
    """
    if al.degenerate:
        return float(al.F_p.min())
    alpha, _ = _max_min_lam(al.F_p, al.v, tol)
    return float(np.min(al.F_p + alpha * al.v))


class WTauNullspaceController(_JbMixin, DynamicControllerPD):
    """PD computed-torque controller redistributing tendon forces along null(W_tau).

    Drop-in replacement for DynamicControllerPD -- same __init__ signature plus F_min and
    uniform_fallback.

    `uniform_fallback` adds max(0, F_min - min F) along the uniform direction 1 after the
    redistribution, which guarantees positivity but costs axial tracking error (J @ 1 is purely
    axial). Off by default so the pure null(W_tau) method is what gets measured.
    """

    def __init__(self, *args, F_min=0.5, la_pre=None, uniform_fallback=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.F_min = F_min
        # Physical pretension per tendon, as set on the RodTendonForce objects. The tension that
        # must stay non-negative is la_pre + la_tau, so pretension is headroom: it relaxes the
        # floor on la_tau to F_min - la_pre, per tendon. The controller already compensates the
        # pretension itself through sys.h, so this changes only the constraint, not the task.
        self.la_pre = (
            np.zeros(self.nla_tau) if la_pre is None
            else np.broadcast_to(np.asarray(la_pre, dtype=float), (self.nla_tau,)).copy()
        )
        self.uniform_fallback = uniform_fallback
        self._alloc_cache = None

    @property
    def floor(self):
        """Effective per-tendon floor on la_tau."""
        return self.F_min - self.la_pre

    def _alloc(self, t, q, u):
        # la_tau runs on every ScipyDAE residual evaluation and la_tau_q / la_tau_u then ask for
        # the same state again; the sys.h evaluation inside _Jb is expensive. Memoize the whole
        # allocation (cf. NullspaceShiftController._alloc).
        cache = self._alloc_cache
        if (
            cache is not None
            and t == cache[0]
            and np.array_equal(q, cache[1])
            and np.array_equal(u, cache[2])
        ):
            return cache[3]

        J, b = self._Jb(t, q, u)
        W = self.W_tau(t, q)
        al = wtau_alloc(J, b, W, F_min=self.floor, inv_damping=self.inv_damping)
        self._alloc_cache = (t, q.copy(), u.copy(), al)
        return al

    ## ----- Forces and control law -----

    def _uniform_shift(self, al):
        if not self.uniform_fallback:
            return 0.0
        return max(0.0, float(np.max(self.floor - al.F)))

    def la_tau(self, t, q, u):
        al = self._alloc(t, q, u)
        return al.F + self._uniform_shift(al)

    def _F_p_q(self, t, q, u, al, J_qk):
        """d F_p / d q for the damped right inverse.

        Same derivation as DynamicControllerPD.la_tau_q, except the einsum contracts with F_p
        rather than with the shifted force la_tau() returns.
        """
        s = al.S_inv @ al.b
        J_qkTs = np.einsum("ajk,a->jk", J_qk, s)  # (n, nq)
        J_qkla = np.einsum("ajk,j->ak", J_qk, al.F_p)  # (m, nq)
        J_pinv_qb = J_qkTs - al.J_pinv @ (J_qkla + al.J @ J_qkTs)
        return J_pinv_qb + al.J_pinv @ self._b_q(t, q, u)

    def la_tau_q(self, t, q, u):
        al = self._alloc(t, q, u)
        J_qk = self._J_qk(t, q)  # (m, n, nq)
        F_p_q = self._F_p_q(t, q, u, al, J_qk)

        if al.degenerate or not al.active:
            G = F_p_q  # alpha == 0, so F == F_p
        else:
            v_q = _dv_dq(al.W, self.W_tau_q(t, q), al.v, al.sigma, al.V)  # (n, nq)
            v, alpha = al.v, al.alpha
            if len(al.active) == 1:
                k = al.active[0]
                alpha_q = (-F_p_q[k] - alpha * v_q[k]) / v[k]
            else:
                i, j = al.active
                alpha_q = ((F_p_q[j] - F_p_q[i]) - alpha * (v_q[i] - v_q[j])) / (v[i] - v[j])
            G = F_p_q + alpha * v_q + np.outer(v, alpha_q)

        if self.uniform_fallback and self._uniform_shift(al) > 0.0:
            # s = F_min - min_i F_i, so ds/dq = -G[k, :] with k the locally fixed argmin
            k = int(np.argmax(self.floor - al.F))
            return G - np.ones((self.nla_tau, 1)) * G[k, :]
        return G

    def la_tau_u(self, t, q, u):
        al = self._alloc(t, q, u)
        F_p_u = al.J_pinv @ self._b_u(t, q, u)  # W_tau is independent of u, so v_u == 0

        if al.degenerate or not al.active:
            G = F_p_u
        else:
            v = al.v
            if len(al.active) == 1:
                k = al.active[0]
                alpha_u = -F_p_u[k] / v[k]
            else:
                i, j = al.active
                alpha_u = (F_p_u[j] - F_p_u[i]) / (v[i] - v[j])
            G = F_p_u + np.outer(v, alpha_u)

        if self.uniform_fallback and self._uniform_shift(al) > 0.0:
            k = int(np.argmax(self.floor - al.F))
            return G - np.ones((self.nla_tau, 1)) * G[k, :]
        return G


def replay_alloc(controller, sol):
    """Re-run the allocator over a solved trajectory.

    Replaying rather than instrumenting the solver means this cannot perturb the integration, and
    it works on a `sol` produced by any controller (cf. compute_la_ts in dynamic_control_test.py).
    """
    qDOF, uDOF = controller.qDOF, controller.uDOF
    la_pre = getattr(controller, "la_pre", np.zeros(controller.nla_tau))
    keys = ("t", "F_p", "F", "F_tot", "F_p_tot", "alpha", "a_lo", "a_hi", "feasible", "floor_max",
            "sigma", "sum_v", "v", "leak_W", "leak_J", "res_p", "WF", "WF_p")
    rec = {k: [] for k in keys}

    for t, q_sys, u_sys in zip(sol.t, sol.q, sol.u):
        q, u = q_sys[qDOF], u_sys[uDOF]
        al = controller._alloc(t, q, u)
        # What la_tau actually commands, which is al.F only when no uniform fallback is applied.
        # Reporting al.F directly would make the fallback invisible and understate the tension.
        F_cmd = controller.la_tau(t, q, u)
        rec["t"].append(t)
        rec["F_p"].append(al.F_p)
        rec["F"].append(F_cmd)
        # what the tendon physically carries: pretension plus the commanded increment
        rec["F_tot"].append(F_cmd + la_pre)
        rec["F_p_tot"].append(al.F_p + la_pre)
        rec["alpha"].append(al.alpha)
        rec["a_lo"].append(al.a_lo)
        rec["a_hi"].append(al.a_hi)
        rec["feasible"].append(al.feasible)
        rec["floor_max"].append(achievable_floor(al))
        rec["sigma"].append(al.sigma)
        rec["sum_v"].append(float(al.v.sum()))
        rec["v"].append(al.v)
        # Measure the leak against what is actually commanded. Under the uniform fallback the
        # extra s*1 term is NOT in the nullspace, so it belongs in this number rather than
        # being quietly excluded by reporting al.leak_W.
        dF = F_cmd - al.F_p
        rec["leak_W"].append(float(np.linalg.norm(al.W @ dF)))
        rec["leak_J"].append(float(np.linalg.norm(al.J @ dF)))
        # the damped-pinv residual that was already there before any redistribution
        rec["res_p"].append(float(np.linalg.norm(al.J @ al.F_p - al.b)))
        rec["WF"].append(al.W @ F_cmd)
        rec["WF_p"].append(al.W @ al.F_p)

    return {k: np.asarray(val) for k, val in rec.items()}
