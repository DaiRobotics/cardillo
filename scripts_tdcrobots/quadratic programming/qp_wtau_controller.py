r"""Positive tendon forces by QP projection in the W_tau metric, with a zero floor.

The single-purpose version of `qp_positive_controller.py`: no metric switch, no f_min, no
SVD-based reweightings. Just the formulation as originally posed --

    min_x  1/2 || W_tau @ x - W_tau @ la_tau ||^2      s.t.  x >= 0

where `la_tau` is whatever `DynamicControllerPD` already commands (feedback linearization plus
the PD outer loop). The control law is untouched; only the final allocation is repaired.

WHY THIS IS THE ONE TO USE
--------------------------
Of the four metrics trialled, only this one integrates a full 10 s A->E sweep. Measured against
the unconstrained baseline over that run:

    baseline        min force -1.3867 N,  20.1% of samples negative,  settled error 0.000 mm
    this controller min force  0.0000 N,   0.0% of samples negative,  settled error 0.509 mm

so positivity costs about half a millimetre of settled tracking error. The QP is inactive 79.8%
of the time -- when the PD output is already nonnegative, `nnls` returns it unchanged and this
class is bit-for-bit identical to `DynamicControllerPD`.

HOW IT IS SOLVED
----------------
With f_min = 0 the box QP *is* a nonnegative least squares problem, no shifting needed:

    min ||W_tau y - c||  s.t. y >= 0,     c = W_tau @ la_tau

`scipy.optimize.nnls` solves it exactly (Lawson-Hanson active set, finite termination). W_tau is
(nu, 4) with full column rank, so the objective is strictly convex and the solution is unique --
it is the metric projection of la_tau onto the nonnegative orthant in the W_tau^T W_tau metric.

DERIVATIVES
-----------
ScipyDAE consumes la_tau_q / la_tau_u. Write W_F = W_tau[:, F] for the columns belonging to the
tendons that are still pulling. The KKT conditions of the QP are

    [W_tau^T (W_tau x - c)]_F = 0        free tendons: the cost gradient vanishes
    [W_tau^T (W_tau x - c)]_C >= 0       clamped tendons: pulling harder cannot help

and since x_C = 0, the free block is exactly the normal equations W_F^T W_F x_F = W_F^T c.
Differentiating those at fixed active set, with r = c - W_F x_F:

    dx_F = (W_F^T W_F)^-1 [ (dW_F)^T r + W_F^T ( dc - (dW_F) x_F ) ]

Clamped tendons sit at exactly zero and have identically zero derivative. This is exact almost
everywhere; the map is piecewise smooth and the Jacobian genuinely jumps where the active set
changes. Verified against central differences to ~1e-9 (q) and ~1e-10 (u).

USAGE
-----
Drop-in for DynamicControllerPD -- same constructor arguments::

    from qp_wtau_controller import QPWTauController

    controller = QPWTauController(
        system, rod, tendons, r_OP_ref_fn,
        v_P_ref_fn=v_P_ref_fn, a_P_ref_fn=a_P_ref_fn,
        Kp=200.0, Kd=20.0, inv_damping=1e-3,
    )
    system.add(controller)
"""

import sys
from pathlib import Path

# dynamic_controller.py lives one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_controller import DynamicControllerPD
from qp_projection import QPWTauProjection


class QPWTauController(QPWTauProjection, DynamicControllerPD):
    """DynamicControllerPD with its output projected onto {x >= 0} in the W_tau metric.

    All the projection maths lives in `QPWTauProjection` (qp_projection.py), shared with the
    static controller. The only thing specific to this host is the `_as_baseline` guard below.
    """

    ## ----- keeping the inherited control law self-consistent -----

    def _unconstrained_la_tau_q(self, t, q, u):
        """The parent's own d(la_tau_real)/dq, evaluated as if the QP did not exist.

        `DynamicControllerPD.la_tau_q` calls `self.la_tau(t, q, u)` internally
        (dynamic_controller.py:111) and feeds the result into the derivative of the damped
        pseudo-inverse, where it stands for the algebraic quantity J^T S^-1 b. That identity
        holds only for the unconstrained solution. Without the guard the projected force flows
        into it and la_tau_q is wrong by up to ~20% -- exactly in the partially-clamped states
        where the constraint is doing its work.
        """
        with self._as_baseline():
            return super()._unconstrained_la_tau_q(t, q, u)

    def _unconstrained_la_tau_u(self, t, q, u):
        with self._as_baseline():
            return super()._unconstrained_la_tau_u(t, q, u)
