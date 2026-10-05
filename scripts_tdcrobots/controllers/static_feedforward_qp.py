# box-constrained (la_t + dla_t >= 0) variant of inverse_statics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import lsq_linear

from controllers.static_feedforward import InverseStaticsFeedforward, StaticModelTwin


def inverse_statics_qp(
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
    n = static_model_twin.n_tendons
    la_t = np.zeros(n) if la_t0 is None else la_t0.copy()

    J_stat = static_model_twin.solve_and_eval_J_stat(la_t)
    e = r_OP_ref - static_model_twin.r_OP_eq()
    e_n = np.linalg.norm(e)

    for k in range(max_iter):
        if verbose:
            print(f"  inverse statics (QP) it {k:2d}: |e| = {e_n * 1e3:9.6f} mm")
        if e_n < tol:
            break

        # min ||J_stat x - e||^2 + damping ||x||^2 with lo <= x <= hi
        A = np.vstack([J_stat, np.sqrt(damping) * np.eye(n)])
        b = np.concatenate([e, np.zeros(n)])
        lo = np.maximum(-la_t, -max_step)
        hi = np.full(n, max_step)
        dla_t = lsq_linear(A, b, bounds=(lo, hi)).x

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
                print("  inverse statics (QP): backtracking ran out of retries")
            break

        la_t = la_t + dla_t
        J_stat, e, e_n = J_stat_try, e_try, np.linalg.norm(e_try)
    else:
        print(
            f"inverse statics (QP) did not reach tol={tol:.1e} for r_OP_ref = "
            f"{r_OP_ref}, |e| = {e_n * 1e3:.6f} mm"
        )

    # leave the twin at the equilibrium of the tension we return
    static_model_twin.solve(la_t)
    return la_t


class InverseStaticsFeedforwardQP(InverseStaticsFeedforward):
    """InverseStaticsFeedforward with the Gauss-Newton step routed through inverse_statics_qp."""

    def inverse_statics(self, r_OP_ref, la_t0=None, **kwargs):
        assert (
            self.static_model_twin is not None
        ), "no static model twin: pass model_factory or static_model_twin to init_feedforward"
        return inverse_statics_qp(self.static_model_twin, r_OP_ref, la_t0=la_t0, **kwargs)


if __name__ == "__main__":
    from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL

    names = ["A", "B", "C", "D", "E"]
    points = [SETPOINT_TABLE[n] for n in names]

    for cls, label in [(InverseStaticsFeedforward, "unconstrained"), (InverseStaticsFeedforwardQP, "QP box-constrained")]:
        host = cls()
        host.init_feedforward(4, model_factory=CommonModel, g_accel=G_ACCEL)
        twin = host.static_model_twin
        pts = host.feedforward_from_setpoints(points, t_hold=2.0, verbose=False)

        errs = []
        for name, la_t in zip(names, pts):
            twin.solve(la_t)
            errs.append(np.linalg.norm(twin.r_OP_eq() - SETPOINT_TABLE[name]))
        errs = np.array(errs)

        print(label)
        print("  min tension [N]:", pts.min(), " max tension [N]:", pts.max())
        print("  tip error max [mm]:", errs.max() * 1e3, " mean [mm]:", errs.mean() * 1e3)
        for name, la_t, err in zip(names, pts, errs):
            print(f"  {name}: la_t = {np.array2string(la_t, precision=4)}, err = {err * 1e3:.4f} mm")
