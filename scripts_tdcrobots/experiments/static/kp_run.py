# static controller runs at fixed Kp values, instability metrics and vtk export
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time

import numpy as np

from experiments.static.static_controller_test_main import simulate, applied_tensions

NAMES = ["A", "B", "C", "D", "E"]
T_HOLD = 5.0


def window_ptp(t, y, t0, t1):
    """peak to peak of y on [t0, t1], nan if the window is empty."""
    mask = (t >= t0) & (t <= t1)
    return float(np.ptp(y[mask])) if mask.sum() > 2 else float("nan")


if __name__ == "__main__":
    # ---- parameters ----
    Kp_list = [8.0]
    # Kp_list = [2.0, 4.0, 6.0, 7.0, 8.0, 10.0]
    t_sim = 2.0  # unstable runs make the adaptive solver slow, keep t_sim short
    fps = 30
    export_vtk = True

    results = []
    for Kp in Kp_list:
        t_wall = time.time()
        sol, controller, rod, refs, model = simulate(
            NAMES,
            smooth=False,
            t_hold=T_HOLD,
            t_sim=t_sim,
            dt=1e-3,
            Kp=Kp,
            Kd=0.0,
            damping_ratio=0.1,
            use_feedforward=False,
            start_point="E",
            variant="plain",
            verbose=True,
        )
        wall = time.time() - t_wall

        t = sol.t
        r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
        r_ref = np.array([refs["r"](ti) for ti in t])
        e = np.linalg.norm(r_OP - r_ref, axis=1)
        la_ts = applied_tensions(controller, sol)

        # solver gave up before t_sim -> blow up
        t_end = t[-1]
        status = "completed" if t_end >= t_sim - 1e-6 else "solver_died"

        # oscillation growth inside the last completed hold: ptp of the error, late vs early half
        hold_idx = int(min(t_end, t_sim) // T_HOLD - (1 if t_end % T_HOLD < 0.5 else 0))
        hold_idx = max(hold_idx, 0)
        h0 = hold_idx * T_HOLD + 0.5  # skip the step transient
        h1 = min((hold_idx + 1) * T_HOLD, t_end)
        mid = 0.5 * (h0 + h1)
        ptp_early = window_ptp(t, e, h0, mid)
        ptp_late = window_ptp(t, e, mid, h1)
        growth = ptp_late / ptp_early if ptp_early and ptp_early > 0 else float("nan")

        print(f"Kp = {Kp:g}: {status}, t_end = {t_end:.3f} s, wall = {wall:.1f} s")
        print("  peak error:", e.max() * 1e3, "mm")
        print("  final error:", e[-1] * 1e3, "mm")
        print("  tension range:", la_ts.min(), la_ts.max(), "N")
        print("  ptp early / late:", ptp_early * 1e3, ptp_late * 1e3, "mm")
        print("  oscillation growth:", growth)
        results.append((Kp, status, t_end, e.max() * 1e3, growth))

        if export_vtk:
            model.system.export(Path(__file__).parent / "vtk", f"Kp_{Kp:g}", sol, fps=fps)
            print("  VTK written to", Path(__file__).parent / "vtk" / f"Kp_{Kp:g}")

    print("\nKp, status, t_end, peak error [mm], growth")
    for row in results:
        print(row)
