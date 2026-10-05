import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL, compute_la_ts, DATA_DIR
from controllers.dynamic_controller import DynamicControllerPD

# nnls projection of the allocation, False = plain damped right inverse
POSITIVE = False


def p2p_sequence(names, t_hold=5.0):
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        return pts[min(int(t // t_hold), len(pts) - 1)]

    return (r_OP_ref_fn, lambda t: np.zeros(3), lambda t: np.zeros(3))


def smooth_p2p_sequence(names, t_hold=5.0, t_move=1.0, r_start=None):
    # quintic blend centred on each setpoint change, r_start ramps in from the actual tip
    pts = [SETPOINT_TABLE[name] for name in names]
    n = len(pts)
    t_transition = 0.5 * t_move

    def smoothing(r_OP0, r_OP1, s, T):
        s = min(max(s, 0.0), 1.0)
        d = r_OP1 - r_OP0
        zd = r_OP0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5)
        zd_dot = d * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        zd_ddot = d * (60 * s - 180 * s**2 + 120 * s**3) / (T**2)
        return zd, zd_dot, zd_ddot

    def ref_fns(t):
        if r_start is not None and t < t_move:
            return smoothing(np.asarray(r_start, dtype=float), pts[0], t / t_move, t_move)

        seg = max(0, min(int(t // t_hold), n - 1))
        t_seg0 = seg * t_hold
        t_seg1 = (seg + 1) * t_hold
        # entering segment
        if seg >= 1 and (t - t_seg0) < t_transition:
            return smoothing(pts[seg - 1], pts[seg],
                             (t - (t_seg0 - t_transition)) / t_move, t_move)
        # about to leave for seg + 1
        if seg <= n - 2 and (t_seg1 - t) < t_transition:
            return smoothing(pts[seg], pts[seg + 1],
                             (t - (t_seg1 - t_transition)) / t_move, t_move)
        return pts[seg], np.zeros(3), np.zeros(3)  # hold

    return (lambda t: ref_fns(t)[0], lambda t: ref_fns(t)[1], lambda t: ref_fns(t)[2])


def rod_q0_at(name):
    # precomputed static equilibrium pose of the setpoint
    return pd.read_csv(DATA_DIR / "p2p_q0_gamma0.csv")[f"q0_{name}"].to_numpy()


def simulate(
    names,
    smooth=True,
    t_move=0.5,
    t_hold=5.0,
    t_sim=25.0,
    dt=1e-3,
    Kp=200.0,
    Kd=20.0,
    damping_ratio=0.1,
    la_pre=0.0,
    inv_damping=1e-3,
    positive=POSITIVE,
    start_point="E",
    verbose=True,
):
    model = CommonModel(damping_ratio=damping_ratio, la_pre=la_pre)
    system, rod, tendons = model.system, model.rod, model.tendons

    # ---- start pose ----
    if start_point is not None:
        rod.q0 = rod_q0_at(start_point).copy()
    r_start = rod.q0[rod.nodalDOF_r[-1]]
    if verbose and start_point is not None:
        err = np.linalg.norm(r_start - SETPOINT_TABLE[start_point])
        print(f"start pose {start_point}: err = {err * 1e3:.4f} mm")

    # ---- reference ----
    r_fn, v_fn, a_fn = p2p_sequence(names, t_hold=t_hold)
    refs = {"r": r_fn, "v": v_fn, "a": a_fn}
    if smooth:
        refs["r"], refs["v"], refs["a"] = smooth_p2p_sequence(
            names, t_hold=t_hold, t_move=t_move, r_start=r_start
        )

    controller = DynamicControllerPD(
        system,
        rod,
        tendons,
        lambda t: refs["r"](t),
        v_P_ref_fn=lambda t: refs["v"](t),
        a_P_ref_fn=lambda t: refs["a"](t),
        Kp=Kp,
        Kd=Kd,
        inv_damping=inv_damping,
        positive=positive,
    )

    system.add(controller)
    system.assemble()

    sol = ScipyDAE(system, t_sim, dt).solve()
    return sol, controller, rod, refs, model


def report(label, sol, controller, rod, refs, names, t_hold, smooth, t_move):
    r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
    r_ref = np.array([refs["r"](t) for t in sol.t])
    e = np.linalg.norm(r_OP - r_ref, axis=1)
    la_ts = compute_la_ts(controller, sol)
    dla = np.abs(np.diff(la_ts, axis=0) / np.diff(sol.t)[:, None])

    print(f"\n--- {label} ---")
    print(f"peak |e| vs reference: {e.max() * 1e3:.2f} mm")
    print(f"tension range: [{la_ts.min():.3f}, {la_ts.max():.3f}] N")
    print(f"peak |dla_tau/dt|: {dla.max():.2f} N/s")
    print("settled error at the end of each hold:")
    for k, name in enumerate(names):
        # a centred blend eats t_move/2 off the end of each hold
        t_end = (k + 1) * t_hold - (0.5 * t_move if smooth else 0.0)
        i = min(np.searchsorted(sol.t, t_end) - 1, len(sol.t) - 1)
        err = np.linalg.norm(r_OP[i] - SETPOINT_TABLE[name])
        print(f"  {name} (t = {sol.t[i]:.2f} s): {err * 1e3:.4e} mm")
    return r_OP, r_ref, e, la_ts


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    smooth = False
    t_move = 1.0  # length of the quintic blend
    t_hold = 2
    t_sim = t_hold * 5
    dt = 1e-3
    Kp = 45.0
    # Kd = 10.0
    Kd = 2 * np.sqrt(Kp)
    damping_ratio = 0.1
    la_pre = 0.0  # pretension, compensated through h
    inv_damping = 1e-3  # damping of the right inverse of J_dyn
    positive = True
    start_point = "E"  # None starts from the straight rod
    # start_point = None

    # run stepped and smoothed back to back
    compare = False

    show_3d = True
    export_vtk = False
    export_fps = 120

    out = Path(__file__).parent

    kwargs = dict(
        t_hold=t_hold,
        t_sim=t_sim,
        dt=dt,
        Kp=Kp,
        Kd=Kd,
        damping_ratio=damping_ratio,
        la_pre=la_pre,
        inv_damping=inv_damping,
        positive=positive,
        start_point=start_point,
    )

    print(f"dynamic PD controller, positive={positive}, start={start_point}, Kp={Kp}, Kd={Kd}, g={G_ACCEL}")

    if compare:
        for sm in (False, True):
            sol, controller, rod, refs, model = simulate(
                names, smooth=sm, t_move=t_move, **kwargs
            )
            report(
                f"smoothed t_move={t_move:g}s" if sm else "stepped",
                sol, controller, rod, refs, names, t_hold, sm, t_move,
            )
        raise SystemExit

    sol, controller, rod, refs, model = simulate(names, smooth=smooth, t_move=t_move, **kwargs)
    r_OP, r_ref, e, la_ts = report(
        f"smoothed t_move={t_move:g}s" if smooth else "stepped",
        sol, controller, rod, refs, names, t_hold, smooth, t_move,
    )

    # ---- paraview export ----
    if export_vtk:
        model.system.export(out, "vtk", sol, fps=export_fps)
        print("VTK written to", out / "vtk")

    # ---- visualization ----
    import matplotlib.pyplot as plt

    tag = (
        ("_pos" if positive else "_plain")
        + (f"_smooth{t_move:g}" if smooth else "_step")
        + (f"_start{start_point}" if start_point else "")
    )
    setpoint_times = [k * t_hold for k in range(1, len(names))]

    fig, ax = plt.subplots(num="TendonForces", figsize=(9, 4.5))
    for k in range(la_ts.shape[1]):
        ax.plot(sol.t, la_ts[:, k], label=f"tendon {k + 1}")
    ax.axhline(0.0, color="k", ls="-", lw=0.8)
    for ts in setpoint_times:
        ax.axvline(ts, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$\lambda_{\tau,i}$ [N]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / f"dynamic_pd_tendon_forces{tag}.png", dpi=150)

    fig, axs = plt.subplots(3, 1, num="xyz", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(sol.t, r_ref[:, i], "b--", label="desired")
        axs[i].plot(sol.t, r_OP[:, i], "r", label="actual")
        axs[i].set_ylabel(f"{lbl} [m]")
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel(r"$t$ [s]")
    fig.tight_layout()
    fig.savefig(out / f"dynamic_pd_tip_tracking{tag}.png", dpi=150)

    # tracking error on a log axis
    fig, ax = plt.subplots(num="Error", figsize=(9, 4.0))
    ax.semilogy(sol.t, np.maximum(e, 1e-12) * 1e3, "r")
    for ts in setpoint_times:
        ax.axvline(ts, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$||\boldsymbol{r}_{n}^{*} - \boldsymbol{r}_n||$ [mm]")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out / f"dynamic_pd_tracking_error{tag}.png", dpi=150)

    print("figures written to", out)

    if show_3d:
        from model.tdcm_li2023 import rod_visualization

        rod_visualization(model, sol)

    plt.show()
