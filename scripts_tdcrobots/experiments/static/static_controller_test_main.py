# static controller (li 2023) on a stepped or quintic blended point to point reference
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from controllers.static_feedforward import StaticModelTwin, inverse_statics

# "plain" or "qp"
CONTROLLER_VARIANT = "plain"


def load_controller(variant):
    if variant == "qp":
        from controllers.static_controller_qp import StaticController, la_t_from_solution
    elif variant == "plain":
        from controllers.static_controller import StaticController, la_t_from_solution
    else:
        raise ValueError(f"unknown controller variant {variant!r}, expected 'plain' or 'qp'")
    return StaticController, la_t_from_solution


StaticController, la_t_from_solution = load_controller(CONTROLLER_VARIANT)


def wrong_model_factory(err_stiff=1.0, err_rho=1.0):
    """model factory with a wrong stiffness / density, used for the static twin only."""

    def factory(**kwargs):
        # stiffness has to go through the constructor, the rod bakes it in at build time
        model = CommonModel(stiff_scale=err_stiff, **kwargs)
        if err_rho != 1.0:
            model.rod_density = model.rod_density * err_rho
        return model

    return factory


def p2p_sequence(names, t_hold=5.0):
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        return pts[min(int(t // t_hold), len(pts) - 1)]

    return (r_OP_ref_fn, lambda t: np.zeros(3), lambda t: np.zeros(3))


def smooth_p2p_sequence(names, t_hold=5.0, t_move=1.0, r_start=None):
    """quintic blend over t_move, centred on each setpoint change."""
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
        # ramp from the actual start pose into the first setpoint
        if r_start is not None and t < t_move:
            return smoothing(np.asarray(r_start, dtype=float), pts[0], t / t_move, t_move)

        seg = max(0, min(int(t // t_hold), n - 1))
        t_seg0 = seg * t_hold
        t_seg1 = (seg + 1) * t_hold
        if seg >= 1 and (t - t_seg0) < t_transition:  # entering this segment
            return smoothing(pts[seg - 1], pts[seg],
                             (t - (t_seg0 - t_transition)) / t_move, t_move)
        if seg <= n - 2 and (t_seg1 - t) < t_transition:  # leaving for seg + 1
            return smoothing(pts[seg], pts[seg + 1],
                             (t - (t_seg1 - t_transition)) / t_move, t_move)
        return pts[seg], np.zeros(3), np.zeros(3)

    return (lambda t: ref_fns(t)[0], lambda t: ref_fns(t)[1], lambda t: ref_fns(t)[2])


def simulate(
    names,
    smooth=True,
    t_move=0.5,
    t_hold=5.0,
    t_sim=25.0,
    dt=1e-3,
    Kp=2.0,
    Kd=0.0,
    damping_ratio=0.1,
    use_feedforward=False,
    start_point="E",
    J_stat_check_dt=1e-3,
    variant=None,
    qp_positive=True,
    qp_back_calculate=True,
    qp_reg=1e-8,
    err_stiff=1.0,
    err_rho=1.0,
    err_g=1.0,
    verbose=True,
):
    """one run, returns (sol, controller, rod, refs, model) with refs = {"r": fn, "v": fn}."""
    variant = CONTROLLER_VARIANT if variant is None else variant
    Controller, _ = load_controller(variant)
    qp_kwargs = (
        dict(positive=qp_positive, back_calculate=qp_back_calculate, qp_reg=qp_reg)
        if variant == "qp"
        else {}
    )

    model = CommonModel(damping_ratio=damping_ratio, la_pre=0.0)
    system, rod, tendons = model.system, model.rod, model.tendons

    # the controller gets a delegate, the smoothed reference is set once the start pose is known
    r_fn, v_fn, _ = p2p_sequence(names, t_hold=t_hold)
    refs = {"r": r_fn, "v": v_fn}

    controller = Controller(
        system,
        rod,
        tendons,
        lambda t: refs["r"](t),
        v_P_ref_fn=lambda t: refs["v"](t),
        Kp=Kp,
        Kd=Kd,
        la_t0=(np.zeros(4) if use_feedforward else np.array([0.5, 0.0, 0.0, 0.0])),
        model_factory=wrong_model_factory(err_stiff=err_stiff, err_rho=err_rho),
        g_accel=err_g * G_ACCEL,
        J_stat_check_dt=J_stat_check_dt,
        **qp_kwargs,
    )
    model_error = (err_stiff, err_rho, err_g) != (1.0, 1.0, 1.0)
    if model_error and verbose:
        # err_rho and err_g both scale the twin's gravity load
        grav = err_rho * err_g
        note = "  == err_h" if abs(grav - err_stiff) < 1e-12 else "  (not a uniform h scaling)"
        print(f"model error (twin only): h_elastic x{err_stiff:g}, h_gravity x{grav:g}{note}")

    if use_feedforward:
        controller.feedforward_from_setpoints(
            [SETPOINT_TABLE[n] for n in names],
            t_hold=t_hold,
            la_t0=np.array([0.5, 0.0, 0.0, 0.0]),
            verbose=verbose,
        )

    # ---- start pose ----
    # the controller commands la_t0 + la_t_ff(0), so offset the state by the feedforward
    q_rod_start, r_start = None, None
    if start_point is not None:
        if model_error:
            # plant starts at rest at the setpoint, so the start tension comes from a true twin
            true_twin = StaticModelTwin(CommonModel, g_accel=G_ACCEL)
            la_t_start = inverse_statics(
                true_twin, SETPOINT_TABLE[start_point], la_t0=np.array([0.5, 0.0, 0.0, 0.0])
            )
            q_rod_start, r_start = true_twin.q_eq(), true_twin.r_OP_eq()
        else:
            la_t_start = controller.inverse_statics(
                SETPOINT_TABLE[start_point], la_t0=np.array([0.5, 0.0, 0.0, 0.0])
            )
        controller.q0 = la_t_start - controller.la_t_ff(0.0)
        # reseed J_stat about the start tension
        controller.set_J_stat(controller.static_model_twin.solve_and_eval_J_stat(la_t_start))
        if r_start is None:
            r_start = controller.static_model_twin.r_OP_eq()
        if verbose:
            print(
                f"start pose {start_point}: la_t = {np.array2string(la_t_start, precision=4)}, "
                f"err = {np.linalg.norm(r_start - SETPOINT_TABLE[start_point]) * 1e3:.4f} mm"
            )

    if smooth:
        refs["r"], refs["v"], _ = smooth_p2p_sequence(
            names,
            t_hold=t_hold,
            t_move=t_move,
            r_start=(r_start if r_start is not None
                     else controller.static_model_twin.r_OP_eq()),
        )

    system.add(controller)
    system.assemble()

    # start from the static equilibrium of the start tension
    q_rod = q_rod_start if q_rod_start is not None else controller.static_model_twin.q_eq()
    q0 = np.concatenate((q_rod, controller.q0))
    system.set_new_initial_state(q0, np.zeros(system.nu))

    sol = ScipyDAE(system, t_sim, dt).solve()
    return sol, controller, rod, refs, model


def applied_tensions(controller, sol):
    """tendon tensions actually applied to the rod, shape (nt, n_tendons)."""
    # with the projection on the state is not the applied force, so replay la_tau
    if getattr(controller, "positive", False):
        return np.array([
            controller.la_tau(t, q[controller.qDOF], u[controller.uDOF])
            for t, q, u in zip(sol.t, sol.q, sol.u)
        ])
    la_t = sol.q[:, controller.my_qDOF]
    return la_t + np.array([controller.la_t_ff(t) for t in sol.t])


def report(label, sol, controller, rod, refs, names, t_hold, smooth, t_move):
    r_OP = sol.q[:, rod.qDOF][:, rod.nodalDOF_r[-1]]
    r_ref = np.array([refs["r"](t) for t in sol.t])
    e = np.linalg.norm(r_OP - r_ref, axis=1)
    la_ts = applied_tensions(controller, sol)
    dla = np.abs(np.diff(la_ts, axis=0) / np.diff(sol.t)[:, None])

    print(f"\n--- {label} ---")
    print(f"  peak |e| vs reference: {e.max() * 1e3:.2f} mm")
    print(f"  tension range: [{la_ts.min():.3f}, {la_ts.max():.3f}] N")
    print(f"  peak |dla_t/dt|: {dla.max():.2f} N/s")
    print("  settled error at the end of each hold:")
    for k, name in enumerate(names):
        # the blend is centred on the setpoint change, so the hold ends t_move/2 early
        t_end = (k + 1) * t_hold - (0.5 * t_move if smooth else 0.0)
        i = min(np.searchsorted(sol.t, t_end) - 1, len(sol.t) - 1)
        err = np.linalg.norm(r_OP[i] - SETPOINT_TABLE[name])
        print(f"    {name} (t = {sol.t[i]:.2f} s): {err * 1e3:.4e} mm")
    return r_OP, r_ref, e, la_ts


if __name__ == "__main__":
    # ---- parameters ----
    names = ["A", "B", "C", "D", "E"]
    smooth = False
    t_move = 0.5  # [s] length of the quintic blend
    t_hold = 2
    # t_sim = 10.0
    t_sim = t_hold * 5
    dt = 1e-3
    Kp = 8.0
    Kd = 0.0
    damping_ratio = 0.1
    use_feedforward = False
    start_point = "E"
    # start_point = None
    J_stat_check_dt = 1e-2

    # "plain": tensions may go negative, "qp": projected onto la_tau >= 0
    variant = "plain"

    # ---- model error, applied to the static twin only ----
    err_stiff = 1.0  # twin's E and G
    err_rho = 1.0    # twin's density
    err_g = 1.0      # twin's gravity acceleration

    # ---- positivity qp options, only for variant == "qp" ----
    qp_positive = True
    qp_back_calculate = True
    qp_reg = 1e-8

    # run stepped and smoothed back to back
    compare = False

    show_3d = True

    # ---- paraview export ----
    export_vtk = True
    export_fps = 30

    kwargs = dict(
        t_hold=t_hold,
        t_sim=t_sim,
        dt=dt,
        Kp=Kp,
        Kd=Kd,
        damping_ratio=damping_ratio,
        use_feedforward=use_feedforward,
        start_point=start_point,
        J_stat_check_dt=J_stat_check_dt,
        variant=variant,
        qp_positive=qp_positive,
        qp_back_calculate=qp_back_calculate,
        qp_reg=qp_reg,
        err_stiff=err_stiff,
        err_rho=err_rho,
        err_g=err_g,
    )

    qp_txt = (
        f"positivity QP on (positive={qp_positive}, back_calculate={qp_back_calculate}, "
        f"qp_reg={qp_reg:g})"
        if variant == "qp"
        else "positivity QP off"
    )
    print(f"static controller '{variant}', feedforward={use_feedforward}, "
          f"start={start_point}, Kp={Kp}, Kd={Kd}")
    print("  " + qp_txt)

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
        model.system.export(Path(__file__).parent, "vtk", sol, fps=export_fps)
        pvds = sorted(p.name for p in (Path(__file__).parent / "vtk").glob("*.pvd"))
        print("VTK written to", Path(__file__).parent / "vtk")
        print(f"  open in ParaView: {', '.join(pvds)}")

    # ---- visualization ----
    import matplotlib.pyplot as plt

    out = Path(__file__).parent
    err_tag = "".join(
        f"_{n}{v:g}" for n, v in (("stiff", err_stiff), ("rho", err_rho), ("g", err_g))
        if v != 1.0
    )
    tag = (
        f"_{variant}"
        + ("_ff" if use_feedforward else "_noff")
        + (f"_smooth{t_move:g}" if smooth else "_step")
        + (f"_start{start_point}" if start_point else "")
        + err_tag
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
    # ax.set_title(f"Tendon forces, static controller ({variant}, Kp={Kp})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / f"smoothed_tendon_forces{tag}.png", dpi=150)

    fig, axs = plt.subplots(3, 1, num="XYZ", figsize=(9, 7), sharex=True)
    for i, lbl in enumerate("xyz"):
        axs[i].plot(sol.t, r_ref[:, i], "b--", label="desired")
        axs[i].plot(sol.t, r_OP[:, i], "r", label="actual")
        axs[i].set_ylabel(f"{lbl} [m]")
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel(r"$t$ [s]")
    # fig.suptitle(f"Tip trajectory tracking, static controller ({variant})")
    fig.tight_layout()
    fig.savefig(out / f"smoothed_tip_tracking{tag}.png", dpi=150)

    # tracking error on a log axis
    fig, ax = plt.subplots(num="Error", figsize=(9, 4.0))
    ax.semilogy(sol.t, np.maximum(e, 1e-12) * 1e3, "r")
    for ts in setpoint_times:
        ax.axvline(ts, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$||\boldsymbol{r}_{OP}^{*} - \boldsymbol{r}_{OP}||$ [mm]")
    # ax.set_title(f"Tracking error, static controller ({variant}, Kp={Kp})")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out / f"smoothed_tracking_error{tag}.png", dpi=150)

    print("figures written to", out)

    # ---- 3d rod animation ----
    if show_3d:
        from model.tdcm_li2023 import rod_visualization

        rod_visualization(model, sol)

    plt.show()
