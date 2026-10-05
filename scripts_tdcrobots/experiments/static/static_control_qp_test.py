# static controller with and without the positivity qp, same plant and gains
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cardillo.solver import ScipyDAE

from model.common_model import CommonModel, SETPOINT_TABLE, G_ACCEL
from controllers.static_controller_qp import StaticController, la_t_from_solution
from controllers.static_feedforward_qp import inverse_statics_qp

NAMES = ["A", "B", "C", "D", "E"]


def p2p_sequence(names, t_hold):
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        return pts[min(int(t // t_hold), len(pts) - 1)]

    return r_OP_ref_fn


def build(positive, Kp, Kd, damping_ratio, J_stat_check_dt, t_hold,
          use_feedforward, ff_qp, start):
    """model, controller and initial state for one configuration."""
    model = CommonModel(damping_ratio=damping_ratio, la_pre=0.0)
    system, rod, tendons = model.system, model.rod, model.tendons
    ref = p2p_sequence(NAMES, t_hold)

    controller = StaticController(
        system, rod, tendons, ref,
        v_P_ref_fn=lambda t: np.zeros(3),
        Kp=Kp, Kd=Kd,
        la_t0=np.array([0.5, 0.0, 0.0, 0.0]),
        model_factory=CommonModel, g_accel=G_ACCEL,
        J_stat_check_dt=J_stat_check_dt,
        positive=positive,
    )

    # ---- feedforward: one inverse statics solve per setpoint ----
    if use_feedforward:
        if ff_qp:
            # box constrained gauss newton, feedforward table is nonnegative
            twin = controller.static_model_twin
            pts, guess = [], np.array([0.5, 0.0, 0.0, 0.0])
            for n in NAMES:
                guess = inverse_statics_qp(twin, SETPOINT_TABLE[n], la_t0=guess)
                pts.append(guess.copy())
            pts = np.asarray(pts)
            nn = len(NAMES)
            controller.set_feedforward(lambda t: pts[min(int(t // t_hold), nn - 1)])
            controller.la_t_ff_pts = pts
        else:
            pts = controller.feedforward_from_setpoints(
                [SETPOINT_TABLE[n] for n in NAMES],
                t_hold=t_hold,
                la_t0=np.array([0.5, 0.0, 0.0, 0.0]),
                verbose=False,
            )
        print("  feedforward min:", np.min(pts), "N", "" if np.min(pts) >= 0 else "(negative)")

    # ---- start pose ----
    # the controller commands la_t0 + la_t_ff(0), so offset the state by the feedforward
    if start is not None:
        la_t_start = controller.inverse_statics(
            SETPOINT_TABLE[start], la_t0=np.array([0.5, 0.0, 0.0, 0.0])
        )
        controller.q0 = la_t_start - controller.la_t_ff(0.0)
        controller.set_J_stat(
            controller.static_model_twin.solve_and_eval_J_stat(la_t_start)
        )
        r_start = controller.static_model_twin.r_OP_eq()
        print(f"  start {start}: la_t = {np.array2string(la_t_start, precision=4)}, "
              f"err = {np.linalg.norm(r_start - SETPOINT_TABLE[start]) * 1e3:.4f} mm")

    system.add(controller)
    system.assemble()
    q0 = np.concatenate((controller.static_model_twin.q_eq(), controller.q0))
    system.set_new_initial_state(q0, np.zeros(system.nu))
    return model, controller, ref


def report(tag, d):
    la = d["la"]
    neg = np.mean(la.min(axis=1) < -1e-9) * 100
    print(f"[{tag}] min force: {la.min():.4f} N, negative: {neg:.1f} %, "
          f"settled: {d['e'][d['settled']].mean() * 1e3:.4f} mm, "
          f"mean: {d['e'].mean() * 1e3:.3f} mm", flush=True)


def plot(tag, d, title, out):
    import matplotlib.pyplot as plt

    t = d["t"]
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1, 1, 1.4]})
    for i, nm in enumerate("XYZ"):
        ax[i].plot(t, d["r_ref"][:, i] * 1e2, "b--", lw=1.2, label="desired")
        ax[i].plot(t, d["r"][:, i] * 1e2, "r", lw=1.2, label="actual")
        ax[i].set_ylabel(nm + " [cm]")
        ax[i].legend(fontsize=8)
        ax[i].grid(True, alpha=0.4)
    ax[0].set_title(title, fontsize=11)

    a = ax[3]
    for k in range(int(d["n_tendons"])):
        line, = a.plot(t, d["la"][:, k], lw=1.1, label="tendon " + str(k + 1))
        if bool(d["use_ff"]):
            a.plot(t, d["ff"][:, k], color=line.get_color(), ls="--", lw=0.9)
    a.axhline(0.0, color="k", lw=1.0)
    for s in range(1, len(NAMES)):
        a.axvline(s * float(d["t_hold"]), color="0.8", ls=":", lw=0.8)
    neg = np.mean(d["la"].min(axis=1) < -1e-9) * 100
    ff_txt = "solid: total, dashed: feedforward" if bool(d["use_ff"]) else "no feedforward"
    a.set_title("tendon tension ({})   min {:.4f} N, {:.1f}% negative".format(
        ff_txt, d["la"].min(), neg), fontsize=10)
    a.set_xlabel("Time [s]")
    a.set_ylabel("Tendon force [N]")
    a.legend(fontsize=8, ncol=4)
    a.grid(True, alpha=0.4)

    fig.tight_layout()
    p = out / (tag + ".png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("  wrote", p)


if __name__ == "__main__":
    # ---- parameters ----
    Kp = 2.0
    Kd = 0.0
    t_sim = 25.0
    t_hold = 5.0
    dt = 1e-3
    damping_ratio = 0.1
    J_stat_check_dt = 1e-2
    start = "E"
    # start = None
    use_feedforward = False
    ff_qp = False  # feedforward via inverse_statics_qp (nonnegative)
    runs = ["plain", "qp"]
    # runs = ["qp"]
    use_cache = True

    out = Path(__file__).parent
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    print(f"Kp={Kp:g} Kd={Kd:g} | t_sim={t_sim:g} t_hold={t_hold:g} dt={dt:g} | "
          f"start={start} | ff={use_feedforward}{' (QP)' if ff_qp else ''}")
    print()

    res = {}
    for tag, positive in (("plain", False), ("qp", True)):
        if tag not in runs:
            continue
        key = (f"sq_{tag}_{Kp}_ff{int(use_feedforward)}{int(ff_qp)}_s{start}"
               f"_t{t_sim:g}_h{t_hold:g}")
        f = cache / (key + ".npz")
        if use_cache and f.exists():
            d = dict(np.load(f, allow_pickle=True))
            print(f"[{tag}] cached")
        else:
            print(f"[{tag}] positive={positive}", flush=True)
            model, controller, ref = build(positive, Kp, Kd, damping_ratio, J_stat_check_dt,
                                           t_hold, use_feedforward, ff_qp, start)
            sol = ScipyDAE(model.system, t_sim, dt).solve()

            # applied tension, not the integrator state
            la = np.array([controller.la_tau(t, q[controller.qDOF], u[controller.uDOF])
                           for t, q, u in zip(sol.t, sol.q, sol.u)])
            ff = np.array([controller.la_t_ff(t) for t in sol.t])
            rod = model.rod
            r = sol.q[:, rod.qDOF].reshape((-1, rod.nnode, 7))[:, -1, 0:3]
            rr = np.array([ref(t) for t in sol.t])
            e = np.linalg.norm(rr - r, axis=1)
            # last quarter of each hold counts as settled
            seg = np.floor(sol.t / t_hold).astype(int)
            settled = np.concatenate([
                np.where(seg == s)[0][-max(1, int(0.25 * np.sum(seg == s))):]
                for s in np.unique(seg)
            ])
            d = dict(t=sol.t, la=la, ff=ff, r=r, r_ref=rr, e=e, settled=settled,
                     n_tendons=model.n_tendons, positive=positive, t_hold=t_hold,
                     use_ff=use_feedforward)
            if use_cache:
                np.savez_compressed(f, **d)
        report(tag, d)
        plot(tag, d, "Static controller, {} (Kp={:g}, start={})".format(
            "positivity QP" if positive else "unconstrained", Kp, start), out)
        res[tag] = d

    if len(res) == 2:
        print()
        for tag in ("plain", "qp"):
            report(tag, res[tag])
        cost = (res["qp"]["e"][res["qp"]["settled"]].mean()
                - res["plain"]["e"][res["plain"]["settled"]].mean()) * 1e3
        print(f"\ncost of positivity: {cost:+.4f} mm settled error")
