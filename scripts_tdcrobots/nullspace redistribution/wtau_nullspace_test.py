"""Runner and verification for WTauNullspaceController.

Staged so failures surface early and cheaply:

    python wtau_nullspace_test.py --probe     sigma_4(W_tau) over the workspace  (~1 min)
    python wtau_nullspace_test.py --fdcheck   analytic vs finite-difference      (~1 min)
    python wtau_nullspace_test.py --short     t_sim = 2, one setpoint            (minutes)
    python wtau_nullspace_test.py             t_sim = 10, A-B-C-D-E

Add --save to write PNGs instead of opening windows.

--fdcheck is the gate: dv/dq (eigenvector perturbation of W_tau^T W_tau) is the genuinely new
algebra here, and ScipyDAE will crawl or diverge if it is wrong.

The headline plot is Plot 2, and it is deliberately not flattering: it shows how much wrench
this redistribution actually leaks, because null(W_tau) is exact only at the straight
configuration.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

if "--save" in sys.argv:  # headless: write PNGs instead of opening windows
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# this folder, then scripts_tdcrobots one level up for the user's original modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))


def _find_nullspace_controller():
    """Locate nullspace_controller.py, used only by --compare / --isolate.

    It has moved between scripts_tdcrobots and its subfolders, so search rather than hard-code,
    and let the caller degrade gracefully when it is not found.
    """
    root = Path(__file__).resolve().parent.parent
    for cand in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        if (cand / "nullspace_controller.py").exists():
            sys.path.append(str(cand))
            return True
    return False

from cardillo.solver import ScipyDAE

from dynamic_control_test import CommonModel, SETPOINT_TABLE
from wtau_nullspace_controller import (
    WTauNullspaceController,
    achievable_floor,
    replay_alloc,
    wtau_alloc,
    wtau_null_dir,
)


# ----------------------------------------------------------------------------------
# 1. Allocator self-test: no solver, seconds to run
# ----------------------------------------------------------------------------------


def selftest(n_trials=2000, n=4, m=3, nu=66, inv_damping=1e-3, seed=0):
    """Random W with a planted near-nullspace, so the branches get exercised realistically."""
    rng = np.random.default_rng(seed)
    stats = dict(feasible=0, infeasible=0, degenerate=0, alpha_zero=0)
    worst = dict(leak=0.0, floor=0.0, sign=0.0, dir=0.0, dir_pert=0.0)

    for _ in range(n_trials):
        # plant a direction z that W nearly annihilates, mimicking the real geometry
        z = rng.normal(size=n)
        z /= np.linalg.norm(z)
        W = rng.normal(size=(nu, n))
        W -= np.outer(W @ z, z)  # exactly annihilates z
        pert = rng.choice([0.0, 1e-3, 1e-2])
        W += pert * np.outer(rng.normal(size=nu), z)  # then perturb, as bending does
        J = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        F_min = rng.choice([0.0, 0.1, 0.5, 2.0, 10.0])

        al = wtau_alloc(J, b, W, F_min=F_min, inv_damping=inv_damping)

        # (a) the direction really is the one W annihilates. Only exact when the kernel is
        # exact; a perturbation of size `pert` moves the smallest singular vector by O(pert),
        # which is the real robot's situation and is tracked separately rather than gated.
        if not al.degenerate:
            miss = 1.0 - abs(al.v @ z)
            key = "dir" if pert == 0.0 else "dir_pert"
            worst[key] = max(worst[key], miss)

        # (b) the leak is exactly |alpha| * sigma_min, i.e. nothing else moved
        if not al.degenerate:
            predicted = abs(al.alpha) * al.sigma[-1]
            worst["leak"] = max(worst["leak"], abs(al.leak_W - predicted) / max(1.0, predicted))

        if al.feasible:
            stats["feasible"] += 1
            worst["floor"] = max(worst["floor"], max(0.0, F_min - al.F.min()))
        else:
            stats["infeasible"] += 1
        if al.degenerate:
            stats["degenerate"] += 1
            continue
        if al.alpha == 0.0:
            stats["alpha_zero"] += 1

        # (c) the SVD's arbitrary sign convention must not reach the output
        al2 = wtau_alloc(J, b, -W, F_min=F_min, inv_damping=inv_damping)
        worst["sign"] = max(
            worst["sign"], np.max(np.abs(al.F - al2.F)) / max(1.0, np.max(np.abs(al.F)))
        )

        # (d) the max-min branch must beat both interval endpoints
        if not al.feasible and al.active and np.isfinite(al.a_lo) and np.isfinite(al.a_hi):
            g = np.min(al.F)
            assert g >= np.min(al.F_p + al.a_lo * al.v) - 1e-9, "max-min alpha is not maximal"
            assert g >= np.min(al.F_p + al.a_hi * al.v) - 1e-9, "max-min alpha is not maximal"

    print("--- allocator self-test ---")
    print(f"  trials                    : {n_trials}")
    print(f"  feasible / infeasible     : {stats['feasible']} / {stats['infeasible']}")
    print(f"  degenerate / alpha == 0   : {stats['degenerate']} / {stats['alpha_zero']}")
    print(f"  max 1 - |v . z|, exact ker: {worst['dir']:.3e}   (must be ~1e-12)")
    print(f"  ... with kernel perturbed : {worst['dir_pert']:.3e}   (O(perturbation), informational)")
    print(f"  max rel leak mismatch     : {worst['leak']:.3e}   (must be ~1e-12)")
    print(f"  max floor violation       : {worst['floor']:.3e}   (must be 0)")
    print(f"  max rel sign-flip drift   : {worst['sign']:.3e}   (must be ~1e-15)")

    ok = (
        worst["dir"] < 1e-8
        and worst["leak"] < 1e-8
        and worst["floor"] < 1e-9
        and worst["sign"] < 1e-9
    )
    print("  RESULT                    :", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------------------
# 2. Simulation
# ----------------------------------------------------------------------------------


def p2p_sequence(names, t_hold=2.0):
    pts = [SETPOINT_TABLE[n] for n in names]

    def r_OP_ref_fn(t):
        return pts[min(int(t // t_hold), len(pts) - 1)]

    return r_OP_ref_fn


def smooth_p2p_sequence(names, t_hold=2.0, t_move=0.5, r_start=None):
    """Quintic-blended setpoint sequence -- same construction as `smooth_p2p_sequence` in the
    __main__ block of dynamic_control_test.py, which is not importable from there.

    Each transition is centred on the setpoint change and spans t_move, so the reference is C2
    and v_P_ref / a_P_ref are consistent with r_OP_ref rather than zero.

    `r_start` optionally ramps from the rod's actual initial tip position into the first
    setpoint over [0, t_move]. Without it the reference still steps discontinuously at t = 0,
    which is the largest transient in the whole run and would mask the effect of smoothing
    everywhere else.
    """
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
        if seg >= 1 and (t - t_seg0) < t_transition:  # entering this segment
            return smoothing(pts[seg - 1], pts[seg],
                             (t - (t_seg0 - t_transition)) / t_move, t_move)
        if seg <= n - 2 and (t_seg1 - t) < t_transition:  # about to leave for seg + 1
            return smoothing(pts[seg], pts[seg + 1],
                             (t - (t_seg1 - t_transition)) / t_move, t_move)
        return pts[seg], np.zeros(3), np.zeros(3)

    return (lambda t: ref_fns(t)[0], lambda t: ref_fns(t)[1], lambda t: ref_fns(t)[2])


def load_q0(name):
    """Equilibrium rod configuration at a setpoint, from p2p_q0_gamma0.csv next to the
    user's dynamic_control_test.py (columns q0_A .. q0_E)."""
    import pandas as pd

    csv = Path(__file__).resolve().parent.parent / "p2p_q0_gamma0.csv"
    return pd.read_csv(csv)[f"q0_{name}"].to_numpy().copy()


def build(r_OP_ref_fn, damping_ratio=0.1, F_min=0.5, la_pre=None, uniform_fallback=False,
          start=None, **extra):
    """Fresh plant + controller. A new CommonModel per run: system.add/assemble mutates.

    `r_OP_ref_fn` may instead be a factory taking the freshly built model and returning
    (r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn) -- needed by the smoothed reference, which has to
    know the rod's actual initial tip position before it can ramp away from it.

    `la_pre` is a per-tendon physical pretension, applied with RodTendonForce.set_force after
    construction so `CommonModel` (which only takes a uniform la_pre) is left untouched.
    """
    model = CommonModel(damping_ratio=damping_ratio, la_pre=0.0)
    if start is not None:
        # must precede system.assemble(); cf. the commented-out "Start at E" line in
        # dynamic_control_test.py
        model.rod.q0 = load_q0(start)
    if la_pre is not None:
        la_pre = np.broadcast_to(np.asarray(la_pre, dtype=float), (len(model.tendons),)).copy()
        for tendon, p in zip(model.tendons, la_pre):
            tendon.set_force(float(p))
        model.la_pre = la_pre

    common = dict(
        v_P_ref_fn=lambda t: np.zeros(3),
        a_P_ref_fn=lambda t: np.zeros(3),
        Kp=200.0,
        Kd=20.0,
        inv_damping=1e-3,
    )
    if callable(r_OP_ref_fn) and getattr(r_OP_ref_fn, "is_factory", False):
        r_OP_ref_fn, common["v_P_ref_fn"], common["a_P_ref_fn"] = r_OP_ref_fn(model)
    common.update(extra)
    controller = WTauNullspaceController(
        model.system, model.rod, model.tendons, r_OP_ref_fn,
        F_min=F_min, la_pre=la_pre, uniform_fallback=uniform_fallback, **common,
    )
    model.system.add(controller)
    model.system.assemble()
    return model, controller, r_OP_ref_fn


def make_refs(names, t_hold=2.0, smooth=False, t_move=0.5, ramp_start=True):
    """Reference builder. Returns a factory so the smoothed variant can read the rod's initial
    tip position off the model that `build` creates."""
    if not smooth:
        return p2p_sequence(list(names), t_hold=t_hold)

    def factory(model):
        r_start = None
        if ramp_start:
            rod = model.rod
            r_start = rod._view_nodal_q(rod.q0)[-1, :3].copy()
        return smooth_p2p_sequence(list(names), t_hold=t_hold, t_move=t_move, r_start=r_start)

    factory.is_factory = True
    return factory


def run(t_sim=10.0, dt=1e-4, names=("A", "B", "C", "D", "E"), t_hold=2.0, F_min=0.5,
        la_pre=None, uniform_fallback=False, smooth=False, t_move=0.5, ramp_start=True,
        start=None):
    refs = make_refs(names, t_hold=t_hold, smooth=smooth, t_move=t_move, ramp_start=ramp_start)
    model, controller, r_OP_ref_fn = build(refs, F_min=F_min, la_pre=la_pre,
                                           uniform_fallback=uniform_fallback, start=start)
    sol = ScipyDAE(model.system, t_sim, dt).solve()
    return model, controller, sol, r_OP_ref_fn


# ----------------------------------------------------------------------------------
# 3. Probe: is there a nullspace at all, over the whole workspace?
# ----------------------------------------------------------------------------------


def probe(t_sim=10.0, dt=1e-3, names=("A", "B", "C", "D", "E"), t_hold=2.0, n_rows=16):
    """sigma(W_tau) and the null direction along a real trajectory."""
    model, controller, sol, _ = run(t_sim=t_sim, dt=dt, names=names, t_hold=t_hold)
    qDOF, uDOF = controller.qDOF, controller.uDOF

    print("\n--- null(W_tau) over the trajectory ---")
    print(f"{'t':>6} {'s1':>9} {'s2':>9} {'s3':>9} {'s4':>10} {'s4/s1':>9} {'s4/s2':>9} "
          f"{'sum(v)':>10}  v")
    step = max(1, len(sol.t) // n_rows)
    s4s1, s4s2, sums = [], [], []
    for i in range(0, len(sol.t), step):
        al = controller._alloc(sol.t[i], sol.q[i][qDOF], sol.u[i][uDOF])
        s = al.sigma
        r1, r2, sv = s[-1] / s[0], s[-1] / s[1], al.v.sum()
        s4s1.append(r1)
        s4s2.append(r2)
        sums.append(abs(sv))
        print(f"{sol.t[i]:6.3f} {s[0]:9.4f} {s[1]:9.5f} {s[2]:9.5f} {s[3]:10.3e} "
              f"{r1:9.2e} {r2:9.2e} {sv:10.2e}  {np.array2string(al.v, precision=3)}")

    print(f"\n  sigma_4/sigma_1  min/max : {min(s4s1):.2e} / {max(s4s1):.2e}")
    print(f"  sigma_4/sigma_2  min/max : {min(s4s2):.2e} / {max(s4s2):.2e}"
          "   <-- leak relative to the LATERAL (steering) authority")
    print(f"  max |sum(v)|             : {max(sums):.2e}"
          "   <-- ~0 means v is an antagonistic pair swap")
    return controller, sol


# ----------------------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------------------


def _shade_infeasible(ax, t, feasible):
    infeas = ~np.asarray(feasible, dtype=bool)
    if not infeas.any():
        return
    edges = np.diff(infeas.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if infeas[0]:
        starts = [0] + starts
    if infeas[-1]:
        ends = ends + [len(t) - 1]
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color="red", alpha=0.10, lw=0,
                   label="_" if s != starts[0] else "positivity unreachable")


def plot_forces(rec, F_min):
    """Plot 1: does redistribution change la_tau, and is the result positive?

    Plots the physical tension la_pre + la_tau, so a pretensioned run is judged on what the
    tendon actually carries.
    """
    t, F_p, F = rec["t"], rec["F_p_tot"], rec["F_tot"]
    fig, ax = plt.subplots(num="WTauTendonForces", figsize=(10, 5))
    _shade_infeasible(ax, t, rec["feasible"])
    for k in range(F.shape[1]):
        c = f"C{k}"
        ax.plot(t, F_p[:, k], c=c, ls="--", lw=1.0, alpha=0.7,
                label=f"tendon {k+1}  $F_p$ (no redistribution)")
        ax.plot(t, F[:, k], c=c, ls="-", lw=1.6, label=f"tendon {k+1}  redistributed")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(F_min, color="k", ls=":", lw=1.0, label=f"$F_{{min}}$ = {F_min:g} N")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tendon force [N]")
    ax.set_title(r"Tendon forces: dashed = damped pinv, solid = shifted along null($W_\tau$)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_leakage(rec):
    """Plot 2: the honest accounting -- what does using null(W_tau) actually cost?"""
    t = rec["t"]
    fig, axes = plt.subplots(2, 1, num="WTauLeakage", figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.semilogy(t, np.maximum(rec["sigma"][:, -1], 1e-20), c="C3", lw=1.5,
                label=r"$\sigma_4(W_\tau)$  (0 $\Rightarrow$ exact nullspace)")
    ax.semilogy(t, rec["sigma"][:, 1], c="C7", lw=1.0, alpha=0.8,
                label=r"$\sigma_2(W_\tau)$  (lateral / steering authority)")
    ax.semilogy(t, rec["sigma"][:, 0], c="C8", lw=1.0, alpha=0.5,
                label=r"$\sigma_1(W_\tau)$  (axial)")
    ax.set_ylabel(r"singular values of $W_\tau$")
    ax.set_title(r"null($W_\tau$) is exact only at the straight configuration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    ax.semilogy(t, np.maximum(rec["leak_W"], 1e-20), c="C3", lw=1.5,
                label=r"$\|W_\tau F - W_\tau F_p\|$  (wrench leaked by the shift)")
    ax.semilogy(t, np.maximum(rec["leak_J"], 1e-20), c="C0", lw=1.4,
                label=r"$\|J F - J F_p\|$  (its share in the tip acceleration)")
    ax.semilogy(t, np.maximum(rec["res_p"], 1e-20), c="C7", lw=1.0, alpha=0.8,
                label=r"$\|J F_p - b\|$  (pre-existing damped-pinv residual)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("residual norm")
    ax.set_title("Cost of redistributing along the near-nullspace, vs the error already present")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()


def plot_shift(rec, F_min):
    """Plot 3: the shift, the feasible band, and how much negativity is removed."""
    t = rec["t"]
    fig, axes = plt.subplots(2, 1, num="WTauShift", figsize=(10, 7), sharex=True)

    ax = axes[0]
    _shade_infeasible(ax, t, rec["feasible"])
    lo = np.clip(rec["a_lo"], -1e3, 1e3)
    hi = np.clip(rec["a_hi"], -1e3, 1e3)
    ax.fill_between(t, lo, hi, where=(rec["a_lo"] <= rec["a_hi"]),
                    color="C2", alpha=0.20, label=r"feasible $[\alpha_{lo}, \alpha_{hi}]$")
    ax.plot(t, rec["alpha"], c="C0", lw=1.4, label=r"$\alpha$ used")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"Nullspace shift $\lambda_{pos} = \lambda + B\alpha$ and the feasible segment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    _shade_infeasible(ax, t, rec["feasible"])
    ax.plot(t, rec["F_p_tot"].min(axis=1), c="C7", ls="--", lw=1.2, label=r"$\min_i F_{p,i}$")
    ax.plot(t, rec["F_tot"].min(axis=1), c="C0", lw=1.6, label=r"$\min_i F_i$")
    ax.plot(t, rec["floor_max"], c="C2", ls="-.", lw=1.0,
            label=r"best achievable $\min_i F_i$")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhline(F_min, color="k", ls=":", lw=1.0, label=f"$F_{{min}}$ = {F_min:g} N")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("smallest tendon force [N]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def report(rec, F_min, la_pre=None):
    t = rec["t"]
    # judge positivity on the PHYSICAL tension la_pre + la_tau that the tendon carries, and on
    # the COMMANDED force -- not on rec["feasible"], which describes the allocation before any
    # uniform fallback and so would understate a run using it
    minF, minFp = rec["F_tot"].min(axis=1), rec["F_p_tot"].min(axis=1)
    feas = minF >= F_min - 1e-9
    s = rec["sigma"]

    print("\n--- null(W_tau) redistribution report ---")
    print(f"  samples                      : {len(t)}  over t in [{t[0]:.3f}, {t[-1]:.3f}] s")
    if la_pre is not None and np.any(la_pre):
        print(f"  pretension per tendon [N]    : {np.array2string(np.asarray(la_pre), precision=3)}")
    print(f"  tension >= F_min={F_min:g}         : {feas.mean()*100:.1f} % of samples")
    print(f"  min tendon tension, before   : {minFp.min():.4f} N")
    print(f"  min tendon tension, after    : {minF.min():.4f} N")
    print(f"  negative samples, before     : {np.mean(minFp < 0)*100:.1f} %")
    print(f"  negative samples, after      : {np.mean(minF < 0)*100:.1f} %")
    print(f"  alpha range                  : [{rec['alpha'].min():.4f}, {rec['alpha'].max():.4f}]")
    print(f"  fraction with alpha == 0     : {np.mean(rec['alpha'] == 0.0)*100:.1f} %")

    print("\n  how exact is the nullspace:")
    print(f"    sigma_4  max               : {s[:, -1].max():.3e}")
    print(f"    sigma_4/sigma_1  max       : {(s[:, -1]/s[:, 0]).max():.3e}")
    print(f"    sigma_4/sigma_2  max       : {(s[:, -1]/s[:, 1]).max():.3e}  <-- vs steering")
    print(f"    max |sum(v)|               : {np.abs(rec['sum_v']).max():.3e}")

    print("\n  what the shift leaked:")
    print(f"    ||W F - W F_p||  med/max   : {np.median(rec['leak_W']):.3e} / {rec['leak_W'].max():.3e}")
    print(f"    ||J F - J F_p||  med/max   : {np.median(rec['leak_J']):.3e} / {rec['leak_J'].max():.3e}")
    print(f"    ||J F_p - b||    med       : {np.median(rec['res_p']):.3e}  (already present)")
    ratio = rec["leak_J"] / np.maximum(rec["res_p"], 1e-300)
    print(f"    leak / pre-existing  med   : {np.median(ratio):.3f}")

    fm = rec["floor_max"]
    print(f"\n  achievable floor min/median  : {fm.min():.4f} / {np.median(fm):.4f} N")
    print("  feasibility vs F_min:")
    for cand in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        print(f"      F_min = {cand:4.2f} N  ->  {np.mean(fm >= cand)*100:5.1f} % of samples achievable")


# ----------------------------------------------------------------------------------
# 4b. Tracking, and head-to-head against the null(J) controller
# ----------------------------------------------------------------------------------


def tracking(model, sol, r_OP_ref_fn):
    """Tip tracking error [mm] over a solution."""
    rod = model.rod
    err = np.array([
        np.linalg.norm(rod._view_nodal_q(q[rod.qDOF])[-1, :3] - r_OP_ref_fn(t))
        for t, q in zip(sol.t, sol.q)
    ]) * 1e3
    return err


def compare(t_sim=10.0, dt=1e-4, names=("A", "B", "C", "D", "E"), t_hold=2.0, F_min=0.5):
    """null(W_tau) vs null(J): does the stronger invariance actually buy anything?

    Same plant, same gains, same F_min; only the redistribution direction differs.
    """
    if not _find_nullspace_controller():
        print("  --compare needs nullspace_controller.py (not found); skipping")
        return []
    from nullspace_controller import NullspaceShiftController, replay_alloc as replay_J

    r_OP_ref_fn = p2p_sequence(list(names), t_hold=t_hold)
    rows = []

    model_w, ctrl_w, _ = build(r_OP_ref_fn, F_min=F_min)
    sol_w = ScipyDAE(model_w.system, t_sim, dt).solve()
    rec_w = replay_alloc(ctrl_w, sol_w)
    rows.append(("null(W_tau)", rec_w["F"], tracking(model_w, sol_w, r_OP_ref_fn),
                 rec_w["feasible"], rec_w["leak_J"]))

    model_j = CommonModel(damping_ratio=0.1, la_pre=0.0)
    ctrl_j = NullspaceShiftController(
        model_j.system, model_j.rod, model_j.tendons, r_OP_ref_fn,
        v_P_ref_fn=lambda t: np.zeros(3), a_P_ref_fn=lambda t: np.zeros(3),
        Kp=200.0, Kd=20.0, inv_damping=1e-3,
    )
    ctrl_j.F_min = F_min
    model_j.system.add(ctrl_j)
    model_j.system.assemble()
    sol_j = ScipyDAE(model_j.system, t_sim, dt).solve()
    rec_j = replay_J(ctrl_j, sol_j)
    rows.append(("null(J)", rec_j["F"], tracking(model_j, sol_j, r_OP_ref_fn),
                 rec_j["feasible"], np.zeros(len(rec_j["t"]))))

    print(f"\n--- null(W_tau) vs null(J),  t_sim = {t_sim} s,  F_min = {F_min} N ---")
    print(f"{'direction':>14} {'%>=F_min':>9} {'%neg':>7} {'min F':>9} "
          f"{'err mean':>9} {'err final':>10} {'leak J med':>11}")
    for name, F, err, feas, leak in rows:
        print(f"{name:>14} {np.mean(feas)*100:8.1f}% {np.mean(F.min(axis=1) < 0)*100:6.1f}% "
              f"{F.min():9.4f} {err.mean():8.3f}mm {err[-1]:9.3f}mm {np.median(leak):11.3e}")
    return rows


def isolate(t_sim=10.0, dt=1e-3, names=("A", "B", "C", "D", "E"), t_hold=2.0, F_min=0.5):
    """Same trajectory, both allocators -- separates the direction from the closed loop.

    `compare` runs two controllers, so each walks its own trajectory and the positivity gap
    could be feedback divergence rather than the choice of B. Here one simulation is run and
    both allocators are replayed on identical (t, q, u), which isolates the allocator itself.
    """
    if not _find_nullspace_controller():
        print("  --isolate needs nullspace_controller.py (not found); skipping")
        return None, None
    from nullspace_controller import nullspace_alloc

    model, controller, sol, _ = run(t_sim=t_sim, dt=dt, names=names, t_hold=t_hold, F_min=F_min)
    qDOF, uDOF = controller.qDOF, controller.uDOF

    n_w = n_j = n = 0
    neg_w = neg_j = 0
    minw, minj, align = [], [], []
    for t, q_sys, u_sys in zip(sol.t, sol.q, sol.u):
        q, u = q_sys[qDOF], u_sys[uDOF]
        al_w = controller._alloc(t, q, u)
        al_j = nullspace_alloc(al_w.J, al_w.b, F_min=F_min, inv_damping=controller.inv_damping)
        n += 1
        n_w += al_w.feasible
        n_j += al_j.feasible
        neg_w += al_w.F.min() < 0
        neg_j += al_j.F.min() < 0
        minw.append(al_w.F.min())
        minj.append(al_j.F.min())
        align.append(abs(al_w.v @ al_j.v))

    print(f"\n--- same trajectory, both allocators (n = {n}, F_min = {F_min} N) ---")
    print(f"{'direction':>14} {'%>=F_min':>9} {'%neg':>7} {'min F':>9}")
    print(f"{'null(W_tau)':>14} {n_w/n*100:8.1f}% {neg_w/n*100:6.1f}% {min(minw):9.4f}")
    print(f"{'null(J)':>14} {n_j/n*100:8.1f}% {neg_j/n*100:6.1f}% {min(minj):9.4f}")
    print(f"  |v_W . v_J|  min/median : {min(align):.6f} / {np.median(align):.6f}")
    return minw, minj


# ----------------------------------------------------------------------------------
# 5. Finite-difference check of the analytic Jacobians
# ----------------------------------------------------------------------------------


def fd_check(controller, t, q, u, n_cols=10, eps=1e-7, seed=0):
    rng = np.random.default_rng(seed)
    al = controller._alloc(t, q, u)
    Jq = controller.la_tau_q(t, q, u)
    Ju = controller.la_tau_u(t, q, u)

    def worst(analytic, x, setter, sig, n_dof):
        # la_tau_q is genuinely sparse (only the last nodes have nonzero columns, inherited from
        # _b_q), so uniform sampling would mostly compare 0 against 0. Test where the derivative
        # lives, falling back to uniform sampling if it is all zero.
        nz = np.where(np.abs(analytic).max(axis=0) > 0)[0]
        pool = nz if len(nz) else np.arange(n_dof)
        cols = rng.choice(pool, size=min(n_cols, len(pool)), replace=False)
        e, switched = 0.0, 0
        for k in cols:
            xp, xm = x.copy(), x.copy()
            xp[k] += eps
            xm[k] -= eps
            # The allocation is piecewise smooth: across a change of active set la_tau has a
            # genuine kink, so a central difference straddling one measures the corner rather
            # than the derivative. Report those columns separately.
            if sig(xp) != sig(xm):
                switched += 1
                continue
            fd = (setter(xp) - setter(xm)) / (2 * eps)
            an = analytic[:, k]
            e = max(e, np.max(np.abs(fd - an)) / max(1.0, np.max(np.abs(an))))
        return e, switched, len(cols)

    def sig_q(x):
        a = controller._alloc(t, x, u)
        return (a.active, a.feasible, a.degenerate)

    def sig_u(x):
        a = controller._alloc(t, q, x)
        return (a.active, a.feasible, a.degenerate)

    eq, sq, nq_ = worst(Jq, q, lambda x: controller.la_tau(t, x, u), sig_q, len(q))
    eu, su, nu_ = worst(Ju, u, lambda x: controller.la_tau(t, q, x), sig_u, len(u))
    print(f"  t = {t:7.4f}  active = {str(al.active):8s} feasible = {str(al.feasible):5s} "
          f"alpha = {al.alpha:+.4f} | rel err  d/dq = {eq:.2e} ({nq_-sq}/{nq_} cols)  "
          f"d/du = {eu:.2e} ({nu_-su}/{nu_} cols)")
    if sq or su:
        print(f"{'':16s}skipped {sq} q- and {su} u-columns straddling an active-set switch")
    return eq, eu


def run_fd_check(t_sim=0.5, dt=1e-4, n_states=4, F_min=0.5, la_pre=None):
    print("--- finite-difference check of la_tau_q / la_tau_u ---")
    print("  (mismatch is expected exactly where the active set switches)")
    model, controller, sol, _ = run(t_sim=t_sim, dt=dt, names=("A",), t_hold=t_sim + 1.0,
                                    F_min=F_min, la_pre=la_pre)
    idxs = np.linspace(len(sol.t) // 4, len(sol.t) - 1, n_states, dtype=int)
    errs = [fd_check(controller, sol.t[i], sol.q[i][controller.qDOF], sol.u[i][controller.uDOF])
            for i in idxs]
    worst = max(max(a, b) for a, b in errs)
    print(f"  worst relative error         : {worst:.3e}   (gate: < 1e-6)")
    print("  RESULT                       :", "PASS" if worst < 1e-6 else "FAIL")
    return worst < 1e-6


# ----------------------------------------------------------------------------------


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = set(argv)

    # --f-min X : the tension floor. 0.0 asks only for non-negative forces, which is both easier
    # to satisfy and cheaper, since alpha = 0 stays feasible more often and no shift is applied.
    F_min = 0.5
    if "--f-min" in argv:
        F_min = float(argv[argv.index("--f-min") + 1])

    # --pre K:V  physical pretension V [N] on tendon K only (0-indexed), e.g. --pre 0:0.5
    # --pre V    the same pretension on every tendon
    # --smooth [--t-move X] : quintic-blended reference with consistent v_P_ref / a_P_ref,
    # plus a ramp out of the rod's actual initial tip position instead of a step at t = 0.
    smooth = "--smooth" in args
    t_move = float(argv[argv.index("--t-move") + 1]) if "--t-move" in argv else 0.5
    ramp_start = "--no-ramp" not in args

    # --start E : begin from the equilibrium configuration at a setpoint instead of straight
    start = argv[argv.index("--start") + 1] if "--start" in argv else None

    la_pre = None
    if "--pre" in argv:
        spec = argv[argv.index("--pre") + 1]
        la_pre = np.zeros(4)
        if ":" in spec:
            k, val = spec.split(":")
            la_pre[int(k)] = float(val)
        else:
            la_pre[:] = float(spec)

    if "--selftest" in args:
        raise SystemExit(0 if selftest() else 1)

    if "--fdcheck" in args:
        raise SystemExit(0 if run_fd_check(F_min=F_min, la_pre=la_pre) else 1)

    if "--probe" in args:
        probe()
        raise SystemExit(0)

    if "--compare" in args:
        compare(F_min=F_min)
        raise SystemExit(0)

    if "--short" in args:
        t_sim, names, t_hold = 2.0, ("A",), 10.0
    elif "--long" in args:  # 5 s per setpoint: every hold reaches steady state
        t_sim, names, t_hold = 25.0, ("A", "B", "C", "D", "E"), 5.0
    elif "--verylong" in args:  # two full sweeps, tests drift and repeatability
        t_sim, names, t_hold = 50.0, ("A", "B", "C", "D", "E", "A", "B", "C", "D", "E"), 5.0
    else:
        t_sim, names, t_hold = 10.0, ("A", "B", "C", "D", "E"), 2.0

    selftest(n_trials=500)

    model, controller, sol, r_OP_ref_fn = run(
        t_sim=t_sim, names=names, t_hold=t_hold, F_min=F_min, la_pre=la_pre,
        uniform_fallback="--fallback" in args,
        smooth=smooth, t_move=t_move, ramp_start=ramp_start, start=start,
    )
    rec = replay_alloc(controller, sol)
    report(rec, F_min, la_pre=la_pre)

    err = tracking(model, sol, r_OP_ref_fn)
    print(f"\n  tip tracking error [mm]      : mean {err.mean():.3f}, "
          f"median {np.median(err):.3f}, final {err[-1]:.3f}")

    plot_forces(rec, F_min)
    plot_leakage(rec)
    plot_shift(rec, F_min)

    if "--save" in args:
        out = Path(__file__).parent
        sfx = (
            (f"_{int(t_sim)}s" if t_sim > 10.0 else "")
            + (f"_fmin{F_min:g}" if F_min != 0.5 else "")
            + ("_pre" + "-".join(f"{p:g}" for p in la_pre) if la_pre is not None else "")
            + (f"_smooth{t_move:g}" if smooth else "")
            + ("_fallback" if "--fallback" in args else "")
            + (f"_start{start}" if start else "")
        )
        for name, fname in (
            ("WTauTendonForces", f"wtau_tendon_forces{sfx}.png"),
            ("WTauLeakage", f"wtau_leakage{sfx}.png"),
            ("WTauShift", f"wtau_shift{sfx}.png"),
        ):
            plt.figure(name).savefig(out / fname, dpi=130)
            print(f"  wrote {out / fname}")
    else:
        plt.show()
