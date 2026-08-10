# Positive tendon forces by QP projection — theory, and what it can and cannot buy

Companion to `qp_positive_controller.py` / `qp_positive_test.py` in this folder. All numbers
below are measured on the 4-tendon, 10-element `CommonModel` from `dynamic_control_test.py`.

Run everything from this directory (`scripts_tdcrobots/quadratic programming/`); both modules
put the parent on `sys.path` themselves, so the unmodified controllers one level up import
normally. Line references like `dynamic_controller.py:100` point at the parent directory.
Simulation results are cached in `qp_cache/`, so re-plotting is instant.

```
python qp_positive_test.py --check                          # KKT + finite-difference gate
python qp_positive_test.py --compare --t-sim 10 --t-hold 2  # the table in section 4
python qp_positive_test.py --replay --t-sim 10              # offline projection analysis
```

---

## 1. The control law being repaired

The plant is

$$M(q)\,\dot u = h(t,q,u) + W_c\lambda_c + W_\tau\lambda_\tau,\qquad \dot q = B(q)\,u$$

with tip position $y = C_1 q$ ($C_1$ selects the last node's $r$, `dynamic_controller.py:34-36`).
Differentiating twice and freezing $B$ and $M$ at their initial values (this is what
`build_M_tilde_inv` does, `dynamic_controller.py:38-44`):

$$\ddot y \;=\; \tilde M^{-1}\big(\tilde h + W_\tau\lambda_\tau\big),\qquad \tilde M^{-1} := (C_1B)M^{-1}\in\mathbb R^{3\times n_u}$$

Define the **task Jacobian** and the **drift**

$$J := \tilde M^{-1}W_\tau\;\in\mathbb R^{3\times4},\qquad \ddot y_0 := -\tilde M^{-1}\tilde h$$

The PD outer loop asks for $\ddot y = a$ with $a = \ddot y_{ref} + K_d(\dot y_{ref}-\dot y) + K_p(y_{ref}-y)$,
so the allocation problem is the linear system

$$\boxed{\;J\lambda_\tau = b,\qquad b := a + \ddot y_0\;}$$

3 equations, 4 unknowns. `DynamicControllerPD` picks the damped least-norm solution
$\lambda_{ref} = J^\top(JJ^\top+\delta I)^{-1}b$ (`dynamic_controller.py:91`). Nothing in that
expression constrains the sign, which is the whole problem: a tendon can only **pull**.

## 2. The QP

$$\min_x\;\tfrac12\lVert A(x-\lambda_{ref})\rVert^2\quad\text{s.t.}\quad x\ge f_{\min}$$

Substituting $y = x - f_{\min}\mathbf 1$ and $c = A(\lambda_{ref}-f_{\min}\mathbf 1)$ turns this into a
plain NNLS, $\min\lVert Ay-c\rVert$ s.t. $y\ge0$, which Lawson–Hanson solves exactly in finitely
many active-set steps. Convex, and strictly convex whenever $A$ has full column rank.

KKT, with $g = A^\top A(x-\lambda_{ref})$: $\;g_i = 0$ where $x_i > f_{\min}$, and $g_i \ge 0$ where
$x_i = f_{\min}$. Verified to machine precision in `--check` (residuals $\le 10^{-15}$).

**Derivatives.** The implicit solver needs $\partial\lambda_\tau/\partial q$ and $\partial\lambda_\tau/\partial u$. On the free
set $F$ the solution satisfies the normal equations $A_F^\top A_F\,y_F = A_F^\top c$; differentiating
at fixed active set,

$$\frac{\partial y_F}{\partial\theta} = (A_F^\top A_F)^{-1}\Big[(\partial_\theta A_F)^\top r + A_F^\top\big(\partial_\theta c - (\partial_\theta A_F)y_F\big)\Big],\qquad r = c - A_Fy_F$$

with clamped rows exactly zero. This is **exact almost everywhere** but the map is only
piecewise smooth — the Jacobian genuinely jumps where the active set changes. `--check` counts
those kink columns separately rather than reporting them as error. Verified against central
differences to $10^{-9}$ (q) and $10^{-11}$ (u).

> **A trap worth recording.** `DynamicControllerPD.la_tau_q` calls `self.la_tau(t,q,u)`
> internally (`dynamic_controller.py:111`) and feeds the result into the derivative of the
> damped pseudo-inverse. Under normal method resolution that call lands on the *subclass*
> override, so a subclass that changes `la_tau` silently corrupts the inherited `la_tau_q` —
> measured here as a 16% error. `QPPositiveController._as_baseline()` suppresses the override
> while the parent runs. Any future subclass of `DynamicControllerPD` that overrides `la_tau`
> has the same problem.

## 3. Why this robot fights back

Four tendons at $\varphi=0,\tfrac\pi2,\pi,\tfrac{3\pi}2$, all routed parallel to the backbone. Measured at
$q_0$:

| quantity | value |
|---|---|
| pairwise column correlations of $W_\tau$ | 0.9981 – 0.9990 |
| Gram $W_\tau^\top W_\tau$ | $\approx 2.0\cdot\mathbf{1}\mathbf{1}^\top$ (every entry 1.996–2.000) |
| singular values of $J$ | $[650.7,\;13.18,\;13.18]$ |
| $J^\top u_{ax}$, $u_{ax}=U_{:,1}$ | $[325.34,\;325.34,\;325.34,\;325.34]$ |

Three consequences, and they are the whole story.

### (a) The achievable axial acceleration is one-signed

$J^\top u_{ax} = \alpha\mathbf 1$ with $\alpha = 325.34 > 0$. Hence for **any** $\lambda\ge0$

$$u_{ax}^\top(J\lambda) = (J^\top u_{ax})^\top\lambda = \alpha\,(\mathbf 1^\top\lambda)\;\ge\;0$$

Pulling on tendons can only accelerate the tip one way along the backbone axis. If the PD law
demands $u_{ax}^\top b < 0$, the task is **infeasible**, not merely awkward. This is the Gordan
certificate already recorded in `nullspace_control_test.py:342-401`, restated.

### (b) The nullspace is orthogonal to the direction positivity cares about

$J$ is $3\times4$, so $\operatorname{null}(J) = \operatorname{span}\{v\}$ is one-dimensional — the redundancy the
nullspace controller tried to exploit. But $Jv=0$ together with $J^\top u_{ax}=\alpha\mathbf 1$ gives

$$\mathbf 1^\top v = \tfrac1\alpha\,(J^\top u_{ax})^\top v = \tfrac1\alpha\,u_{ax}^\top(Jv) = 0$$

Measured: $\langle\mathbf 1,v\rangle/\lVert v\rVert$ between $10^{-16}$ and $10^{-5}$. So **redistributing along
the nullspace cannot change $\mathbf 1^\top\lambda$ at all** — and $\mathbf 1^\top\lambda$ is exactly the quantity
(a) says positivity constrains. Free redundancy and the binding constraint are orthogonal.
That is why `NullspaceShiftController` bottoms out at $-0.67$ N; it is not a tuning failure.

### (c) Therefore the QP collapses to a vertex

Since $W_\tau^\top W_\tau\approx2\cdot\mathbf 1\mathbf 1^\top$, the objective is dominated by
$2\big(\mathbf 1^\top(x-\lambda_{ref})\big)^2$. Minimizing that over $x\ge0$:

- if $\mathbf 1^\top\lambda_{ref} < 0$, the best feasible uniform component is $0$, giving $x = \mathbf 0$;
- if $\mathbf 1^\top\lambda_{ref} > 0$, the near-rank-1 Gram makes the optimum a **single-tendon vertex**.

Measured over five test states (`--check`): $x=\mathbf 0$ in four of them, and $x=[0,\,5.66,\,0,\,0]$ in
the fifth — exactly the predicted on/off behaviour. The metric weights the axial direction
$\sigma_1/\sigma_2 = 49\times$ more than the two lateral ones, so the QP spends its entire budget
matching the one component that positivity has already made unmatchable, and discards the
bending information that actually steers the tip.

$A = J$ behaves the same ($x = [0,5.94,0,0]$) for the same reason: $J$ inherits the near-rank-1
structure from $W_\tau$.

## 4. What works

### Ranking of metrics, same state, same $\lambda_{ref} = [0.471,\,0.897,\,-0.459,\,-0.885]$

| metric $A$ | $x$ | Jacobian | comment |
|---|---|---|---|
| $W_\tau$ (as requested) | $[0,\;0.025,\;0,\;0]$ | exact, $10^{-9}$ | worst: axial weighted most heavily |
| $J$ | $[0,\;0.025,\;0,\;0]$ | exact, $10^{-9}$ | same collapse |
| `whitened` | $[0.258,\;1.110,\;0,\;0]$ | exact, $10^{-9}$ | all three task directions weighted equally |
| `axial` = $(I-u_{ax}u_{ax}^\top)J$ | $[0.930,\;1.782,\;0,\;0]$ | **wrong**, $0.3$–$62$ | Clarke as a QP; analysis only |

The ordering is not an accident: it is monotone in how much the metric de-weights $u_{ax}$.

**On differentiating the whitened metric.** The natural writing $A=\Sigma^{-1}U^\top J = V^\top$ is
not differentiable here: $\sigma_2 = \sigma_3$ *exactly* (both 13.18), so $U$'s lateral columns are
only defined up to a rotation and standard SVD-derivative formulas divide by $\sigma_2^2-\sigma_3^2 = 0$.
The objective is fine though — only $A^\top A$ enters the QP, and

$$A^\top A \;=\; J^\top U\Sigma^{-2}U^\top J \;=\; J^\top (JJ^\top)^{-1} J \;=:\; P$$

the orthogonal projector onto $\operatorname{row}(J)$, which contains no SVD at all. Since $P$ is
symmetric idempotent, $P^\top P = P$, so taking $A = P$ gives the identical QP with an exact
closed-form derivative ($M = S^{-1}J$, $S = JJ^\top$, $D = \partial J/\partial q_k$):

$$\partial P = D^\top M + M^\top D - M^\top\!\big(DJ^\top + JD^\top\big)M$$

Implemented as `d_rowproj`. This drops the measured Jacobian error from $8.7\times10^{-2}$–$35$
(frozen-SVD) to $1.3\times10^{-9}$, which is what makes `whitened` usable in a closed loop at all.
The `axial` metric has no such escape — it needs $u_{ax}$ itself, not just $A^\top A$ — so it stays
analysis-only.

### The one genuine knob: uniform pretension

$J\mathbf 1$ is **purely axial** — measured lateral/axial ratio $8.6\times10^{-15}$ at $q_0$, $\sim10^{-8}$ under
perturbation. So adding $s\mathbf 1$ to any allocation:

- raises **every** tendon force by exactly $s$, hence buys positivity outright;
- costs **only** axial task error, leaving the two lateral (bending) directions untouched.

That makes $f_{\min}$ the only lever that genuinely trades a well-defined quantity for
positivity, and it is why `MinimalAxialShiftController`'s closed form
$s^\star = \max(0,\,f_{\min}-\min_i\lambda_i)$ is the right shape of answer. In QP language: use
$f_{\min}>0$, not $f_{\min}=0$.

### Closed loop — it works, and much better than the static analysis suggests

Full A→B→C→D→E sweep, `t_sim=10`, `t_hold=2`, `dt=1e-4`, `ScipyDAE`. "settled" averages the last
25% of each hold window, excluding the step transients that otherwise dominate the mean.

| configuration | min force | % negative | mean err | settled err | completed |
|---|---|---|---|---|---|
| baseline (unconstrained) | **−1.3867 N** | **20.1%** | 5.979 mm | 0.000 mm | 10.0 s |
| QP `W_tau`, `f_min=0` | 0.0000 N | 0.0% | 7.760 mm | 0.509 mm | 10.0 s |
| QP `W_tau`, `f_min=0.5` | 0.5000 N | 0.0% | 8.200 mm | 1.038 mm | 10.0 s |
| QP `whitened`, `f_min=0.5` | 0.5000 N | 0.0% | 8.369 mm | 2.465 mm | **died t=6.06** |
| QP `J`, `f_min=0.5` | 0.5000 N | 0.0% | 32.926 mm | 34.322 mm | **died t=4.01** |

Positivity holds exactly, and the tip still converges to within half a millimetre. The static
analysis is pessimistic because it ignores feedback: when the QP under-delivers, the tracking
error grows, which grows $\lVert b\rVert$, which pushes $\mathbf 1^\top\lambda_{ref}$ positive, at which
point the tendons engage. The loop self-corrects what the allocation cannot. **The price of
positivity on this robot is about 0.5 mm of settled tracking error** — far cheaper than the
vertex-collapse picture in §3(c) would predict.

**Only the $W_\tau$ metric survives the full run.** `ScipyDAE` returns a short solution instead of
raising when it gives up, so this is easy to miss — `truncated()` in the test harness now checks
`t[-1]` against `t_sim` and the comparison plot greys out the dead region. The error figures for
the two truncated rows are averages over the part that ran and are not comparable to the others.

The `J` metric is worst: it develops a sustained **limit cycle** through the first setpoint
(clearly visible in the tip trace, roughly 7 oscillations over 0–2 s) before the solver gives up
at t=4.01. Its objective has only three rows and drops the $W_\tau$ row weighting that keeps the
$W_\tau$ metric anchored, so its allocation is free to wander along the nullspace — and it does,
oscillating rather than settling. This is the same failure mode as the limit-cycle behaviour
seen elsewhere in this project, reached from a new direction.

### A rank-deficiency trap

The undamped row-space projector $P = J^\top(JJ^\top)^{-1}J$ raises `LinAlgError: Singular matrix`
partway through a closed-loop run: $\sigma_3\to0$ at some configurations along the trajectory (already
visible in `--check`, where $\sigma$ degrades to $[650.9,\,6.59,\,0.60]$ under perturbation). $S$ therefore
carries the same Tikhonov term the controller already uses, $S = JJ^\top+\delta I$. This costs
idempotence — the objective becomes $\lVert P_\delta(x-\lambda_{ref})\rVert^2$, weighting each row direction
by $(\sigma^2/(\sigma^2+\delta))^2$ — which is arguably the better metric anyway: it fades out
directions the robot has nearly lost authority over rather than amplifying them.

## 5. Summary

**Works**

- The QP is correctly posed, convex, and solved exactly; KKT to $10^{-15}$.
- Analytic Jacobians are exact a.e. and verified to $10^{-9}$ — `ScipyDAE` converges normally.
- Positivity is guaranteed unconditionally: `min force = 0.0000 N`, 0% negative samples, against
  a baseline that spends **20.1% of the run pushing** and reaches −1.39 N.
- Reduces to the unmodified `DynamicControllerPD` bit-for-bit whenever no constraint binds
  (verified: $\lvert x-\lambda_{ref}\rvert = 10^{-13}$, Jacobian agreement $10^{-9}$).
- Closed loop over the full 10 s sweep costs about **0.5 mm of settled tracking error** — the
  headline result, and a genuinely cheap price.
- $f_{\min}$ behaves exactly as the theory says: `f_min=0.5` holds every tendon at or above
  0.5 N for the whole run, at 1.0 mm settled error.

**Does not work**

- The $W_\tau$ metric *judged statically*. Near-rank-1 Gram ⇒ the QP is essentially a scalar
  problem in $\mathbf 1^\top x$ and its output collapses to $\mathbf 0$ or to a single-tendon
  vertex. It is rescued by feedback, not by the allocation — see §4.
- **Every reweighted metric, in closed loop.** `J` limit-cycles and dies at t=4.01; `whitened`
  dies at t=6.06 despite an exact Jacobian and a well-conditioned objective. Reweighting away
  from $W_\tau$ makes the allocation less anchored, not more capable — the redundancy it frees up
  is the redundancy §3(b) proves is useless, so the extra freedom buys oscillation rather than
  accuracy. This is the main surprise of the study: §3's static ranking is *exactly inverted* by
  the closed loop.
- The `axial` metric even as a controller candidate. Its $A$ depends on $u_{ax}$, which rotates
  with $q$; freezing it gives Jacobian errors of 0.3–62, so it is `--replay`/`--check` only.
- The undamped row-space projector: outright `LinAlgError` mid-run.
- Any hope that the QP recovers the *task*. It cannot: by (a) the demanded acceleration is often
  outside the reachable cone, so *no* algorithm — QP, NNLS, nullspace, or otherwise — can hold
  $J\lambda = b$ with $\lambda \ge 0$. The QP chooses *how* to fail, not *whether* to.
- Nullspace redistribution as a positivity mechanism, for the reason in (b). Already known in
  this repo; the QP result is an independent confirmation from a different direction.

## 6. Why `wtau_nullspace_controller.py` cannot do this, measured

`nullspace_vs_qp.py` runs the unconstrained baseline once and analyses 4001 samples of it with
both allocators. Both start from the same $F_p$ and both aim to preserve $W_\tau F_p$; they differ
only in what they are allowed to search:

| | search set | problem type |
|---|---|---|
| `wtau_nullspace` | the **line** $\{F_p+\alpha v\}$, $W_\tau v\approx0$ | feasibility — the set may be empty |
| QP | the **cone** $\{x\ge f_{\min}\}$ | optimization — never empty, $f_{\min}\mathbf 1$ always feasible |

### Result

| | value |
|---|---|
| $\lvert\mathbf 1^\top v\rvert$, max | $4.8\times10^{-4}$ |
| $\lvert\mathbf 1^\top F - \mathbf 1^\top F_p\rvert$, max | $4.3\times10^{-4}$ |
| nullspace best achievable min force | min **−1.0055 N**, median 1.78 N, max 10.44 N |
| samples where nullspace **cannot** reach 0 N | **17.6%** (704 / 4001) |
| QP achieved min force | 0.0000 N, **0.0%** below zero |

### Two distinct obstructions, and the small one is not the important one

**(a) The sum is conserved.** $v\approx(1,-1,1,-1)/2$ is an antagonistic pair swap, so
$\mathbf 1^\top v\approx0$ and $\mathbf 1^\top F$ is invariant along the whole line (confirmed:
$4.3\times10^{-4}$). Since $F\ge0\Rightarrow\mathbf 1^\top F\ge0$, any configuration with
$\mathbf 1^\top F_p<0$ is *provably* infeasible for every $\alpha$. Confirmed exactly: all 17 such
samples have `achievable_floor` $<0$, no exceptions.

**But this accounts for only 17 of the 704 failures — 2.4%.** It is the clean argument, not the
operative one.

**(b) One scalar against four inequalities.** Of the 3984 samples with $\mathbf 1^\top F_p\ge0$,
**687 still cannot be made positive** — 17.2% of the run. $\mathbf 1^\top F_p\ge0$ is necessary but
nowhere near sufficient: a 1-D line generically misses a 4-facet cone. This dominates (a) by
about 40:1 and is the real reason the method fails.

So the honest summary is *dimensional*, not algebraic: the redundancy available to a 4-tendon,
3-DOF robot is one scalar, and positivity is four constraints. The sum invariant is a sharp
special case that happens to be rare here.

### What the nullspace method is genuinely better at

It preserves the wrench more faithfully than the QP, exactly as designed:

| | mean | max |
|---|---|---|
| nullspace leak $\lVert W_\tau(F-F_p)\rVert$ | $1.19\times10^{-3}$ | $7.9\times10^{-3}$ |
| QP residual $\lVert W_\tau(x-F_p)\rVert$ | $4.79\times10^{-3}$ | $9.5\times10^{-1}$ |

Both are ~$10^{-15}$ for most of the run and spike only in two brief windows (startup and the
t≈4.1 setpoint jump) — i.e. exactly where the constraint binds. So this is not "QP good,
nullspace bad". It is a **fidelity-versus-feasibility trade**: the nullspace method is nearly
exact when it succeeds and simply fails 17.6% of the time; the QP always succeeds and pays up to
$0.95$ of wrench residual in the brief windows where it must.

One caveat on the leak: $\sigma_4/\sigma_1 \le 2.8\times10^{-3}$ looks negligible, but against the
lateral singular values $\sigma_2=\sigma_3\approx0.081$ it reaches $9.8\times10^{-2}$, and the leaked tip
acceleration $\lVert J(F-F_p)\rVert$ averages $0.25$ and peaks at $1.68$ m/s². The kernel is exact
only at the straight configuration.

**Recommendation**

Use `metric="W_tau"` — the formulation as originally posed — with $f_{\min}$ set to whatever
minimum tension the hardware needs to keep tendons taut. It costs ~0.5 mm settled at
$f_{\min}=0$ and ~1.0 mm at $f_{\min}=0.5$, guarantees positivity unconditionally, and is the
only variant that integrates to completion.

The reweightings in §4 are worth keeping in the file as analysis tools (`--replay` quantifies
what each one throws away) but not as controllers. The original instinct — match the full
generalized force $W_\tau\lambda_\tau$, not some projection of it — turned out to be the right one, for a
reason the static analysis actively argued against: the $W_\tau$ metric's "over-weighting" of the
axial direction is what keeps the allocation pinned down, and the feedback loop handles the
axial infeasibility on its own.
