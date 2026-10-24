import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.linalg import eig, cdf2rdf
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous

from solve_dae.integrate._dae.radau import (
    DAEDenseOutput,
    radau_constants,
    predict_factor,
    RadauDAE,
)


def solve_collocation_system(
    fun,
    t,
    y,
    h,
    Yp0,
    scale,
    tol,
    LU_real,
    LU_complex,
    solve_lu,
    C,
    T,
    TI,
    A,
    newton_max_iter,
    fun_vec=None,
):
    s, n = Yp0.shape
    ncs = s // 2
    tau = t + h * C

    Yp = Yp0.copy()
    Y = y + h * A.dot(Yp)

    F = np.empty((s, n))

    dY_norm_old = None
    dW_dot = np.empty_like(F)
    converged = False
    thq_old = None
    rate = None
    for k in range(newton_max_iter):
        if fun_vec is not None:
            F = np.asarray(fun_vec(tau, Y, Yp))
        else:
            for i in range(s):
                F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        G = TI @ F
        G_real = -G[0]
        G_complex = np.empty((ncs, n), dtype=complex)
        for i in range(ncs):
            G_complex[i] = -(G[2 * i + 1] + 1j * G[2 * i + 2])

        dW_dot_real = solve_lu(LU_real, G_real)
        dW_dot_complex = np.empty_like(G_complex)
        for i in range(ncs):
            dW_dot_complex[i] = solve_lu(LU_complex[i], G_complex[i])

        dW_dot[0] = dW_dot_real
        for i in range(ncs):
            dW_dot[2 * i + 1] = dW_dot_complex[i].real
            dW_dot[2 * i + 2] = dW_dot_complex[i].imag

        dYp = T.dot(dW_dot)
        dY = h * A.dot(dYp)

        Yp += dYp
        Y += dY

        dY_norm = norm(dY / scale)
        if dY_norm_old is not None:
            thq = dY_norm / dY_norm_old
            rate = thq if thq_old is None else np.sqrt(thq * thq_old)
            thq_old = thq

        if rate is not None and (
            rate >= 1 or rate ** (newton_max_iter - k) / (1 - rate) * dY_norm > tol
        ):
            break

        if dY_norm == 0 or rate is not None and rate / (1 - rate) * dY_norm < tol:
            converged = True
            break

        dY_norm_old = dY_norm

    return converged, k + 1, Y, Yp, rate


class MyRadauDAE(RadauDAE):
    def __init__(self, *args, fun_vec=None, **kwargs):

        RadauDAE.__init__(self, *args, **kwargs)

        def _fun_vec(*args):
            self.nfev += self.stages
            return fun_vec(*args)

        self.fun_vec = _fun_vec if fun_vec is not None else None

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp

        s = self.stages
        MU_REAL = self.MU_REAL
        MU_COMPLEX = self.MU_COMPLEX
        C = self.C
        T = self.T
        TI = self.TI
        A = self.A
        v_implicit = self.v_implicit
        v_explicit = self.v_explicit
        b_hat1_implicit = self.b_hat1_implicit
        P2 = self.P2

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol
        newton_tol = self.newton_tol
        newton_max_iter = self.newton_max_iter

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        Jy = self.Jy
        Jyp = self.Jyp
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        factor = None
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Yp0 = np.tile(yp, s).reshape(s, -1)
            else:
                # improved guess by extrapolating the collocation polynomial
                Yp0 = self.sol(t + h * C)[1].T
            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(Jyp + h * MU_REAL * Jy)
                    LU_complex = [self.lu(Jyp + h * MU * Jy) for MU in MU_COMPLEX]

                converged, n_iter, Y, Yp, rate = solve_collocation_system(
                    self.fun,
                    t,
                    y,
                    h,
                    Yp0,
                    scale,
                    newton_tol,
                    LU_real,
                    LU_complex,
                    self.solve_lu,
                    C,
                    T,
                    TI,
                    A,
                    newton_max_iter,
                    fun_vec=self.fun_vec,
                )

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                self.nrejct += 1
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = Y[-1]
            yp_new = Yp[-1]

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol

            # error of collocation polynomial of order s
            error_collocation = y - P2[0] @ Y

            # embedded error measures
            if self.newton_iter_embedded == 0:
                # explicit embedded method with R(z) = +oo for z -> oo
                error_embedded = h * (MU_REAL * yp + v_explicit @ Yp)
            elif self.newton_iter_embedded == 1:
                # compute implicit embedded method with a single Newton iteration;
                # R(z) = b_hat1 / b_hats2 = DAMPING_RATIO_ERROR_ESTIMATE for z -> oo
                yp_hat_new = (v_implicit @ Yp - b_hat1_implicit * yp) / MU_REAL
                F = self.fun(t_new, y_new, yp_hat_new)
                error_embedded = -h * MU_REAL * self.solve_lu(LU_real, F)
            else:
                # compute implicit embedded method with `newton_iter_embedded` iterations
                yp_hat_new0 = (
                    -(y / h + b_hat1_implicit * yp + self.b_hat_implicit @ Yp) / MU_REAL
                )
                y_hat_new = y_new.copy()  # initial guess
                for _ in range(self.newton_iter_embedded):
                    yp_hat_new = yp_hat_new0 + y_hat_new / (h * MU_REAL)
                    F = self.fun(t_new, y_hat_new, yp_hat_new)
                    y_hat_new -= h * MU_REAL * self.solve_lu(LU_real, F)

                error_embedded = y_hat_new - y_new

            # mix embedded error with collocation error as proposed in Guglielmi2001/ Guglielmi2003
            error = self.continuous_error_weight * np.abs(error_collocation) ** (
                (s + 1) / s
            ) + (1 - self.continuous_error_weight) * np.abs(error_embedded)
            error_norm = norm(error / scale)

            safety = 0.9 * (2 * newton_max_iter + 1) / (2 * newton_max_iter + n_iter)

            if error_norm > 1:
                self.nrejct += 1
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
                h_abs *= safety * factor

                LU_real = None
                LU_complex = None
            else:
                self.naccpt += 1
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = (
            jac is not None
            and n_iter > self.jac_recompute_iter
            and rate > self.jac_recompute_rate
        )

        if factor is None:
            factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
            factor = safety * factor

        # do not alter step-size in deadband
        if (
            not recompute_jac
            and self.controller_deadband[0] <= factor <= self.controller_deadband[1]
        ):
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        if recompute_jac:
            Jy, Jyp = self.jac(t_new, y_new, yp_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y
        self.yp_old = yp

        self.t = t_new
        self.y = y_new
        self.yp = yp_new

        self.Y = Y
        self.Yp = Yp

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.Jy = Jy
        self.Jyp = Jyp

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message


import inspect
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.integrate._ivp.ivp import prepare_events, find_active_events

from solve_dae.integrate._dae.dae import (
    DaeSolver,
    BDFDAE,
    RadauDAE,
    DaeSolution,
    EPS,
    MESSAGES,
    DaeResult,
)

METHODS = {
    "BDF": BDFDAE,
    "Radau": MyRadauDAE,
}


def solve_dae(
    fun,
    t_span,
    y0,
    yp0,
    method="Radau",
    t_eval=None,
    dense_output=False,
    events=None,
    var_index=None,
    vectorized=False,
    args=None,
    fun_vec=None,
    **options,
):
    if var_index is not None:
        raise NotImplementedError(
            "var_index is not implemented yet: error estimates are not "
            "currently scaled by differentiation index. Pass None (the "
            "default), which treats every equation as index <= 1."
        )

    if method not in METHODS and not (
        inspect.isclass(method) and issubclass(method, DaeSolver)
    ):
        raise ValueError(f"`method` must be one of {METHODS} or DaeSolver class.")

    t0, t_bound = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters. Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        def fun(t, y, yp, fun=fun):
            return fun(t, y, yp, *args)

        jac = options.get("jac")
        if callable(jac):
            options["jac"] = lambda t, y, yp: jac(t, y, yp, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, t_bound)) or np.any(t_eval > max(t0, t_bound)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if t_bound > t0 and np.any(d <= 0) or t_bound < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if t_bound > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(
        fun, t0, y0, yp0, t_bound, vectorized=vectorized, fun_vec=fun_vec, **options
    )

    if t_eval is None:
        ts = [t0]
        ys = [y0]
        yps = [yp0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
        yps = []
    else:
        ts = []
        ys = []
        yps = []

    interpolants = []

    if events is not None:
        events, max_events, event_dir = prepare_events(events)
        event_count = np.zeros(len(events))
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [
                lambda t, y, yp, event=event: event(t, y, yp, *args) for event in events
            ]
        g = [event(t0, y0, yp0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
        yp_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None
        yp_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == "finished":
            status = 0
        elif solver.status == "failed":
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y
        yp = solver.yp

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            g_new = [event(t, y, yp) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                event_count[active_events] += 1
                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, event_count, max_events, t_old, t
                )

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    ye, ype = sol(te)
                    y_events[e].append(ye)
                    yp_events[e].append(ype)

                if terminate:
                    status = 1
                    t = roots[-1]
                    y, yp = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
            yps.append(yp)
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side="right")
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side="left")
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                y, yp = sol(t_eval_step)
                ys.append(y)
                yps.append(yp)
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]
        yp_events = [np.asarray(ype) for ype in yp_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
        yps = np.vstack(yps).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)
        yps = np.hstack(yps)

    if dense_output:
        if t_eval is None:
            sol = DaeSolution(ts, interpolants)
        else:
            sol = DaeSolution(ti, interpolants)
    else:
        sol = None

    return DaeResult(
        t=ts,
        y=ys,
        yp=yps,
        sol=sol,
        t_events=t_events,
        y_events=y_events,
        yp_events=yp_events,
        nfev=solver.nfev,
        njev=solver.njev,
        nlu=solver.nlu,
        nlusolve=solver.nlusolve,
        naccpt=solver.naccpt,
        nrejct=solver.nrejct,
        nstep=solver.naccpt + solver.nrejct,
        status=status,
        message=message,
        success=status >= 0,
    )
