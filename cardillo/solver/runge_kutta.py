import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp as _solve_ivp


def runge_kutta_4(dydt, y0, t0, tf, h, step_callback=lambda t, y: None, verbose=True):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    if verbose:
        pbar = tqdm(total=n, desc="Runge-Kutta 4th Order", unit="step")
    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * dydt(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t[i + 1] = t[i] + h

        step_callback(t[i + 1], y[i + 1])
        if verbose:
            pbar.update(1)

    return t, y


def runge_kutta_3_8(dydt, y0, t0, tf, h, step_callback=lambda t, y: None, verbose=True):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    f13 = 1 / 3
    f23 = 2 / 3
    if verbose:
        pbar = tqdm(total=n, desc="Runge-Kutta 3/8 Rule", unit="step")
    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + f13 * h, y[i] + f13 * k1)
        k3 = h * dydt(t[i] + f23 * h, y[i] - f13 * k1 + k2)
        k4 = h * dydt(t[i] + h, y[i] + k1 - k2 + k3)

        y[i + 1] = y[i] + (k1 + 3 * k2 + 3 * k3 + k4) / 8
        t[i + 1] = t[i] + h

        step_callback(t[i + 1], y[i + 1])
        if verbose:
            pbar.update(1)
    return t, y


def solve_ivp(
    dydt,
    y0,
    t0,
    tf,
    dt,
    method="BDF",
    step_callback=lambda t, y: None,
    verbose=True,
    rtol=1.0e-3,
    atol=1.0e-6,
    **kwargs,
):

    def _event(t, y):
        step_callback(t, y)
        return 1

    if verbose:
        pbar = tqdm(total=100, desc=f"IVP {method}", unit="pct")
        dydt.pbar_i = 0
        dydt.pbar_dt = (tf - t0) / 100

        def _dydt(t, y):
            i = int(np.floor((t - t0 + dydt.pbar_dt / 2) / dydt.pbar_dt))
            pbar.update(i - dydt.pbar_i)
            pbar.set_description(f"IVP {method}: {t:0.2e} s < {tf:0.2e} s")
            dydt.pbar_i = i
            return dydt(t, y)

    else:
        _dydt = dydt
    t_eval = np.arange(t0, tf, dt)
    t_eval[-1] = min(t_eval[-1], tf)  # Ensure the last time point does not exceed t_end
    sol = _solve_ivp(
        _dydt,
        (t0, tf),
        y0,
        t_eval=t_eval,
        method=method,
        dense_output=False,
        events=[_event],
        rtol=rtol,
        atol=atol,
        **kwargs,
    )
    assert sol.success, f"{method} solver failed: {sol.message}"
    if verbose:
        pbar.close()

    return sol.t, sol.y.T
