import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp


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


def solve_ivp_sequence(
    dydt,
    y0,
    t0,
    tf,
    dt_sequence=1e-2,
    method="RK45",
    max_step=1e-3,
    rtol=1.0e-8, atol=1.0e-10,
    step_callback=lambda t, y: None,
    ivp_callback=lambda t, y: None,
    verbose=True,
):
    results = {"t": [], "y": [], "event_times": []}
    t_current = t0
    y_current = y0

    def _step_callback(t, y):
        step_callback(t, y)
        return 1

    if verbose:
        pbar = tqdm(
            total=int((tf - t0) / dt_sequence ), desc=f"Method {method}", unit="event"
        )
    i = 0
    while t_current < tf:
        t_end = min(t0 + (i + 1) * dt_sequence , tf)
        t_eval = np.arange(t_current, t_end + max_step/2, max_step)
        t_eval[-1] = t_end  # Ensure the last time point is exactly t_end
        sol = solve_ivp(
            dydt,
            (t_current, t_end),
            y_current,
            t_eval=t_eval,
            method=method,
            dense_output=False,
            events=[_step_callback],
            max_step=max_step,
            rtol=rtol, atol=atol
        )
        t = sol.t
        y = sol.y.T
        # save results
        results["t"].extend(t[:-1])
        results["y"].extend(y[:-1])

        t_current = t[-1]
        y_current = y[-1]

        ivp_callback(t_current, y_current)
        if verbose:
            pbar.update(1)

        i += 1
        if t_current >= tf:
            results["t"].extend([t_current])
            results["y"].extend([y_current])
            break

    return np.array(results["t"]), np.array(results["y"])
