import numpy as np


def runge_kutta_4(dydt, y0, t0, tf, h, step_callback=lambda t, y: None):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * dydt(t[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * dydt(t[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t[i + 1] = t[i] + h

        step_callback(t[i + 1], y[i + 1])

    return t, y


def runge_kutta_3_8(dydt, y0, t0, tf, h, step_callback=lambda t, y: None):
    n = int((tf - t0) / h)

    t = np.zeros(n + 1)
    y = np.zeros((n + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    f13 = 1 / 3
    f23 = 2 / 3
    for i in range(n):
        k1 = h * dydt(t[i], y[i])
        k2 = h * dydt(t[i] + f13 * h, y[i] + f13 * k1)
        k3 = h * dydt(t[i] + f23 * h, y[i] - f13 * k1 + k2)
        k4 = h * dydt(t[i] + h, y[i] + k1 - k2 + k3)

        y[i + 1] = y[i] + (k1 + 3 * k2 + 3 * k3 + k4) / 8
        t[i + 1] = t[i] + h

        step_callback(t[i + 1], y[i + 1])
    return t, y
