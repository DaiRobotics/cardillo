import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Sphere, Box
from cardillo.math import axis_angle2quat, e3, cross3, A_IB_basic, ax2skew
from cardillo.math.approx_fprime import approx_fprime
from cardillo.forces import Force
from cardillo.solver import ScipyDAE


class RollingCondition:
    """Rolling condition for rigid disc:
    - impenetrability on position level.
    - nonholonomic no sliding on velocity level."""

    def __init__(self, board: Frame, ball: RigidBody, la_g0=None, la_gamma0=None):
        self.mazeboard = board
        self.ball = ball

        self.nla_g = 1
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.nla_gamma = 2
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = np.concatenate([self.ball.qDOF, self.mazeboard.qDOF])
        self.uDOF = np.concatenate([self.ball.uDOF, self.mazeboard.uDOF])
        self.nq1 = self.ball.nq
        self.nu1 = self.ball.nu
        self.nq2 = self.mazeboard.nq
        self.nu2 = self.mazeboard.nu
        self.nq = self.nq1 + self.nq2
        self.nu = self.nu1 + self.nu2

    def r_CP(self, t, q):
        A_IB = self.mazeboard.A_IB(t)

        return -A_IB[:, 2] * self.ball.radius

    #################
    # non penetration
    #################
    def g(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        r_OB = self.mazeboard.r_OP(t)
        A_IB = self.mazeboard.A_IB(t)
        B_r_BC = A_IB.T @ (r_OC - r_OB)
        return e3 @ B_r_BC - self.ball.radius

    def g_dot(self, t, q, u):
        r_OC = self.ball.r_OP(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t)
        A_IB = self.mazeboard.A_IB(t)

        B_r_BC = A_IB.T @ (r_OC - r_OB)
        v_C = self.ball.v_P(t, q, u)
        v_B = self.mazeboard.v_P(t)
        B_Omega_B = self.mazeboard.B_Omega(t)
        return e3 @ (A_IB.T @ (v_C - v_B) - cross3(B_Omega_B, B_r_BC))

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_dot_q = approx_fprime(
            q, lambda q: self.g_dot(t, q, u), method="cs", eps=1.0e-15
        )

        return g_dot_q @ self.ball.q_dot(t, q, u) + self.g_dot_u(t, q) @ u_dot

    def g_q(self, t, q):
        return approx_fprime(
            q, lambda q: self.g(t, q), method="cs", eps=1.0e-15
        ).reshape(self.nla_g, self.ball.nq)

    def g_qq_dense(self, t, q):
        return approx_fprime(q, lambda q: self.g_q(t, q), method="3-point").reshape(
            self.nla_g, self.ball.nq, self.ball.nq
        )

    def g_q_T_mu_q(self, t, q, mu_g):
        return np.einsum("ijk,i", self.g_qq_dense(t, q), mu_g)

    def g_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_dot(t, q, u)).reshape(
            self.nla_g, self.ball.nq
        )

    def W_g(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        J_P1 = self.ball.J_P(
            t,
            q[: self.nq1],
        )

        r_OB = self.mazeboard.r_OP(t)
        J_P2 = self.mazeboard.J_P(t)
        A_IB = self.mazeboard.A_IB(t)
        B_J_R2 = self.mazeboard.B_J_R(t)

        B_r_BC = A_IB.T @ (r_OC - r_OB)
        J_P = np.zeros((3, self.nu))
        J_P[:, : self.nu1] = J_P1
        J_P[:, self.nu1 :] = -J_P2
        J_R = np.zeros((3, self.nu))
        J_R[:, self.nu1 :] = B_J_R2
        return e3 @ (A_IB.T @ J_P + ax2skew(B_r_BC) @ J_R)

    def Wla_g_q(self, t, q, la_g):
        return approx_fprime(q, lambda q: self.W_g(t, q) * la_g)

    ########################
    # no in plane velocities
    ########################

    def gamma(self, t, q, u):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        A_IK = self.ball.A_IB(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t)
        A_IB = self.mazeboard.A_IB(t)

        r_CP = -A_IB[:, 2] * self.ball.radius

        B_r_BP = A_IB.T @ (r_OC - r_OB)
        B_r_BP[2] = 0  # project to plane of maze board

        v_P = self.ball.v_P(t, q[: self.nq1], u[: self.nu1], B_r_CP=A_IK.T @ r_CP)
        v_P2 = self.mazeboard.v_P(t, B_r_CP=B_r_BP)
        return A_IB.T[:2] @ (v_P - v_P2)

    def gamma_dot(self, t, q, u, u_dot):
        gamma_q = approx_fprime(
            q, lambda q: self.gamma(t, q, u), method="cs", eps=1.0e-15
        )
        gamma_u = self.gamma_u(t, q)

        return gamma_q @ self.ball.q_dot(t, q, u) + gamma_u @ u_dot

    def gamma_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.gamma(t, q, u))

    def gamma_dot_q(self, t, q, u, u_dot):
        raise NotImplementedError("")

    def gamma_u(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        A_IK = self.ball.A_IB(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t)
        A_IB = self.mazeboard.A_IB(t)

        r_CP = -A_IB[:, 2] * self.ball.radius

        B_r_BP = A_IB.T @ (r_OC - r_OB)
        B_r_BP[2] = 0  # project to plane of maze board

        J_P1 = self.ball.J_P(t, q[: self.nq1], B_r_CP=A_IK.T @ r_CP)
        J_P2 = self.mazeboard.J_P(t, B_r_CP=B_r_BP)
        J_P = np.zeros((3, self.nu))
        J_P[:, : self.nu1] = J_P1
        J_P[:, self.nu1 :] = -J_P2
        return A_IB.T[:2] @ J_P

    def W_gamma(self, t, q):
        return self.gamma_u(t, q).T

    def Wla_gamma_q(self, t, q, la_gamma):
        return approx_fprime(q, lambda q: self.gamma_u(t, q).T @ la_gamma)


def disc(mass, radius, q0=None, u0=None):
    width = radius / 100
    A = 1 / 4 * mass * radius**2
    C = 1 / 2 * mass * radius**2
    B_Theta_C = np.diag(np.array([A, C, A]))

    disc = Sphere(RigidBody)(
        radius,
        A_BM=A_IB_basic(-np.pi / 2).x,
        B_r_CP=np.array([0, width / 2, 0]),
        mass=mass,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
    )
    return disc


def disc_boundary(disc, t, q, n=100):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
    B_r_CP = disc.radius * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
    return np.repeat(disc.r_OP(t, q), n).reshape(3, n) + disc.A_IB(t, q) @ B_r_CP


if __name__ == "__main__":
    """Analytical analysis of the rolling motion of a disc, see Lesaux2005
    Section 5 and 6.

    References
    ==========
    Lesaux2005: https://doi.org/10.1007/s00332-004-0655-4
    """

    ############
    # parameters
    ############
    gravity = 9.81  # gravity
    m = 0.3048  # disc mass

    # disc radius
    r = 0.05

    # inertia of the disc, Lesaux2005 before (5.3)
    A = B = 0.25 * m * r**2
    C = 0.5 * m * r**2

    ####################
    # initial conditions
    ####################
    case = "spinning"
    case = "rolling"

    if case == "spinning":
        # inclination angle is 0
        beta0 = 0
        beta_dot0 = 0

        # initial rolling velocity
        gamma_dot0 = 0
        # initial spinning velocity
        alpha_dot0 = 1

        # simulation time
        t1 = 2 * np.pi / np.abs(alpha_dot0)

    elif case == "rolling":
        # inclination angle is 0
        beta0 = 0
        beta_dot0 = 0

        # initial rolling velocity
        gamma_dot0 = 0
        # initial spinning velocity
        alpha_dot0 = 0

        # simulation time
        t1 = 2  # simulation time

    else:
        raise ValueError(
            f"Invalid initial condition case: '{case}'. Valid cases are 'spinning', 'rolling', 'circular'."
        )

    # center of mass
    x0 = 0
    y0 = -r * np.sin(np.deg2rad(0))
    z0 = r * np.cos(np.deg2rad(0))

    # angular velocity
    B_Omega0 = np.array(
        [beta_dot0, alpha_dot0 * np.sin(beta0) + gamma_dot0, alpha_dot0 * np.cos(beta0)]
    )

    # center of mass velocity
    v_C0 = np.array(
        [r * gamma_dot0 + r * alpha_dot0 * np.sin(beta0), -r * beta_dot0, 0]
    )

    # initial conditions
    t0 = 0
    p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
    q0 = np.array((x0, y0, z0, *p0))
    u0 = np.concatenate((v_C0, B_Omega0))

    #################
    # assemble system
    #################
    # create floor (Box only for visualization purposes)
    T = 1

    def A_IB_floor(t):
        return A_IB_basic(np.deg2rad(30) * np.sin(2 * np.pi * t / T)).x
        return A_IB_basic(np.deg2rad(30)).x

    R = 2
    board = Box(Frame)(
        dimensions=[2.2 * R, 2.2 * R, 0.0001],
        name="floor",
        A_IB=A_IB_floor,
    )

    # create disc
    disc = disc(m, r, q0, u0)

    # create rolling condition
    rolling_condition = RollingCondition(board, disc)

    # gravity
    f_g = Force(lambda t: np.array([0, 0, -m * gravity]), disc)

    # assemble system
    system = System()
    system.add(disc, rolling_condition, f_g, board)
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-2  # time step

    sol = ScipyDAE(system, t1, dt).solve()

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates
    u = sol.u  # velocity coordinates

    # compute bilateral constraint quantities
    g = np.array([system.g(ti, qi) for ti, qi in zip(t, q)])
    g = np.array([system.g(ti, qi) for ti, qi in zip(t, q)])
    g_dot = np.array([system.g_dot(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    gamma = np.array([system.gamma(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])

    #################
    # post-processing
    #################
    B_r_OP = np.array([board.A_IB(ti).T @ qi[:3] for ti, qi in zip(t, q)])
    r_OP = sol.q[:, disc.qDOF]
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    fig.suptitle("Evolution of constraint quantities")
    # g
    ax[0].plot(t, r_OP[:, 0])
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$x$")
    ax[0].grid()

    ax[1].plot(t, r_OP[:, 1])
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$y$")
    ax[1].grid()

    ax[2].plot(t, r_OP[:, 2])
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$z$")
    ax[2].grid()

    # plots
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 7))
    fig.suptitle("Evolution of constraint quantities")
    # g
    ax[0].plot(t, g)
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$g$")
    ax[0].grid()

    # g_dot
    ax[1].plot(t, g_dot)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel("$\\dot{g}$")
    ax[1].grid()

    # gamma_x
    ax[2].plot(t, gamma[:, 0])
    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$\\gamma_x$")
    ax[2].grid()

    # gamma_y
    ax[3].plot(t, gamma[:, 1])
    ax[3].set_xlabel("$t$")
    ax[3].set_ylabel("$\\gamma_y$")
    ax[3].grid()

    plt.tight_layout()
    plt.show()

    from cardillo.visualization.vtk_render2 import Plotter

    plotter = Plotter(system, (1000, 1000))
    plotter.render_solution(sol, True, 0.2)
    # animation
    t = t
    q = q

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = R
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    from collections import deque

    slowmotion = 1
    fps = 200
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        x_S, y_S, z_S = disc.r_OP(t, q)

        A_IB = disc.A_IB(t, q)
        d1 = A_IB[:, 0] * r
        d2 = A_IB[:, 1] * r
        d3 = A_IB[:, 2] * r

        (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
        (bdry,) = ax.plot([], [], [], "-k")
        (trace,) = ax.plot([], [], [], "--k")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, bdry, trace, d1_, d2_, d3_

    COM, bdry, trace, d1_, d2_, d3_ = create(0, q[0])

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        global x_trace, y_trace, z_trace
        if t == t0:
            x_trace = deque([])
            y_trace = deque([])
            z_trace = deque([])

        x_S, y_S, z_S = disc.r_OP(t, q)

        x_bdry, y_bdry, z_bdry = disc_boundary(disc, t, q)

        x_t, y_t, z_t = disc.r_OP(t, q) + rolling_condition.r_CP(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IB = disc.A_IB(t, q)
        d1 = A_IB[:, 0] * r
        d2 = A_IB[:, 1] * r
        d3 = A_IB[:, 2] * r

        COM.set_data(np.array([x_S]), np.array([y_S]))
        COM.set_3d_properties(np.array([z_S]))

        bdry.set_data(np.array(x_bdry), np.array(y_bdry))
        bdry.set_3d_properties(np.array(z_bdry))

        trace.set_data(np.array(x_trace), np.array(y_trace))
        trace.set_3d_properties(np.array(z_trace))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, bdry, trace, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()
    exit()
    # vtk-export
    dir_name = Path(__file__).parent
    e = system.export(dir_name, "vtk", sol)
    # additionally export body fixed frame
    e.export_contr(disc, file_name="A_IB", base_export=True)
