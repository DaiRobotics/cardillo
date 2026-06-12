import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from vtk import VTK_VERTEX

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Sphere, Box
from cardillo.math import axis_angle2quat, e3, cross3, A_IB_basic, ax2skew, Log_SO3_quat
from cardillo.math.approx_fprime import approx_fprime
from cardillo.forces import Force
from cardillo.solver import ScipyDAE, BackwardEuler, Moreau


class MazeBoard:
    def __init__(
        self,
        name="mazaboard",
        **kwargs,
    ):
        """Frame parameterized by time dependent position and orientation.

        Parameters
        ----------
        name : str
            Name of frame.
        """
        self.name = name

        self.nq = 2
        self.nu = 2
        self.q0 = np.zeros(2, float)
        self.u0 = np.zeros(2, float)

        self.constant_mass_matrix = True

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.eye(2)

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return np.eye(2)

    #####################
    # auxiliary functions
    #####################
    def local_qDOF_P(self, xi=None):
        return np.arange(self.nq)

    def local_uDOF_P(self, xi=None):
        return np.arange(self.nu)

    def A_IB(self, t, q, xi=None):
        return A_IB_basic(q[0]).x @ A_IB_basic(q[1]).y

    def A_IB_q(self, t, q, xi=None):
        return np.concatenate(
            (
                (A_IB_basic(q[0]).dx @ A_IB_basic(q[1]).y)[..., None],
                (A_IB_basic(q[0]).x @ A_IB_basic(q[1]).dy)[..., None],
            ),
            axis=-1,
        )

    def r_OP(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return self.A_IB(t, q) @ B_r_CP

    def r_OP_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        return B_r_CP @ self.A_IB_q(t, q)

    def v_P(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        A_IB_q = self.A_IB_q(t, q)
        return A_IB_q @ u @ B_r_CP

    def v_P_q(self, t, q, u, xi=None, B_r_CP=np.zeros(3)):
        raise NotImplementedError

    def J_P(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        J_P = np.zeros((3, 2))
        A_IB_q = self.A_IB_q(t, q)
        J_P[:, 0] = A_IB_q @ np.array([1, 0]) @ B_r_CP
        J_P[:, 1] = A_IB_q @ np.array([0, 1]) @ B_r_CP
        return J_P

    def J_P_q(self, t, q, xi=None, B_r_CP=np.zeros(3)):
        raise
        return np.empty((3, 0, 0))

    def a_P(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        raise NotImplementedError

    def a_P_q(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        raise NotImplementedError

    def a_P_u(self, t, q, u, u_dot, xi=None, B_r_CP=np.zeros(3)):
        raise NotImplementedError

    def B_Omega(self, t, q, u, xi=None):
        B_omega_IB = np.array([0, u[1], 0]) + A_IB_basic(q[1]).y.T @ np.array(
            [u[0], 0, 0]
        )
        return B_omega_IB

    def B_Omega_q(self, t, q, u, xi=None):
        B_Omega_q = np.zeros((3, 2))
        B_Omega_q[:, 1] = A_IB_basic(q[1]).dy.T @ np.array([u[0], 0, 0])
        return B_Omega_q

    def B_J_R(self, t, q, xi=None):
        B_J_R = np.zeros((3, 2))
        B_J_R[:, 0] = A_IB_basic(q[1]).y.T @ np.array([1, 0, 0])
        B_J_R[1, 1] = 1
        return B_J_R

    def B_J_R_q(self, t, q, xi=None):
        B_J_R_q = np.zeros((3, 2, 2))
        B_J_R_q[:, 0, 1] = A_IB_basic(q[1]).dy.T @ np.array([1, 0, 0])
        return B_J_R_q


class Controller:
    def __init__(self, ball: RigidBody, board: MazeBoard):
        self.ball = ball
        self.board = board
        self.Kd_angle = 1e3
        self.Kp_angle = 0.25 * self.Kd_angle**2
        self.q_des = np.zeros(2)

    def assembler_callback(self):
        self.qDOF = np.concatenate((self.board.qDOF, self.ball.qDOF))
        self.uDOF = np.concatenate((self.board.uDOF, self.ball.uDOF))
        self.nq1 = len(self.board.qDOF)
        self.nu1 = len(self.board.uDOF)
        self.nq = len(self.qDOF)
        self.nu = len(self.uDOF)

    def step_callback(self, t, q, u):
        self.ball_control(t, q, u)
        return q, u

    @staticmethod
    def ball_traj(t):
        T = 5
        rho = 0.02
        x = rho * (np.cos(2 * np.pi * t / T) - 1)
        x_t = -(2 * np.pi / T) * rho * np.sin(2 * np.pi * t / T)
        x_tt = -((2 * np.pi / T) ** 2) * rho * np.cos(2 * np.pi * t / T)
        y = rho * np.sin(2 * np.pi * t / T)
        y_t = (2 * np.pi / T) * rho * np.cos(2 * np.pi * t / T)
        y_tt = -((2 * np.pi / T) ** 2) * rho * np.sin(2 * np.pi * t / T)
        r_ref = np.array([x, y])
        r_ref_t = np.array([x_t, y_t])
        r_ref_tt = np.array([x_tt, y_tt])

        # r_ref = np.array([x, y * 0])
        # r_ref_t = np.array([x_t, y_t * 0])
        # r_ref_tt = np.array([x_tt,  y_tt * 0])

        # r_ref = np.array([0.02, 0])
        # r_ref_t = np.zeros_like(r_ref)
        # r_ref_tt = np.zeros_like(r_ref)
        return r_ref, r_ref_t, r_ref_tt

    def ball_control(self, t, q, u):
        # board
        B_Omega_B = self.board.B_Omega(t, q[: self.nq1], u[: self.nq1])
        r_OB = self.board.r_OP(t, q[: self.nq1])
        A_IB = self.board.A_IB(t, q[: self.nq1])
        v_B = self.board.v_P(t, q[: self.nq1], u[: self.nu1])

        # ball
        r_OC = self.ball.r_OP(t, q[self.nq1 :])
        v_C = self.ball.v_P(t, q[self.nq1 :], u[self.nu1 :])
        B_Omega_B = self.board.B_Omega(t, q[: self.nq1], u[: self.nu1])

        B_r_BC = A_IB.T @ (r_OC - r_OB)
        B_r_BC_t = A_IB.T @ (v_C - v_B) - cross3(B_Omega_B, B_r_BC)

        # PD Controller for ball position
        r = B_r_BC[:2]
        r_t = B_r_BC_t[:2]
        r_ref, r_ref_t, r_ref_tt = self.ball_traj(t)
        # desired angles
        err = r_ref - r
        err_t = r_ref_t - r_t
        self.q_des[0] = -(r_ref_tt[1] + 5 * err_t[1] + 0.25 * 25 * err[1]) / 9.81
        self.q_des[1] = (r_ref_tt[0] + 5 * err_t[0] + 0.25 * 25 * err[0]) / 9.81

    def angle_control(self, t, q, u):
        u_des = np.zeros(2)
        return self.Kd_angle * (u_des - u[: self.nu1]) + self.Kp_angle * (
            self.q_des - q[: self.nq1]
        )

    def h(self, t, q, u):
        # PD Controller for board angles
        h = np.zeros(self.nu)
        h[: self.nu1] = self.angle_control(t, q, u)
        return h

    def h_q(self, t, q, u):
        h_q = np.zeros((self.nu, self.nq))
        h_q[: self.nu1, : self.nq1] = -np.eye(2) * self.Kp_angle
        return h_q

    def h_u(self, t, q, u):
        h_u = np.zeros((self.nu, self.nu))
        h_u[: self.nu1, : self.nu1] = -np.eye(2) * self.Kd_angle
        return h_u


class RollingCondition:
    """Rolling condition for rigid ball:
    - impenetrability on position level.
    - nonholonomic no sliding on velocity level."""

    def __init__(
        self,
        board: MazeBoard,
        ball: RigidBody,
        la_g0=None,
        la_gamma0=None,
    ):
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
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])

        return -A_IB[:, 2] * self.ball.radius

    #################
    # non penetration
    #################
    def g(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t, q[self.nq1 :])
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])

        B_r_BC = A_IB.T @ (r_OC - r_OB)
        return e3 @ B_r_BC - self.ball.radius

    def g_dot(self, t, q, u):
        r_OC = self.ball.r_OP(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t, q[self.nq1 :])
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])

        B_r_BC = A_IB.T @ (r_OC - r_OB)
        v_C = self.ball.v_P(t, q[: self.nq1], u[: self.nu1])
        v_B = self.mazeboard.v_P(t, q[self.nq1 :], u[self.nu1 :])
        B_Omega_B = self.mazeboard.B_Omega(t, q[self.nq1 :], u[self.nu1 :])
        return e3 @ (A_IB.T @ (v_C - v_B) - cross3(B_Omega_B, B_r_BC))

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_dot_q = approx_fprime(
            q, lambda q: self.g_dot(t, q, u), method="cs", eps=1.0e-15
        )
        q_dot = np.concatenate(
            (
                self.ball.q_dot(t, q[: self.nq1], u[: self.nu1]),
                self.mazeboard.q_dot(t, q[self.nq1 :], u[self.nu1 :]),
            )
        )
        return g_dot_q @ q_dot + self.g_dot_u(t, q) @ u_dot

    def g_q(self, t, q):
        return approx_fprime(
            q, lambda q: self.g(t, q), method="cs", eps=1.0e-15
        ).reshape(self.nla_g, self.nq)

    def g_qq_dense(self, t, q):
        return approx_fprime(q, lambda q: self.g_q(t, q), method="3-point").reshape(
            self.nla_g, self.nq, self.nq
        )

    def g_q_T_mu_q(self, t, q, mu_g):
        return np.einsum("ijk,i", self.g_qq_dense(t, q), mu_g)

    def g_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_dot(t, q, u)).reshape(
            self.nla_g, self.nq
        )

    def W_g(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        J_P1 = self.ball.J_P(
            t,
            q[: self.nq1],
        )

        r_OB = self.mazeboard.r_OP(t, q[self.nq1 :])
        J_P2 = self.mazeboard.J_P(t, q[self.nq1 :])
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])
        B_J_R2 = self.mazeboard.B_J_R(t, q[self.nq1 :])

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

        r_OB = self.mazeboard.r_OP(t, q[self.nq1 :])
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])

        r_CP = -A_IB[:, 2] * self.ball.radius

        B_r_BP = A_IB.T @ (r_OC - r_OB)
        B_r_BP[2] = 0  # project to plane of maze board

        v_P = self.ball.v_P(t, q[: self.nq1], u[: self.nu1], B_r_CP=A_IK.T @ r_CP)
        v_P2 = self.mazeboard.v_P(t, q[self.nq1 :], u[self.nu1 :], B_r_CP=B_r_BP)
        return A_IB.T[:2] @ (v_P - v_P2)

    def gamma_dot(self, t, q, u, u_dot):
        gamma_q = approx_fprime(
            q, lambda q: self.gamma(t, q, u), method="cs", eps=1.0e-15
        )
        gamma_u = self.gamma_u(t, q)

        q_dot = np.concatenate(
            (
                self.ball.q_dot(t, q[: self.nq1], u[: self.nu1]),
                self.mazeboard.q_dot(t, q[self.nq1 :], u[self.nu1 :]),
            )
        )
        return gamma_q @ q_dot + gamma_u @ u_dot

    def gamma_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.gamma(t, q, u))

    def gamma_dot_q(self, t, q, u, u_dot):
        raise NotImplementedError("")

    def gamma_u(self, t, q):
        r_OC = self.ball.r_OP(t, q[: self.nq1])
        A_IK = self.ball.A_IB(t, q[: self.nq1])

        r_OB = self.mazeboard.r_OP(t, q[self.nq1 :])
        A_IB = self.mazeboard.A_IB(t, q[self.nq1 :])

        r_CP = -A_IB[:, 2] * self.ball.radius

        B_r_BP = A_IB.T @ (r_OC - r_OB)
        B_r_BP[2] = 0  # project to plane of maze board

        J_P1 = self.ball.J_P(t, q[: self.nq1], B_r_CP=A_IK.T @ r_CP)
        J_P2 = self.mazeboard.J_P(t, q[self.nq1 :], B_r_CP=B_r_BP)
        J_P = np.zeros((3, self.nu))
        J_P[:, : self.nu1] = J_P1
        J_P[:, self.nu1 :] = -J_P2
        return A_IB.T[:2] @ J_P

    def W_gamma(self, t, q):
        return self.gamma_u(t, q).T

    def Wla_gamma_q(self, t, q, la_gamma):
        return approx_fprime(q, lambda q: self.gamma_u(t, q).T @ la_gamma)


def ball_boundary(ball, t, q, n=100):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
    B_r_CP = ball.radius * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
    return (
        np.repeat(ball.r_OP(t, q[ball.qDOF]), n).reshape(3, n)
        + ball.A_IB(t, q[ball.qDOF]) @ B_r_CP
    )


if __name__ == "__main__":
    """Analytical analysis of the rolling motion of a ball, see Lesaux2005
    Section 5 and 6.

    References
    ==========
    Lesaux2005: https://doi.org/10.1007/s00332-004-0655-4
    """

    ############
    # parameters
    ############
    gravity = 9.81  # gravity
    m = 0.01  # ball mass

    # ball radius
    r = 5e-3

    ####################
    # initial conditions
    ####################
    # inclination angle is 0
    beta0 = 0
    beta_dot0 = 0

    # initial rolling velocity
    gamma_dot0 = 0

    # initial spinning velocity
    alpha_dot0 = 0

    # simulation time
    t1 = 10  # simulation time

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
    q0 = np.array((0, 0, r, *p0))
    u0 = np.concatenate((v_C0, B_Omega0))

    #################
    # assemble system
    #################
    # create ball
    width = r / 100
    A = 0.25 * m * r**2
    B_Theta_C = np.diag(np.array([A, A, A]))

    ball = Sphere(RigidBody)(
        r,
        A_BM=A_IB_basic(-np.pi / 2).x,
        # B_r_CP=np.array([0, width / 2, 0]),
        mass=m,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
    )

    # create maza_board (Box only for visualization purposes)
    T = 1

    def A_IB_maza_board(t):
        return A_IB_basic(np.deg2rad(30) * np.sin(2 * np.pi * t / T)).x
        return A_IB_basic(np.deg2rad(30)).x

    R = 0.1
    # board = Box(Frame)(
    #     dimensions=[2.2 * R, 2.2 * R, 0.0001],
    #     name="maza_board",
    #     A_IB=A_IB_maza_board,
    # )
    board = Box(MazeBoard)(
        dimensions=[2.2 * R, 2.2 * R, 0.0001],
        name="maza_board",
    )

    # create rolling condition
    rolling_condition = RollingCondition(board, ball)

    # gravity
    f_g = Force(lambda t: np.array([0, 0, -m * gravity]), ball)

    # controller
    controller = Controller(ball, board)

    # assemble system
    system = System()
    system.add(ball, f_g, board, rolling_condition, controller)
    system.assemble()

    ############
    # simulation
    ############
    dt = 2.0e-2  # time step

    # sol = BackwardEuler(system, t1, dt).solve()
    sol = BackwardEuler(system, t1, dt).solve()

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

    # board
    r_OB = np.array([board.r_OP(ti, qi[board.qDOF]) for ti, qi in zip(t, q)])
    A_IB = np.array([board.A_IB(ti, qi[board.qDOF]) for ti, qi in zip(t, q)])

    # ball
    r_OC = np.array([ball.r_OP(ti, qi[ball.qDOF]) for ti, qi in zip(t, q)])
    B_r_BC = np.array([A.T @ (r1 - r2) for A, r1, r2 in zip(A_IB, r_OC, r_OB)])

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 7))
    fig.suptitle("Position")

    traj = np.array(list(map(Controller.ball_traj, t)))
    r_ref, r_ref_t, r_ref_tt = np.swapaxes(traj, 0, 1)
    # g
    ax[0, 0].plot(t, r_ref[:, 0], "-r")
    ax[0, 0].plot(t, B_r_BC[:, 0])
    ax[0, 0].set_xlabel("$t$")
    ax[0, 0].set_ylabel("$x$")
    ax[0, 0].grid()

    ax[0, 1].plot(t, r_ref[:, 1], "-r")
    ax[0, 1].plot(t, B_r_BC[:, 1])
    ax[0, 1].set_xlabel("$t$")
    ax[0, 1].set_ylabel("$y$")
    ax[0, 1].grid()

    ax[1, 0].plot(t, np.rad2deg(q[:, board.qDOF[0]]))
    ax[1, 0].set_xlabel("$t$")
    ax[1, 0].set_ylabel("$alpha$")
    ax[1, 0].grid()

    ax[1, 1].plot(t, np.rad2deg(q[:, board.qDOF[1]]))
    ax[1, 1].set_xlabel("$t$")
    ax[1, 1].set_ylabel("$beta$")
    ax[1, 1].grid()

    ax[2, 0].plot(t, np.rad2deg(u[:, board.uDOF[0]]))
    ax[2, 0].set_xlabel("$t$")
    ax[2, 0].set_ylabel("$alpha dot$")
    ax[2, 0].grid()

    ax[2, 1].plot(t, np.rad2deg(u[:, board.uDOF[1]]))
    ax[2, 1].set_xlabel("$t$")
    ax[2, 1].set_ylabel("$beta dot$")
    ax[2, 1].grid()

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

    # from cardillo.visualization.vtk_render2 import Plotter

    # plotter = Plotter(system, (1000, 1000))
    # plotter.render_solution(sol, True, 0.2)
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
        x_S, y_S, z_S = ball.r_OP(t, q[ball.qDOF])

        A_IB = ball.A_IB(t, q[ball.qDOF])
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

        x_S, y_S, z_S = ball.r_OP(t, q[ball.qDOF])

        x_bdry, y_bdry, z_bdry = ball_boundary(ball, t, q)

        x_t, y_t, z_t = ball.r_OP(t, q[ball.qDOF]) + rolling_condition.r_CP(
            t, q[rolling_condition.qDOF]
        )

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IB = ball.A_IB(t, q[ball.qDOF])
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
    # exit()
    # vtk-export
    dir_name = Path(__file__).parent
    e = system.export(dir_name, "vtk", sol)
    # additionally export body fixed frame
    e.export_contr(ball, file_name="A_IB", base_export=True)
