import warnings

import numpy as np

from .math import ax2skew, cross3
from .rods.discreteRod import DiscreteRod


def len_slice(x):
    if isinstance(x, slice):
        if x.step is None:
            return x.stop - x.start
        else:
            (x.stop - x.start + x.step - 1) // x.step
    else:
        return len(x)


def concatenate_qDOF(object):
    qDOF1 = object.subsystem1.qDOF
    qDOF2 = object.subsystem2.qDOF
    local_qDOF1 = object.subsystem1.local_qDOF_P(object.xi1)
    local_qDOF2 = object.subsystem2.local_qDOF_P(object.xi2)

    object.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
    if isinstance(local_qDOF1, slice):
        object._nq1 = len_slice(local_qDOF1)
    else:
        object._nq1 = len(local_qDOF1)
    if isinstance(local_qDOF2, slice):
        object._nq2 = len_slice(local_qDOF2)
    else:
        object._nq2 = len(local_qDOF2)
    object._nq = object._nq1 + object._nq2

    return local_qDOF1, local_qDOF2


def concatenate_uDOF(object):
    uDOF1 = object.subsystem1.uDOF
    uDOF2 = object.subsystem2.uDOF
    local_uDOF1 = object.subsystem1.local_uDOF_P(object.xi1)
    local_uDOF2 = object.subsystem2.local_uDOF_P(object.xi2)

    object.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
    object._nu1 = nu1 = len_slice(local_uDOF1)
    object._nu2 = nq1 = len_slice(local_uDOF2)
    # object._nu2 = len(local_uDOF2)
    object._nu = object._nu1 + object._nu2

    return local_uDOF1, local_uDOF2


class RigidConnection:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        r_OJ0=None,
        A_IJ0=None,
        xi1=None,
        xi2=None,
        name="rigid_connection",
        **kwargs,
    ):
        self.name = name
        projection_pairs_rotation = [(1, 2), (2, 0), (0, 1)]
        self.subsystem1 = subsystem1
        self.subsystem2 = subsystem2
        self.xi1 = xi1
        self.xi2 = xi2
        self.r_OJ0 = r_OJ0
        self.A_IJ0 = A_IJ0

        # guard against flawed constrained_axes input
        self.nla_g_rot = len(projection_pairs_rotation)
        for pair in projection_pairs_rotation:
            assert len(np.unique(pair)) == 2
            for i in pair:
                assert i in [0, 1, 2]

        self.nla_g = 3 + self.nla_g_rot
        self.projection_pairs = projection_pairs_rotation

        self.constrain_orientation = self.nla_g_rot > 0

        if "name" in kwargs:
            self.name = kwargs.get("name")

    def assembler_callback(self):
        local_qDOF1, local_qDOF2 = concatenate_qDOF(self)
        concatenate_uDOF(self)

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
        )

        # check for A_IB of subsystem 1
        if hasattr(self.subsystem1, "A_IB"):
            A_IB10 = self.subsystem1.A_IB(
                self.subsystem1.t0, self.subsystem1.q0[local_qDOF1], self.xi1
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP10

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB10

            self.B1_r_P1J0 = A_IB10.T @ (self.r_OJ0 - r_OP10)
            self.A_K1J0 = A_IB10.T @ self.A_IJ0
        else:
            self.B1_r_P1J0 = np.zeros(3)
            self.A_K1J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

        # check for A_IB of subsystem 2
        if hasattr(self.subsystem2, "A_IB"):
            A_IB20 = self.subsystem2.A_IB(
                self.subsystem2.t0, self.subsystem2.q0[local_qDOF2], self.xi2
            )

            if self.r_OJ0 is None:
                self.r_OJ0 = r_OP20

            if self.A_IJ0 is None:
                self.A_IJ0 = A_IB20

            self.B2_r_P2J0 = A_IB20.T @ (self.r_OJ0 - r_OP20)
            self.A_K2J0 = A_IB20.T @ self.A_IJ0
        else:
            self.B2_r_P2J0 = np.zeros(3)
            self.A_K2J0 = None  # unused
            assert self.nla_g_rot == 0  # Spherical case

    # auxiliary functions
    def r_OJ1(self, t, q):
        return self.subsystem1.r_OP(t, q[: self._nq1], self.xi1, self.B1_r_P1J0)

    def r_OJ2(self, t, q):
        return self.subsystem2.r_OP(t, q[self._nq1 :], self.xi2, self.B2_r_P2J0)

    def r_OJ1_q1(self, t, q):
        return self.subsystem1.r_OP_q(t, q[: self._nq1], self.xi1, self.B1_r_P1J0)

    def r_OJ2_q2(self, t, q):
        return self.subsystem2.r_OP_q(t, q[self._nq1 :], self.xi2, self.B2_r_P2J0)

    def v_J1(self, t, q, u):
        return self.subsystem1.v_P(
            t, q[: self._nq1], u[: self._nu1], self.xi1, self.B1_r_P1J0
        )

    def v_J2(self, t, q, u):
        return self.subsystem2.v_P(
            t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B2_r_P2J0
        )

    def v_J1_q1(self, t, q, u):
        return self.subsystem1.v_P_q(
            t, q[: self._nq1], u[: self._nu1], self.xi1, self.B1_r_P1J0
        )

    def v_J2_q2(self, t, q, u):
        return self.subsystem2.v_P_q(
            t, q[self._nq1 :], u[self._nu1 :], self.xi2, self.B2_r_P2J0
        )

    def a_J1(self, t, q, u, u_dot):
        return self.subsystem1.a_P(
            t,
            q[: self._nq1],
            u[: self._nu1],
            u_dot[: self._nu1],
            self.xi1,
            self.B1_r_P1J0,
        )

    def a_J2(self, t, q, u, u_dot):
        return self.subsystem2.a_P(
            t,
            q[self._nq1 :],
            u[self._nu1 :],
            u_dot[self._nu1 :],
            self.xi2,
            self.B2_r_P2J0,
        )

    def a_J1_q1(self, t, q, u, u_dot):
        return self.subsystem1.a_P_q(
            t,
            q[: self._nq1],
            u[: self._nu1],
            u_dot[: self._nu1],
            self.xi1,
            self.B1_r_P1J0,
        )

    def a_J2_q2(self, t, q, u, u_dot):
        return self.subsystem2.a_P_q(
            t,
            q[self._nq1 :],
            u[self._nu1 :],
            u_dot[self._nu1 :],
            self.xi2,
            self.B2_r_P2J0,
        )

    def a_J1_u1(self, t, q, u, u_dot):
        return self.subsystem1.a_P_u(
            t,
            q[: self._nq1],
            u[: self._nu1],
            u_dot[: self._nu1],
            self.xi1,
            self.B1_r_P1J0,
        )

    def a_J2_u2(self, t, q, u, u_dot):
        return self.subsystem2.a_P_u(
            t,
            q[self._nq1 :],
            u[self._nu1 :],
            u_dot[self._nu1 :],
            self.xi2,
            self.B2_r_P2J0,
        )

    def J_J1(self, t, q):
        return self.subsystem1.J_P(t, q[: self._nq1], self.xi1, self.B1_r_P1J0)

    def J_J2(self, t, q):
        return self.subsystem2.J_P(t, q[self._nq1 :], self.xi2, self.B2_r_P2J0)

    def J_J1_q1(self, t, q):
        return self.subsystem1.J_P_q(t, q[: self._nq1], self.xi1, self.B1_r_P1J0)

    def J_J2_q2(self, t, q):
        return self.subsystem2.J_P_q(t, q[self._nq1 :], self.xi2, self.B2_r_P2J0)

    def A_IB1(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IB1 = self.subsystem1.A_IB(t, q[: self._nq1], self.xi1)
        return self._A_IB1

    def A_IB2(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IB2 = self.subsystem2.A_IB(t, q[self._nq1 :], self.xi2)
        return self._A_IB2

    def A_IB_q1(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IB_q1 = self.subsystem1.A_IB_q(t, q[: self._nq1], self.xi1)
        return self._A_IB_q1

    def A_IB_q2(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IB_q2 = self.subsystem2.A_IB_q(t, q[self._nq1 :], self.xi2)
        return self._A_IB_q2

    def A_IJ1(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IJ1 = self.A_IB1(t, q) @ self.A_K1J0
        return self._A_IJ1

    def A_IJ2(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IJ2 = self.A_IB2(t, q) @ self.A_K2J0
        return self._A_IJ2

    def A_IJ1_q1(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IJ1_q1 = self.A_K1J0.T @ self.A_IB_q1(t, q)
        return self._A_IJ1_q1

    def A_IJ2_q2(self, t, q):
        # if self._t != t or self._q.tobytes() != q.tobytes():
        self._A_IJ2_q2 = self.A_K2J0.T @ self.A_IB_q2(t, q)
        return self._A_IJ2_q2

    def B_Omega1(self, t, q, u):
        # if (
        #     self._t != t
        #     or self._q.tobytes() != q.tobytes()
        #     or self._u.tobytes() != u.tobytes()
        # ):
        self._B_Omega1 = self.subsystem1.B_Omega(
            t, q[: self._nq1], u[: self._nu1], self.xi1
        )
        return self._B_Omega1

    def B_Omega2(self, t, q, u):
        # if (
        #     self._t != t
        #     or self._q.tobytes() != q.tobytes()
        #     or self._u.tobytes() != u.tobytes()
        # ):
        self._B_Omega2 = self.subsystem2.B_Omega(
            t, q[self._nq1 :], u[self._nu1 :], self.xi2
        )
        return self._B_Omega2

    def Omega1(self, t, q, u):
        # if (
        #     self._t != t
        #     or self._q.tobytes() != q.tobytes()
        #     or self._u.tobytes() != u.tobytes()
        # ):
        self._Omega1 = self.A_IB1(t, q) @ self.B_Omega1(t, q, u)
        return self._Omega1

    def Omega2(self, t, q, u):
        # if (
        #     self._t != t
        #     or self._q.tobytes() != q.tobytes()
        #     or self._u.tobytes() != u.tobytes()
        # ):
        self._Omega2 = self.A_IB2(t, q) @ self.B_Omega2(t, q, u)
        return self._Omega2

    def Omega1_q1(self, t, q, u):
        return (self.B_Omega1(t, q, u) @ self.A_IB_q1(t, q)) + self.A_IB1(
            t, q
        ) @ self.subsystem1.B_Omega_q(t, q[: self._nq1], u[: self._nu1], self.xi1)

    def Omega2_q2(self, t, q, u):
        return (self.B_Omega2(t, q, u) @ self.A_IB_q2(t, q)) + self.A_IB2(
            t, q
        ) @ self.subsystem2.B_Omega_q(t, q[self._nq1 :], u[self._nu1 :], self.xi2)

    def Psi1(self, t, q, u, u_dot):
        return self.A_IB1(t, q) @ self.subsystem1.B_Psi(
            t, q[: self._nq1], u[: self._nu1], u_dot[: self._nu1], self.xi1
        )

    def Psi2(self, t, q, u, u_dot):
        return self.A_IB2(t, q) @ self.subsystem2.B_Psi(
            t, q[self._nq1 :], u[self._nu1 :], u_dot[self._nu1 :], self.xi2
        )

    def Psi1_q1(self, t, q, u, u_dot):
        return (
            self.subsystem1.B_Psi(
                t, q[: self._nq1], u[: self._nu1], u_dot[: self._nu1], self.xi1
            )
            @ self.A_IB_q1(t, q)
        ) + self.A_IB1(t, q) @ self.subsystem1.B_Psi_q(
            t, q[: self._nq1], u[: self._nu1], u_dot[: self._nu1], self.xi1
        )

    def Psi2_q2(self, t, q, u, u_dot):
        return (
            self.subsystem2.B_Psi(
                t, q[self._nq1 :], u[self._nu1 :], u_dot[self._nu1 :], self.xi2
            )
            @ self.A_IB_q2(t, q)
        ) + self.A_IB2(t, q) @ self.subsystem2.B_Psi_q(
            t, q[self._nq1 :], u[self._nu1 :], u_dot[self._nu1 :], self.xi2
        )

    def Psi1_u1(self, t, q, u, u_dot):
        return self.A_IB1(t, q) @ self.subsystem1.B_Psi_u(
            t, q[: self._nq1], u[: self._nu1], u_dot[: self._nu1], self.xi1
        )

    def Psi2_u2(self, t, q, u, u_dot):
        return self.A_IB2(t, q) @ self.subsystem2.B_Psi_u(
            t, q[self._nq1 :], u[self._nu1 :], u_dot[self._nu1 :], self.xi2
        )

    def J_R1(self, t, q):
        return self.A_IB1(t, q) @ self.subsystem1.B_J_R(t, q[: self._nq1], self.xi1)

    def J_R2(self, t, q):
        return self.A_IB2(t, q) @ self.subsystem2.B_J_R(t, q[self._nq1 :], self.xi2)

    def J_R1_q1(self, t, q):
        return (
            self.subsystem1.B_J_R(t, q[: self._nq1], self.xi1).T @ self.A_IB_q1(t, q)
        ) + (
            self.subsystem1.B_J_R_q(t, q[: self._nq1], self.xi1).T @ self.A_IB1(t, q).T
        ).T

    def J_R2_q2(self, t, q):
        return (
            self.subsystem2.B_J_R(t, q[self._nq1 :], self.xi2).T @ self.A_IB_q2(t, q)
        ) + (
            self.subsystem2.B_J_R_q(t, q[self._nq1 :], self.xi2).T @ self.A_IB2(t, q).T
        ).T

    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=float)
        g[:3] = self.r_OJ2(t, q) - self.r_OJ1(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            for i, (a, b) in enumerate(self.projection_pairs):
                g[3 + i] = A_IJ1[:, a] @ A_IJ2[:, b]

        return g

    def g_q(self, t, q):
        g_q = np.zeros((self.nla_g, self._nq), dtype=float)
        nq1 = self._nq1

        g_q[:3, :nq1] = -self.r_OJ1_q1(t, q)
        g_q[:3, nq1:] = self.r_OJ2_q2(t, q)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                g_q[3 + i, :nq1] = A_IJ2[:, b] @ A_IJ1_q1[:, a]
                g_q[3 + i, nq1:] = A_IJ1[:, a] @ A_IJ2_q2[:, b]

        return g_q

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=float)
        g_dot[:3] = self.v_J2(t, q, u) - self.v_J1(t, q, u)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs):
                n = cross3(A_IJ1[:, a], A_IJ2[:, b])
                g_dot[3 + i] = n @ Omega21
        return g_dot

    def g_dot_q(self, t, q, u):
        g_dot_q = np.zeros((self.nla_g, self._nq), dtype=float)
        nq1 = self._nq1
        g_dot_q[:3, :nq1] = -self.v_J1_q1(t, q, u)
        g_dot_q[:3, nq1:] = self.v_J2_q2(t, q, u)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)
            Omega1_q1 = self.Omega1_q1(t, q, u)
            Omega2_q2 = self.Omega2_q2(t, q, u)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_dot_q[3 + i, :nq1] = (
                    n @ Omega1_q1 - Omega21 @ ax2skew(e_b) @ A_IJ1_q1[:, a]
                )
                g_dot_q[3 + i, nq1:] = (
                    -n @ Omega2_q2 + Omega21 @ ax2skew(e_a) @ A_IJ2_q2[:, b]
                )

        return g_dot_q

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=np.float64)
        g_ddot[:3] = self.a_J2(t, q, u, u_dot) - self.a_J1(t, q, u, u_dot)

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            Omega1 = self.Omega1(t, q, u)
            Omega2 = self.Omega2(t, q, u)
            Omega21 = Omega1 - Omega2
            Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                g_ddot[3 + i] = (
                    cross3(cross3(Omega1, e_a), e_b) + cross3(e_a, cross3(Omega2, e_b))
                ) @ Omega21 + n @ Psi21

        return g_ddot

    def W_g(self, t, q):
        W_g = np.zeros((self._nu, self.nla_g), dtype=float)
        nu1 = self._nu1
        W_g[:nu1, :3] = -self.J_J1(t, q).T
        W_g[nu1:, :3] = self.J_J2(t, q).T

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)
            J_R1 = self.J_R1(t, q)
            J_R2 = self.J_R2(t, q)

            n = np.array(
                [cross3(A_IJ1[:, a], A_IJ2[:, b]) for a, b in self.projection_pairs]
            )
            W_g[:nu1, 3:] = (n @ J_R1).T
            W_g[nu1:, 3:] = (-n @ J_R2).T
            # for i, (a, b) in enumerate(self.projection_pairs):
            #     n = cross3(A_IJ1[:, a], A_IJ2[:, b])
            #     W_g[:nu1, 3 + i] = n @ J_R1
            #     W_g[nu1:, 3 + i] = -n @ J_R2
        return W_g

    def Wla_g_q(self, t, q, la_g):
        Wla_g_q = np.zeros((self._nu, self._nq), dtype=float)
        nq1 = self._nq1
        nu1 = self._nu1

        Wla_g_q[:nu1, :nq1] -= (self.J_J1_q1(t, q).T @ la_g[:3]).T
        Wla_g_q[nu1:, nq1:] += (self.J_J2_q2(t, q).T @ la_g[:3]).T

        if self.constrain_orientation:
            A_IJ1 = self.A_IJ1(t, q)
            A_IJ2 = self.A_IJ2(t, q)

            A_IJ1_q1 = self.A_IJ1_q1(t, q)
            A_IJ2_q2 = self.A_IJ2_q2(t, q)

            J_R1 = self.J_R1(t, q)
            J_R2 = self.J_R2(t, q)
            J_R1_q1 = self.J_R1_q1(t, q)
            J_R2_q2 = self.J_R2_q2(t, q)

            for i, (a, b) in enumerate(self.projection_pairs):
                e_a, e_b = A_IJ1[:, a], A_IJ2[:, b]
                n = cross3(e_a, e_b)
                n_q1 = -ax2skew(e_b) @ A_IJ1_q1[:, a]
                n_q2 = ax2skew(e_a) @ A_IJ2_q2[:, b]
                Wla_g_q[:nu1, :nq1] += (
                    la_g[3 + i] * (J_R1_q1.T @ n).T + la_g[3 + i] * J_R1.T @ n_q1
                )
                Wla_g_q[:nu1, nq1:] += la_g[3 + i] * J_R1.T @ n_q2
                Wla_g_q[nu1:, :nq1] -= la_g[3 + i] * J_R2.T @ n_q1
                Wla_g_q[nu1:, nq1:] -= (
                    la_g[3 + i] * (J_R2_q2.T @ n).T + la_g[3 + i] * J_R2.T @ n_q2
                )

        return Wla_g_q
