import numpy as np
from jax import jit, numpy as jnp, vmap

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods.discreteRod import ElementKinematics
from cardillo.rods import DiscreteRod

E3 = np.eye(3)


class RodTendonForce:
    @staticmethod
    @jit
    def _n_vert(r_OP_vert):
        n_line = r_OP_vert[1:] - r_OP_vert[:-1]
        n_line /= jnp.linalg.norm(n_line, axis=1, keepdims=True)
        return jnp.concatenate(
            (-n_line[0, None], n_line[:-1] - n_line[1:], n_line[-1, None])
        )

    # @staticmethod
    # @jit
    # def _W12(r_OP1, r_OP2, J_P1, J_P2):
    #     n = r_OP2 - r_OP1
    #     n /= jnp.sqrt(n @ n)
    #     W1 = -J_P1.T @ n
    #     W2 = J_P2.T @ n
    #     return W1, W2

    @staticmethod
    @jit
    def _W12(n_diff, J_P):
        return J_P.T @ n_diff

    @staticmethod
    @jit
    def _W12_q(r_OP1, r_OP2, J_P1, J_P2, r_OP1_q1, r_OP2_q2, J_P1_q1, J_P2_q2):
        n = r_OP2 - r_OP1
        l = jnp.sqrt(n @ n)
        n /= l

        tmp = jnp.outer(n, n) - jnp.eye(3)
        n_q1 = tmp @ r_OP1_q1 / l
        n_q2 = -tmp @ r_OP2_q2 / l

        J_P1_T_n_q1 = (J_P1_q1.T @ n).T + J_P1.T @ n_q1
        J_P1_T_n_q2 = J_P1.T @ n_q2
        J_P2_T_n_q1 = J_P2.T @ n_q1
        J_P2_T_n_q2 = (J_P2_q2.T @ n).T + J_P2.T @ n_q2

        W12_q = jnp.concatenate(
            (
                -jnp.concatenate((J_P1_T_n_q1, J_P1_T_n_q2), axis=1),
                jnp.concatenate((J_P2_T_n_q1, J_P2_T_n_q2), axis=1),
            )
        )

        return W12_q

    @staticmethod
    @jit
    def __W_l(alpha_vert, q_vert, B_r_CPs):
        r_OP_vert = ElementKinematics.r_OP_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_vert = ElementKinematics.J_P_batch(alpha_vert, q_vert, B_r_CPs)

        n_line = r_OP_vert[1:] - r_OP_vert[:-1]
        l_line = jnp.linalg.norm(n_line, axis=1, keepdims=True)
        n_line /= l_line

        n_vert = jnp.concatenate(
            (-n_line[0, None], n_line[:-1] - n_line[1:], n_line[-1, None])
        )

        return jnp.einsum("ijk,ij->ik", J_P_vert, n_vert).ravel()

    @staticmethod
    @jit
    def __W_l_q(alpha_vert, q_vert, B_r_CPs):
        r_OP_vert = ElementKinematics.r_OP_batch(alpha_vert, q_vert, B_r_CPs)
        r_OP_q_vert = ElementKinematics.r_OP_q_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_vert = ElementKinematics.J_P_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_q_vert = ElementKinematics.J_P_q_batch(alpha_vert, q_vert, B_r_CPs)

        n_line = r_OP_vert[1:] - r_OP_vert[:-1]
        l_line = jnp.linalg.norm(n_line, axis=1, keepdims=True)
        n_line /= l_line

        tmp = jnp.einsum("bi,bj->bij", n_line, n_line) - jnp.eye(3)
        tmp /= l_line[..., None]
        n_line_q_prev = tmp @ r_OP_q_vert[:-1]
        n_line_q_next = -tmp @ r_OP_q_vert[1:]

        n_vert = jnp.concatenate(
            (-n_line[0, None], n_line[:-1] - n_line[1:], n_line[-1, None])
        )

        # n_vert = n_prev - n_next

        Z = jnp.zeros((1, 3, 14))
        n_vert_q = jnp.concatenate(
            (
                jnp.concatenate(
                    (Z, -n_line_q_prev[None, 0], -n_line_q_next[None, 0]), axis=-1
                ),
                jnp.concatenate(
                    (
                        n_line_q_prev[:-1],
                        n_line_q_next[:-1] - n_line_q_prev[1:],
                        -n_line_q_next[1:],
                    ),
                    axis=-1,
                ),
                jnp.concatenate(
                    (n_line_q_prev[None, -1], n_line_q_next[None, -1], Z), axis=-1
                ),
            )
        )

        J_P_n_q = jnp.einsum("ijk,ijl->ikl", J_P_vert, n_vert_q)
        J_P_q_n = jnp.einsum("ijkl,ij->ikl", J_P_q_vert, n_vert)
        W_l_q = J_P_n_q.at[:, :, 14:28].add(J_P_q_n)
        return W_l_q

    @staticmethod
    @jit
    def __W_l_q2(alpha_vert, q_vert, B_r_CPs):
        r_OP_vert = ElementKinematics.r_OP_batch(alpha_vert, q_vert, B_r_CPs)
        r_OP_q_vert = ElementKinematics.r_OP_q_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_vert = ElementKinematics.J_P_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_q_vert = ElementKinematics.J_P_q_batch(alpha_vert, q_vert, B_r_CPs)
        return RodTendonForce._W12_qs(
            r_OP_vert[:-1],
            r_OP_vert[1:],
            J_P_vert[:-1],
            J_P_vert[1:],
            r_OP_q_vert[:-1],
            r_OP_q_vert[1:],
            J_P_q_vert[:-1],
            J_P_q_vert[1:],
        )

    _W12s = jit(vmap(_W12.__func__))
    _W12_qs = jit(vmap(_W12_q.__func__))

    def __init__(
        self,
        rod: DiscreteRod,
        xis,
        B_r_CPs=None,
        name="tendon",
    ) -> None:
        self.rod = rod
        self.xis = xis
        self.n_vert = len(xis)
        self.B_r_CPs = (
            np.zeros((self.n_vert, 3)) if B_r_CPs is None else np.asarray(B_r_CPs)
        )
        self.name = name

        alpha_vert = np.array([self.rod._alpha(xi) for xi in self.xis])

        self._W_l = jit(
            lambda q: RodTendonForce.__W_l(
                alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )

        self._W_l_q = jit(
            lambda q: RodTendonForce.__W_l_q(
                alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )
        self._W_l_q2 = jit(
            lambda q: RodTendonForce.__W_l_q2(
                alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )

        self.r_OP_vert = jit(
            lambda q: ElementKinematics.r_OP_batch(
                alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )

        self._nq = np.arange(self.n_vert * 14 + 1, step=14)
        self._nu = np.arange(self.n_vert * 12 + 1, step=12)

        self._W_q_coo2 = CooMatrix((self.n_vert * 12, self.n_vert * 14))
        for k in range(self.n_vert - 1):
            nu0, nu1, nu2 = self._nu[k : k + 3]
            nq0, nq1, nq2 = self._nq[k : k + 3]
            self._W_q_coo2[nu0:nu2, nq0:nq2] = np.empty((nu2 - nu0, nq2 - nq0))

        self._h_q_coo = CooMatrix((self.n_vert * 12, self.n_vert * 14))
        for k in range(self.n_vert):
            u1, u2 = self._nu[k : k + 2]
            if k == 0:
                q1 = self._nq[k]
                q2 = self._nq[k + 2]
            elif k == self.n_vert - 1:
                q1 = self._nq[k - 1]
                q2 = self._nq[k + 1]
            else:
                q1 = self._nq[k - 1]
                q2 = self._nq[k + 2]

            self._h_q_coo[u1:u2, q1:q2] = np.empty((u2 - u1, q2 - q1))

    def assembler_callback(self):
        rod = self.rod
        els = [rod._element_number(xi) for xi in self.xis]
        self.qDOF = np.concatenate([rod.qDOF[rod.elDOF[el]] for el in els])
        self.uDOF = np.concatenate([rod.uDOF[rod.elDOF_u[el]] for el in els])

    def W_l(self, t, q):
        W = self._W_l(q)
        return W.__array__()

    def h(self, t, q, u):
        return -self.la(t) * self.W_l(t, q)

    def h_q(self, t, q, u):
        h_q_coo = self._h_q_coo
        W_l_q = self._W_l_q(q).__array__()
        h_q = W_l_q * (-self.la(t))
        h_q_coo.data = np.concatenate(
            (h_q[0, :, 14:].ravel(), h_q[1:-1].ravel(), h_q[-1, :, :-14].ravel())
        )
        return h_q_coo

    def la(self, t):
        return 0.0

    def set_force(self, force):
        self.la = force if callable(force) else lambda t: force
