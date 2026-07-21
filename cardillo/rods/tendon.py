import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from jax import jit, numpy as jnp

from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods.discreteRod import ElementKinematics
from cardillo.rods import DiscreteRod


class RodTendonBasic:
    def __init__(
        self, rod: DiscreteRod, xis, B_r_CPs=None, name="tendon", color=(0, 200, 50)
    ) -> None:
        self.rod = rod
        self.xis = xis
        self.n_vert = len(xis)
        self.B_r_CPs = (
            np.zeros((self.n_vert, 3)) if B_r_CPs is None else np.asarray(B_r_CPs)
        )
        self.name = name
        self._color = color

        self.alpha_vert = np.array([self.rod._alpha(xi) for xi in self.xis])

        self.r_OP_vert = jit(
            lambda q: ElementKinematics.r_OP_batch(
                self.alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )

        self._W_t_jit = jit(
            lambda q: RodTendonBasic._W_t(
                self.alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )
        )

        self._Wla_t_q_jit = jit(self._Wla_t_q)

        self._Wla_t_q_coo = CooMatrix((self.n_vert * 12, self.n_vert * 14))
        for k in range(self.n_vert):
            u1, u2 = 12 * k, 12 * (k + 1)
            q1, q2 = 14 * (k - 1), 14 * (k + 2)
            if k == 0:
                q1 = 0
            elif k == self.n_vert - 1:
                q2 = 14 * self.n_vert

            self._Wla_t_q_coo[u1:u2, q1:q2] = np.empty((u2 - u1, q2 - q1))

        # for visualization
        self._init_poly_data()

    @staticmethod
    @jit
    def _W_t(alpha_vert, q_vert, B_r_CPs):
        r_OP_vert = ElementKinematics.r_OP_batch(alpha_vert, q_vert, B_r_CPs)
        J_P_vert = ElementKinematics.J_P_batch(alpha_vert, q_vert, B_r_CPs)

        n_line = r_OP_vert[1:] - r_OP_vert[:-1]
        l_line = jnp.linalg.norm(n_line, axis=1, keepdims=True)
        n_line /= l_line

        n_vert = jnp.concatenate(
            (n_line[0, None], -n_line[:-1] + n_line[1:], -n_line[-1, None])
        )

        return jnp.einsum("ijk,ij->ik", J_P_vert, n_vert).ravel()

    @staticmethod
    @jit
    def _W_l_q(alpha_vert, q_vert, B_r_CPs):
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
            (n_line[0, None], -n_line[:-1] + n_line[1:], -n_line[-1, None])
        )

        Z = jnp.zeros((1, 3, 14))
        n_vert_q = jnp.concatenate(
            (
                jnp.concatenate(
                    (Z, n_line_q_prev[None, 0], n_line_q_next[None, 0]), axis=-1
                ),
                jnp.concatenate(
                    (
                        -n_line_q_prev[:-1],
                        -n_line_q_next[:-1] + n_line_q_prev[1:],
                        n_line_q_next[1:],
                    ),
                    axis=-1,
                ),
                jnp.concatenate(
                    (-n_line_q_prev[None, -1], -n_line_q_next[None, -1], Z), axis=-1
                ),
            )
        )

        J_P_n_q = jnp.einsum("ijk,ijl->ikl", J_P_vert, n_vert_q)
        J_P_q_n = jnp.einsum("ijkl,ij->ikl", J_P_q_vert, n_vert)
        W_l_q = J_P_n_q.at[:, :, 14:28].add(J_P_q_n)
        return W_l_q

    def _Wla_t_q(self, q, la):
        W_l_q = RodTendonBasic._W_l_q(
            self.alpha_vert, q.reshape((self.n_vert, -1)), self.B_r_CPs
        )
        Wla_q = W_l_q * la
        return jnp.concatenate(
            (Wla_q[0, :, 14:].ravel(), Wla_q[1:-1].ravel(), Wla_q[-1, :, :-14].ravel())
        )

    def _init_poly_data(self):
        self._poly_data = vtk.vtkPolyData()
        # points
        self._points = np.empty((self.n_vert, 3), dtype=float)
        array = numpy_to_vtk(self._points, deep=False)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(array)

        self._poly_data.SetPoints(vtk_points)

        # cells
        self._poly_data.Allocate(1)
        self._poly_data.InsertNextCell(
            vtk.VTK_LINE, self.n_vert, np.arange(self.n_vert)
        )

    def export(self, sol_i, **kwargs):
        self._update_poly_data(sol_i)
        return self._poly_data


class RodTendonForce(RodTendonBasic):
    def __init__(
        self, rod: DiscreteRod, xis, B_r_CPs=None, name="tendon", color=(0, 200, 50)
    ) -> None:
        super().__init__(rod, xis, B_r_CPs, name, color)
        self.nla_tau = 1

        def W_tau(t, q):
            return self._W_t_jit(q)

        self.W_tau = W_tau

        def Wla_tau_q(t, q, u):
            coo = self._Wla_t_q_coo
            coo.data = self._Wla_t_q_jit(q, self.la_tau(t, q, u))
            return coo

        self.Wla_tau_q = Wla_tau_q

        self.Wla_tau_u = lambda t, q, u: None

    def assembler_callback(self):
        rod = self.rod
        els = np.array([rod._element_number(xi) for xi in self.xis])
        self.qDOF = np.concatenate([rod.qDOF[rod.elDOF[el]] for el in els])
        self.uDOF = np.concatenate([rod.uDOF[rod.elDOF_u[el]] for el in els])

    def la_tau(self, t, q, u):
        return 0.0

    def _update_poly_data(self, sol_i):
        q = sol_i.q[self.qDOF]
        self._points[:] = self.r_OP_vert(q)
        self._poly_data.Modified()


class RodTendonForceIntegrator(RodTendonBasic):
    def __init__(
        self, rod: DiscreteRod, xis, B_r_CPs=None, name="tendon", color=(0, 200, 50)
    ) -> None:
        super().__init__(rod, xis, B_r_CPs, name, color)
        self.nla_tau = 1
        self.nq = 1
        self.q0 = np.zeros(1)

        def W_tau(t, q):
            return self._W_t_jit(q[:-1])

        self.W_tau = W_tau

        def Wla_tau_q(t, q, u):
            coo = self._Wla_t_q_coo
            coo.data = self._Wla_t_q_jit(q[:-1], self.la_tau(t, q, u))
            return coo

        self.Wla_tau_q = Wla_tau_q

        self.Wla_tau_u = lambda t, q, u: None

    def assembler_callback(self):
        rod = self.rod
        els = np.array([rod._element_number(xi) for xi in self.xis])
        self.qDOF = np.concatenate(
            [rod.qDOF[rod.elDOF[el]] for el in els] + [self.my_qDOF]
        )
        self.uDOF = np.concatenate([rod.uDOF[rod.elDOF_u[el]] for el in els])

    def q_dot(self, t, q, u):
        return 0.0

    def la_tau(self, t, q, u):
        return q[-1]

    def _update_poly_data(self, sol_i):
        q = sol_i.q[self.qDOF]
        self._points[:] = self.r_OP_vert(q[:-1])
        self._poly_data.Modified()
