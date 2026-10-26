import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from warnings import warn

from jax import jit, numpy as jnp

from cardillo.contributions import TauContribution
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.rods.discreteRod import ElementKinematics, NodeKinematics
from cardillo.rods import DiscreteRod
from cardillo.visualization.vtk_render2 import RuntimeVisualBase


class RodTendonExport(RuntimeVisualBase):
    def __init__(
        self, n_vert, tube_radius=1e-3, name="tendon", color=(0, 200, 50), opacity=1
    ):
        self.name = name
        self._color = color
        self._opacity = opacity
        self._tube_radius = tube_radius
        self._init_polydata(n_vert)

    def _init_polydata(self, n_vert):
        self._polydata = vtk.vtkPolyData()
        # points
        self._points = np.empty((n_vert, 3), dtype=float)
        array = numpy_to_vtk(self._points, deep=False)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(array)

        self._polydata.SetPoints(vtk_points)

        # cells
        self._polydata.Allocate(1)
        self._polydata.InsertNextCell(vtk.VTK_LINE, n_vert, np.arange(n_vert))

    def _update_polydata(self, sol_i):
        q = sol_i.q[self.qDOF]
        self._points[:] = self._r_OP_verts(q)
        self._polydata.Modified()

    def export(self, sol_i, **kwargs):
        self._update_polydata(sol_i)
        return self._polydata

    def to_vtk_actors(self):

        poly_data = self._polydata

        filter = vtk.vtkTubeFilter()
        filter.SetRadius(self._tube_radius)
        filter.SetInputData(poly_data)
        filter.SetNumberOfSides(8)
        filter.CappingOn()

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(([c / 255 for c in self._color]))
        actor.GetProperty().SetOpacity(self._opacity)

        return [actor]

    def update_vtk_actors(self, sol_i):
        self._update_polydata(sol_i)


class RodTendonKinematics(RodTendonExport):
    def __init__(
        self,
        rod: DiscreteRod,
        xis,
        B_r_CPs=None,
        name="tendon",
        color=(0, 200, 50),
        **kwargs
    ) -> None:
        super().__init__(len(xis), name=name, color=color, **kwargs)
        self._jaxed = True
        self.rod = rod
        self.xis = xis
        self.n_vert = len(xis)
        self.B_r_CPs = (
            np.zeros((self.n_vert, 3)) if B_r_CPs is None else np.asarray(B_r_CPs)
        )

        self._alpha_verts = np.array([self.rod._alpha(xi) for xi in self.xis])
        self._alpha_verts = np.round(
            self._alpha_verts, decimals=10
        )  # avoid numerical issues with alpha=0 or 1

        # determine if the tendon vertices are all at the element ends (alpha=0 or 1)
        _simple_vert = True
        for alpha in self._alpha_verts:
            if alpha > 0 and alpha < 1:
                _simple_vert = False
                break
        else:
            self._alpha_verts = np.astype(self._alpha_verts, int)
        self._vert_at_node = _simple_vert

        self._r_OP_verts_jit = jit(self._r_OP_verts_jax)

        self._W_t_jit = jit(self._W_t_jax)

        self._W_t_q_jit = jit(self._W_t_q_jax)

        self._nq_el = _nq_el = 7 if _simple_vert else 14
        self._nu_el = _nu_el = 6 if _simple_vert else 12
        self._W_t_q_coo = CooMatrix((self.n_vert * _nu_el, self.n_vert * _nq_el))
        for k in range(self.n_vert):
            u1, u2 = _nu_el * k, _nu_el * (k + 1)
            q1, q2 = _nq_el * (k - 1), _nq_el * (k + 2)
            if k == 0:
                q1 = 0
            elif k == self.n_vert - 1:
                q2 = _nq_el * self.n_vert

            self._W_t_q_coo[u1:u2, q1:q2] = np.empty((u2 - u1, q2 - q1))

    def _r_OP_verts_jax(self, q):
        if self._vert_at_node:
            return NodeKinematics.r_OP_batch(q.reshape((self.n_vert, -1)), self.B_r_CPs)
        else:
            return ElementKinematics.r_OP_batch(
                self._alpha_verts, q.reshape((self.n_vert, -1)), self.B_r_CPs
            )

    def _r_OP_verts(self, q):
        return self._r_OP_verts_jit(q)

    def _W_t_jax(self, q):
        q_vert = q.reshape((self.n_vert, -1))
        if self._vert_at_node:
            r_OP_vert = NodeKinematics.r_OP_batch(q_vert, self.B_r_CPs)
            J_P_vert = NodeKinematics.J_P_batch(q_vert, self.B_r_CPs)
        else:
            r_OP_vert = ElementKinematics.r_OP_batch(
                self._alpha_verts, q_vert, self.B_r_CPs
            )
            J_P_vert = ElementKinematics.J_P_batch(
                self._alpha_verts, q_vert, self.B_r_CPs
            )
        n_line = r_OP_vert[1:] - r_OP_vert[:-1]
        l_line = jnp.linalg.norm(n_line, axis=1, keepdims=True)
        n_line /= l_line

        n_vert = jnp.concatenate(
            (n_line[0, None], -n_line[:-1] + n_line[1:], -n_line[-1, None])
        )

        return jnp.einsum("ijk,ij->ik", J_P_vert, n_vert).ravel()

    def W_t(self, q):
        return self._W_t_jit(q)

    def _W_t_q_jax(self, q):
        alpha_vert = self._alpha_verts
        q_vert = q.reshape((self.n_vert, -1))
        if self._vert_at_node:
            r_OP_vert = NodeKinematics.r_OP_batch(q_vert, self.B_r_CPs)
            r_OP_q_vert = NodeKinematics.r_OP_q_batch(q_vert, self.B_r_CPs)
            J_P_vert = NodeKinematics.J_P_batch(q_vert, self.B_r_CPs)
            J_P_q_vert = NodeKinematics.J_P_q_batch(q_vert, self.B_r_CPs)
        else:
            r_OP_vert = ElementKinematics.r_OP_batch(alpha_vert, q_vert, self.B_r_CPs)
            r_OP_q_vert = ElementKinematics.r_OP_q_batch(
                alpha_vert, q_vert, self.B_r_CPs
            )
            J_P_vert = ElementKinematics.J_P_batch(alpha_vert, q_vert, self.B_r_CPs)
            J_P_q_vert = ElementKinematics.J_P_q_batch(alpha_vert, q_vert, self.B_r_CPs)

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

        _nq_el = self._nq_el
        Z = jnp.zeros((1, 3, _nq_el))
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
        W_t_q = J_P_n_q.at[:, :, _nq_el : 2 * _nq_el].add(J_P_q_n)
        return jnp.concatenate(
            (
                W_t_q[0, :, _nq_el:].ravel(),
                W_t_q[1:-1].ravel(),
                W_t_q[-1, :, :-_nq_el].ravel(),
            )
        )

    def W_t_q(self, q):
        coo = self._W_t_q_coo
        coo.data = self._W_t_q_jit(q)
        return coo

    def assembler_callback(self):
        rod = self.rod
        els = np.array([rod._element_number(xi) for xi in self.xis])
        if not self._vert_at_node:
            warn(
                "❌ Some tendon vertices are not at the element ends, leading to a more complex DOF mapping. Please check if this is necessary."
            )
            self.qDOF = np.concatenate([rod.qDOF[rod.elDOF[el]] for el in els])
            self.uDOF = np.concatenate([rod.uDOF[rod.elDOF_u[el]] for el in els])
        else:
            self.qDOF = np.concatenate(
                [
                    rod.qDOF[rod.elDOF[el]][al * 7 : (al + 1) * 7]
                    for al, el in zip(self._alpha_verts, els)
                ]
            )
            self.uDOF = np.concatenate(
                [
                    rod.uDOF[rod.elDOF_u[el]][al * 6 : (al + 1) * 6]
                    for al, el in zip(self._alpha_verts, els)
                ]
            )


class RodTendonForce(RodTendonKinematics, TauContribution):
    def __init__(
        self, rod: DiscreteRod, xis, B_r_CPs=None, name="tendon", color=(0, 200, 50)
    ) -> None:
        TauContribution.__init__(self, nla_tau=1, ntau=1)
        RodTendonKinematics.__init__(
            self, rod, xis, B_r_CPs=B_r_CPs, name=name, color=color
        )

        self._Wla_tau_q_coo = CooMatrix((self.n_vert * 12, self.n_vert * 14))
        self._Wla_tau_q_coo.col = self._W_t_q_coo.col.copy()
        self._Wla_tau_q_coo.row = self._W_t_q_coo.row.copy()

        self.__Wla_tau_q_jit = jit(self.Wla_tau_q_jax)

    def assembler_callback(self):
        super().assembler_callback()

        self.W_tau_jax_row = np.arange(len(self.uDOF), dtype=int)
        self.W_tau_jax_col = np.zeros(len(self.uDOF), dtype=int)

        self.Wla_tau_q_jax_row = self._Wla_tau_q_coo.row
        self.Wla_tau_q_jax_col = self._Wla_tau_q_coo.col

    def W_tau_jax(self, t, q):
        return self._W_t_jit(q)

    def W_tau(self, t, q):
        return self._W_t_jit(q)

    def Wla_tau_q_jax(self, t, q, u, tau):
        return self._W_t_q_jit(q) * tau

    def Wla_tau_q(self, t, q, u, tau):
        coo = self._Wla_tau_q_coo
        coo.data = self.__Wla_tau_q_jit(t, q, u, tau=tau)
        return coo

    def la_tau_jax(self, t, q, u, tau):
        return tau

    def la_tau(self, t, q, u, tau):
        return tau

    def tau(self, t):
        return 0.0
