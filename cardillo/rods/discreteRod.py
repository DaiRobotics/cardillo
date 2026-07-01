from abc import ABC
import numpy as np
from numpy.lib.stride_tricks import as_strided

from jax import vmap, jit, jacrev
from jax import numpy as jnp

from cardillo.math_numba import (
    norm,
    cross3,
    Log_SO3_quat,
)
from cardillo import math_jax

from cardillo.math import A_IB_basic
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.check_time_derivatives import check_time_derivatives
from cardillo.utility.cachetools import MyLRUCache
from cardillo.visualization.vtk_render2 import VisualDiscreteRod

from ._cross_section import CrossSectionInertias

E3 = jnp.eye(3, dtype=jnp.float64)
Z3 = jnp.zeros((3, 3))
Z34 = jnp.zeros((3, 4))


def _slice_to_array(s):
    if isinstance(s, slice):
        return np.arange(*s.indices(s.stop))
    elif isinstance(s, list):
        return [_slice_to_array(el) for el in s]


def _combine_indices(rows_list, cols_list):
    # rows_list and cols_list are lists of slices or arrays that define the submatrices of the COO matrix.
    ptr = np.empty(len(rows_list) + 1, dtype=int)
    ptr[0] = 0
    # count number
    for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
        if isinstance(rows, slice):
            start, stop, step = rows.indices(rows.stop)
            nrow = (stop - start) // step
        else:
            nrow = len(rows)
        if isinstance(cols, slice):
            start, stop, step = cols.indices(cols.stop)
            ncol = (stop - start) // step
        else:
            ncol = len(cols)
        ptr[i + 1] = ptr[i] + nrow * ncol
    rows_combined = np.empty(ptr[-1], dtype=int)
    cols_combined = np.empty(ptr[-1], dtype=int)
    # set rows and cols
    for i, (rows, cols) in enumerate(zip(rows_list, cols_list)):
        if isinstance(rows, slice):
            rows = _slice_to_array(rows)
        if isinstance(cols, slice):
            cols = _slice_to_array(cols)
        rows_combined[ptr[i] : ptr[i + 1]] = rows.repeat(len(cols))
        cols_combined[ptr[i] : ptr[i + 1]] = np.tile(cols, len(rows))
    return ptr, rows_combined, cols_combined


class ElementKinematics(ABC):

    @staticmethod
    @jit
    def r_OP(alpha, q, B_r_CP):
        A_IB = ElementKinematics.A_IB(alpha, q)
        r_OC0, r_OC1 = q[:3], q[7:10]
        r_OC = r_OC0 + alpha * (r_OC1 - r_OC0)
        return r_OC + A_IB @ B_r_CP

    @staticmethod
    @jit
    def r_OP_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.A_IB_q(alpha, q)
        r_OC_q = jnp.concatenate(
            [
                (1.0 - alpha) * E3,  # cols 0:3
                Z34,  # cols 3:7
                alpha * E3,  # cols 7:10
                Z34,  # cols 10:14
            ],
            axis=1,
        )
        return r_OC_q + B_r_CP @ A_IB_q

    @staticmethod
    @jit
    def v_P(alpha, q, u, B_r_CP):
        A_IB = ElementKinematics.A_IB(alpha, q)
        B_Omega = ElementKinematics.B_Omega(alpha, u)
        v_C0 = u[:3]
        v_C1 = u[6:9]
        v_C = v_C0 + alpha * (v_C1 - v_C0)
        return v_C + A_IB @ math_jax.cross3(B_Omega, B_r_CP)

    @staticmethod
    @jit
    def v_P_q(alpha, q, u, B_r_CP):
        A_IB_q = ElementKinematics.A_IB_q(alpha, q)
        B_Omega = ElementKinematics.B_Omega(alpha, u)
        return math_jax.cross3(B_Omega, B_r_CP) @ A_IB_q

    @staticmethod
    @jit
    def J_P(alpha, q, B_r_CP):
        A_IB = ElementKinematics.A_IB(alpha, q)
        B_r_CP_tilde = math_jax.ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde

        return jnp.concatenate(
            [
                (1.0 - alpha) * E3,
                -(1.0 - alpha) * r_CP_tilde,
                alpha * E3,
                -alpha * r_CP_tilde,
            ],
            axis=1,
        )

    @staticmethod
    @jit
    def J_P_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.A_IB_q(alpha, q)
        B_r_CP_tilde = math_jax.ax2skew(B_r_CP)
        r_CP_tilde_q = B_r_CP_tilde.T @ A_IB_q

        J_P_q = jnp.zeros((3, 12, 14))
        J_P_q = J_P_q.at[:, 3:6].set(-(1.0 - alpha) * r_CP_tilde_q)
        J_P_q = J_P_q.at[:, 9:12].set(-alpha * r_CP_tilde_q)
        return J_P_q

    @staticmethod
    @jit
    def a_P(alpha, q, u, u_dot, B_r_CP):
        A_IB = ElementKinematics.A_IB(alpha, q)
        B_Omega = ElementKinematics.B_Omega(alpha, u)
        B_Psi = ElementKinematics.B_Psi(alpha, u_dot)
        a_C0 = u_dot[:3]
        a_C1 = u_dot[6:9]
        a_C = a_C0 + alpha * (a_C1 - a_C0)
        return a_C + A_IB @ (
            math_jax.cross3(B_Psi, B_r_CP)
            + math_jax.cross3(B_Omega, math_jax.cross3(B_Omega, B_r_CP))
        )

    @staticmethod
    @jit
    def A_IB(alpha, q):
        P0, P1 = q[3:7], q[10:]
        P = P0 + alpha * (P1 - P0)
        return math_jax.Exp_SO3_quat_norm(P)

    @staticmethod
    @jit
    def A_IB_q(alpha, q):
        P0, P1 = q[3:7], q[10:]
        P = P0 + alpha * (P1 - P0)

        A_P = math_jax.Exp_SO3_quat_P_norm(P)

        A_IB_q = jnp.concatenate(
            [
                jnp.zeros((3, 3, 3)),
                (1.0 - alpha) * A_P,
                jnp.zeros((3, 3, 3)),
                alpha * A_P,
            ],
            axis=-1,
        )  # (3,3,14)

        return A_IB_q

    @staticmethod
    @jit
    def B_Omega(alpha, u):
        B_Omega_1 = u[3:6]
        B_Omega_2 = u[9:12]
        return B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)

    @staticmethod
    @jit
    def B_Psi(alpha, u_dot):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        B_Psi_1 = u_dot[3:6]
        B_Psi_2 = u_dot[9:12]
        B_Psi = B_Psi_1 + alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi

    @staticmethod
    @jit
    def B_J_R(alpha):
        return jnp.array(
            [
                [0, 0, 0, 1 - alpha, 0, 0, 0, 0, 0, alpha, 0, 0],
                [0, 0, 0, 0, 1 - alpha, 0, 0, 0, 0, 0, alpha, 0],
                [0, 0, 0, 0, 0, 1 - alpha, 0, 0, 0, 0, 0, alpha],
            ]
        )

    r_OP_batch = jit(vmap(r_OP.__func__))
    r_OP_q_batch = jit(vmap(r_OP_q.__func__))
    J_P_batch = jit(vmap(J_P.__func__))
    J_P_q_batch = jit(vmap(J_P_q.__func__))

    B_Omega_q = jnp.zeros((3, 14))
    B_J_R_q = jnp.zeros((3, 12, 14))
    B_Psi_q = jnp.zeros((3, 14))
    B_Psi_u = jnp.zeros((3, 12))


class DiscreteRod:
    def __init__(
        self,
        cross_section,
        material_model,
        nelement,
        Q,
        *,
        q0=None,
        u0=None,
        cross_section_inertias=CrossSectionInertias(),
        name="discrete_rod",
        damping_ratio=0.0,
    ):
        self._jaxed = True
        self.cross_section = cross_section
        self.material_model = material_model
        self.nelement = nelement
        self.nnode = nelement + 1
        self.name = name
        self._damp_ratio = damping_ratio
        self._damping = damping_ratio > 0

        # centerline parameter of nodes
        self.xi_node = np.linspace(0, 1, self.nnode)

        # stiffness matrices
        assert (
            cross_section._variable == material_model._variable
        ), "cross_section and material_model must both be variable or both be constant!"
        if material_model._variable:
            K_ga_els = []
            K_ka_els = []
            C_n_els = []
            C_m_els = []
            for el in range(nelement):
                xi = 0.5 * (self.xi_node[el] + self.xi_node[el + 1])
                K_ga_els.append(material_model.C_n(xi))
                K_ka_els.append(material_model.C_m(xi))
                C_n_els.append(material_model.C_n_inv(xi))
                C_m_els.append(material_model.C_m_inv(xi))
        else:
            K_ga_els = [material_model.C_n] * nelement
            K_ka_els = [material_model.C_m] * nelement
            C_n_els = [material_model.C_n_inv] * nelement
            C_m_els = [material_model.C_m_inv] * nelement
        self.K_ga_els = np.array(K_ga_els)
        self.K_ka_els = np.array(K_ka_els)
        self.C_n_els = np.array(C_n_els)
        self.C_m_els = np.array(C_m_els)
        if self._damping:
            self.K_ga_damp_els = self.K_ga_els * self._damp_ratio
            self.K_ka_damp_els = self.K_ka_els * self._damp_ratio
            self.C_n_damp_els = self.C_n_els / self._damp_ratio
            self.C_m_damp_els = self.C_m_els / self._damp_ratio

        # total DOFs
        self.nq = 7 * self.nnode
        self.nu = 6 * self.nnode
        self.nla_S = self.nnode
        self.nla_c = self.nelement * (6 if not self._damping else 12)

        self.q0 = Q if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        # slices of DOFs
        self.elDOF = [slice(7 * el, 7 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_u = [slice(6 * el, 6 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_la_c = [
            (
                slice(6 * el, 6 * (el + 1))
                if not self._damping
                else slice(12 * el, 12 * (el + 1))
            )
            for el in range(self.nelement)
        ]
        self.nodalDOF = [slice(7 * n, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r = [slice(7 * n, 7 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p = [slice(7 * n + 3, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_u = [slice(6 * n, 6 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r_u = [slice(6 * n, 6 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p_u = [slice(6 * n + 3, 6 * (n + 1)) for n in range(self.nnode)]

        # element lengths
        self.L_els = np.array(
            [
                norm(Q[self.nodalDOF_r[el + 1]] - Q[self.nodalDOF_r[el]])
                for el in range(self.nelement)
            ]
        )
        # rod length
        self.L = np.sum(self.L_els)

        self.__jit_func__()

        # reference strain
        _, self.B_gamma0, self.B_kappa0 = self._eval(Q)

        self.__init_coo__(cross_section_inertias)

    def __jit_func__(self):
        self._eval = jit(
            lambda q: DiscreteRod._eval_els(DiscreteRod._gen_element_q(q), self.L_els)
        )

        self.q_dot = jit(self._q_dot)

        _q_dot_q = jit(self._q_dot_q)

        def q_dot_q(t, q, u):
            self._q_dot_q_coo.data = _q_dot_q(t, q, u)
            return self._q_dot_q_coo

        self.q_dot_q = q_dot_q

        self.h = jit(self._h)

        _h_u = jit(self._h_u)

        def h_u(t, q, u):
            self._h_u_coo.data = _h_u(t, q, u)
            return self._h_u_coo

        self.h_u = h_u

        self.la_c = jit(self._la_c)

        self.c = jit(self._c)

        _c_q = jit(self._c_q)

        def c_q(t, q, u, la_c):
            self._c_q_coo.data = _c_q(t, q, u, la_c)
            return self._c_q_coo

        self.c_q = c_q

        if self._damping:
            _c_u = jit(
                lambda q, u: DiscreteRod._c_damp_u_els(
                    DiscreteRod._gen_element_q(q),
                    DiscreteRod._gen_element_u(u),
                    self.L_els,
                ).ravel()
            )

            def c_u(t, q, u, la_c):
                self._c_u_coo.data = _c_u(q, u)
                return self._c_u_coo

            self.c_u = c_u

        _W_c = jit(self._W_c)

        def W_c(t, q):
            self._W_c_coo.data = _W_c(t, q)
            return self._W_c_coo

        self.W_c = W_c

        _Wla_c_q = jit(self._Wla_c_q)

        def Wla_c_q(t, q, la_c):
            self._Wla_c_q_coo.data = _Wla_c_q(t, q, la_c)
            return self._Wla_c_q_coo

        self.Wla_c_q = Wla_c_q

    def __init_coo__(self, cross_section_inertias):
        # M
        self.constant_mass_matrix = True
        _M_coo = CooMatrix((self.nu, self.nu))
        row1 = col1 = np.array(_slice_to_array(self.nodalDOF_r_u)).flatten()
        ptr, row2, col2 = _combine_indices(self.nodalDOF_p_u, self.nodalDOF_p_u)
        _M_coo.row = np.concatenate((row1, row2))
        _M_coo.col = np.concatenate((col1, col2))
        _M_coo.data = np.empty_like(_M_coo.col, dtype=float)
        self._B_Theta_C = []
        for n in range(self.nnode):
            if n == 0:
                L_node = self.L_els[0] / 2
            elif n == self.nnode - 1:
                L_node = self.L_els[n - 1] / 2
            else:
                L_node = (self.L_els[n] + self.L_els[n - 1]) / 2
            if cross_section_inertias._variable:
                xi = self.xi_node[n]
                mass = cross_section_inertias.A_rho0(xi) * L_node
                B_Theta_C = cross_section_inertias.B_I_rho0(xi) * L_node
            else:
                mass = cross_section_inertias.A_rho0 * L_node
                B_Theta_C = cross_section_inertias.B_I_rho0 * L_node
            self._B_Theta_C.append(B_Theta_C)
            _M_coo.data[3 * n : 3 * (n + 1)] = mass
            _M_coo.data[self.nnode * 3 + ptr[n] : self.nnode * 3 + ptr[n + 1]] = (
                B_Theta_C.flatten()
            )
        self._M_coo = _M_coo.asformat("coo")
        self._M_coo.eliminate_zeros()
        self._B_Theta_C = np.array(self._B_Theta_C)

        # c_la_c
        _c_la_c_coo = CooMatrix((self.nla_c, self.nla_c))
        _, _c_la_c_coo.row, _c_la_c_coo.col = _combine_indices(
            self.elDOF_la_c, self.elDOF_la_c
        )
        c_la_c_els = (
            np.zeros((self.nelement, 6, 6), dtype=float)
            if not self._damping
            else np.zeros((self.nelement, 12, 12), dtype=float)
        )
        for el in range(self.nelement):
            c_la_c = c_la_c_els[el]
            c_la_c[:3, :3] = self.C_n_els[el]
            c_la_c[3:6, 3:6] = self.C_m_els[el]
            if self._damping:
                c_la_c[6:9, 6:9] = self.C_n_damp_els[el]
                c_la_c[9:, 9:] = self.C_m_damp_els[el]
            c_la_c *= self.L_els[el]
        _c_la_c_coo.data = c_la_c_els.ravel()
        self._c_la_c_coo = _c_la_c_coo.asformat("coo")
        self._c_la_c_coo.eliminate_zeros()

        # c_q
        self._c_q_coo = CooMatrix((self.nla_c, self.nq))
        _, self._c_q_coo.row, self._c_q_coo.col = _combine_indices(
            self.elDOF_la_c, self.elDOF
        )
        # c_u
        self._c_u_coo = CooMatrix((self.nla_c, self.nu))
        _, self._c_u_coo.row, self._c_u_coo.col = _combine_indices(
            self.elDOF_la_c, self.elDOF_u
        )
        # W_c
        self._W_c_coo = CooMatrix((self.nu, self.nla_c))
        _, self._W_c_coo.row, self._W_c_coo.col = _combine_indices(
            self.elDOF_u, self.elDOF_la_c
        )
        # Wla_c_q
        self._Wla_c_q_coo = CooMatrix((self.nu, self.nq))
        _, self._Wla_c_q_coo.row, self._Wla_c_q_coo.col = _combine_indices(
            self.elDOF_u, self.elDOF
        )
        # q_dot_q
        self._q_dot_q_coo = CooMatrix((self.nq, self.nq))
        _, self._q_dot_q_coo.row, self._q_dot_q_coo.col = _combine_indices(
            self.nodalDOF_p, self.nodalDOF_p
        )
        # q_dot_u
        self._q_dot_u_coo = CooMatrix((self.nq, self.nu))
        self._q_dot_u_coo.row = np.array(_slice_to_array(self.nodalDOF_r)).flatten()
        self._q_dot_u_coo.col = np.array(_slice_to_array(self.nodalDOF_r_u)).flatten()
        self._q_dot_u_coo.data = np.ones((len(self._q_dot_u_coo.col),), dtype=float)
        # h_u
        self._h_u_coo = CooMatrix((self.nu, self.nu))
        _, self._h_u_coo.row, self._h_u_coo.col = _combine_indices(
            self.nodalDOF_p_u, self.nodalDOF_p_u
        )
        # g_S_q
        self._g_S_q_coo = CooMatrix((self.nla_S, self.nq))
        _, self._g_S_q_coo.row, self._g_S_q_coo.col = _combine_indices(
            np.arange(self.nnode)[:, None],
            self.nodalDOF_p,
        )

    @staticmethod
    @jit
    def _gen_element_q(q):
        nelement = q.shape[0] // 7 - 1
        q_nodes = q[: (nelement + 1) * 7].reshape(nelement + 1, 7)
        return jnp.concatenate([q_nodes[:-1], q_nodes[1:]], axis=1)  # (nelement, 14)

    @staticmethod
    @jit
    def _gen_element_u(u):
        nelement = u.shape[0] // 6 - 1
        u_nodes = u[: (nelement + 1) * 6].reshape(nelement + 1, 6)
        return jnp.concatenate([u_nodes[:-1], u_nodes[1:]], axis=1)  # (nelement, 12)

    @staticmethod
    @jit
    def _c_damp_u_el(qe, ue, Le):
        B_gamma_dot_ue, B_kappa_dot_ue = DiscreteRod._eval_dot_u_el(qe, ue, Le)
        return jnp.concatenate(
            [jnp.zeros((6, 12)), B_gamma_dot_ue, B_kappa_dot_ue], axis=0
        ) * (-Le)

    _c_damp_u_els = jit(vmap(_c_damp_u_el.__func__))

    @staticmethod
    @jit
    def _eval_common(qe, Le):
        inv_Le = 1.0 / Le

        r_OC0 = qe[:3]
        r_OC1 = qe[7:10]
        P0 = qe[3:7]
        P1 = qe[10:14]

        r_OC_s = (r_OC1 - r_OC0) * inv_Le

        P = 0.5 * (P0 + P1)
        P_s = (P1 - P0) * inv_Le

        A_IB = math_jax.Exp_SO3_quat_norm(P)
        T = math_jax.T_SO3_quat_norm(P)

        return r_OC_s, P, P_s, A_IB, T

    @staticmethod
    @jit
    def _eval_dot_common(ue, Le):
        inv_Le = 1.0 / Le

        v_C0 = ue[:3]
        v_C1 = ue[6:9]
        B_Omega_0 = ue[3:6]
        B_Omega_1 = ue[9:12]

        v_C_s = (v_C1 - v_C0) * inv_Le

        B_Omega = 0.5 * (B_Omega_0 + B_Omega_1)
        B_Omega_s = (B_Omega_1 - B_Omega_0) * inv_Le

        return v_C_s, B_Omega, B_Omega_s

    @staticmethod
    @jit
    def _eval_el(qe, Le):
        r_OC_s, P, P_s, A_IB, T = DiscreteRod._eval_common(qe, Le)

        B_gamma = A_IB.T @ r_OC_s
        B_kappa = T @ P_s

        return A_IB, B_gamma, B_kappa

    _eval_els = jit(vmap(_eval_el.__func__))

    @staticmethod
    @jit
    def _eval_q_el(qe, Le):
        r_OC_s, P, P_s, A_IB, T = DiscreteRod._eval_common(qe, Le)
        inv_Le = 1.0 / Le

        # A_IB_qe
        A_P_2 = 0.5 * math_jax.Exp_SO3_quat_P_norm(P)
        A_IB_qe = jnp.concatenate(
            [jnp.zeros((3, 3, 3)), A_P_2, jnp.zeros((3, 3, 3)), A_P_2],
            axis=-1,
        )  # (3,3,14)

        # T_qe
        T_P_2 = 0.5 * math_jax.T_SO3_quat_P_norm(P)  # (3,4,4)
        T_qe = jnp.concatenate(
            [jnp.zeros((3, 4, 3)), T_P_2, jnp.zeros((3, 4, 3)), T_P_2],
            axis=-1,
        )  # (3,4,14)

        # B_gamma_qe <- B_gamma = A_IB.T @ r_OC_s
        A_IB_T = A_IB.T
        A_IB_T__r_OC_s_qe = (
            jnp.concatenate([-A_IB_T, Z34, A_IB_T, Z34], axis=1) * inv_Le
        )
        B_gamma_qe = jnp.tensordot(A_IB_qe, r_OC_s, axes=[[0], [0]]) + A_IB_T__r_OC_s_qe

        # B_kappa_qe <- B_kappa = T @ P_s
        T__P_s_qe = jnp.concatenate([Z3, -T, Z3, T], axis=1) * inv_Le
        B_kappa_qe = jnp.tensordot(T_qe, P_s, axes=[[1], [0]]) + T__P_s_qe

        return A_IB_qe, B_gamma_qe, B_kappa_qe

    _eval_q_els = jit(vmap(_eval_q_el.__func__))

    @staticmethod
    @jit
    def _eval_dot_el(qe, ue, Le):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        v_C_s, B_Omega, B_Omega_s = DiscreteRod._eval_dot_common(ue, Le)

        B_gamma_dot = A_IB.T @ v_C_s - math_jax.cross3(B_Omega, B_gamma)
        B_kappa_dot = B_Omega_s - math_jax.cross3(B_Omega, B_kappa)

        return B_gamma_dot, B_kappa_dot

    @staticmethod
    @jit
    def _eval_dot_q_el(qe, ue, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)

        v_C_s, B_Omega, B_Omega_s = DiscreteRod._eval_dot_common(ue, Le)

        B_gamma_dot_qe = (
            jnp.tensordot(A_IB_qe, v_C_s, axes=[[0], [0]])
            - math_jax.ax2skew(B_Omega) @ B_gamma_qe
        )
        B_kappa_dot_qe = -math_jax.ax2skew(B_Omega) @ B_kappa_qe

        return B_gamma_dot_qe, B_kappa_dot_qe

    # TODO: analytical function
    _eval_dot_u_el = jit(jacrev(_eval_dot_el.__func__, argnums=1))

    def _element_number(self, xi):
        num = jnp.int64(xi * self.nelement)
        return num - jnp.int64(num >= self.nelement)

    def _view_element_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nelement, 14), strides=(stride * 7, stride))

    def _view_element_la_c(self, la_c):
        return la_c.reshape((self.nelement, -1))

    def _view_nodal_q(self, q):
        return q.reshape((self.nnode, 7))

    def _view_nodal_u(self, u):
        return u.reshape((self.nnode, 6))

    # def nodes(self, q):
    #     """Returns nodal position coordinates"""
    #     q_body = q[self.qDOF]
    #     return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    @staticmethod
    def straight_configuration(
        nelement,
        L,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    ):
        nnode = nelement + 1
        x0 = np.linspace(0, L, num=nnode)
        y0 = np.zeros(nnode)
        z0 = np.zeros(nnode)
        r_OC = np.vstack((x0, y0, z0))
        r_OC = r_OP0 + (A_IB0 @ r_OC).T
        P = np.repeat(Log_SO3_quat(A_IB0)[None, :], nnode, axis=0)
        return np.hstack((r_OC, P)).flatten()

    @staticmethod
    def serret_frenet_configuration(
        nelement,
        r_OP,
        r_OP_xi,
        r_OP_xixi,
        xi1,
        alpha=0.0,
        r_OP0=jnp.zeros(3, dtype=float),
        A_IB0=jnp.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for a pre-curved rod along curve r_OP. The cross-section orientations are based on the Serret-Frenet equations and afterwards rotated by alpha."""
        nnodes_r = nelement + 1

        r_OP, r_OP_xi, r_OP_xixi = check_time_derivatives(r_OP, r_OP_xi, r_OP_xixi)
        alpha, _, _ = check_time_derivatives(alpha, None, None)

        xis = np.linspace(0, xi1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((nnodes_r, 3))
        p0 = np.zeros((nnodes_r, 4))

        for i, xii in enumerate(xis):
            r0[i] = r_OP0 + A_IB0 @ r_OP(xii)
            r_xi = r_OP_xi(xii)
            r_xixi = r_OP_xixi(xii)
            ex = r_xi / norm(r_xi)
            ey = r_xixi - ex * (ex @ r_xixi)
            ey = ey / norm(ey)
            A_B0B = np.vstack([ex, ey, cross3(ex, ey)]).T
            A_IB = A_IB0 @ A_B0B @ A_IB_basic(alpha(xii)).x
            p0[i] = Log_SO3_quat(A_IB)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[i] @ p0[i + 1]
            if inner < 0:
                p0[i + 1] *= -1

        return np.concatenate([r0, p0], axis=1).flatten()

    @staticmethod
    def pose_configuration(
        nelement,
        B0_r_C0Ci,
        A_B0Bi,
        r_OC0=jnp.zeros(3, dtype=float),
        A_IB0=jnp.eye(3, dtype=float),
    ):
        """Compute generalized position coordinates for a pre-curved rod with centerline curve r_OP and orientation of A_IB."""
        nnodes_r = nelement + 1

        assert callable(B0_r_C0Ci), "r_OP must be callable!"
        assert callable(A_B0Bi), "A_IB must be callable!"

        xis = np.linspace(0, 1, nnodes_r)

        # nodal positions and unit quaternions
        r0 = np.zeros((nnodes_r, 3))
        p0 = np.zeros((nnodes_r, 4))

        for i, xii in enumerate(xis):
            r0[i] = r_OC0 + A_IB0 @ B0_r_C0Ci(xii)
            A_IBi = A_IB0 @ A_B0Bi(xii)
            p0[i] = Log_SO3_quat(A_IBi)

        # check for the right quaternion hemisphere
        for i in range(nnodes_r - 1):
            inner = p0[i] @ p0[i + 1]
            if inner < 0:
                p0[i + 1] *= -1

        return np.concatenate([r0, p0], axis=1).flatten()

    #####################
    # kinematic equations
    #####################
    @staticmethod
    @jit
    def _q_dot_node(q, u):
        T = math_jax.T_SO3_inv_quat(q[3:]) @ u[3:]
        return jnp.concatenate([u[:3], T])

    _q_dot_nodes = jit(vmap(_q_dot_node.__func__))

    def _q_dot(self, t, q, u):
        return DiscreteRod._q_dot_nodes(q.reshape((-1, 7)), u.reshape((-1, 6))).ravel()

    @staticmethod
    @jit
    def _p_dot_p_node(q, u):
        return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:])

    _p_dot_p_nodes = jit(vmap(_p_dot_p_node.__func__))

    def _q_dot_q(self, t, q, u):
        return DiscreteRod._p_dot_p_nodes(
            q.reshape((-1, 7)), u.reshape((-1, 6))
        ).ravel()

    def q_dot_u(self, t, q):
        T_SO3_inv_quat_nodes = math_jax.T_SO3_inv_quat_batch(
            self._view_nodal_q(q)[:, 3:]
        ).__array__()
        # TODO: speed up
        for n in range(self.nnode):
            nodalDOF_p = self.nodalDOF_p[n]
            nodalDOF_p_u = self.nodalDOF_p_u[n]
            self._q_dot_u_coo[n, nodalDOF_p, nodalDOF_p_u] = T_SO3_inv_quat_nodes[n]
        return self._q_dot_u_coo

    def step_callback(self, t, q, u):
        p = q.reshape((self.nnode, 7))[:, 3:]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self._M_coo

    @staticmethod
    @jit
    def _h_node(u, B_Theta_C):
        B_omega_IB = u[3:]
        tmp = B_Theta_C @ B_omega_IB
        cross = math_jax.cross3(tmp, B_omega_IB)
        return jnp.pad(cross, (3, 0))

    _h_nodes = jit(vmap(_h_node.__func__))

    def _h(self, t, q, u):
        return DiscreteRod._h_nodes(u.reshape((-1, 6)), self._B_Theta_C).ravel()

    @staticmethod
    @jit
    def _h_u_node(B_omega_IB, B_Theta_C):
        return (
            math_jax.ax2skew(B_Theta_C @ B_omega_IB)
            - math_jax.ax2skew(B_omega_IB) @ B_Theta_C
        )

    _h_u_nodes = jit(vmap(_h_u_node.__func__))

    def _h_u(self, t, q, u):
        return DiscreteRod._h_u_nodes(
            u.reshape((-1, 6))[:, 3:], self._B_Theta_C
        ).ravel()

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        return np.einsum("ij,ij->i", p, p) - 1

    def g_S_q(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        self._g_S_q_coo.data = (2 * p).ravel()
        return self._g_S_q_coo

    ############
    # compliance
    ############
    @staticmethod
    @jit
    def _la_c_el(qe, Le, B_gamma0, B_kappa0, K_ga, K_ka):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        B_n = K_ga @ (B_gamma - B_gamma0)
        B_m = K_ka @ (B_kappa - B_kappa0)
        return jnp.concatenate([B_n, B_m])

    _la_c_els = jit(vmap(_la_c_el.__func__))

    @staticmethod
    @jit
    def _la_c_damp_el(qe, ue, Le, B_gamma0, B_kappa0, K_ga, K_ka, K_ga_damp, K_ka_damp):
        la_c_12 = DiscreteRod._la_c_el(qe, Le, B_gamma0, B_kappa0, K_ga, K_ka)
        B_gamma_dot, B_kappa_dot = DiscreteRod._eval_dot_el(qe, ue, Le)
        B_n_damp = K_ga_damp @ B_gamma_dot
        B_m_damp = K_ka_damp @ B_kappa_dot
        return jnp.concatenate([la_c_12, B_n_damp, B_m_damp])

    _la_c_damp_els = jit(vmap(_la_c_damp_el.__func__))

    def _la_c(self, t, q, u):
        return (
            DiscreteRod._la_c_els(
                DiscreteRod._gen_element_q(q),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.K_ga_els,
                self.K_ka_els,
            ).ravel()
            if not self._damping
            else DiscreteRod._la_c_damp_els(
                DiscreteRod._gen_element_q(q),
                DiscreteRod._gen_element_u(u),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.K_ga_els,
                self.K_ka_els,
                self.K_ga_damp_els,
                self.K_ka_damp_els,
            ).ravel()
        )

    @staticmethod
    @jit
    def _c_el(qe, la_c, Le, B_gamma0, B_kappa0, C_n, C_m):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        B_n, B_m = la_c[:3], la_c[3:]

        c1 = (C_n @ B_n - (B_gamma - B_gamma0)) * Le
        c2 = (C_m @ B_m - (B_kappa - B_kappa0)) * Le

        return jnp.concatenate([c1, c2])

    _c_els = jit(vmap(_c_el.__func__))

    @staticmethod
    @jit
    def _c_damp_el(qe, ue, la_c, Le, B_gamma0, B_kappa0, C_n, C_m, C_n_damp, C_m_damp):
        c12 = DiscreteRod._c_el(qe, la_c[:6], Le, B_gamma0, B_kappa0, C_n, C_m)
        B_gamma_dot, B_kappa_dot = DiscreteRod._eval_dot_el(qe, ue, Le)
        B_n_damp, B_m_damp = la_c[6:9], la_c[9:]

        c3 = (C_n_damp @ B_n_damp - B_gamma_dot) * Le
        c4 = (C_m_damp @ B_m_damp - B_kappa_dot) * Le

        return jnp.concatenate([c12, c3, c4])

    _c_damp_els = jit(vmap(_c_damp_el.__func__))

    def _c(self, t, q, u, la_c):
        return (
            DiscreteRod._c_els(
                DiscreteRod._gen_element_q(q),
                la_c.reshape((self.nelement, -1)),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.C_n_els,
                self.C_m_els,
            ).ravel()
            if not self._damping
            else DiscreteRod._c_damp_els(
                DiscreteRod._gen_element_q(q),
                DiscreteRod._gen_element_u(u),
                la_c.reshape((self.nelement, -1)),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.C_n_els,
                self.C_m_els,
                self.C_n_damp_els,
                self.C_m_damp_els,
            ).ravel()
        )

    def c_la_c(self):
        return self._c_la_c_coo

    @staticmethod
    @jit
    def _c_q_el(qe, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)
        return jnp.concatenate([B_gamma_qe, B_kappa_qe], axis=0) * (-Le)

    _c_q_els = jit(vmap(_c_q_el.__func__))

    @staticmethod
    @jit
    def _c_damp_q_el(qe, ue, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)
        B_gamma_dot_qe, B_kappa_dot_qe = DiscreteRod._eval_dot_q_el(qe, ue, Le)
        return jnp.concatenate(
            [B_gamma_qe, B_kappa_qe, B_gamma_dot_qe, B_kappa_dot_qe], axis=0
        ) * (-Le)

    _c_damp_q_els = jit(vmap(_c_damp_q_el.__func__))

    def _c_q(self, t, q, u, la_c):
        return (
            DiscreteRod._c_q_els(DiscreteRod._gen_element_q(q), self.L_els).ravel()
            if not self._damping
            else DiscreteRod._c_damp_q_els(
                DiscreteRod._gen_element_q(q),
                DiscreteRod._gen_element_u(u),
                self.L_els,
            ).ravel()
        )

    @staticmethod
    @jit
    def _W_c_el(qe, Le):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        s1 = 0.5 * Le * math_jax.ax2skew(B_gamma)
        s2 = 0.5 * Le * math_jax.ax2skew(B_kappa)

        row1 = jnp.concatenate([A_IB, Z3], axis=1)
        row2 = jnp.concatenate([s1, E3 + s2], axis=1)
        row3 = jnp.concatenate([-A_IB, Z3], axis=1)
        row4 = jnp.concatenate([s1, -E3 + s2], axis=1)

        return jnp.concatenate([row1, row2, row3, row4], axis=0)

    _W_c_els = jit(vmap(_W_c_el.__func__))

    @staticmethod
    @jit
    def _W_c_damp_el(qe, Le):
        W_c_el = DiscreteRod._W_c_el(qe, Le)
        return jnp.concatenate([W_c_el, W_c_el], axis=1)

    _W_c_damp_els = jit(vmap(_W_c_damp_el.__func__))

    def _W_c(self, t, q):
        return (
            DiscreteRod._W_c_els(DiscreteRod._gen_element_q(q), self.L_els).ravel()
            if not self._damping
            else DiscreteRod._W_c_damp_els(
                DiscreteRod._gen_element_q(q), self.L_els
            ).ravel()
        )

    @staticmethod
    @jit
    def _Wla_c_q_el(qe, la_c, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)
        B_n = la_c[:3]
        B_m = la_c[3:]

        A_IB_B_n__qe = B_n @ A_IB_qe

        common = (
            -0.5
            * Le
            * (
                jnp.cross(B_n[:, None], B_gamma_qe, axis=0)
                + jnp.cross(B_m[:, None], B_kappa_qe, axis=0)
            )
        )

        return jnp.concatenate([A_IB_B_n__qe, common, -A_IB_B_n__qe, common], axis=0)

    _Wla_c_q_els = jit(vmap(_Wla_c_q_el.__func__))

    @staticmethod
    @jit
    def _Wla_c_q_damp_el(qe, la_c, Le):
        _la_c = la_c[:6] + la_c[6:]
        return DiscreteRod._Wla_c_q_el(qe, _la_c, Le)

    _Wla_c_q_damp_els = jit(vmap(_Wla_c_q_damp_el.__func__))

    def _Wla_c_q(self, t, q, la_c):
        return (
            DiscreteRod._Wla_c_q_els(
                DiscreteRod._gen_element_q(q),
                la_c.reshape((self.nelement, -1)),
                self.L_els,
            ).ravel()
            if not self._damping
            else DiscreteRod._Wla_c_q_damp_els(
                DiscreteRod._gen_element_q(q),
                la_c.reshape((self.nelement, -1)),
                self.L_els,
            ).ravel()
        )

    def _alpha(self, xi):
        num = self._element_number(xi)
        xi_node = jnp.array(self.xi_node)
        return (xi - xi_node[num]) / (xi_node[num + 1] - xi_node[num])

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def local_qDOF_P(self, xi):
        el = self._element_number(xi)
        return self.elDOF[el]

    def local_uDOF_P(self, xi):
        el = self._element_number(xi)
        return self.elDOF_u[el]

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qe, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.r_OP(alpha, qe, B_r_CP)

    def r_OP_q(self, t, qe, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.r_OP_q(alpha, qe, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.v_P(alpha, qe, ue, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.v_P_q(alpha, qe, ue, B_r_CP)

    def J_P(self, t, qe, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.J_P(alpha, qe, B_r_CP)

    def J_P_q(self, t, qe, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.J_P_q(alpha, qe, B_r_CP)

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=jnp.zeros(3, dtype=float)):
        alpha = self._alpha(xi)
        return ElementKinematics.a_P(alpha, qe, ue, ue_dot, B_r_CP)

    def A_IB(self, t, qe, xi):
        alpha = self._alpha(xi)
        return ElementKinematics.A_IB(alpha, qe)

    def A_IB_q(self, t, qe, xi):
        alpha = self._alpha(xi)
        return ElementKinematics.A_IB_q(alpha, qe)

    def B_Omega(self, t, qe, ue, xi):
        alpha = self._alpha(xi)
        return ElementKinematics.B_Omega(alpha, ue)

    def B_Omega_q(self, t, qe, ue, xi):
        return ElementKinematics.B_Omega_q

    def B_J_R(self, t, qe, xi):
        alpha = self._alpha(xi)
        return ElementKinematics.B_J_R(alpha)

    def B_J_R_q(self, t, qe, xi):
        return ElementKinematics.B_J_R_q

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        alpha = self._alpha(xi)
        return ElementKinematics.B_Psi(alpha, ue_dot)

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        return ElementKinematics.B_Psi_q

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        return ElementKinematics.B_Psi_u

    def export(self, sol_i, **kwargs):
        if not hasattr(self, "_visual_twin"):
            self._visual_twin = VisualDiscreteRod(self)
        self._visual_twin.update_visual_state(sol_i)
        return self._visual_twin._ugrid
