import numpy as np
from numpy.lib.stride_tricks import as_strided

import jax
from jax import vmap, jit
from jax import numpy as jnp
from numba import njit

from cardillo.math_numba import (
    norm,
    cross3,
    ax2skew,
    Log_SO3_quat,
    Exp_SO3_quat,
    Exp_SO3_quat_P,
)
from cardillo import math_jax

from cardillo.math import A_IB_basic
from cardillo.utility.coo_matrix import CooMatrix
from cardillo.utility.check_time_derivatives import check_time_derivatives
from cardillo.utility.cachetools import MyLRUCache
from cardillo.visualization.vtk_render2 import VisualDiscreteRod

from ._cross_section import CrossSectionInertias

jax.config.update("jax_enable_x64", True)

eye3 = jnp.eye(3, dtype=jnp.float64)
zeros3 = jnp.zeros((3, 3))


_nla_c_el = 6  # 6/12


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


class ElementKinematics:
    def __init__(self, xi, alpha):
        self.xi = xi
        self.alpha = alpha

        # allocate memery
        self._B_Omega_q = np.zeros((3, 14), dtype=float)
        self._B_J_R = np.zeros((3, 12), dtype=float)
        self._B_J_R[0, 3] = self._B_J_R[1, 4] = self._B_J_R[2, 5] = 1 - alpha
        self._B_J_R[0, 9] = self._B_J_R[1, 10] = self._B_J_R[2, 11] = alpha
        self._B_J_R_q = np.zeros((3, 12, 14), dtype=float)
        self._B_Psi_q = np.zeros((3, 14), dtype=float)
        self._B_Psi_u = np.zeros((3, 12), dtype=float)

        self._A_IB_cache = MyLRUCache(maxsize=5)
        self._A_IB_q_cache = MyLRUCache(maxsize=5)

    ##########################
    # r_OP / A_IB contribution
    ##########################

    def r_OP(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q)
        return _r_OP(self.alpha, q, A_IB, B_r_CP)

    def r_OP_q(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q)
        return _r_OP_q(self.alpha, A_IB_q, B_r_CP)

    def v_P(self, t, q, u, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q)
        return _v_P(self.alpha, A_IB, u, self.B_Omega(t, q, u), B_r_CP)

    def v_P_q(self, t, q, u, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q)
        B_Omega = self.B_Omega(t, q, u)
        return _v_P_q(A_IB_q, B_Omega, B_r_CP)

    def J_P(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q)
        return _J_P(self.alpha, A_IB, B_r_CP)

    def J_P_q(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        A_IB_q = self.A_IB_q(t, q)
        return _J_P_q(self.alpha, A_IB_q, B_r_CP)

    def a_P(self, t, q, u, u_dot, B_r_CP=np.zeros(3, dtype=float)):
        A_IB = self.A_IB(t, q)
        B_Omega = self.B_Omega(t, q, u)
        B_Psi = self.B_Psi(t, q, u, u_dot)
        return _a_P(self.alpha, u_dot, A_IB, B_Omega, B_Psi, B_r_CP)

    def a_P_q(self, t, q, u, u_dot, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, q, u, xi)
    #     B_Psi = self.B_Psi(t, q, u, u_dot, xi)
    #     a_P_q = np.einsum(
    #         "ijk,j->ik",
    #         self.A_IB_q(t, q, xi),
    #         cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP)),
    #     )
    #     return a_P_q

    def a_P_u(self, t, q, u, u_dot, B_r_CP=None):
        raise

    #     B_Omega = self.B_Omega(t, q, u, xi)
    #     local = -self.A_IB(t, q, xi) @ (
    #         ax2skew(cross3(B_Omega, B_r_CP)) + ax2skew(B_Omega) @ ax2skew(B_r_CP)
    #     )

    #     N, _ = self.basis_functions_r(xi)
    #     a_P_u = np.zeros((3, self.nu_element), dtype=float)
    #     for node in range(self.nnodes_element_r):
    #         a_P_u[:, self.nodalDOF_element_p_u[node]] += N[node] * local

    #     return a_P_u

    def A_IB(self, t, q):
        key = q.tobytes()
        A_IB = self._A_IB_cache[key]
        if A_IB is None:
            A_IB = _A_IB(self.alpha, q)
            self._A_IB_cache[key] = A_IB
        return A_IB

    def A_IB_q(self, t, q):
        key = q.tobytes()
        A_IB_q = self._A_IB_q_cache[key]
        if A_IB_q is None:
            A_IB_q = _A_IB_q(self.alpha, q)
            self._A_IB_q_cache[key] = A_IB_q
        return A_IB_q

    def B_Omega(self, t, q, u):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        angular velocities in the B-frame.
        """
        return _B_Omega(self.alpha, u)

    def B_Omega_q(self, t, q, u):
        return self._B_Omega_q

    def B_J_R(self, t, q):
        return self._B_J_R

    def B_J_R_q(self, t, q):
        return self._B_J_R_q

    def B_Psi(self, t, q, u, u_dot):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        B_Psi_1 = u_dot[3:6]
        B_Psi_2 = u_dot[9:12]
        B_Psi = B_Psi_1 + self.alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi

    def B_Psi_q(self, t, q, u, u_dot):
        return self._B_Psi_q

    def B_Psi_u(self, t, q, u, u_dot):
        return self._B_Psi_u


@njit(cache=True)
def is_nonzero3(v):
    return v[0] != 0.0 or v[1] != 0.0 or v[2] != 0.0


@njit(cache=True)
def _r_OP(alpha, q, A_IB, B_r_CP):
    r_OC0, r_OC1 = q[:3], q[7:10]
    r_OP = (1 - alpha) * r_OC0 + alpha * r_OC1
    if is_nonzero3(B_r_CP):
        r_OP += A_IB @ B_r_CP
    return r_OP


@njit(cache=True)
def _r_OP_q(alpha, A_IB_q, B_r_CP):
    r_OP_q = np.zeros((3, 14), dtype=float)
    r_OP_q[0, 0] = r_OP_q[1, 1] = r_OP_q[2, 2] = 1 - alpha
    r_OP_q[0, 7] = r_OP_q[1, 8] = r_OP_q[2, 9] = alpha
    if is_nonzero3(B_r_CP):
        for i in range(3):
            r_OP_q[i] += B_r_CP @ A_IB_q[i]
    return r_OP_q


@njit(cache=True)
def _v_P(alpha, A_IB, u, B_Omega, B_r_CP):
    v_C0 = u[:3]
    v_C1 = u[6:9]
    v_C = v_C0 + alpha * (v_C1 - v_C0)

    if is_nonzero3(B_r_CP):
        return v_C + A_IB @ cross3(B_Omega, B_r_CP)
    else:
        return v_C


@njit(cache=True)
def _v_P_q(A_IB_q, B_Omega, B_r_CP):
    v_P_q = np.zeros((3, 14), dtype=float)
    if is_nonzero3(B_r_CP):
        cross = cross3(B_Omega, B_r_CP)
        for i in range(3):
            v_P_q[i] = cross @ A_IB_q[i]
    return v_P_q


@njit(cache=True)
def _J_P(alpha, A_IB, B_r_CP):
    J_P = np.zeros((3, 12), dtype=float)
    J_P[0, 0] = J_P[1, 1] = J_P[2, 2] = 1 - alpha
    J_P[0, 6] = J_P[1, 7] = J_P[2, 8] = alpha
    if is_nonzero3(B_r_CP):
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde
        J_P[:, 3:6] = -(1 - alpha) * r_CP_tilde
        J_P[:, 9:12] = -alpha * r_CP_tilde
    return J_P


@njit(cache=True)
def _J_P_q(alpha, A_IB_q, B_r_CP):
    J_P_q = np.zeros((3, 12, 14), dtype=float)
    if is_nonzero3(B_r_CP):
        B_r_CP_tilde = ax2skew(B_r_CP)
        r_CP_tilde_q = np.zeros((3, 3, 14), dtype=float)
        for i in range(3):
            r_CP_tilde_q[i] = B_r_CP_tilde.T @ A_IB_q[i]
        J_P_q[:, 3:6] = -(1 - alpha) * r_CP_tilde_q
        J_P_q[:, 9:12] = -alpha * r_CP_tilde_q
    return J_P_q


@njit(cache=True)
def _a_P(alpha, u_dot, A_IB, B_Omega, B_Psi, B_r_CP):
    a_C0 = u_dot[:3]
    a_C1 = u_dot[6:9]
    a_C = a_C0 + alpha * (a_C1 - a_C0)
    if is_nonzero3(B_r_CP):
        return a_C + A_IB @ (
            cross3(B_Psi, B_r_CP) + cross3(B_Omega, cross3(B_Omega, B_r_CP))
        )
    else:
        return a_C


@njit(cache=True)
def _A_IB(alpha, q):
    P0, P1 = q[3:7], q[10:]
    P = (1 - alpha) * P0 + alpha * P1
    return Exp_SO3_quat(P, normalize=True)


@njit(cache=True)
def _A_IB_q(alpha, q):
    P0, P1 = q[3:7], q[10:]
    P = (1 - alpha) * P0 + alpha * P1

    P_q = np.zeros((4, 14), dtype=float)

    P_q[0, 3] = P_q[1, 4] = P_q[2, 5] = P_q[3, 6] = 1 - alpha
    P_q[0, 10] = P_q[1, 11] = P_q[2, 12] = P_q[3, 13] = alpha

    A_P = Exp_SO3_quat_P(P, normalize=True)
    A_IB_q = np.empty((3, 3, 14))
    for i in range(3):
        A_IB_q[i] = A_P[i] @ P_q
    return A_IB_q


@njit(cache=True)
def _B_Omega(alpha, u):
    B_Omega_1 = u[3:6]
    B_Omega_2 = u[9:12]
    return B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)


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
    ):
        # manual caches
        # self._eval_cache = self._deval_cache = np.empty(0).tobytes()
        self._eval_cache = MyLRUCache(maxsize=10)
        self._deval_cache = MyLRUCache(maxsize=10)

        self.cross_section = cross_section
        # super().__init__(cross_section)
        self.material_model = material_model
        self.nelement = nelement
        self.nnode = nelement + 1
        self.name = name

        # centerline parameter of nodes
        self.xi_node = np.linspace(0, 1, self.nnode)

        #
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

        # total DOFs
        self.nq = 7 * self.nnode
        self.nu = 6 * self.nnode
        self.nla_S = self.nnode
        self.nla_c = self.nelement * _nla_c_el

        self.q0 = Q if q0 is None else np.asarray(q0)
        self.u0 = np.zeros(self.nu, dtype=float) if u0 is None else np.asarray(u0)
        self.la_S0 = np.zeros(self.nla_S, dtype=float)

        # slices of DOFs
        self.elDOF = [slice(7 * el, 7 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_u = [slice(6 * el, 6 * (el + 2)) for el in range(self.nelement)]
        self.elDOF_la_c = [
            slice(_nla_c_el * el, _nla_c_el * (el + 1)) for el in range(self.nelement)
        ]
        self.nodalDOF = [slice(7 * n, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r = [slice(7 * n, 7 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p = [slice(7 * n + 3, 7 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_u = [slice(6 * n, 6 * (n + 1)) for n in range(self.nnode)]
        self.nodalDOF_r_u = [slice(6 * n, 6 * n + 3) for n in range(self.nnode)]
        self.nodalDOF_p_u = [slice(6 * n + 3, 6 * (n + 1)) for n in range(self.nnode)]

        self.set_reference_strains(Q)

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
                w = self.L_els[0] / 2
            elif n == self.nnode - 1:
                w = self.L_els[n - 1] / 2
            else:
                w = (self.L_els[n] + self.L_els[n - 1]) / 2
            if cross_section_inertias._variable:
                xi = self.xi_node[n]
                mass = cross_section_inertias.A_rho0(xi) * w
                B_Theta_C = cross_section_inertias.B_I_rho0(xi) * w
            else:
                mass = cross_section_inertias.A_rho0 * w
                B_Theta_C = cross_section_inertias.B_I_rho0 * w
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
        c_la_c_els = np.zeros((self.nelement, _nla_c_el, _nla_c_el), dtype=float)
        for el in range(self.nelement):
            c_la_c = c_la_c_els[el]
            c_la_c[:3, :3] = self.C_n_els[el]
            c_la_c[3:6, 3:6] = self.C_m_els[el]
            if _nla_c_el == 12:
                c_la_c[6:9, 6:9] = self.C_n_els[el]
                c_la_c[9:, 9:] = self.C_m_els[el]
            c_la_c *= self.L_els[el]
        _c_la_c_coo.data = c_la_c_els.ravel()
        self._c_la_c_coo = _c_la_c_coo.asformat("coo")
        self._c_la_c_coo.eliminate_zeros()

        self._markers = {}

        # allocate memery
        self._B_Omega_q = np.zeros((3, 14), dtype=float)
        self._B_J_R = np.zeros((3, 12), dtype=float)
        self._B_J_R_q = np.zeros((3, 12, 14), dtype=float)
        self._B_Psi_q = np.zeros((3, 14), dtype=float)
        self._B_Psi_u = np.zeros((3, 12), dtype=float)
        # CooMatrix
        self._c_q_coo = CooMatrix((self.nla_c, self.nq))
        _, self._c_q_coo.row, self._c_q_coo.col = _combine_indices(
            self.elDOF_la_c, self.elDOF
        )

        self._W_c_coo = CooMatrix((self.nu, self.nla_c))
        _, self._W_c_coo.row, self._W_c_coo.col = _combine_indices(
            self.elDOF_u, self.elDOF_la_c
        )
        self._Wla_c_q_coo = CooMatrix((self.nu, self.nq))
        _, self._Wla_c_q_coo.row, self._Wla_c_q_coo.col = _combine_indices(
            self.elDOF_u, self.elDOF
        )

        self._q_dot_q_coo = CooMatrix((self.nq, self.nq))
        _, self._q_dot_q_coo.row, self._q_dot_q_coo.col = _combine_indices(
            self.nodalDOF_p, self.nodalDOF_p
        )
        self._q_dot_u_coo = CooMatrix((self.nq, self.nu))
        self._q_dot_u_coo.row = np.array(_slice_to_array(self.nodalDOF_r)).flatten()
        self._q_dot_u_coo.col = np.array(_slice_to_array(self.nodalDOF_r_u)).flatten()
        self._q_dot_u_coo.data = np.ones((len(self._q_dot_u_coo.col),), dtype=float)
        self._h_u_coo = CooMatrix((self.nu, self.nu))
        _, self._h_u_coo.row, self._h_u_coo.col = _combine_indices(
            self.nodalDOF_p_u, self.nodalDOF_p_u
        )
        self._g_S_q_coo = CooMatrix((self.nla_S, self.nq))
        _, self._g_S_q_coo.row, self._g_S_q_coo.col = _combine_indices(
            np.arange(self.nnode)[:, None],
            self.nodalDOF_p,
        )

        # cache
        self._alpha_cache = MyLRUCache(maxsize=self.nnode * 10)
        self._eval_kinematics_cache = MyLRUCache(maxsize=self.nnode * 10)

    def set_reference_strains(self, Q):
        self.L_els = np.array(
            [
                norm(Q[self.nodalDOF_r[el + 1]] - Q[self.nodalDOF_r[el]])
                for el in range(self.nelement)
            ]
        )
        self.L = np.sum(self.L_els)
        _, self.B_Gamma0, self.B_Kappa0 = self._eval_els(Q)
        self.B_Ga_Ka0 = np.concatenate((self.B_Gamma0, self.B_Kappa0), axis=1)

    def element_number(self, xi):
        num = int(xi * self.nelement)
        return num if num < self.nelement else num - 1

    def element_interval(self, el):
        return (self.xi_node[el], self.xi_node[el + 1])

    def _view_element_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nelement, 14), strides=(stride * 7, stride))

    def _view_element_la_c(self, la_c):
        return la_c.reshape((self.nelement, _nla_c_el))

    def _view_nodal_q(self, q):
        return q.reshape((self.nnode, 7))

    def _view_nodal_u(self, u):
        return u.reshape((self.nnode, 6))

    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def get_marker(self, xi):
        try:
            mk = self._markers[xi]
        except KeyError:
            alpha = self._alpha(xi)
            mk = ElementKinematics(xi, alpha)
            self._markers[xi] = mk
        if hasattr(self, "qDOF") and not hasattr(mk, "qDOF"):
            num = self.element_number(xi)
            mk.t0 = self.t0
            mk.q0 = self.q0[self.elDOF[num]]
            mk.qDOF = self.qDOF[self.elDOF[num]]
            mk.uDOF = self.uDOF[self.elDOF_u[num]]
        return mk

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
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
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
        r_OC0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
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

    def assembler_callback(self):
        for xi, mk in self._markers.items():
            num = self.element_number(xi)
            mk.t0 = self.t0
            mk.q0 = self.q0[self.elDOF[num]]
            mk.u0 = self.u0[self.elDOF_u[num]]
            mk.qDOF = self.qDOF[self.elDOF[num]]
            mk.uDOF = self.uDOF[self.elDOF_u[num]]

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return np.asarray(
            _q_dot_nodes(self._view_nodal_q(q), self._view_nodal_u(u))
        ).ravel()

    def q_dot_q(self, t, q, u):
        p_dot_p_nodes = np.asarray(
            _p_dot_p_nodes(self._view_nodal_q(q), self._view_nodal_u(u))
        )
        self._q_dot_q_coo.data = p_dot_p_nodes.ravel()
        # for n in range(self.nnode):
        #     nodalDOF_p = self.nodalDOF_p[n]
        #     self._q_dot_q_coo[n, nodalDOF_p, nodalDOF_p] = p_dot_q_nodes[n]
        return self._q_dot_q_coo

    def q_dot_u(self, t, q):
        T_SO3_inv_quat_nodes = np.asarray(
            math_jax.T_SO3_inv_quat_batch(self._view_nodal_q(q)[:, 3:], False)
        )
        for n in range(self.nnode):
            nodalDOF_p = self.nodalDOF_p[n]
            nodalDOF_p_u = self.nodalDOF_p_u[n]
            self._q_dot_u_coo[n, nodalDOF_p, nodalDOF_p_u] = T_SO3_inv_quat_nodes[n]
        return self._q_dot_u_coo

    def step_callback(self, t, q, u):
        p = self._view_nodal_q(q)[:, 3:]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        return self._M_coo

    def h(self, t, q, u):
        return np.asarray(_h_nodes(self._view_nodal_u(u), self._B_Theta_C)).ravel()

    def h_u(self, t, q, u):
        h_u_nodes = np.asarray(
            _h_u_nodes(self._view_nodal_u(u)[:, 3:], self._B_Theta_C)
        )
        self._h_u_coo.data = h_u_nodes.ravel()
        # for n in range(self.nnode):
        #     nodalDOF_p_u = self.nodalDOF_p_u[n]
        #     self._h_u_coo[n, nodalDOF_p_u, nodalDOF_p_u] = h_u_nodes[n]
        return self._h_u_coo

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        return np.sum(p**2, axis=1) - 1

    def g_S_q(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        self._g_S_q_coo.data = (2 * p).ravel()
        # for n in range(self.nnode):
        #     nodalDOF_p = self.nodalDOF_p[n]
        #     self._g_S_q_coo[n, n, nodalDOF_p] = 2 * p[n]
        return self._g_S_q_coo

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        _, B_Gamma, B_Kappa = self._eval_els(q)
        la_c_el = _la_c_els(
            B_Gamma,
            B_Kappa,
            self.L_els,
            self.B_Gamma0,
            self.B_Kappa0,
            self.K_ga_els,
            self.K_ka_els,
        )
        return np.asarray(la_c_el).ravel()

    def c(self, t, q, u, la_c):
        _, B_Gamma, B_Kappa = self._eval_els(q)
        return np.asarray(
            _c_els(
                B_Gamma,
                B_Kappa,
                self._view_element_la_c(la_c),
                self.L_els,
                self.B_Gamma0,
                self.B_Kappa0,
                self.C_n_els,
                self.C_m_els,
            )
        ).ravel()

    def c_la_c(self):
        return self._c_la_c_coo

    def c_q(self, t, q, u, la_c):
        _, B_Gamma_qe, B_Kappa_qe = self._deval_els(q)
        c_q_els = np.asarray(_c_q_els(B_Gamma_qe, B_Kappa_qe, self.L_els))
        self._c_q_coo.data = c_q_els.ravel()
        # for el in range(self.nelement):
        #     elDOF = self.elDOF[el]
        #     elDOF_la_c = self.elDOF_la_c[el]
        #     self._c_q_coo[el, elDOF_la_c, elDOF] = c_q_els[el]
        return self._c_q_coo

    def W_c(self, t, q):
        A_IB, B_Gamma, B_Kappa = self._eval_els(q)
        W_c_els = np.asarray(_W_c_els(A_IB, B_Gamma, B_Kappa, self.L_els))
        self._W_c_coo.data = W_c_els.ravel()
        # for el in range(self.nelement):
        #     elDOF_u = self.elDOF_u[el]
        #     elDOF_la_c = self.elDOF_la_c[el]
        #     self._W_c_coo[el, elDOF_u, elDOF_la_c] = W_c_els[el]
        return self._W_c_coo

    def Wla_c_q(self, t, q, la_c):
        A_IB_qe, B_Gamma_qe, B_Kappa_qe = self._deval_els(q)
        Wla_c_q_els = np.asarray(
            _Wla_c_q_els(
                A_IB_qe,
                B_Gamma_qe,
                B_Kappa_qe,
                self._view_element_la_c(la_c),
                self.L_els,
            )
        )
        self._Wla_c_q_coo.data = Wla_c_q_els.ravel()
        # for el in range(self.nelement):
        #     elDOF = self.elDOF[el]
        #     elDOF_u = self.elDOF_u[el]
        #     self._Wla_c_q_coo[el, elDOF_u, elDOF] = Wla_c_q_els[el]
        return self._Wla_c_q_coo

    # @cachedmethod(lambda self: self._alpha_cache, key=lambda self, xi: xi)
    def _alpha(self, xi):
        num = self.element_number(xi)
        return (xi - self.xi_node[num]) / (self.xi_node[num + 1] - self.xi_node[num])

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        el = self.element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        el = self.element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.r_OP(t, qe, B_r_CP)

    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.r_OP_q(t, qe, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.v_P(t, qe, ue, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.v_P_q(t, qe, ue, B_r_CP)

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.J_P(t, qe, B_r_CP)

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.J_P_q(t, qe, B_r_CP)

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        mk = self.get_marker(xi)
        return mk.a_P(t, qe, ue, ue_dot, B_r_CP)

    def A_IB(self, t, qe, xi):
        mk = self.get_marker(xi)
        return mk.A_IB(t, qe)

    def A_IB_q(self, t, qe, xi):
        mk = self.get_marker(xi)
        return mk.A_IB_q(t, qe)

    def B_Omega(self, t, qe, ue, xi):
        mk = self.get_marker(xi)
        return mk.B_Omega(t, qe, ue)

    def B_Omega_q(self, t, qe, ue, xi):
        mk = self.get_marker(xi)
        return mk.B_Omega_q(t, qe, ue)

    def B_J_R(self, t, qe, xi):
        mk = self.get_marker(xi)
        return mk.B_J_R(t, qe)

    def B_J_R_q(self, t, qe, xi):
        mk = self.get_marker(xi)
        return mk.B_J_R_q(t, qe)

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        mk = self.get_marker(xi)
        return mk.B_Psi(t, qe, ue, ue_dot)

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        mk = self.get_marker(xi)
        return mk.B_Psi_q(t, qe, ue, ue_dot)

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        mk = self.get_marker(xi)
        return mk.B_Psi_u(t, qe, ue, ue_dot)

    def _eval_els(self, q):
        key = q.tobytes()
        eval_els = self._eval_cache[key]
        if eval_els is None:
            eval_els = _eval_els(self._view_element_q(q), self.L_els)
            self._eval_cache[key] = eval_els
        return eval_els

    def _deval_els(self, q):
        key = q.tobytes()
        deval_els = self._deval_cache[key]
        if deval_els is None:
            deval_els = _deval_els(self._view_element_q(q), self.L_els)
            self._deval_cache[key] = deval_els
        return deval_els

    def export(self, sol_i, **kwargs):
        if not hasattr(self, "_visual_twin"):
            self._visual_twin = VisualDiscreteRod(self)
        self._visual_twin.update_visual_state(sol_i)
        return self._visual_twin._ugrid


def _h_node(u, B_Theta_C):
    B_omega_IB = u[3:]
    tmp = B_Theta_C @ B_omega_IB
    cross = math_jax.cross3(tmp, B_omega_IB)
    return jnp.pad(cross, (3, 0))


_h_nodes = jit(vmap(_h_node))


def _h_u_node(B_omega_IB, B_Theta_C):
    return (
        math_jax.ax2skew(B_Theta_C @ B_omega_IB)
        - math_jax.ax2skew(B_omega_IB) @ B_Theta_C
    )


_h_u_nodes = jit(vmap(_h_u_node))


def _q_dot_node(q, u):
    T = math_jax.T_SO3_inv_quat(q[3:], normalize=False) @ u[3:]
    return jnp.concatenate([u[:3], T])


_q_dot_nodes = jit(vmap(_q_dot_node))


def _p_dot_p_node(q, u):
    return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:], normalize=False)


_p_dot_p_nodes = jit(vmap(_p_dot_p_node))


def _la_c_el(B_Gamma, B_Kappa, Le, B_Gamma0, B_Kappa0, K_ga, K_ka):
    # TODO: add damping
    B_n = K_ga @ (B_Gamma - B_Gamma0) * Le
    B_m = K_ka @ (B_Kappa - B_Kappa0) * Le

    # TODO:add damping
    return jnp.concatenate([B_n, B_m])


_la_c_els = jit(vmap(_la_c_el))


def _W_c_el(A_IB, B_Gamma, B_Kappa, Le):
    # A_IB, B_Gamma, B_Kappa = _eval(qe, Le)
    s1 = 0.5 * Le * math_jax.ax2skew(B_Gamma)
    s2 = 0.5 * Le * math_jax.ax2skew(B_Kappa)

    # TODO:add damping
    row1 = jnp.concatenate([A_IB, zeros3], axis=1)
    row2 = jnp.concatenate([s1, eye3 + s2], axis=1)
    row3 = jnp.concatenate([-A_IB, zeros3], axis=1)
    row4 = jnp.concatenate([s1, -eye3 + s2], axis=1)

    return jnp.concatenate([row1, row2, row3, row4], axis=0)


_W_c_els = jit(vmap(_W_c_el))


def _Wla_c_q_el(A_IB_qe, B_Gamma_qe, B_Kappa_qe, la_c, Le):
    B_n = la_c[:3]
    B_m = la_c[3:]

    W0 = B_n @ A_IB_qe

    common = (
        -0.5
        * Le
        * (
            jnp.cross(B_n[:, None], B_Gamma_qe, axis=0)
            + jnp.cross(B_m[:, None], B_Kappa_qe, axis=0)
        )
    )

    return jnp.concatenate([W0, common, -W0, common])


_Wla_c_q_els = jit(vmap(_Wla_c_q_el))


def _c_el(B_Gamma, B_Kappa, la_c, Le, B_Gamma0, B_Kappa0, C_n, C_m):
    B_n, B_m = la_c[:3], la_c[3:]

    c_n = (C_n @ B_n - (B_Gamma - B_Gamma0)) * Le
    c_m = (C_m @ B_m - (B_Kappa - B_Kappa0)) * Le

    # TODO:add damping
    return jnp.concatenate([c_n, c_m])


_c_els = jit(vmap(_c_el))


def _c_q_el(B_Gamma_qe, B_Kappa_qe, Le):
    c_n_qe = -B_Gamma_qe * Le
    c_m_qe = -B_Kappa_qe * Le
    return jnp.concatenate([c_n_qe, c_m_qe])


_c_q_els = jit(vmap(_c_q_el))


def _eval(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]
    r_OC1 = qe[7:10]
    P1 = qe[10:14]

    inv_Le = 1.0 / Le

    r_OC_s = (r_OC1 - r_OC0) * inv_Le

    P = 0.5 * (P0 + P1)
    P_s = (P1 - P0) * inv_Le

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    B_Gamma = A_IB.T @ r_OC_s

    B_Kappa = T @ P_s
    return A_IB, B_Gamma, B_Kappa


_eval_els = jit(vmap(_eval))


def _deval(qe, Le):
    r_OC0 = qe[:3]
    P0 = qe[3:7]
    r_OC1 = qe[7:10]
    P1 = qe[10:14]

    inv_Le = 1.0 / Le

    r_OC_s = (r_OC1 - r_OC0) * inv_Le

    P = (P0 + P1) / 2
    P_s = (P1 - P0) * inv_Le
    P_qe = 0.5 * jnp.hstack(
        (jnp.zeros((4, 3)), jnp.eye(4), jnp.zeros((4, 3)), jnp.eye(4))
    )

    A_IB = math_jax.Exp_SO3_quat(P, normalize=True)
    A_IB_T = A_IB.T
    A_IB_qe = math_jax.Exp_SO3_quat_P(P, normalize=True) @ P_qe
    #
    T = math_jax.T_SO3_quat(P, normalize=True)
    T_P = math_jax.T_SO3_quat_P(P, normalize=True)

    # B_Gamma = A_IB.T @ r_OC_s
    term2 = (
        jnp.concatenate([-A_IB_T, jnp.zeros((3, 4)), A_IB_T, jnp.zeros((3, 4))], axis=1)
        * inv_Le
    )

    B_Gamma_qe = jnp.einsum("k,kij", r_OC_s, A_IB_qe) + term2

    # B_Kappa = T @ P_s
    term2 = (
        jnp.concatenate([jnp.zeros((3, 3)), -T, jnp.zeros((3, 3)), T], axis=1) * inv_Le
    )
    B_Kappa_qe = P_s @ T_P @ P_qe + term2

    return A_IB_qe, B_Gamma_qe, B_Kappa_qe


_deval_els = jit(vmap(_deval))
