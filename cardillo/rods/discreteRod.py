import numpy as np
from numpy.lib.stride_tricks import as_strided

from jax import vmap, jit, jacfwd
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


class ElementKinematics:

    @staticmethod
    @jit
    def __r_OP(alpha, q, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        r_OC0, r_OC1 = q[:3], q[7:10]
        r_OP = r_OC0 + alpha * (r_OC1 - r_OC0)
        return r_OP + A_IB @ B_r_CP

    @staticmethod
    @jit
    def __r_OP_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        r_OP_q = jnp.concatenate(
            [
                (1.0 - alpha) * E3,  # cols 0:3
                Z34,  # cols 3:7
                alpha * E3,  # cols 7:10
                Z34,  # cols 10:14
            ],
            axis=1,
        )
        return r_OP_q + B_r_CP @ A_IB_q

    @staticmethod
    @jit
    def __v_P(alpha, q, u, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        v_C0 = u[:3]
        v_C1 = u[6:9]
        v_C = v_C0 + alpha * (v_C1 - v_C0)
        return v_C + A_IB @ math_jax.cross3(B_Omega, B_r_CP)

    @staticmethod
    @jit
    def __v_P_q(alpha, q, u, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        cross = math_jax.cross3(B_Omega, B_r_CP)
        v_P_q = cross @ A_IB_q
        return v_P_q

    @staticmethod
    @jit
    def __J_P(alpha, q, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
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
    def __J_P_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        B_r_CP_tilde = math_jax.ax2skew(B_r_CP)
        r_CP_tilde_q = B_r_CP_tilde.T @ A_IB_q

        J_P_q = jnp.zeros((3, 12, 14))
        J_P_q = J_P_q.at[:, 3:6].set(-(1.0 - alpha) * r_CP_tilde_q)
        J_P_q = J_P_q.at[:, 9:12].set(-alpha * r_CP_tilde_q)
        return J_P_q

    @staticmethod
    @jit
    def __a_P(alpha, q, u, u_dot, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        B_Psi = ElementKinematics.__B_Psi(alpha, u_dot)
        a_C0 = u_dot[:3]
        a_C1 = u_dot[6:9]
        a_C = a_C0 + alpha * (a_C1 - a_C0)
        return a_C + A_IB @ (
            math_jax.cross3(B_Psi, B_r_CP)
            + math_jax.cross3(B_Omega, math_jax.cross3(B_Omega, B_r_CP))
        )

    @staticmethod
    @jit
    def __A_IB(alpha, q):
        P0, P1 = q[3:7], q[10:]
        P = P0 + alpha * (P1 - P0)
        return math_jax.Exp_SO3_quat_norm(P)

    @staticmethod
    @jit
    def __A_IB_q(alpha, q):
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
    def __B_Omega(alpha, u):
        B_Omega_1 = u[3:6]
        B_Omega_2 = u[9:12]
        return B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)

    @staticmethod
    @jit
    def __B_Psi(alpha, u_dot):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        B_Psi_1 = u_dot[3:6]
        B_Psi_2 = u_dot[9:12]
        B_Psi = B_Psi_1 + alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi

    def __init__(self, alpha):

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
        self._r_OP = jit(lambda q, B_r_CP: ElementKinematics.__r_OP(alpha, q, B_r_CP))
        self._r_OP_q = jit(
            lambda q, B_r_CP: ElementKinematics.__r_OP_q(alpha, q, B_r_CP)
        )
        self._v_P = jit(
            lambda q, u, B_r_CP: ElementKinematics.__v_P(alpha, q, u, B_r_CP)
        )
        self._v_P_q = jit(
            lambda q, u, B_r_CP: ElementKinematics.__v_P_q(alpha, q, u, B_r_CP)
        )
        self._J_P = jit(lambda q, B_r_CP: ElementKinematics.__J_P(alpha, q, B_r_CP))
        self._J_P_q = jit(lambda q, B_r_CP: ElementKinematics.__J_P_q(alpha, q, B_r_CP))
        self._a_P = jit(
            lambda q, u, u_dot, B_r_CP: ElementKinematics.__a_P(
                alpha, q, u, u_dot, B_r_CP
            )
        )
        self._A_IB = jit(lambda q: ElementKinematics.__A_IB(alpha, q))
        self._A_IB_q = jit(lambda q: ElementKinematics.__A_IB_q(alpha, q))
        self._B_Omega = jit(lambda u: ElementKinematics.__B_Omega(alpha, u))
        self._B_Psi = jit(lambda u_dot: ElementKinematics.__B_Psi(alpha, u_dot))

    ##########################
    # r_OP / A_IB contribution
    ##########################

    def r_OP(self, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._r_OP(q, B_r_CP).__array__()

    def r_OP_q(self, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._r_OP_q(q, B_r_CP).__array__()

    def v_P(self, q, u, B_r_CP=np.zeros(3, dtype=float)):
        return self._v_P(q, u, B_r_CP).__array__()

    def v_P_q(self, q, u, B_r_CP=np.zeros(3, dtype=float)):
        return self._v_P_q(q, u, B_r_CP).__array__()

    def J_P(self, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._J_P(q, B_r_CP).__array__()

    def J_P_q(self, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._J_P_q(q, B_r_CP).__array__()

    def a_P(self, q, u, u_dot, B_r_CP=np.zeros(3, dtype=float)):
        return self._a_P(q, u, u_dot, B_r_CP).__array__()

    def A_IB(self, q):
        key = q.tobytes()
        A_IB = self._A_IB_cache[key]
        if A_IB is None:
            A_IB = self._A_IB(q).__array__()
            self._A_IB_cache[key] = A_IB
        return A_IB

    def A_IB_q(self, q):
        key = q.tobytes()
        A_IB_q = self._A_IB_q_cache[key]
        if A_IB_q is None:
            A_IB_q = self._A_IB_q(q).__array__()
            self._A_IB_q_cache[key] = A_IB_q
        return A_IB_q

    def B_Omega(self, u):
        return self._B_Omega(u).__array__()

    def B_Omega_q(self):
        return self._B_Omega_q

    def B_J_R(self):
        return self._B_J_R

    def B_J_R_q(self):
        return self._B_J_R_q

    def B_Psi(self, u_dot):
        return self._B_Psi(u_dot).__array__()

    def B_Psi_q(self):
        return self._B_Psi_q

    def B_Psi_u(self):
        return self._B_Psi_u


class DiscreteRod:
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
    def _q_dot_node(q, u):
        T = math_jax.T_SO3_inv_quat(q[3:]) @ u[3:]
        return jnp.concatenate([u[:3], T])

    @staticmethod
    @jit
    def _p_dot_p_node(q, u):
        return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:])

    @staticmethod
    @jit
    def _h_node(u, B_Theta_C):
        B_omega_IB = u[3:]
        tmp = B_Theta_C @ B_omega_IB
        cross = math_jax.cross3(tmp, B_omega_IB)
        return jnp.pad(cross, (3, 0))

    @staticmethod
    @jit
    def _h_u_node(B_omega_IB, B_Theta_C):
        return (
            math_jax.ax2skew(B_Theta_C @ B_omega_IB)
            - math_jax.ax2skew(B_omega_IB) @ B_Theta_C
        )

    @staticmethod
    @jit
    def _la_c_el(qe, Le, B_gamma0, B_kappa0, K_ga, K_ka):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        B_n = K_ga @ (B_gamma - B_gamma0)
        B_m = K_ka @ (B_kappa - B_kappa0)
        return jnp.concatenate([B_n, B_m])

    @staticmethod
    @jit
    def _la_c_damp_el(qe, ue, Le, B_gamma0, B_kappa0, K_ga, K_ka, K_ga_damp, K_ka_damp):
        la_c_12 = DiscreteRod._la_c_el(qe, Le, B_gamma0, B_kappa0, K_ga, K_ka)
        B_gamma_dot, B_kappa_dot = DiscreteRod._eval_dot_el(qe, ue, Le)
        B_n_damp = K_ga_damp @ B_gamma_dot
        B_m_damp = K_ka_damp @ B_kappa_dot
        return jnp.concatenate([la_c_12, B_n_damp, B_m_damp])

    @staticmethod
    @jit
    def _c_el(qe, la_c, Le, B_gamma0, B_kappa0, C_n, C_m):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        B_n, B_m = la_c[:3], la_c[3:]

        c1 = (C_n @ B_n - (B_gamma - B_gamma0)) * Le
        c2 = (C_m @ B_m - (B_kappa - B_kappa0)) * Le

        return jnp.concatenate([c1, c2])

    @staticmethod
    @jit
    def _c_damp_el(qe, ue, la_c, Le, B_gamma0, B_kappa0, C_n, C_m, C_n_damp, C_m_damp):
        c12 = DiscreteRod._c_el(qe, la_c[:6], Le, B_gamma0, B_kappa0, C_n, C_m)
        B_gamma_dot, B_kappa_dot = DiscreteRod._eval_dot_el(qe, ue, Le)
        B_n_damp, B_m_damp = la_c[6:9], la_c[9:]

        c3 = (C_n_damp @ B_n_damp - B_gamma_dot) * Le
        c4 = (C_m_damp @ B_m_damp - B_kappa_dot) * Le

        return jnp.concatenate([c12, c3, c4])

    @staticmethod
    @jit
    def _c_q_el(qe, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)
        return jnp.concatenate([B_gamma_qe, B_kappa_qe], axis=0) * (-Le)

    @staticmethod
    @jit
    def _c_damp_q_el(qe, ue, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._eval_q_el(qe, Le)
        B_gamma_dot_qe, B_kappa_dot_qe = DiscreteRod._eval_dot_q_el(qe, ue, Le)
        return jnp.concatenate(
            [B_gamma_qe, B_kappa_qe, B_gamma_dot_qe, B_kappa_dot_qe], axis=0
        ) * (-Le)

    @staticmethod
    @jit
    def _c_damp_u_el(qe, ue, Le):
        B_gamma_dot_ue, B_kappa_dot_ue = DiscreteRod._eval_dot_u_el(qe, ue, Le)
        return jnp.concatenate(
            [jnp.zeros((6, 12)), B_gamma_dot_ue, B_kappa_dot_ue], axis=0
        ) * (-Le)

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

    @staticmethod
    @jit
    def _W_c_damp_el(qe, Le):
        W_c_el = DiscreteRod._W_c_el(qe, Le)
        return jnp.concatenate([W_c_el, W_c_el], axis=1)

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

    @staticmethod
    @jit
    def _Wla_c_q_damp_el(qe, la_c, Le):
        _la_c = la_c[:6] + la_c[6:]
        return DiscreteRod._Wla_c_q_el(qe, _la_c, Le)

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
    _eval_dot_u_el = jit(jacfwd(_eval_dot_el.__func__, argnums=1))

    _q_dot_nodes = jit(vmap(_q_dot_node.__func__))
    _p_dot_p_nodes = jit(vmap(_p_dot_p_node.__func__))

    _h_nodes = jit(vmap(_h_node.__func__))
    _h_u_nodes = jit(vmap(_h_u_node.__func__))

    _la_c_els = jit(vmap(_la_c_el.__func__))
    _la_c_damp_els = jit(vmap(_la_c_damp_el.__func__))
    _c_els = jit(vmap(_c_el.__func__))
    _c_damp_els = jit(vmap(_c_damp_el.__func__))
    _c_q_els = jit(vmap(_c_q_el.__func__))
    _c_damp_q_els = jit(vmap(_c_damp_q_el.__func__))
    _c_damp_u_els = jit(vmap(_c_damp_u_el.__func__))
    _W_c_els = jit(vmap(_W_c_el.__func__))
    _W_c_damp_els = jit(vmap(_W_c_damp_el.__func__))
    _Wla_c_q_els = jit(vmap(_Wla_c_q_el.__func__))
    _Wla_c_q_damp_els = jit(vmap(_Wla_c_q_damp_el.__func__))
    _eval_els = jit(vmap(_eval_el.__func__))
    _eval_q_els = jit(vmap(_eval_q_el.__func__))

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

        self._kinematics_els = {}

        self.__init_coo__(cross_section_inertias)

        # add derivative c_u
        if self._damping:

            def c_u(t, q, u, la_c):
                self._c_u_coo.data = self._c_u(q, u).__array__().ravel()
                return self._c_u_coo

            self.c_u = c_u

    def __jit_func__(self):
        self._eval = jit(
            lambda q: DiscreteRod._eval_els(DiscreteRod._gen_element_q(q), self.L_els)
        )

        self._q_dot = jit(
            lambda q, u: DiscreteRod._q_dot_nodes(
                q.reshape((-1, 7)), u.reshape((-1, 6))
            )
        )

        self._q_dot_q = jit(
            lambda q, u: DiscreteRod._p_dot_p_nodes(
                q.reshape((-1, 7)), u.reshape((-1, 6))
            )
        )

        self._h = jit(
            lambda u: DiscreteRod._h_nodes(u.reshape((-1, 6)), self._B_Theta_C)
        )

        self._h_u = jit(
            lambda u: DiscreteRod._h_u_nodes(u.reshape((-1, 6))[:, 3:], self._B_Theta_C)
        )

        self._la_c = jit(
            lambda q, u: (
                DiscreteRod._la_c_els(
                    DiscreteRod._gen_element_q(q),
                    self.L_els,
                    self.B_gamma0,
                    self.B_kappa0,
                    self.K_ga_els,
                    self.K_ka_els,
                )
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
                )
            )
        )

        self._c = jit(
            lambda q, u, la_c: (
                DiscreteRod._c_els(
                    DiscreteRod._gen_element_q(q),
                    la_c.reshape((self.nelement, -1)),
                    self.L_els,
                    self.B_gamma0,
                    self.B_kappa0,
                    self.C_n_els,
                    self.C_m_els,
                )
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
                )
            )
        )

        self._c_q = jit(
            lambda q, u: (
                DiscreteRod._c_q_els(DiscreteRod._gen_element_q(q), self.L_els)
                if not self._damping
                else DiscreteRod._c_damp_q_els(
                    DiscreteRod._gen_element_q(q),
                    DiscreteRod._gen_element_u(u),
                    self.L_els,
                )
            )
        )

        self._c_u = jit(
            lambda q, u: DiscreteRod._c_damp_u_els(
                DiscreteRod._gen_element_q(q), DiscreteRod._gen_element_u(u), self.L_els
            )
        )

        self._W_c = jit(
            lambda q: (
                DiscreteRod._W_c_els(DiscreteRod._gen_element_q(q), self.L_els)
                if not self._damping
                else DiscreteRod._W_c_damp_els(
                    DiscreteRod._gen_element_q(q), self.L_els
                )
            )
        )

        self._Wla_c_q = jit(
            lambda q, la_c: (
                DiscreteRod._Wla_c_q_els(
                    DiscreteRod._gen_element_q(q),
                    la_c.reshape((self.nelement, -1)),
                    self.L_els,
                )
                if not self._damping
                else DiscreteRod._Wla_c_q_damp_els(
                    DiscreteRod._gen_element_q(q),
                    la_c.reshape((self.nelement, -1)),
                    self.L_els,
                )
            )
        )

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

    def _element_number(self, xi):
        num = int(xi * self.nelement)
        return num if num < self.nelement else num - 1

    def _view_element_q(self, q):
        stride = q.strides[0]
        return as_strided(q, shape=(self.nelement, 14), strides=(stride * 7, stride))

    def _view_element_la_c(self, la_c):
        return la_c.reshape((self.nelement, -1))

    def _view_nodal_q(self, q):
        return q.reshape((self.nnode, 7))

    def _view_nodal_u(self, u):
        return u.reshape((self.nnode, 6))

    def nodes(self, q):
        """Returns nodal position coordinates"""
        q_body = q[self.qDOF]
        return np.array([q_body[nodalDOF] for nodalDOF in self.nodalDOF_r]).T

    def __el_kinematics(self, xi):
        try:
            el = self._kinematics_els[xi]
        except KeyError:
            el = ElementKinematics(self._alpha(xi))
            self._kinematics_els[xi] = el
        if hasattr(self, "qDOF") and not hasattr(el, "qDOF"):
            num = self._element_number(xi)
            el.t0 = self.t0
            el.q0 = self.q0[self.elDOF[num]]
            el.qDOF = self.qDOF[self.elDOF[num]]
            el.uDOF = self.uDOF[self.elDOF_u[num]]
        return el

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
        for xi, mk in self._kinematics_els.items():
            num = self._element_number(xi)
            mk.t0 = self.t0
            mk.q0 = self.q0[self.elDOF[num]]
            mk.u0 = self.u0[self.elDOF_u[num]]
            mk.qDOF = self.qDOF[self.elDOF[num]]
            mk.uDOF = self.uDOF[self.elDOF_u[num]]

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        return self._q_dot(q, u).__array__().ravel()

    def q_dot_q(self, t, q, u):
        self._q_dot_q_coo.data = self._q_dot_q(q, u).__array__().ravel()
        return self._q_dot_q_coo

    def q_dot_u(self, t, q):
        T_SO3_inv_quat_nodes = math_jax.T_SO3_inv_quat_batch(
            self._view_nodal_q(q)[:, 3:]
        ).__array__()
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
        return self._h(u).__array__().ravel()

    def h_u(self, t, q, u):
        self._h_u_coo.data = self._h_u(u).__array__().ravel()
        return self._h_u_coo

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        return np.sum(p**2, axis=1) - 1

    def g_S_q(self, t, q):
        p = self._view_nodal_q(q)[:, 3:]
        self._g_S_q_coo.data = (2 * p).__array__().ravel()
        return self._g_S_q_coo

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c_el = self._la_c(q, u)
        return la_c_el.__array__().ravel()

    def c(self, t, q, u, la_c):
        return self._c(q, u, la_c).__array__().ravel()

    def c_la_c(self):
        return self._c_la_c_coo

    def c_q(self, t, q, u, la_c):
        self._c_q_coo.data = self._c_q(q, u).__array__().ravel()
        return self._c_q_coo

    def W_c(self, t, q):
        self._W_c_coo.data = self._W_c(q).__array__().ravel()
        return self._W_c_coo

    def Wla_c_q(self, t, q, la_c):
        self._Wla_c_q_coo.data = self._Wla_c_q(q, la_c).__array__().ravel()
        return self._Wla_c_q_coo

    # @cachedmethod(lambda self: self._alpha_cache, key=lambda self, xi: xi)
    def _alpha(self, xi):
        num = self._element_number(xi)
        return (xi - self.xi_node[num]) / (self.xi_node[num + 1] - self.xi_node[num])

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    def elDOF_P(self, xi):
        el = self._element_number(xi)
        return self.elDOF[el]

    def elDOF_P_u(self, xi):
        el = self._element_number(xi)
        return self.elDOF_u[el]

    def local_qDOF_P(self, xi):
        return self.elDOF_P(xi)

    def local_uDOF_P(self, xi):
        return self.elDOF_P_u(xi)

    ##########################
    # r_OP / A_IB contribution
    ##########################
    def r_OP(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).r_OP(qe, B_r_CP)

    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).r_OP_q(qe, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).v_P(qe, ue, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).v_P_q(qe, ue, B_r_CP)

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).J_P(qe, B_r_CP)

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).J_P_q(qe, B_r_CP)

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).a_P(qe, ue, ue_dot, B_r_CP)

    def A_IB(self, t, qe, xi):
        return self.__el_kinematics(xi).A_IB(qe)

    def A_IB_q(self, t, qe, xi):
        return self.__el_kinematics(xi).A_IB_q(qe)

    def B_Omega(self, t, qe, ue, xi):
        return self.__el_kinematics(xi).B_Omega(ue)

    def B_Omega_q(self, t, qe, ue, xi):
        return self.__el_kinematics(xi).B_Omega_q()

    def B_J_R(self, t, qe, xi):
        return self.__el_kinematics(xi).B_J_R()

    def B_J_R_q(self, t, qe, xi):
        return self.__el_kinematics(xi).B_J_R_q()

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi(ue_dot)

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi_q()

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi_u()

    def export(self, sol_i, **kwargs):
        if not hasattr(self, "_visual_twin"):
            self._visual_twin = VisualDiscreteRod(self)
        self._visual_twin.update_visual_state(sol_i)
        return self._visual_twin._ugrid
