import numpy as np
from numpy.lib.stride_tricks import as_strided

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

    @staticmethod
    def __r_OP(alpha, q, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        r_OC0, r_OC1 = q[:3], q[7:10]
        r_OP = r_OC0 + alpha * (r_OC1 - r_OC0)
        return r_OP + A_IB @ B_r_CP


    @staticmethod
    def __r_OP_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        I3 = jnp.eye(3)
        Z4 = jnp.zeros((3, 4), I3.dtype)
        r_OP_q = jnp.concatenate(
            [
                (1.0 - alpha) * I3,          # cols 0:3
                Z4,                          # cols 3:7
                alpha * I3,                  # cols 7:10
                Z4,                          # cols 10:14
            ],
            axis=1,
        )
        return r_OP_q + B_r_CP @ A_IB_q


    @staticmethod
    def __v_P(alpha, q, u, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        v_C0 = u[:3]
        v_C1 = u[6:9]
        v_C = v_C0 + alpha * (v_C1 - v_C0)
        return v_C + A_IB @ math_jax.cross3(B_Omega, B_r_CP)


    @staticmethod
    def __v_P_q(alpha, q, u, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        cross = math_jax.cross3(B_Omega, B_r_CP)
        v_P_q = cross @ A_IB_q
        return v_P_q


    @staticmethod
    def __J_P(alpha, q, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        B_r_CP_tilde = math_jax.ax2skew(B_r_CP)
        r_CP_tilde = A_IB @ B_r_CP_tilde

        I3 = jnp.eye(3)

        return jnp.concatenate(
            [
                (1.0 - alpha) * I3,
                -(1.0 - alpha) * r_CP_tilde,
                alpha * I3,
                -alpha * r_CP_tilde,
            ],
            axis=1,
        )


    @staticmethod
    def __J_P_q(alpha, q, B_r_CP):
        A_IB_q = ElementKinematics.__A_IB_q(alpha, q)
        B_r_CP_tilde = math_jax.ax2skew(B_r_CP)
        r_CP_tilde_q = B_r_CP_tilde.T @ A_IB_q
        
        J_P_q = jnp.zeros((3, 12, 14))
        J_P_q = J_P_q.at[:, 3:6].set(-(1.0 - alpha) * r_CP_tilde_q)
        J_P_q = J_P_q.at[:, 9:12].set(-alpha * r_CP_tilde_q)
        return J_P_q


    @staticmethod
    def __a_P(alpha, q, u, u_dot, B_r_CP):
        A_IB = ElementKinematics.__A_IB(alpha, q)
        B_Omega = ElementKinematics.__B_Omega(alpha, u)
        B_Psi = ElementKinematics.__B_Psi(alpha, u_dot)
        a_C0 = u_dot[:3]
        a_C1 = u_dot[6:9]
        a_C = a_C0 + alpha * (a_C1 - a_C0)
        return a_C + A_IB @ (
            math_jax.cross3(B_Psi, B_r_CP) + math_jax.cross3(B_Omega, math_jax.cross3(B_Omega, B_r_CP))
        )


    @staticmethod
    def __A_IB(alpha, q):
        P0, P1 = q[3:7], q[10:]
        P = P0 + alpha * (P1 - P0)
        return math_jax.Exp_SO3_quat_norm(P)


    @staticmethod
    def __A_IB_q(alpha, q):
        P0, P1 = q[3:7], q[10:]
        P = P0 + alpha * (P1 - P0)

        A_P = math_jax.Exp_SO3_quat_P_norm(P)
        A_IB_q = jnp.zeros((3, 3, 14))
        A_IB_q = A_IB_q.at[..., 3:7].set((1.0 - alpha) * A_P)
        A_IB_q = A_IB_q.at[..., 10:14].set(alpha * A_P)

        return A_IB_q


    @staticmethod
    def __B_Omega(alpha, u):
        B_Omega_1 = u[3:6]
        B_Omega_2 = u[9:12]
        return B_Omega_1 + alpha * (B_Omega_2 - B_Omega_1)

    @staticmethod
    def __B_Psi(alpha, u_dot):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        time derivative of the angular velocities in the B-frame.
        """
        B_Psi_1 = u_dot[3:6]
        B_Psi_2 = u_dot[9:12]
        B_Psi = B_Psi_1 + alpha * (B_Psi_2 - B_Psi_1)
        return B_Psi
    
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
        self._r_OP = jit(lambda q, B_r_CP: ElementKinematics.__r_OP(self.alpha, q, B_r_CP))
        self._r_OP_q = jit(lambda q, B_r_CP: ElementKinematics.__r_OP_q(self.alpha, q, B_r_CP))
        self._v_P = jit(lambda q, u, B_r_CP: ElementKinematics.__v_P(self.alpha, q, u, B_r_CP))
        self._v_P_q = jit(lambda q, u, B_r_CP: ElementKinematics.__v_P_q(self.alpha, q, u, B_r_CP))
        self._J_P = jit(lambda q, B_r_CP: ElementKinematics.__J_P(self.alpha, q, B_r_CP))
        self._J_P_q = jit(lambda q, B_r_CP: ElementKinematics.__J_P_q(self.alpha, q, B_r_CP))
        self._a_P = jit(lambda q, u, u_dot, B_r_CP: ElementKinematics.__a_P(self.alpha, q, u, u_dot, B_r_CP))
        self._A_IB = jit(lambda q: ElementKinematics.__A_IB(self.alpha, q))
        self._A_IB_q = jit(lambda q: ElementKinematics.__A_IB_q(self.alpha, q))
        self._B_Omega = jit(lambda u: ElementKinematics.__B_Omega(self.alpha, u))
        self._B_Psi = jit(lambda u_dot: ElementKinematics.__B_Psi(self.alpha, u_dot))

    ##########################
    # r_OP / A_IB contribution
    ##########################

    def r_OP(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._r_OP(q, B_r_CP)

    def r_OP_q(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._r_OP_q(q, B_r_CP)

    def v_P(self, t, q, u, B_r_CP=np.zeros(3, dtype=float)):
        return np.asarray(self._v_P(q, u, B_r_CP))

    def v_P_q(self, t, q, u, B_r_CP=np.zeros(3, dtype=float)):
        return self._v_P_q(q, u, B_r_CP)

    def J_P(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        return self._J_P(q, B_r_CP)

    def J_P_q(self, t, q, B_r_CP=np.zeros(3, dtype=float)):
        return np.asarray(self._J_P_q(q, B_r_CP))

    def a_P(self, t, q, u, u_dot, B_r_CP=np.zeros(3, dtype=float)):
        return self._a_P(q, u, u_dot, B_r_CP)

    def A_IB(self, t, q):
        key = q.tobytes()
        A_IB = self._A_IB_cache[key]
        if A_IB is None:
            A_IB = np.asarray(self._A_IB(q))
            self._A_IB_cache[key] = A_IB
        return A_IB

    def A_IB_q(self, t, q):
        key = q.tobytes()
        A_IB_q = self._A_IB_q_cache[key]
        if A_IB_q is None:
            A_IB_q = np.asarray(self._A_IB_q(q))
            self._A_IB_q_cache[key] = A_IB_q
        return A_IB_q

    def B_Omega(self, t, q, u):
        """Since we use Petrov-Galerkin method we only interpolate the nodal
        angular velocities in the B-frame.
        """
        return self._B_Omega(u)

    def B_Omega_q(self, t, q, u):
        return self._B_Omega_q

    def B_J_R(self, t, q):
        return self._B_J_R

    def B_J_R_q(self, t, q):
        return self._B_J_R_q

    def B_Psi(self, t, q, u, u_dot):
        return self._B_Psi(u_dot)

    def B_Psi_q(self, t, q, u, u_dot):
        return self._B_Psi_q

    def B_Psi_u(self, t, q, u, u_dot):
        return self._B_Psi_u



class DiscreteRod:
    @jit
    def _gen_element_q(q):
        nelement = q.shape[0] // 7 - 1
        q_nodes = q[: (nelement + 1) * 7].reshape(nelement + 1, 7)
        return jnp.concatenate([q_nodes[:-1], q_nodes[1:]], axis=1)  # (nelement, 14)

    @staticmethod
    def _q_dot_node(q, u):
        T = math_jax.T_SO3_inv_quat(q[3:]) @ u[3:]
        return jnp.concatenate([u[:3], T])

    @staticmethod
    def _p_dot_p_node(q, u):
        return u[3:] @ math_jax.T_SO3_inv_quat_P(q[3:])

    @staticmethod
    def _h_node(u, B_Theta_C):
        B_omega_IB = u[3:]
        tmp = B_Theta_C @ B_omega_IB
        cross = math_jax.cross3(tmp, B_omega_IB)
        return jnp.pad(cross, (3, 0))

    @staticmethod
    def _h_u_node(B_omega_IB, B_Theta_C):
        return (
            math_jax.ax2skew(B_Theta_C @ B_omega_IB)
            - math_jax.ax2skew(B_omega_IB) @ B_Theta_C
        )

    @staticmethod
    def _la_c_el(qe, Le, B_gamma0, B_kappa0, K_ga, K_ka):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        # TODO: add damping
        B_n = K_ga @ (B_gamma - B_gamma0) * Le
        B_m = K_ka @ (B_kappa - B_kappa0) * Le

        # TODO:add damping
        return jnp.concatenate([B_n, B_m])

    @staticmethod
    def _c_el(qe, la_c, Le, B_gamma0, B_kappa0, C_n, C_m):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        B_n, B_m = la_c[:3], la_c[3:]

        c_n = (C_n @ B_n - (B_gamma - B_gamma0)) * Le
        c_m = (C_m @ B_m - (B_kappa - B_kappa0)) * Le

        # TODO:add damping
        return jnp.concatenate([c_n, c_m])

    @staticmethod
    def _c_q_el(qe, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._deval_el(qe, Le)
        c_n_qe = -B_gamma_qe * Le
        c_m_qe = -B_kappa_qe * Le
        return jnp.concatenate([c_n_qe, c_m_qe])

    @staticmethod
    def _W_c_el(qe, Le):
        A_IB, B_gamma, B_kappa = DiscreteRod._eval_el(qe, Le)
        s1 = 0.5 * Le * math_jax.ax2skew(B_gamma)
        s2 = 0.5 * Le * math_jax.ax2skew(B_kappa)

        # TODO:add damping
        row1 = jnp.concatenate([A_IB, zeros3], axis=1)
        row2 = jnp.concatenate([s1, eye3 + s2], axis=1)
        row3 = jnp.concatenate([-A_IB, zeros3], axis=1)
        row4 = jnp.concatenate([s1, -eye3 + s2], axis=1)

        return jnp.concatenate([row1, row2, row3, row4], axis=0)

    @staticmethod
    def _Wla_c_q_el(qe, la_c, Le):
        A_IB_qe, B_gamma_qe, B_kappa_qe = DiscreteRod._deval_el(qe, Le)
        B_n = la_c[:3]
        B_m = la_c[3:]

        W0 = B_n @ A_IB_qe

        common = (
            -0.5
            * Le
            * (
                jnp.cross(B_n[:, None], B_gamma_qe, axis=0)
                + jnp.cross(B_m[:, None], B_kappa_qe, axis=0)
            )
        )

        return jnp.concatenate([W0, common, -W0, common])

    @staticmethod
    def _eval_el(qe, Le):
        r_OC0 = qe[:3]
        P0 = qe[3:7]
        r_OC1 = qe[7:10]
        P1 = qe[10:14]

        inv_Le = 1.0 / Le

        r_OC_s = (r_OC1 - r_OC0) * inv_Le

        P = 0.5 * (P0 + P1)
        P_s = (P1 - P0) * inv_Le

        A_IB = math_jax.Exp_SO3_quat_norm(P)
        #
        T = math_jax.T_SO3_quat_norm(P)
        B_gamma = A_IB.T @ r_OC_s

        B_kappa = T @ P_s
        return A_IB, B_gamma, B_kappa

    @staticmethod
    def _deval_el(qe, Le):
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

        A_IB = math_jax.Exp_SO3_quat_norm(P)
        A_IB_T = A_IB.T

        A_P = math_jax.Exp_SO3_quat_P_norm(P)
        A_IB_qe = jnp.zeros((3, 3, 14))
        A_IB_qe = A_IB_qe.at[..., 3:7].set(0.5 * A_P)
        A_IB_qe = A_IB_qe.at[..., 10:14].set(0.5 * A_P)

        #
        T = math_jax.T_SO3_quat_norm(P)
        T_P = math_jax.T_SO3_quat_P_norm(P)

        # B_gamma = A_IB.T @ r_OC_s
        term2 = (
            jnp.concatenate(
                [-A_IB_T, jnp.zeros((3, 4)), A_IB_T, jnp.zeros((3, 4))], axis=1
            )
            * inv_Le
        )

        B_gamma_qe = jnp.einsum("k,kij", r_OC_s, A_IB_qe) + term2

        # B_kappa = T @ P_s
        term2 = (
            jnp.concatenate([jnp.zeros((3, 3)), -T, jnp.zeros((3, 3)), T], axis=1)
            * inv_Le
        )
        B_kappa_qe = P_s @ T_P @ P_qe + term2

        return A_IB_qe, B_gamma_qe, B_kappa_qe

    _q_dot_nodes = jit(vmap(_q_dot_node.__func__))
    _p_dot_p_nodes = jit(vmap(_p_dot_p_node.__func__))

    _h_nodes = jit(vmap(_h_node.__func__))
    _h_u_nodes = jit(vmap(_h_u_node.__func__))

    _la_c_els = jit(vmap(_la_c_el.__func__))
    _c_els = jit(vmap(_c_el.__func__))
    _c_q_els = jit(vmap(_c_q_el.__func__))
    _W_c_els = jit(vmap(_W_c_el.__func__))
    _Wla_c_q_els = jit(vmap(_Wla_c_q_el.__func__))
    _eval_els = jit(vmap(_eval_el.__func__))
    _deval_els = jit(vmap(_deval_el.__func__))

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
        self.cross_section = cross_section
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

        # element lengths
        self.L_els = np.array(
            [
                norm(Q[self.nodalDOF_r[el + 1]] - Q[self.nodalDOF_r[el]])
                for el in range(self.nelement)
            ]
        )
        # rod length
        self.L = np.sum(self.L_els)

        # reference strain
        _, self.B_gamma0, self.B_kappa0 = DiscreteRod._eval_els(DiscreteRod._gen_element_q(Q), self.L_els)
        self.B_Ga_Ka0 = np.concatenate((self.B_gamma0, self.B_kappa0), axis=1)

        # jit functions for each instance
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
            lambda q: DiscreteRod._la_c_els(
                DiscreteRod._gen_element_q(q),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.K_ga_els,
                self.K_ka_els,
            )
        )
        self._c = jit(
            lambda q, la_c: DiscreteRod._c_els(
                DiscreteRod._gen_element_q(q),
                la_c.reshape((nelement, _nla_c_el)),
                self.L_els,
                self.B_gamma0,
                self.B_kappa0,
                self.C_n_els,
                self.C_m_els,
            )
        )
        self._c_q = jit(
            lambda q: DiscreteRod._c_q_els(DiscreteRod._gen_element_q(q), self.L_els)
        )
        self._W_c = jit(
            lambda q: DiscreteRod._W_c_els(DiscreteRod._gen_element_q(q), self.L_els)
        )
        self._Wla_c_q = jit(
            lambda q, la_c: DiscreteRod._Wla_c_q_els(
                DiscreteRod._gen_element_q(q),
                la_c.reshape((nelement, _nla_c_el)),
                self.L_els,
            )
        )

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

        self._kinematics_els = {}

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

    def __el_kinematics(self, xi):
        try:
            el = self._kinematics_els[xi]
        except KeyError:
            alpha = self._alpha(xi)
            el = ElementKinematics(xi, alpha)
            self._kinematics_els[xi] = el
        if hasattr(self, "qDOF") and not hasattr(el, "qDOF"):
            num = self.element_number(xi)
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
        return self._q_dot(q, u).ravel()

    def q_dot_q(self, t, q, u):
        self._q_dot_q_coo.data = self._q_dot_q(q, u).ravel()
        return self._q_dot_q_coo

    def q_dot_u(self, t, q):
        T_SO3_inv_quat_nodes = np.asarray(
            math_jax.T_SO3_inv_quat_batch(self._view_nodal_q(q)[:, 3:])
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
        return self._h(u).ravel()

    def h_u(self, t, q, u):
        self._h_u_coo.data = self._h_u(u).ravel()
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
        return self._g_S_q_coo

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c_el = self._la_c(q)
        return la_c_el.ravel()

    def c(self, t, q, u, la_c):
        return self._c(q, la_c).ravel()

    def c_la_c(self):
        return self._c_la_c_coo

    def c_q(self, t, q, u, la_c):
        self._c_q_coo.data = self._c_q(q).ravel()
        return self._c_q_coo

    def W_c(self, t, q):
        self._W_c_coo.data = self._W_c(q).ravel()
        return self._W_c_coo

    def Wla_c_q(self, t, q, la_c):
        self._Wla_c_q_coo.data = self._Wla_c_q(q, la_c).ravel()
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
        return self.__el_kinematics(xi).r_OP(t, qe, B_r_CP)

    def r_OP_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).r_OP_q(t, qe, B_r_CP)

    def v_P(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).v_P(t, qe, ue, B_r_CP)

    def v_P_q(self, t, qe, ue, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).v_P_q(t, qe, ue, B_r_CP)

    def J_P(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).J_P(t, qe, B_r_CP)

    def J_P_q(self, t, qe, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).J_P_q(t, qe, B_r_CP)

    def a_P(self, t, qe, ue, ue_dot, xi, B_r_CP=np.zeros(3, dtype=float)):
        return self.__el_kinematics(xi).a_P(t, qe, ue, ue_dot, B_r_CP)

    def A_IB(self, t, qe, xi):
        return self.__el_kinematics(xi).A_IB(t, qe)

    def A_IB_q(self, t, qe, xi):
        return self.__el_kinematics(xi).A_IB_q(t, qe)

    def B_Omega(self, t, qe, ue, xi):
        return self.__el_kinematics(xi).B_Omega(t, qe, ue)

    def B_Omega_q(self, t, qe, ue, xi):
        return self.__el_kinematics(xi).B_Omega_q(t, qe, ue)

    def B_J_R(self, t, qe, xi):
        return self.__el_kinematics(xi).B_J_R(t, qe)

    def B_J_R_q(self, t, qe, xi):
        return self.__el_kinematics(xi).B_J_R_q(t, qe)

    def B_Psi(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi(t, qe, ue, ue_dot)

    def B_Psi_q(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi_q(t, qe, ue, ue_dot)

    def B_Psi_u(self, t, qe, ue, ue_dot, xi):
        return self.__el_kinematics(xi).B_Psi_u(t, qe, ue, ue_dot)


    def export(self, sol_i, **kwargs):
        if not hasattr(self, "_visual_twin"):
            self._visual_twin = VisualDiscreteRod(self)
        self._visual_twin.update_visual_state(sol_i)
        return self._visual_twin._ugrid
