import numpy as np
from scipy.optimize import nnls

from cardillo.actuators._base import BaseActuator


class DynamicControllerPD(BaseActuator):
    """Feedback linearization + PD, tendon forces projected onto la_tau >= 0 by a QP."""

    def __init__(self, system, rod, tendons, r_OP_ref_fn, v_P_ref_fn=None, a_P_ref_fn=None, Kp=0.0, Kd=0.0, inv_damping=1e-3, positive=True, qp_tol=1e-8, qp_reg=1e-8, name="dynamic_controller"):
        if a_P_ref_fn is None:
            a_P_ref_fn = lambda t: np.zeros(3)
        if v_P_ref_fn is None:
            v_P_ref_fn = lambda t: np.zeros(3)
        tau = lambda t: np.concatenate([r_OP_ref_fn(t), v_P_ref_fn(t), a_P_ref_fn(t)])
        super().__init__(rod, tau, nla_tau=len(tendons), ntau=9)
        self.system = system
        self.rod = rod
        self.tendons = tendons
        self.Kp = Kp
        self.Kd = Kd
        self.inv_damping = inv_damping
        self.name = name

        self.M_tilde_inv = None
        self._c_la_c_inv = None
        self.nnls_tol = 1e-8

        # ---- positivity QP ----
        self.positive = positive
        self.qp_tol = qp_tol  # tendon counts as free above this force
        self.qp_reg = qp_reg  # eps anchor, relative to ||W_tau||
        self._core_cache = None

    def assembler_callback(self):
        super().assembler_callback()
        rod = self.rod
        self._q_off = rod.qDOF[0]
        self._u_off = rod.uDOF[0]
        C_1 = np.zeros((3, rod.nq))
        C_1[:, rod.nodalDOF_r[-1]] = np.eye(3)
        self.C_1 = C_1

    def build_M_tilde_inv(self, t, q_sys):
        if self.M_tilde_inv is None:
            rod = self.rod
            q_rod = q_sys[rod.qDOF]
            B = rod.q_dot_u(t, q_rod).toarray()
            M = rod.M(t, q_rod).toarray()
            self.M_tilde_inv = (self.C_1 @ B) @ np.linalg.inv(M)

    def system_state(self, q, u=None):
        sys = self.system
        q_sys = np.zeros(sys.nq)
        q_sys[self.qDOF] = q
        if u is None:
            return sys, q_sys
        u_sys = np.zeros(sys.nu)
        u_sys[self.uDOF] = u
        return sys, q_sys, u_sys

    ## ----- Force Directions -----

    def W_tau(self, t, q):
        _, q_sys = self.system_state(q)
        W_tau = np.zeros((self._nu, self.nla_tau))
        for j, td in enumerate(self.tendons):
            np.add.at(W_tau[:, j], td.uDOF - self._u_off, -td.W_l(t, q_sys[td.qDOF]))
        return W_tau

    def W_tau_q(self, t, q):
        _, q_sys = self.system_state(q)
        W_tau_q = np.zeros((self._nu, self.nla_tau, self._nq))
        for j, td in enumerate(self.tendons):
            W_l_q = td.W_l_q(t, q_sys[td.qDOF]).toarray()
            np.add.at(
                W_tau_q[:, j, :],
                ((td.uDOF - self._u_off)[:, None], (td.qDOF - self._q_off)[None, :]),
                -W_l_q,
            )
        return W_tau_q

    ## ----- one evaluation shared by la_tau / la_tau_q / la_tau_u -----

    def _core(self, t, q, u):
        key = (t, q.tobytes(), u.tobytes())
        if self._core_cache is not None and self._core_cache[0] == key:
            return self._core_cache[1]

        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)
        J_dyn = self.M_tilde_inv @ W_tau
        S_inv = np.linalg.inv(J_dyn @ J_dyn.T + self.inv_damping * np.eye(3))
        J_dyn_pinv = J_dyn.T @ S_inv

        # sys.h is the expensive part, evaluated once here
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)
        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]

        tau_ref = self.tau(t)  # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)
        b = a + y_0_ddot

        la_tau_real = J_dyn_pinv @ b  # unconstrained allocation

        core = dict(sys=sys, q_sys=q_sys, u_sys=u_sys, W_tau=W_tau, J_dyn=J_dyn, S_inv=S_inv,
                    J_dyn_pinv=J_dyn_pinv, b=b, la_tau_real=la_tau_real)

        if self.positive:
            # min ||W_tau (x - la_tau_real)||^2 + eps ||x - la_tau_real||^2, x >= 0
            n = self.nla_tau
            sqrt_eps = np.sqrt(self.qp_reg * max(np.linalg.norm(W_tau, 2), 1e-300))
            A = np.vstack([W_tau, sqrt_eps * np.eye(n)])
            c = np.concatenate([W_tau @ la_tau_real, sqrt_eps * la_tau_real])
            la_tau_pos, _ = nnls(A, c)
            core.update(A=A, c=c, la_tau_pos=la_tau_pos, F=np.where(la_tau_pos > self.qp_tol)[0], sqrt_eps=sqrt_eps)

        self._core_cache = (key, core)
        return core

    ## ----- derivatives of the unconstrained allocation -----

    def _b_q_from_core(self, t, core):
        sys, q_sys, u_sys = core["sys"], core["q_sys"], core["u_sys"]
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_q = -self._c_la_c_inv @ sys.c_q(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_q = (
            sys.h_q(t, q_sys, u_sys).toarray()
            + sys.Wla_c_q(t, q_sys, la_c).toarray()
            + W_c @ la_c_q
        )
        a_q = np.zeros((3, self._nq))
        a_q[:, self.rod.nodalDOF_r[-1]] = -self.Kp * np.eye(3)
        return a_q - self.M_tilde_inv @ h_tilde_q

    def _b_u_from_core(self, t, core):
        sys, q_sys, u_sys = core["sys"], core["q_sys"], core["u_sys"]
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_u = -self._c_la_c_inv @ sys.c_u(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_u = sys.h_u(t, q_sys, u_sys).toarray() + W_c @ la_c_u
        a_u = np.zeros((3, self._nu))
        a_u[:, self.rod.nodalDOF_r_u[-1]] = -self.Kd * np.eye(3)
        return a_u - self.M_tilde_inv @ h_tilde_u

    def _la_tau_real_q(self, t, q, core, dW_tau):
        # la_tau_real = J^T S^-1 b, uses J^T s = la_tau_real
        J_dyn, J_dyn_pinv, S_inv, b = core["J_dyn"], core["J_dyn_pinv"], core["S_inv"], core["b"]
        s = S_inv @ b
        J_dyn_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, dW_tau)
        J_dyn_qkTs = np.einsum("ajk,a->jk", J_dyn_qk, s)  # J_qk.T @ s
        J_dyn_qkla = np.einsum("ajk,j->ak", J_dyn_qk, core["la_tau_real"])  # J_qk @ la_tau
        J_dyn_pinv_qb = J_dyn_qkTs - J_dyn_pinv @ (J_dyn_qkla + J_dyn @ J_dyn_qkTs)
        return J_dyn_pinv_qb + J_dyn_pinv @ self._b_q_from_core(t, core)

    def _la_tau_real_u(self, t, core):
        return core["J_dyn_pinv"] @ self._b_u_from_core(t, core)

    ## ----- active-set sensitivity of the QP -----

    def _sensitivity(self, core, dA, dc):
        # free set F: A_F^T A_F x_F = A_F^T c, clamped tendons keep a zero row
        A, la_tau_pos, F, c = core["A"], core["la_tau_pos"], core["F"], core["c"]
        out = np.zeros((self.nla_tau, dc.shape[1]))
        if len(F) == 0:
            return out

        A_F = A[:, F]
        A_F_pinv = np.linalg.pinv(A_F)

        if dA is None:
            out[F, :] = A_F_pinv @ dc
            return out

        dA_F = dA[:, F, :]
        r = c - A_F @ la_tau_pos[F]
        resolve = A_F_pinv @ (dc - np.einsum("afk,f->ak", dA_F, la_tau_pos[F]))
        residual = (A_F_pinv @ A_F_pinv.T) @ np.einsum("afk,a->fk", dA_F, r)
        out[F, :] = resolve + residual
        return out

    ## ----- Forces and control law -----

    def la_tau(self, t, q, u):
        core = self._core(t, q, u)
        return core["la_tau_pos"] if self.positive else core["la_tau_real"]

    def la_tau_q(self, t, q, u):
        core = self._core(t, q, u)
        dW_tau = self.W_tau_q(t, q)
        dla = self._la_tau_real_q(t, q, core, dW_tau)
        if not self.positive:
            return dla
        # the sqrt_eps * I block of A is constant
        dA = np.concatenate([dW_tau, np.zeros((self.nla_tau,) + dW_tau.shape[1:])], axis=0)
        dc = np.vstack([
            np.einsum("ijk,j->ik", dW_tau, core["la_tau_real"]) + core["W_tau"] @ dla,
            core["sqrt_eps"] * dla,
        ])
        return self._sensitivity(core, dA, dc)

    def la_tau_u(self, t, q, u):
        core = self._core(t, q, u)
        dla = self._la_tau_real_u(t, core)
        if not self.positive:
            return dla
        # W_tau depends on q only, so dA = 0
        dc = np.vstack([core["W_tau"] @ dla, core["sqrt_eps"] * dla])
        return self._sensitivity(core, None, dc)

    def step_callback(self, t, q, u):
        return q, u
