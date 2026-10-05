import numpy as np
from scipy.optimize import nnls
from scipy.special import erf

from cardillo.actuators._base import BaseActuator


class DynamicControllerSMC(BaseActuator):
    """Sliding mode controller (Rucker 2022), tendon forces projected onto x >= f_min by a QP."""

    def __init__(self, system, rod, tendons, r_OP_ref_fn, v_P_ref_fn=None, a_P_ref_fn=None, alpha=40.0, k=80.0, c=40.0, inv_damping=1e-3, positive=True, f_min=0.0, qp_tol=1e-8, qp_reg=1e-8, name="dynamic_controller_smc"):
        if a_P_ref_fn is None:
            a_P_ref_fn = lambda t: np.zeros(3)
        if v_P_ref_fn is None:
            v_P_ref_fn = lambda t: np.zeros(3)
        tau = lambda t: np.concatenate([r_OP_ref_fn(t), v_P_ref_fn(t), a_P_ref_fn(t)])
        super().__init__(rod, tau, nla_tau=len(tendons), ntau=9)
        self.system = system
        self.rod = rod
        self.tendons = tendons
        self.alpha = alpha  # sliding surface time constant 1/alpha
        self.k = k          # switching gain
        self.c = c          # erf smoothing steepness
        self.inv_damping = inv_damping
        self.name = name

        self.M_tilde_inv = None
        self._c_la_c_inv = None

        # ---- positivity QP ----
        self.positive = positive
        self.f_min = f_min  # lower bound on la_tau, use -la_pre with a pretensioned model
        self.qp_tol = qp_tol  # tendon counts as free above this bound
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

    ## ----- sliding mode outer loop -----

    def outer_loop(self, t, q, u):
        # e = r_ref - r_OP, s = e_dot + alpha e, a = a_ref + alpha e_dot + k erf(c s)
        tau_ref = self.tau(t)  # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        e = tau_ref[:3] - r_OP
        e_dot = tau_ref[3:6] - v_P
        s = e_dot + self.alpha * e
        a = tau_ref[6:] + self.alpha * e_dot + self.k * erf(self.c * s)
        D = self.k * self.c * (2.0 / np.sqrt(np.pi)) * np.exp(-((self.c * s) ** 2))  # d(k erf(c s))/ds
        return a, D

    ## ----- one evaluation shared by la_tau / la_tau_q / la_tau_u -----

    def _core(self, t, q, u):
        key = (t, q.tobytes(), u.tobytes())
        if self._core_cache is not None and self._core_cache[0] == key:
            return self._core_cache[1]

        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)
        J = self.M_tilde_inv @ W_tau
        S_inv = np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
        J_pinv = J.T @ S_inv

        # sys.h is the expensive part, evaluated once here
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)
        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]

        a, D = self.outer_loop(t, q, u)
        b = a + y_0_ddot

        la_tau_real = J_pinv @ b  # damped pseudo-inverse (Eq. 13)

        core = dict(sys=sys, q_sys=q_sys, u_sys=u_sys, W_tau=W_tau, J=J, S_inv=S_inv,
                    J_pinv=J_pinv, b=b, D=D, la_tau_real=la_tau_real)

        if self.positive:
            n = self.nla_tau
            sqrt_eps = np.sqrt(self.qp_reg * max(np.linalg.norm(W_tau, 2), 1e-300))
            A = np.vstack([W_tau, sqrt_eps * np.eye(n)])
            # y = x - f_min >= 0 turns the box QP into a plain nnls
            shifted = la_tau_real - self.f_min
            cvec = np.concatenate([W_tau @ shifted, sqrt_eps * shifted])
            y, _ = nnls(A, cvec)
            x = y + self.f_min
            core.update(A=A, c=cvec, x=x, y=y, F=np.where(y > self.qp_tol)[0],
                        sqrt_eps=sqrt_eps)

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
        # ds/dr_OP = alpha I, so da_i/dr_OP_j = -alpha D_i delta_ij
        a_q = np.zeros((3, self._nq))
        a_q[:, self.rod.nodalDOF_r[-1]] = -self.alpha * np.diag(core["D"])
        return a_q - self.M_tilde_inv @ h_tilde_q

    def _b_u_from_core(self, t, core):
        sys, q_sys, u_sys = core["sys"], core["q_sys"], core["u_sys"]
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_u = -self._c_la_c_inv @ sys.c_u(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_u = sys.h_u(t, q_sys, u_sys).toarray() + W_c @ la_c_u
        # ds/dv_P = I, so da_i/dv_P_j = -(alpha + D_i) delta_ij
        a_u = np.zeros((3, self._nu))
        a_u[:, self.rod.nodalDOF_r_u[-1]] = -(self.alpha * np.eye(3) + np.diag(core["D"]))
        return a_u - self.M_tilde_inv @ h_tilde_u

    def _la_tau_real_q(self, t, q, core, dW_tau):
        # la_tau_real = J^T S^-1 b, uses J^T s = la_tau_real
        J, J_pinv, S_inv, b = core["J"], core["J_pinv"], core["S_inv"], core["b"]
        s_vec = S_inv @ b
        J_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, dW_tau)
        J_qkTs = np.einsum("ajk,a->jk", J_qk, s_vec)  # J_qk.T @ s
        J_qkla = np.einsum("ajk,j->ak", J_qk, core["la_tau_real"])  # J_qk @ la_tau
        J_pinv_qb = J_qkTs - J_pinv @ (J_qkla + J @ J_qkTs)
        return J_pinv_qb + J_pinv @ self._b_q_from_core(t, core)

    def _la_tau_real_u(self, t, core):
        return core["J_pinv"] @ self._b_u_from_core(t, core)

    ## ----- active-set sensitivity of the QP -----

    def _sensitivity(self, core, dA, dc):
        # free set F in the shifted variable y = x - f_min, clamped tendons keep a zero row
        A, y, F, cvec = core["A"], core["y"], core["F"], core["c"]
        out = np.zeros((self.nla_tau, dc.shape[1]))
        if len(F) == 0:
            return out

        A_F = A[:, F]
        A_F_pinv = np.linalg.pinv(A_F)

        if dA is None:
            out[F, :] = A_F_pinv @ dc
            return out

        dA_F = dA[:, F, :]
        r = cvec - A_F @ y[F]
        resolve = A_F_pinv @ (dc - np.einsum("afk,f->ak", dA_F, y[F]))
        residual = (A_F_pinv @ A_F_pinv.T) @ np.einsum("afk,a->fk", dA_F, r)
        out[F, :] = resolve + residual
        return out

    ## ----- Forces and control law -----

    def la_tau(self, t, q, u):
        core = self._core(t, q, u)
        return core["x"] if self.positive else core["la_tau_real"]

    def la_tau_q(self, t, q, u):
        core = self._core(t, q, u)
        dW_tau = self.W_tau_q(t, q)
        dla = self._la_tau_real_q(t, q, core, dW_tau)
        if not self.positive:
            return dla
        # the sqrt_eps * I block of A is constant
        dA = np.concatenate([dW_tau, np.zeros((self.nla_tau,) + dW_tau.shape[1:])], axis=0)
        dc = np.vstack([
            np.einsum("ijk,j->ik", dW_tau, core["la_tau_real"] - self.f_min)
            + core["W_tau"] @ dla,
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
