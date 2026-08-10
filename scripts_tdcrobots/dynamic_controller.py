import numpy as np
from scipy.optimize import nnls

from cardillo.actuators._base import BaseActuator


# ยะหยา
class DynamicControllerPD(BaseActuator):

    def __init__(self, system, rod, tendons, r_OP_ref_fn, v_P_ref_fn=None, a_P_ref_fn=None, Kp=0.0, Kd=0.0, inv_damping=1e-3, name="dynamic_controller"):
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

    ## ----- Forces and control law -----

    def la_tau(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)

        # Full Actuation
        # J_inv = np.linalg.inv(self.M_tilde_inv @ W_tau)

        # Underactuation
        J = self.M_tilde_inv @ W_tau
        J_inv = J.T @ np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))

        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]

        tau_ref = self.tau(t) # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)

        la_tau = J_inv @ (a + y_0_ddot)
        # la_tau *= 0

        # la_tau, _ = nnls(J, a + y_0_ddot) # Non-Negative Least Squares
        return la_tau


    def la_tau_q(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        la_tau = self.la_tau(t, q, u)
        W_tau = self.W_tau(t, q)
        J = self.M_tilde_inv @ W_tau

        # 3 Tendons Case
        # J_inv = np.linalg.inv(J)

        # b_q for both cases
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())

        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_q = -self._c_la_c_inv @ sys.c_q(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_q = sys.h_q(t, q_sys, u_sys).toarray() + sys.Wla_c_q(t, q_sys, la_c).toarray() + W_c @ la_c_q

        a_q = np.zeros((3, self._nq))
        a_q[:, self.rod.nodalDOF_r[-1]] = -self.Kp * np.eye(3)
        b_q = a_q - self.M_tilde_inv @ h_tilde_q

        # 3 Tendons
        # Jla_tau_q = self.M_tilde_inv @ np.einsum("ijk,j->ik", self.W_tau_q(t, q), la_tau)

        # la_tau_q = J_inv @ (b_q - Jla_tau_q)

        # Underactuated Case
        S_inv = np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
        J_pinv = J.T @ S_inv
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)
        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]
        tau_ref = self.tau(t) # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)
        b = a + y_0_ddot

        s = S_inv @ b
        J_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, self.W_tau_q(t,q))
        J_qkTs = np.einsum("ajk,a->jk", J_qk, s) # J_qk.T @ s
        J_qkla = np.einsum("ajk,j->ak", J_qk, la_tau) # J_qk @ la_tau
        J_pinv_qb = J_qkTs - J_pinv @ (J_qkla + J @ J_qkTs) # J_pinv_q @ b

        la_tau_q = J_pinv_qb + J_pinv @ b_q

        # Non-Negative Least Squares
        # F = np.where(la_tau > self.nnls_tol)[0] # free tendons (postice)
        # la_tau_q = np.zeros((self.nla_tau, self._nq))
        # if len(F) == 0:
        #     return la_tau_q

        # h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)
        # y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]
        # tau_ref = self.tau(t) # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        # r_OP = self.rod._view_nodal_q(q)[-1, :3]
        # v_P = self.rod._view_nodal_u(u)[-1, :3]
        # a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)
        # b = a + y_0_ddot

        # JF = J[:, F] # Pick out the positive forces
        # A_inv = np.linalg.inv(JF.T @ JF)
        # JF_pinv = A_inv @ JF.T
        # la_tau_F = la_tau[F]
        # r = b - JF @ la_tau_F

        # J_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, self.W_tau_q(t,q))
        # JF_qk = J_qk[:, F, :]
        # la_tau_F_q = A_inv @ np.einsum("afk,a->fk", JF_qk, r) + JF_pinv @ (b_q - np.einsum("afk,f->ak", JF_qk, la_tau_F))
        # la_tau_q[F, :] = la_tau_F_q

        return la_tau_q



    def la_tau_u(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)

        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_u = -self._c_la_c_inv @ sys.c_u(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()

        h_tilde_u = sys.h_u(t, q_sys, u_sys).toarray() + W_c @ la_c_u

        a_u = np.zeros((3, self._nu))
        a_u[:, self.rod.nodalDOF_r_u[-1]] = -self.Kd * np.eye(3)
        b_u = a_u - self.M_tilde_inv @ h_tilde_u

        J = self.M_tilde_inv @ W_tau

        # 3 Tendons
        # J_inv = np.linalg.inv(J)
        # la_tau_u = J_inv @ b_u

        # Underactuation
        J_pinv = J.T @ np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
        la_tau_u = J_pinv @ b_u

        # Non-Negative Least Squares
        # F = np.where(self.la_tau(t, q, u) > self.nnls_tol)[0]
        # la_tau_u = np.zeros((self.nla_tau, self._nu))
        # if len(F) == 0:
        #     return la_tau_u

        # JF = J[:, F]
        # JF_pinv = np.linalg.inv(JF.T @ JF) @ JF.T
        # la_tau_u[F, :] = JF_pinv @ b_u

        return la_tau_u

    def step_callback(self, t, q, u):
        return q, u

class _JbMixin:
    def _Jb(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)
        W_tau = self.W_tau(t, q)
        J = self.M_tilde_inv @ W_tau
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)
        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]
        tau_ref = self.tau(t)
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)
        return J, a + y_0_ddot

    def _b_q(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)
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

    def _b_u(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_u = -self._c_la_c_inv @ sys.c_u(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_u = sys.h_u(t, q_sys, u_sys).toarray() + W_c @ la_c_u
        a_u = np.zeros((3, self._nu))
        a_u[:, self.rod.nodalDOF_r_u[-1]] = -self.Kd * np.eye(3)
        return a_u - self.M_tilde_inv @ h_tilde_u

    def _J_qk(self, t, q):
        return np.einsum("ai,ijk->ajk", self.M_tilde_inv, self.W_tau_q(t, q))

class DynamicControllerPID(BaseActuator):

    def __init__(self, system, rod, tendons, r_OP_ref_fn, v_P_ref_fn=None, a_P_ref_fn=None, Kp=0.0, Kd=0.0, Ki=0.0, i_max=np.inf, integral_leak=0.0, inv_damping=1e-3, method="nnls", name="dynamic_controller_pid"):
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
        self.Ki = Ki
        self.i_max = i_max                  # componentwise anti-windup clamp
        self.integral_leak = integral_leak  # exponential forgetting rate [1/s]
        self.inv_damping = inv_damping
        # "nnls": non-negative tensions via active-set least squares
        # "pinv": damped pseudo-inverse, tensions may go negative
        self.method = method
        self.name = name

        self.M_tilde_inv = None
        self._c_la_c_inv = None
        self.nnls_tol = 1e-8

        # integral state, advanced in step_callback only
        self.e_int = np.zeros(3)
        self._t_last = None
        self._e_last = None
        self._la_tau_cache = None  # (t, q, u, la_tau) of the last evaluation
    
    def reset_integral(self):
        """Clear the accumulated integral (e.g. before a new reference leg)."""
        self.e_int = np.zeros(3)
        self._t_last = None
        self._e_last = None
        self._la_tau_cache = None

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

    ## ----- Forces and control law -----

    def _rod_q(self, q):
        """Rod part of the generalized coordinates handed to this actuator."""
        return q

    def _e_int(self, q):
        """Value of the integral state; here an attribute, not part of q."""
        return self.e_int

    def position_error(self, t, q):
        """e = r_ref(t) - r_OP(q) of the rod tip."""
        return self.tau(t)[:3] - self.rod._view_nodal_q(self._rod_q(q))[-1, :3]

    def outer_loop(self, t, q, u):
        """PID feedback law, with the integral state held frozen."""
        tau_ref = self.tau(t)  # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(self._rod_q(q))[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        return (
            tau_ref[6:]
            + self.Kd * (tau_ref[3:6] - v_P)
            + self.Kp * (tau_ref[:3] - r_OP)
            + self.Ki * self._e_int(q)
        )

    def _a_q(self):
        """da/dq. The Ki * e_int term is frozen over a step => it drops out
        and this is the same -Kp * I as in the PD case."""
        a_q = np.zeros((3, self._nq))
        a_q[:, self.rod.nodalDOF_r[-1]] = -self.Kp * np.eye(3)
        return a_q

    def _a_u(self):
        """da/du, the same -Kd * I as in the PD case."""
        a_u = np.zeros((3, self._nu))
        a_u[:, self.rod.nodalDOF_r_u[-1]] = -self.Kd * np.eye(3)
        return a_u

    def _h_tilde(self, sys, t, q_sys, u_sys):
        """Generalized forces the model-based term compensates. Override to
        hide a disturbance from the controller (cf. BlindFFController in
        disturbance_run.py)."""
        return sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)

    def la_tau(self, t, q, u):
        # la_tau_q / la_tau_u and repeated solver residual evaluations call
        # this with identical arguments; skip the nnls + h evaluation then.
        cache = self._la_tau_cache
        if (
            cache is not None
            and t == cache[0]
            and np.array_equal(q, cache[1])
            and np.array_equal(u, cache[2])
        ):
            return cache[3].copy()

        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)
        h = self._h_tilde(sys, t, q_sys, u_sys)

        # Underactuation
        J = self.M_tilde_inv @ W_tau

        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]

        b = self.outer_loop(t, q, u) + y_0_ddot

        if self.method == "pinv":
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
            la_tau = J_pinv @ b
        else:
            la_tau, _ = nnls(J, b)  # Non-Negative Least Squares

        self._la_tau_cache = (t, q.copy(), u.copy(), la_tau.copy())
        return la_tau

    def la_tau_q(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        la_tau = self.la_tau(t, q, u)
        W_tau = self.W_tau(t, q)
        J = self.M_tilde_inv @ W_tau

        # b_q for both cases
        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())

        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_q = -self._c_la_c_inv @ sys.c_q(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()
        h_tilde_q = sys.h_q(t, q_sys, u_sys).toarray() + sys.Wla_c_q(t, q_sys, la_c).toarray() + W_c @ la_c_q

        b_q = self._a_q() - self.M_tilde_inv @ h_tilde_q

        h = self._h_tilde(sys, t, q_sys, u_sys)
        y_0_ddot = -self.M_tilde_inv @ h[self.uDOF]
        b = self.outer_loop(t, q, u) + y_0_ddot

        if self.method == "pinv":
            S_inv = np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
            J_pinv = J.T @ S_inv
            s = S_inv @ b

            J_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, self.W_tau_q(t, q))
            J_qkTs = np.einsum("ajk,a->jk", J_qk, s)       # J_qk.T @ s
            J_qkla = np.einsum("ajk,j->ak", J_qk, la_tau)  # J_qk @ la_tau
            J_pinv_qb = J_qkTs - J_pinv @ (J_qkla + J @ J_qkTs)

            return J_pinv_qb + J_pinv @ b_q

        # Non-Negative Least Squares
        F = np.where(la_tau > self.nnls_tol)[0]  # free tendons (positive)
        la_tau_q = np.zeros((self.nla_tau, self._nq))
        if len(F) == 0:
            return la_tau_q

        JF = J[:, F]  # Pick out the positive forces
        A_inv = np.linalg.inv(JF.T @ JF)
        JF_pinv = A_inv @ JF.T
        la_tau_F = la_tau[F]
        r = b - JF @ la_tau_F

        J_qk = np.einsum("ai,ijk->ajk", self.M_tilde_inv, self.W_tau_q(t, q))
        JF_qk = J_qk[:, F, :]
        la_tau_F_q = A_inv @ np.einsum("afk,a->fk", JF_qk, r) + JF_pinv @ (b_q - np.einsum("afk,f->ak", JF_qk, la_tau_F))
        la_tau_q[F, :] = la_tau_F_q

        return la_tau_q

    def la_tau_u(self, t, q, u):
        sys, q_sys, u_sys = self.system_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)

        if self._c_la_c_inv is None:
            self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
        la_c = sys.la_c(t, q_sys, u_sys)
        la_c_u = -self._c_la_c_inv @ sys.c_u(t, q_sys, u_sys, la_c).toarray()
        W_c = sys.W_c(t, q_sys).toarray()

        h_tilde_u = sys.h_u(t, q_sys, u_sys).toarray() + W_c @ la_c_u

        b_u = self._a_u() - self.M_tilde_inv @ h_tilde_u

        J = self.M_tilde_inv @ W_tau

        if self.method == "pinv":
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.inv_damping * np.eye(3))
            return J_pinv @ b_u

        # Non-Negative Least Squares
        F = np.where(self.la_tau(t, q, u) > self.nnls_tol)[0]
        la_tau_u = np.zeros((self.nla_tau, self._nu))
        if len(F) == 0:
            return la_tau_u

        JF = J[:, F]
        JF_pinv = np.linalg.inv(JF.T @ JF) @ JF.T
        la_tau_u[F, :] = JF_pinv @ b_u

        return la_tau_u

    def step_callback(self, t, q, u):
        """Advance the integral state by one accepted solver step
        (trapezoidal rule), then invalidate the la_tau cache."""
        e = self.position_error(t, q)

        if self._t_last is None:  # first call, from Solver.__init__ at t0
            self._t_last = t
            self._e_last = e
            return q, u

        dt = t - self._t_last
        if dt > 0.0 and self.Ki != 0.0:
            e_int = self.e_int
            if self.integral_leak:
                e_int = e_int * np.exp(-self.integral_leak * dt)
            e_int = e_int + 0.5 * dt * (self._e_last + e)  # trapezoidal
            self.e_int = np.clip(e_int, -self.i_max, self.i_max)
            self._la_tau_cache = None

        if dt >= 0.0:
            self._t_last = t
            self._e_last = e

        return q, u

class ClarkeShiftController(_JbMixin, DynamicControllerPID):

    def __init__(self, *args, f_min=0.5, leak_iters=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_min = f_min
        self.leak_iters = leak_iters

    def _alloc(self, t, q, u):
        J, b = self._Jb(t, q, u)
        U, _, _ = np.linalg.svd(J)
        u_ax = U[:, 0]
        P = np.eye(3) - np.outer(u_ax, u_ax)
        J_lat = P @ J
        S_inv = np.linalg.inv(J_lat @ J_lat.T + self.inv_damping * np.eye(3))
        J_pinv = J_lat.T @ S_inv
        b_lat = P @ b
        la0 = J_pinv @ b_lat
        shift = max(0.0, self.f_min - la0.min())
        leak = J_lat @ np.ones(self.nla_tau)
        for _ in range(self.leak_iters):
            la0 = J_pinv @ (b_lat - shift * leak)
            shift = max(0.0, self.f_min - la0.min())
        k = int(np.argmin(la0))
        return J, b, P, J_lat, S_inv, J_pinv, la0, shift, k

    def la_tau(self, t, q, u):
        *_, la0, shift, _ = self._alloc(t, q, u)
        return la0 + shift

    def la_tau_q(self, t, q, u):
        # full damped-pinv derivative on the projected task (P treated as
        # locally constant), plus the shift term d/dq[-la0[k]] * 1
        J, b, P, J_lat, S_inv, J_pinv, la0, shift, k = self._alloc(t, q, u)
        b_lat = P @ b
        b_q = P @ self._b_q(t, q, u)
        J_qk = np.einsum("wa,ajk->wjk", P, self._J_qk(t, q))
        s = S_inv @ b_lat
        J_qkTs = np.einsum("ajk,a->jk", J_qk, s)
        J_qkla = np.einsum("ajk,j->ak", J_qk, la0)
        J_pinv_qb = J_qkTs - J_pinv @ (J_qkla + J_lat @ J_qkTs)
        la0_q = J_pinv_qb + J_pinv @ b_q
        if shift > 0.0:
            return la0_q + np.ones((self.nla_tau, 1)) * -la0_q[k, :]
        return la0_q

    def la_tau_u(self, t, q, u):
        J, b, P, J_lat, S_inv, J_pinv, la0, shift, k = self._alloc(t, q, u)
        la0_u = J_pinv @ (P @ self._b_u(t, q, u))
        if shift > 0.0:
            return la0_u + np.ones((self.nla_tau, 1)) * -la0_u[k, :]
        return la0_u

