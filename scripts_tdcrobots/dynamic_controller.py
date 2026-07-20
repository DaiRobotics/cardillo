import numpy as np

from cardillo.actuators._base import BaseActuator

class DynamicControllerNew(BaseActuator):

    def __init__(self, system, rod, tendons, r_OP_ref, v_P_ref=None, a_P_ref=None, Kp=0.0, Kd=0.0, name="dynamic_controller"):
        if a_P_ref is None:
            a_P_ref = lambda t: np.zeros(3)
        if v_P_ref is None:
            v_P_ref = lambda t: np.zeros(3)
        tau = lambda t: np.concatenate([r_OP_ref(t), v_P_ref(t), a_P_ref(t)])
        super().__init__(rod, tau, nla_tau=len(tendons), ntau=9)
        self.system = system
        self.rod = rod
        self.tendons = tendons
        self.Kp = Kp
        self.Kd = Kd
        self.name = name

        self.M_inv2 = None
        self._c_la_c_inv = None

    def assembler_callback(self):
        super().assembler_callback()
        rod = self.rod
        self._q_off = rod.qDOF[0]
        self._u_off = rod.uDOF[0]
        C_1 = np.zeros((3, rod.nq))
        C_1[:, rod.nodalDOF_r[-1]] = np.eye(3)
        self.C_1 = C_1

    def build_M_tilde_inv(self, t, q_sys):
        if self.M_inv2 is None:
            rod = self.rod
            q_rod = q_sys[rod.qDOF]
            B = rod.q_dot_u(t, q_rod).toarray()
            M = rod.M(t, q_rod).toarray()
            self.M_inv2 = (self.C_1 @ B) @ np.linalg.inv(M)
    
    def _scatter_state(self, q, u=None):
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
        _, q_sys = self._scatter_state(q)
        W_tau = np.zeros((self._nu, self.nla_tau))
        for j, td in enumerate(self.tendons):
            np.add.at(W_tau[:, j], td.uDOF - self._u_off, -td.W_l(t, q_sys[td.qDOF]))
        return W_tau
    
    def W_tau_q(self, t, q):
        _, q_sys = self._scatter_state(q)
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
        sys, q_sys, u_sys = self._scatter_state(q, u)
        self.build_M_tilde_inv(t, q_sys)

        W_tau = self.W_tau(t, q)
        h = sys.h(t, q_sys, u_sys) + sys.W_c(t, q_sys) @ sys.la_c(t, q_sys, u_sys)

        J_inv = np.linalg.inv(self.M_inv2 @ W_tau)
        y_0_ddot = -self.M_inv2 @ h[self.uDOF]

        tau_ref = self.tau(t) # tau_ref = [r_OP_ref_fn, v_P_ref_fn, a_P_ref_fn]
        r_OP = self.rod._view_nodal_q(q)[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        a = tau_ref[6:] + self.Kd * (tau_ref[3:6] - v_P) + self.Kp * (tau_ref[:3] - r_OP)

        return J_inv @ (a + y_0_ddot)
    

    def la_tau_q(self, t, q, u):
            sys, q_sys, u_sys = self._scatter_state(q, u)
            self.build_M_tilde_inv(t, q_sys)

            la = self.la_tau(t, q, u)
            W_t = self.W_tau(t, q)
            Jla_tau_q = np.einsum("ijk,j->ik", self.W_tau_q(t, q), la)

            if self._c_la_c_inv is None:
                self._c_la_c_inv = np.linalg.inv(sys.c_la_c().toarray())
            
            la_c = sys.la_c(t, q_sys, u_sys)
            la_c_q = -self._c_la_c_inv @ sys.c_q(t, q_sys, u_sys, la_c).toarray()
            W_c = sys.W_c(t, q_sys).toarray()
            h_tilde_q = sys.h_q(t, q_sys, u_sys).toarray() + sys.Wla_c_q(t, q_sys, la_c).toarray() + W_c @ la_c_q

            a_q = np.zeros((3, self._nq))
            a_q[:, self.rod.nodalDOF_r[-1]] = -self.Kp * np.eye(3)

            J_inv = np.linalg.inv(self.M_inv2 @ W_t)
            return J_inv @ (a_q - self.M_inv2 @ h_tilde_q - self.M_inv2 @ Jla_tau_q)
    
    def la_tau_u(self, t, q, u):
        sys, q_sys, u_sys = self._scatter_state(q, u)
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

        J_inv = np.linalg.inv(self.M_inv2 @ W_tau)
        return J_inv @ (a_u - self.M_inv2 @ h_tilde_u)
    
    def step_callback(self, t, q, u):
        return q, u