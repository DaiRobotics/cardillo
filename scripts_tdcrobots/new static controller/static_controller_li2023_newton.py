import numpy as np

from cardillo.actuators._base import BaseActuator

from static_feedforward import InverseStaticsFeedforward


def la_t_from_solution(controller, sol):
    """Tendon tensions la_tau(t) of a finished simulation, shape (nt, n_tendons)."""
    la_t = sol.q[:, controller.my_qDOF]
    la_t_ff = np.array([controller.la_t_ff(t) for t in sol.t])
    return la_t + la_t_ff


class StaticControllerLi2023(InverseStaticsFeedforward, BaseActuator):
    def __init__(
        self,
        system,
        rod,
        tendons,
        r_OP_ref_fn,
        v_P_ref_fn=None,
        Kp=0.0,
        Kd=0.0,
        la_t0=None,
        la_t_ff=None,
        J_stat=None,
        pinv_damping=1e-10,
        model_factory=None,
        static_model_twin=None,
        g_accel=9.81,
        J_stat_check_dt=1e-2,
        name="static_controller",
        **static_model_twin_kwargs,
    ):
        if v_P_ref_fn is None:
            v_P_ref_fn = lambda t: np.zeros(3)
        tau = lambda t: np.concatenate([r_OP_ref_fn(t), v_P_ref_fn(t)])
        super().__init__(rod, tau, nla_tau=len(tendons), ntau=6)

        self.system = system
        self.rod = rod
        self.tendons = tendons
        self.name = name

        # own generalized coordinates: the tendon tensions
        self.nq = len(tendons)
        self.q0 = (np.zeros(self.nq) if la_t0 is None else np.asarray(la_t0, float).copy())
        assert self.q0.shape == (self.nq,)

        self.Kp = Kp
        self.Kd = Kd
        self.pinv_damping = pinv_damping
        self.J_stat_check_dt = J_stat_check_dt
        self.t_jac_last = -np.inf

        self.reseed_J_stat = False
        self.init_feedforward(
            len(tendons),
            static_model_twin=static_model_twin,
            model_factory=model_factory,
            la_t_ff=la_t_ff,
            g_accel=g_accel,
            **static_model_twin_kwargs,
        )
        
        if J_stat is None:
            assert (
                self.static_model_twin is not None
            ), "pass either J_stat or model_factory / static_model_twin"
            J_stat = self.static_model_twin.solve_and_eval_J_stat(self.q0 + self.la_t_ff(0.0))
        self.set_J_stat(J_stat)
        self.reseed_J_stat = True

    ## ----- assembly -----

    def set_J_stat(self, J_stat):
        self.J_stat = J_stat
        self.J_stat_inv = J_stat.T @ np.linalg.solve(J_stat @ J_stat.T + self.pinv_damping * np.eye(J_stat.shape[0]), np.eye(J_stat.shape[0]))

    def _on_feedforward_changed(self):
        """Re-seed J_stat at the new starting tension.

        This also leaves the twin at that static equilibrium, which is what the
        caller wants as the initial rod configuration.
        """
        if self.reseed_J_stat and self.static_model_twin is not None:
            self.set_J_stat(
                self.static_model_twin.solve_and_eval_J_stat(self.q0 + self.la_t_ff(0.0))
            )

    def assembler_callback(self):
        rod = self.rod
        self.qDOF = np.concatenate([self.my_qDOF, rod.qDOF])
        self._nq = len(self.qDOF)
        self.uDOF = rod.uDOF
        self._nu = len(self.uDOF)

        self._td_qDOF = [
            self.nq + np.searchsorted(rod.qDOF, td.qDOF) for td in self.tendons
        ]
        self._td_uDOF = [np.searchsorted(rod.uDOF, td.uDOF) for td in self.tendons]

        self._tip_r_idx = self.nq + np.arange(rod.nodalDOF_r[-1].start, rod.nodalDOF_r[-1].stop)
        self._tip_v_idx = np.arange(rod.nodalDOF_r_u[-1].start, rod.nodalDOF_r_u[-1].stop)

    def rod_q(self, q):
        # q = [la_t, q_rod]
        return q[self.nq :]

    ## ----- Force Directions -----

    def W_tau(self, t, q):
        W_tau = np.zeros((self._nu, self.nla_tau))
        for j, (td, uDOF, qDOF) in enumerate(
            zip(self.tendons, self._td_uDOF, self._td_qDOF)
        ):
            np.add.at(W_tau[:, j], uDOF, -td.W_l(t, q[qDOF]))
        return W_tau

    def W_tau_q(self, t, q):
        W_tau_q = np.zeros((self._nu, self.nla_tau, self._nq))
        for j, (td, uDOF, qDOF) in enumerate(
            zip(self.tendons, self._td_uDOF, self._td_qDOF)
        ):
            W_l_q = td.W_l_q(t, q[qDOF]).toarray()
            np.add.at(W_tau_q[:, j, :], (uDOF[:, None], qDOF[None, :]), -W_l_q)
        return W_tau_q

    ## ----- Tendon Forces -----

    def la_tau(self, t, q, u):
        return q[: self.nq] + self.la_t_ff(t)

    def la_tau_q(self, t, q, u):
        la_tau_q = np.zeros((self.nla_tau, self._nq))
        la_tau_q[:, : self.nq] = np.eye(self.nq)
        return la_tau_q

    def la_tau_u(self, t, q, u):
        return np.zeros((self.nla_tau, self._nu))

    ## ----- control law -----

    def feedback(self, t, q, u):
        """Desired tip velocity Kp * (r_ref - r_OP) + Kd * (v_ref - v_P)."""
        tau_ref = self.tau(t)  # tau_ref = [r_OP_ref, v_P_ref]
        r_OP = self.rod._view_nodal_q(self.rod_q(q))[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        return self.Kp * (tau_ref[:3] - r_OP) + self.Kd * (tau_ref[3:] - v_P)

    def q_dot(self, t, q, u):
        return self.J_stat_inv @ self.feedback(t, q, u)

    def q_dot_q(self, t, q, u):
        q_dot_q = np.zeros((self.nq, self._nq))
        q_dot_q[:, self._tip_r_idx] = -self.Kp * self.J_stat_inv
        return q_dot_q

    def q_dot_u(self, t, q):
        q_dot_u = np.zeros((self.nq, self._nu))
        q_dot_u[:, self._tip_v_idx] = -self.Kd * self.J_stat_inv
        return q_dot_u

    def step_callback(self, t, q, u):
        if self.static_model_twin is not None and t - self.t_jac_last >= self.J_stat_check_dt:
            self.t_jac_last = t
            la_t = q[: self.nq] + self.la_t_ff(t)
            try:
                self.set_J_stat(self.static_model_twin.solve_and_eval_J_stat(la_t))
            except Exception as e:
                print(f"{self.name}: J_stat refresh failed at t={t}: {e}")
        return q, u
