"""Static (kinematics-based) tendon controller of Li et al. (2023).

The controller of `tdcm_li2023.TendonForceControl` rewritten as a proper
cardillo actuator: the tendon tensions are the controller's own generalized
coordinates and enter the equations of motion through `W_tau @ la_tau` instead
of being pushed into the tendons with `RodTendonForce.set_force` from inside
`q_dot`. All derivatives (`W_tau_q`, `la_tau_q`, `la_tau_u`, `q_dot_q`,
`q_dot_u`) are provided analytically, so implicit solvers such as `ScipyDAE`
see a consistent Jacobian.

Control law (Li et al., 2023): with the static compliance Jacobian
`Gamma = d r_OP / d la_t` of the tip, evaluated at an equilibrium by
`tdcm_li2023.eval_gamma`,

    la_t_dot = Gamma^+ @ (Kp * (r_ref - r_OP) + Kd * (v_ref - v_P))
    la_tau   = la_t + la_t_ff(t)

i.e. the feedback loop is an integrator on the tendon tensions.

Usage::

    from tdcm_li2023 import StaticModel, eval_gamma
    static_model = StaticModel()
    Gamma, _, _ = eval_gamma(static_model, la_t0)

    controller = StaticControllerLi2023(
        system, rod, tendons, Gamma, r_OP_ref_fn,
        Kp=Kp, Kd=Kd, la_t0=la_t0,
        gamma_fn=lambda t, la_t: eval_gamma(static_model, la_t)[0],
    )
    system.add(controller)
    system.assemble()
    sol = ScipyDAE(system, t_sim, dt).solve()
    la_ts = la_t_from_solution(controller, sol)

Note: the tendons must not carry a force of their own (never call
`set_force` on them); the whole tendon load is applied by this controller.
Use `la_t_ff` for a constant pretension instead.
"""

import numpy as np

from cardillo.actuators._base import BaseActuator


def damped_right_inverse(Gamma, damping=0.0):
    """Gamma^T (Gamma Gamma^T + damping * I)^-1, the (damped) right inverse."""
    Gamma = np.atleast_2d(Gamma)
    n_task = Gamma.shape[0]
    return Gamma.T @ np.linalg.solve(
        Gamma @ Gamma.T + damping * np.eye(n_task), np.eye(n_task)
    )


def la_t_from_solution(controller, sol):
    """Tendon tensions la_tau(t) of a finished simulation, shape (nt, n_tendons)."""
    la_t = sol.q[:, controller.my_qDOF]
    la_t_ff = np.array([controller.la_t_ff(t) for t in sol.t])
    return la_t + la_t_ff


class StaticControllerLi2023(BaseActuator):
    def __init__(
        self,
        system,
        rod,
        tendons,
        Gamma,
        r_OP_ref_fn,
        v_P_ref_fn=None,
        Kp=0.0,
        Kd=0.0,
        la_t0=None,
        la_t_ff=None,
        la_t_min=0.0,
        la_t_max=np.inf,
        inv_damping=1e-10,
        gamma_fn=None,
        gamma_eps=1.0,
        gamma_check_dt=0.1,
        name="static_controller_li2023",
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
        self.q0 = np.zeros(self.nq) if la_t0 is None else np.asarray(la_t0, float).copy()

        self.Kp = Kp
        self.Kd = Kd
        self.inv_damping = inv_damping
        self.la_t_ff = la_t_ff if callable(la_t_ff) else lambda t: np.zeros(self.nq)

        self.la_t_min = np.broadcast_to(np.asarray(la_t_min, float), (self.nq,)).copy()
        self.la_t_max = np.broadcast_to(np.asarray(la_t_max, float), (self.nq,)).copy()
        self.sat_tol = 1e-12

        self.set_Gamma(Gamma)
        self.gamma_fn = gamma_fn
        self.gamma_eps = gamma_eps
        self.gamma_check_dt = gamma_check_dt
        self._last_gamma_check_t = -np.inf

    ## ----- assembly -----

    def set_Gamma(self, Gamma):
        """Store a new compliance Jacobian and its damped right inverse."""
        self.Gamma = np.atleast_2d(np.asarray(Gamma, float))
        self.Gamma_inv = damped_right_inverse(self.Gamma, self.inv_damping)

    def assembler_callback(self):
        rod = self.rod
        self.qDOF = np.concatenate([self.my_qDOF, rod.qDOF])
        self._nq = len(self.qDOF)
        self.uDOF = rod.uDOF
        self._nu = len(self.uDOF)

        # local (contribution) indices of the tendon DOFs; rod.qDOF / rod.uDOF
        # are sorted, the rod block starts at self.nq in q and at 0 in u
        self._td_qDOF = [self.nq + np.searchsorted(rod.qDOF, td.qDOF) for td in self.tendons]
        self._td_uDOF = [np.searchsorted(rod.uDOF, td.uDOF) for td in self.tendons]

        # tip position / velocity DOFs, again in local indices
        r_slice = rod.nodalDOF_r[-1]
        v_slice = rod.nodalDOF_r_u[-1]
        self._tip_r = self.nq + np.arange(r_slice.start, r_slice.stop)
        self._tip_v = np.arange(v_slice.start, v_slice.stop)

        # anti-windup mask of the last accepted step, see q_dot_u
        self._free_last = np.ones(self.nq, dtype=bool)

    def _rod_q(self, q):
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

    ## ----- control law: integrator on the tendon tensions -----

    def outer_loop(self, t, q, u):
        """Desired tip velocity Kp * (r_ref - r_OP) + Kd * (v_ref - v_P)."""
        tau_ref = self.tau(t)  # tau_ref = [r_OP_ref, v_P_ref]
        r_OP = self.rod._view_nodal_q(self._rod_q(q))[-1, :3]
        v_P = self.rod._view_nodal_u(u)[-1, :3]
        return self.Kp * (tau_ref[:3] - r_OP) + self.Kd * (tau_ref[3:] - v_P)

    def _free(self, q, la_t_dot):
        """Anti-windup mask: components not pushed further beyond their bound."""
        la_t = q[: self.nq]
        at_min = (la_t <= self.la_t_min + self.sat_tol) & (la_t_dot < 0.0)
        at_max = (la_t >= self.la_t_max - self.sat_tol) & (la_t_dot > 0.0)
        return ~(at_min | at_max)

    def q_dot(self, t, q, u):
        la_t_dot = self.Gamma_inv @ self.outer_loop(t, q, u)
        return np.where(self._free(q, la_t_dot), la_t_dot, 0.0)

    def q_dot_q(self, t, q, u):
        la_t_dot = self.Gamma_inv @ self.outer_loop(t, q, u)
        q_dot_q = np.zeros((self.nq, self._nq))
        q_dot_q[:, self._tip_r] = -self.Kp * self.Gamma_inv
        return q_dot_q * self._free(q, la_t_dot)[:, None]

    def q_dot_u(self, t, q):
        # u is not available here, so the rate needed for the saturation mask
        # cannot be evaluated; reuse the mask of the last accepted step.
        q_dot_u = np.zeros((self.nq, self._nu))
        q_dot_u[:, self._tip_v] = -self.Kd * self.Gamma_inv
        return q_dot_u * self._free_last[:, None]

    ## ----- accepted step: refresh the mask and Gamma -----

    def step_callback(self, t, q, u):
        la_t_dot = self.Gamma_inv @ self.outer_loop(t, q, u)
        self._free_last = self._free(q, la_t_dot)
        # The bounds are enforced by the anti-windup mask in q_dot, which is
        # part of the residual the solver integrates. This clamp is only
        # cleanup: ScipyDAE runs step_callback as a solve_dae event, so the
        # write-back of q into the solver state is not guaranteed.
        q[: self.nq] = np.clip(q[: self.nq], self.la_t_min, self.la_t_max)

        if self.gamma_fn is not None and t - self._last_gamma_check_t >= self.gamma_check_dt:
            self._last_gamma_check_t = t
            la_t = np.clip(q[: self.nq] + self.la_t_ff(t), self.la_t_min, self.la_t_max)
            try:
                Gamma = np.atleast_2d(np.asarray(self.gamma_fn(t, la_t), float))
                dGamma = Gamma - self.Gamma
                if np.linalg.norm(dGamma @ np.linalg.pinv(self.Gamma), 2) >= self.gamma_eps:
                    self.set_Gamma(Gamma)
            except Exception as e:  # a failed static solve must not kill the run
                print(f"{self.name}: Gamma refresh failed at t={t}: {e}")

        return q, u
