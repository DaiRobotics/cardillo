import numpy as np
from scipy.optimize import nnls

from cardillo.actuators._base import BaseActuator

from controllers.static_feedforward import InverseStaticsFeedforward


def la_t_from_solution(controller, sol):
    """Tendon tensions la_tau(t) of a finished simulation, shape (nt, n_tendons)."""
    la_t = sol.q[:, controller.my_qDOF]
    la_t_ff = np.array([controller.la_t_ff(t) for t in sol.t])
    return la_t + la_t_ff


class StaticController(InverseStaticsFeedforward, BaseActuator):
    """Static controller (integrator on la_t), tendon forces projected onto la_tau >= 0 by a QP."""

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
        positive=True,
        back_calculate=True,
        qp_tol=1e-8,
        qp_reg=1e-8,
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

        # ---- positivity QP ----
        self.positive = positive
        self.back_calculate = back_calculate  # anti-windup: write projected force back into the state
        self.qp_tol = qp_tol  # tendon counts as free above this force
        self.qp_reg = qp_reg  # eps anchor, relative to ||W_tau||
        self._qp_cache = None

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
        # reseed J_stat at the new starting tension
        self._qp_cache = None
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

    ## ----- positivity QP -----

    def _core(self, t, q):
        # cached on (t, q): la_tau_real and W_tau do not depend on u
        key = (t, q.tobytes())
        if self._qp_cache is not None and self._qp_cache[0] == key:
            return self._qp_cache[1]

        la_tau_real = q[: self.nq] + self.la_t_ff(t)
        W_tau = self.W_tau(t, q)
        core = dict(W_tau=W_tau, la_tau_real=la_tau_real)

        if self.positive:
            # min ||W_tau (x - la_tau_real)||^2 + eps ||x - la_tau_real||^2, x >= 0
            n = self.nla_tau
            sqrt_eps = np.sqrt(self.qp_reg * max(np.linalg.norm(W_tau, 2), 1e-300))
            A = np.vstack([W_tau, sqrt_eps * np.eye(n)])
            c = np.concatenate([W_tau @ la_tau_real, sqrt_eps * la_tau_real])
            la_tau_pos, _ = nnls(A, c)
            core.update(A=A, c=c, la_tau_pos=la_tau_pos, F=np.where(la_tau_pos > self.qp_tol)[0], sqrt_eps=sqrt_eps)

        self._qp_cache = (key, core)
        return core

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

    ## ----- Tendon Forces -----

    def la_tau(self, t, q, u):
        core = self._core(t, q)
        return core["la_tau_pos"] if self.positive else core["la_tau_real"]

    def la_tau_q(self, t, q, u):
        # d(la_tau_real)/dq is the identity on the tension block
        dla = np.zeros((self.nla_tau, self._nq))
        dla[:, : self.nq] = np.eye(self.nq)
        if not self.positive:
            return dla

        core = self._core(t, q)
        dW_tau = self.W_tau_q(t, q)
        # the sqrt_eps * I block of A is constant
        dA = np.concatenate([dW_tau, np.zeros((self.nla_tau,) + dW_tau.shape[1:])], axis=0)
        dc = np.vstack([
            np.einsum("ijk,j->ik", dW_tau, core["la_tau_real"]) + core["W_tau"] @ dla,
            core["sqrt_eps"] * dla,
        ])
        return self._sensitivity(core, dA, dc)

    def la_tau_u(self, t, q, u):
        # no velocity dependence, before or after the projection
        return np.zeros((self.nla_tau, self._nu))

    ## ----- control law -----

    def feedback(self, t, q, u):
        # desired tip velocity Kp (r_ref - r_OP) + Kd (v_ref - v_P)
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
        if self.positive and self.back_calculate:
            # anti-windup: state = applied force, done before the J_stat reseed
            la_tau_pos = self._core(t, q)["la_tau_pos"]
            q = q.copy()
            q[: self.nq] = la_tau_pos - self.la_t_ff(t)
            self._qp_cache = None

        if self.static_model_twin is not None and t - self.t_jac_last >= self.J_stat_check_dt:
            self.t_jac_last = t
            la_t = q[: self.nq] + self.la_t_ff(t)
            try:
                self.set_J_stat(self.static_model_twin.solve_and_eval_J_stat(la_t))
            except Exception as e:
                print(f"{self.name}: J_stat refresh failed at t={t}: {e}")
        return q, u
