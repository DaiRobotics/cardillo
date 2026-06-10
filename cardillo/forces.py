import numpy as np
from .math import norm, outer3
from .utility.coo_matrix import CooMatrix
from .utility.cachetools import MyLRUCache

from .visualization import VisualTendon


class TendonForce:
    def __init__(
        self,
        subsystem_list,
        connectivity,
        xi_list=None,
        B_r_CP_list=None,
        name="tendon",
    ) -> None:
        self.subsystems = subsystem_list
        self.connectivity = connectivity
        self.xis = self.n_subsystems * [(0,)] if xi_list is None else xi_list
        self.Bi_r_CPis = (
            self.n_subsystems * [np.zeros(3)] if B_r_CP_list is None else B_r_CP_list
        )
        self.name = name

        self.n_subsystems = len(subsystem_list)

        self.r_OPk_cache = MyLRUCache(maxsize=self.n_subsystems * 5)
        self.r_OPk_qk_cache = MyLRUCache(maxsize=self.n_subsystems * 5)
        self.J_Pk_cache = MyLRUCache(maxsize=self.n_subsystems * 5)
        self.J_Pk_qk_cache = MyLRUCache(maxsize=self.n_subsystems * 5)

    def assembler_callback(self):
        self._nq: list[int] = []
        self._nu: list[int] = []

        self.qDOF = np.array([], dtype=int)
        self.uDOF = np.array([], dtype=int)

        for sys, xi in zip(self.subsystems, self.xis):
            self._nq.append(len(self.qDOF))
            local_qDOFi = sys.local_qDOF_P(xi)
            self.qDOF = np.concatenate([self.qDOF, sys.qDOF[local_qDOFi]])

            self._nu.append(len(self.uDOF))
            local_uDOFi = sys.local_uDOF_P(xi)
            self.uDOF = np.concatenate([self.uDOF, sys.uDOF[local_uDOFi]])
        self._nq.append(len(self.qDOF))
        self._nu.append(len(self.uDOF))
        self._nq = np.array(self._nq, int)
        self._nu = np.array(self._nu, int)

        self.nq_val = [
            np.arange(*self._nq[i : i + 2]) for i in range(self.n_subsystems)
        ]

        self.nu_val = [
            np.arange(*self._nu[i : i + 2]) for i in range(self.n_subsystems)
        ]

        self._W_q_coo = CooMatrix((self._nu[-1], self._nq[-1]))

    def r_OPk(self, t, q, k):
        qk = q[self.nq_val[k]]
        key = (k, t, qk.tobytes())
        ret = self.r_OPk_cache[key]
        if ret is None:
            ret = self.subsystems[k].r_OP(t, qk, self.xis[k], self.Bi_r_CPis[k])
            self.r_OPk_cache[key] = ret
        return ret

    def r_OPk_qk(self, t, q, k):
        qk = q[self.nq_val[k]]
        key = (k, t, qk.tobytes())
        ret = self.r_OPk_qk_cache[key]
        if ret is None:
            ret = self.subsystems[k].r_OP_q(t, qk, self.xis[k], self.Bi_r_CPis[k])
            self.r_OPk_qk_cache[key] = ret
        return ret
        return self.subsystems[k].r_OP_q(
            t, q[self.nq_val[k]], self.xis[k], self.Bi_r_CPis[k]
        )

    def v_Pk(self, t, q, u, k):
        return self.subsystems[k].v_P(
            t,
            q[self.nq_val[k]],
            u[self.nu_val[k]],
            self.xis[k],
            self.Bi_r_CPis[k],
        )

    def v_Pk_qk(self, t, q, u, k):
        return self.subsystems[k].v_P_q(
            t,
            q[self.nq_val[k]],
            u[self.nu_val[k]],
            self.xis[k],
            self.Bi_r_CPis[k],
        )

    def J_Pk(self, t, q, k):
        qk = q[self.nq_val[k]]
        key = (k, t, qk.tobytes())
        ret = self.J_Pk_cache[key]
        if ret is None:
            ret = self.subsystems[k].J_P(t, qk, self.xis[k], self.Bi_r_CPis[k])
            self.J_Pk_cache[key] = ret
        return ret

    def J_Pk_qk(self, t, q, k):
        qk = q[self.nq_val[k]]
        key = (k, t, qk.tobytes())
        ret = self.J_Pk_qk_cache[key]
        if ret is None:
            ret = self.subsystems[k].J_P_q(t, qk, self.xis[k], self.Bi_r_CPis[k])
            self.J_Pk_qk_cache[key] = ret
        return ret

    def r_PiPj(self, t, q, i, j):
        return self.r_OPk(t, q, j) - self.r_OPk(t, q, i)

    def _nij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        l = norm(r_PiPj)
        return r_PiPj / l

    def _nij_qij(self, t, q, i, j):
        r_PiPj = self.r_PiPj(t, q, i, j)
        gij = norm(r_PiPj)
        tmp = outer3(r_PiPj, r_PiPj) / (gij**3)
        r_OPi_qi = self.r_OPk_qk(t, q, i)
        r_OPj_qj = self.r_OPk_qk(t, q, j)
        n_qi = -r_OPi_qi / gij + tmp @ r_OPi_qi
        n_qj = r_OPj_qj / gij - tmp @ r_OPj_qj
        return n_qi, n_qj

    def l(self, t, q):
        g = 0
        for i, j in self.connectivity:
            g += norm(self.r_OPk(t, q, j) - self.r_OPk(t, q, i))
        return g

    def l_q(self, t, q):
        g_q = np.zeros((self._nq[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            g_q[self.nq_val[i]] += -nij @ self.r_OPk_qk(t, q, i)
            g_q[self.nq_val[j]] += nij @ self.r_OPk_qk(t, q, j)
        return g_q

    def l_dot(self, t, q, u):
        gamma = 0
        for i, j in self.connectivity:
            gamma += self._nij(t, q, i, j) @ (
                self.v_Pk(t, q, u, j) - self.v_Pk(t, q, u, i)
            )
        return gamma

    def l_dot_q(self, t, q, u):
        gamma_q = np.zeros((self._nq[-1]), dtype=np.common_type(q, u))
        for i, j in self.connectivity:
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            nij = self._nij(t, q, i, j)
            vi, vj = self.v_Pk(t, q, u, i), self.v_Pk(t, q, u, j)
            gamma_q[self.nq_val[i]] += (vj - vi) @ nij_qi - nij @ self.v_Pk_qk(
                t, q, u, i
            )
            gamma_q[self.nq_val[j]] += (vj - vi) @ nij_qj - nij @ self.v_Pk_qk(
                t, q, u, j
            )
        return gamma_q

    def W_l(self, t, q):
        W = np.zeros((self._nu[-1]), dtype=q.dtype)
        for i, j in self.connectivity:
            nij = self._nij(t, q, i, j)
            W[self.nu_val[i]] += -self.J_Pk(t, q, i).T @ nij
            W[self.nu_val[j]] += self.J_Pk(t, q, j).T @ nij
        return W

    def W_l_q(self, t, q):
        for n, (i, j) in enumerate(self.connectivity):
            nui, nui1, nuj, nuj1 = self._nu[[i, i + 1, j, j + 1]]
            nqi, nqi1, nqj, nqj1 = self._nq[[i, i + 1, j, j + 1]]
            nij = self._nij(t, q, i, j)
            nij_qi, nij_qj = self._nij_qij(t, q, i, j)
            J_Pi = self.J_Pk(t, q, i)
            J_Pj = self.J_Pk(t, q, j)
            self._W_q_coo[n * 4, nui:nui1, nqi:nqi1] = (
                -self.J_Pk_qk(t, q, i).T @ nij
            ).T - J_Pi.T @ nij_qi
            self._W_q_coo[n * 4 + 1, nuj:nuj1, nqj:nqj1] = (
                self.J_Pk_qk(t, q, j).T @ nij
            ).T + J_Pj.T @ nij_qj
            self._W_q_coo[n * 4 + 2, nui:nui1, nqj:nqj1] = -J_Pi.T @ nij_qj
            self._W_q_coo[n * 4 + 3, nuj:nuj1, nqi:nqi1] = J_Pj.T @ nij_qi
        return self._W_q_coo

    def h(self, t, q, u):
        return -self.la(t) * self.W_l(t, q)

    def h_q(self, t, q, u):
        return -self.la(t) * self.W_l_q(t, q)

    def la(self, t):
        return 0.0

    def set_force(self, force):
        self.la = force if callable(force) else lambda t: force

    def export(self, sol_i, **kwargs):
        if not hasattr(self, "visual_twin"):
            self.visual_twin = VisualTendon(self, radius=1e-3, color=(0, 200, 50))
        self.visual_twin.update_visual_state(sol_i)
        return self.visual_twin._poly_data


class Force:
    r"""Force represented w.r.t. I-basis

    Parameters
    ----------
    force : np.ndarray (3,)
        Force w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which force acts.
    xi : #TODO
    B_r_CP : np.ndarray (3,)
        Position vector of point of attack (P) w.r.t. center of mass (C) in body-fixed B-basis.
    name : str
        Name of contribution.
    """

    def __init__(
        self, force, subsystem, xi=np.zeros(3), B_r_CP=np.zeros(3), name="force"
    ):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.r_OP = lambda t, q: subsystem.r_OP(t, q, xi, B_r_CP)
        self.J_P = lambda t, q: subsystem.J_P(t, q, xi, B_r_CP)
        self.J_P_q = lambda t, q: subsystem.J_P_q(t, q, xi, B_r_CP)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.force(t) @ self.J_P(t, q)

    def h_q(self, t, q, u):
        return np.einsum("i,ijk->jk", self.force(t), self.J_P_q(t, q))


class B_Moment:
    r"""Moment represented w.r.t. body-fixed B-basis

    Parameters
    ----------
    moment : np.ndarray (3,)
        Moment w.r.t. body-fixed B-basis as a callable function of time t.
    subsystem : object
        Object on which moment acts.
    xi : #TODO
    name : str
        Name of contribution.
    """

    def __init__(self, moment, subsystem, xi=np.zeros(3), name="moment"):
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

    def h(self, t, q, u):
        return self.moment(t) @ self.B_J_R(t, q)

    def h_q(self, t, q, u):
        return np.einsum("i,ijk->jk", self.moment(t), self.B_J_R_q(t, q))
