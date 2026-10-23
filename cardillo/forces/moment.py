import numpy as np
from numpy import einsum, zeros
from vtk import VTK_VERTEX
from jax import jit, numpy as jnp
from cardillo.contributions import HContribution


class B_Moment(HContribution):
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

    def __init__(self, moment, subsystem, xi=zeros(3), name="B_moment"):
        HContribution.__init__(self)
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

        self._jaxed = subsystem._jaxed if hasattr(subsystem, "_jaxed") else False

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

        self.h_q_jax_row = np.repeat(np.arange(len(self.uDOF)), len(self.qDOF))
        self.h_q_jax_col = np.tile(np.arange(len(self.qDOF)), len(self.uDOF))

    def h_jax(self, t, q, u):
        return self.moment(t) @ self.B_J_R(t, q)

    def h(self, t, q, u):
        return self.moment(t) @ self.B_J_R(t, q)

    def h_q_jax(self, t, q, u):
        return jnp.einsum("i,ijk->jk", self.moment(t), self.B_J_R_q(t, q)).ravel()

    def h_q(self, t, q, u):
        return einsum("i,ijk->jk", self.moment(t), self.B_J_R_q(t, q))

    def export(self, sol_i, **kwargs):
        r_OP = self.subsystem.r_OP(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        A_IB = self.subsystem.A_IB(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        I_M = A_IB @ self.moment(sol_i.t)

        points = [r_OP]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(M=[I_M])
        return points, cells, None, cell_data


class Moment(HContribution):
    r"""Moment represented w.r.t. I-basis

    Parameters
    ----------
    moment : np.ndarray (3,)
        Moment w.r.t. inertial I-basis as a callable function of time t.
    subsystem : object
        Object on which moment acts.
    xi : #TODO
    name : str
        Name of contribution.
    """

    def __init__(self, moment, subsystem, xi=zeros(3), name="moment"):
        HContribution.__init__(self)
        if not callable(moment):
            self.moment = lambda t: moment
        else:
            self.moment = moment
        self.subsystem = subsystem
        self.xi = xi
        self.name = name

        self.A_IB = lambda t, q: subsystem.A_IB(t, q, xi=xi)
        self.A_IB_q = lambda t, q: subsystem.A_IB_q(t, q, xi=xi)
        self.B_J_R = lambda t, q: subsystem.B_J_R(t, q, xi=xi)
        self.B_J_R_q = lambda t, q: subsystem.B_J_R_q(t, q, xi=xi)

        self._jaxed = subsystem._jaxed if hasattr(subsystem, "_jaxed") else False

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P(self.xi)]
        self.uDOF = self.subsystem.uDOF[self.subsystem.local_uDOF_P(self.xi)]

        self.h_q_jax_row = np.repeat(np.arange(len(self.uDOF)), len(self.qDOF))
        self.h_q_jax_col = np.tile(np.arange(len(self.qDOF)), len(self.uDOF))

    def h_jax(self, t, q, u):
        return (self.moment(t) @ self.A_IB(t, q)) @ self.B_J_R(t, q)

    def h(self, t, q, u):
        return (self.moment(t) @ self.A_IB(t, q)) @ self.B_J_R(t, q)

    def h_q_jax(self, t, q, u):
        I_M = self.moment(t)
        return (
            jnp.einsum("i,ijl,jk->kl", I_M, self.A_IB_q(t, q), self.B_J_R(t, q)).ravel()
            + jnp.einsum("i,ijk->jk", I_M @ self.A_IB(t, q), self.B_J_R_q(t, q)).ravel()
        )

    def h_q(self, t, q, u):
        I_M = self.moment(t)
        return einsum(
            "i,ijl,jk->kl", I_M, self.A_IB_q(t, q), self.B_J_R(t, q)
        ) + einsum("i,ijk->jk", I_M @ self.A_IB(t, q), self.B_J_R_q(t, q))

    def export(self, sol_i, **kwargs):
        r_OP = self.subsystem.r_OP(sol_i.t, sol_i.q[self.qDOF], xi=self.xi)
        I_M = self.moment(sol_i.t)

        points = [r_OP]
        cells = [(VTK_VERTEX, [0])]
        cell_data = dict(M=[I_M])
        return points, cells, None, cell_data
