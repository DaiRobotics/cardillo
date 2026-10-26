from abc import ABC, abstractmethod

import numpy as np
from jax import numpy as jnp


def _empty_jax(*args, **kwargs):
    return jnp.array([])


def _empty_index():
    arr = np.array([], dtype=int)
    arr.flags.writeable = False
    return arr


def add_jax_funs(*names):
    def decorator(cls):
        for name in names:
            defaults = {
                f"{name}_jax": _empty_jax,
                f"{name}_jax_row": _empty_index(),
                f"{name}_jax_col": _empty_index(),
            }
            for attr, value in defaults.items():
                setattr(cls, attr, value)
        return cls

    return decorator


class _Contribution(ABC):
    pass


@add_jax_funs("W_tau", "la_tau", "Wla_tau_q", "Wla_tau_u")
class TauContribution(_Contribution, ABC):
    def __init__(self, nla_tau, ntau=0):
        self.nla_tau = nla_tau
        self.ntau = ntau

    @abstractmethod
    def W_tau(self, t, q):
        pass

    @abstractmethod
    def la_tau(self, t, q, u, tau):
        pass

    def Wla_tau_q(self, t, q, u, tau):
        pass

    def Wla_tau_u(self, t, q, u, tau):
        pass

    def tau(self, t):
        """
        Optional function to compute the control input tau at time t. This function should be implemented in the derived class if ntau > 0.
        """
        if self.ntau:
            raise NotImplementedError(
                "this function should be implemented in the derived class"
            )


@add_jax_funs("c", "c_la_c", "c_q", "c_u", "W_c", "Wla_c_q")
class CContribution(_Contribution, ABC):
    def __init__(self, nla_c):
        self.nla_c = nla_c

    @abstractmethod
    def la_c(self, t, q, u):
        pass

    @abstractmethod
    def c(self, t, q, u, la_c):
        pass

    @abstractmethod
    def c_la_c(self):
        pass

    def c_q(self, t, q, u, la_c):
        pass

    def c_u(self, t, q, u, la_c):
        pass

    @abstractmethod
    def W_c(self, t, q):
        pass

    def Wla_c_q(self, t, q, la_c):
        pass


@add_jax_funs("h", "h_q", "h_u")
class HContribution(_Contribution, ABC):
    @abstractmethod
    def h(self, t, q, u):
        pass

    def h_q(self, t, q, u):
        pass

    def h_u(self, t, q, u):
        pass


@add_jax_funs("g", "g_q", "g_dot", "g_dot_q", "g_dot_u", "W_g", "Wla_g_q")
class GContribution(_Contribution, ABC):
    def __init__(self, nla_g):
        self.nla_g = nla_g

    @abstractmethod
    def g(self, t, q):
        pass

    def g_q(self, t, q):
        pass

    def g_dot(self, t, q, u):
        pass

    def g_dot_q(self, t, q, u):
        pass

    def g_dot_u(self, t, q, u):
        pass

    @abstractmethod
    def W_g(self, t, q):
        pass

    def Wla_g_q(self, t, q, la_g):
        pass
