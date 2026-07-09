import jax.numpy as jnp
from jax import jit, jacfwd, vmap

from .algebra import ax2skew

eye3 = jnp.eye(3, dtype=jnp.float64)


@jit
def _Exp_SO3_quat_norm(P):
    p0, p = P[0], P[1:]
    P2 = P @ P

    p_tilde = ax2skew(p)
    # return jnp.where(
    #     normalize,
    #     eye3 + (2.0 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p)),
    #     (p0**2 - p @ p) * eye3 + jnp.outer(p, 2.0 * p) + 2.0 * p0 * ax2skew(p),
    # )
    return eye3 + (2.0 / P2) * (p0 * ax2skew(p) + p_tilde @ p_tilde)


def Exp_SO3_quat(P, normalize=True):
    if normalize:
        return _Exp_SO3_quat_norm(P)
    else:
        raise NotImplementedError


_Exp_SO3_quat_norm_batch = jit(vmap(_Exp_SO3_quat_norm))


def Exp_SO3_quat_batch(P, normalize=True):
    if normalize:
        return _Exp_SO3_quat_norm_batch(P)
    else:
        raise NotImplementedError


_Exp_SO3_quat_P_norm = jit(jacfwd(_Exp_SO3_quat_norm))


def Exp_SO3_quat_P(P, normalize=True):
    if normalize:
        return _Exp_SO3_quat_P_norm(P)
    else:
        raise NotImplementedError


Exp_SO3_quat_P_norm_batch = jit(vmap(jacfwd(_Exp_SO3_quat_norm)))


def Exp_SO3_quat_batch(P, normalize=True):
    if normalize:
        return _Exp_SO3_quat_norm_batch(P)
    else:
        raise NotImplementedError


@jit
def _T_SO3_quat_norm(P):
    p0, p = P[0], P[1:]

    # return jnp.where(
    #     normalize,
    #     (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
    #     2 * (P @ P) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
    # )
    return (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))


def T_SO3_quat(P, normalize=True):
    if normalize:
        return _T_SO3_quat_norm(P)
    else:
        raise NotImplementedError


_T_SO3_quat_norm_batch = jit(vmap(_T_SO3_quat_norm))


def T_SO3_quat_batch(P, normalize=True):
    if normalize:
        return _T_SO3_quat_norm_batch(P)
    else:
        raise NotImplementedError


_T_SO3_quat_P_norm = jit(jacfwd(_T_SO3_quat_norm))


def T_SO3_quat_P(P, normalize=True):
    if normalize:
        return _T_SO3_quat_P_norm(P)
    else:
        raise NotImplementedError


_T_SO3_quat_P_norm_batch = jit(vmap(jacfwd(_T_SO3_quat_norm)))


def T_SO3_quat_P_batch(P, normalize=True):
    if normalize:
        return _T_SO3_quat_P_norm_batch(P)
    else:
        raise NotImplementedError


@jit
def _T_SO3_inv_quat(P):
    p0, p = P[0], P[1:]
    # return jnp.where(
    #     normalize,
    #     0.5 * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
    #     1 / (2 * (P @ P) ** 2) * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
    # )
    return 1 / (2 * (P @ P) ** 2) * jnp.vstack((-p, p0 * eye3 + ax2skew(p)))


def T_SO3_inv_quat(P, normalize=True):
    if normalize:
        raise NotImplementedError
    else:
        return _T_SO3_inv_quat(P)


_T_SO3_inv_quat_batch = jit(vmap(_T_SO3_inv_quat))


def T_SO3_inv_quat_batch(P, normalize=True):
    if normalize:
        raise NotImplementedError
    else:
        return _T_SO3_inv_quat_batch(P)


_T_SO3_inv_quat_P = jit(jacfwd(_T_SO3_inv_quat))


def T_SO3_inv_quat_P(P, normalize=True):
    if normalize:
        raise NotImplementedError
    else:
        return _T_SO3_inv_quat_P(P)


_T_SO3_inv_quat_P_batch = jit(vmap(jacfwd(_T_SO3_inv_quat)))


def T_SO3_inv_quat_P_batch(P, normalize=True):
    if normalize:
        raise NotImplementedError
    else:
        return _T_SO3_inv_quat_P_batch(P)
