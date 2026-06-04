import jax.numpy as jnp
from jax import jit, jacfwd, vmap

from .algebra import ax2skew, ax2skew_squared


eye3 = jnp.eye(3, dtype=jnp.float64)


@jit
def Exp_SO3_quat_norm(P):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    P2 = P @ P

    return eye3 + (2.0 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p))
    # return jnp.where(
    #     normalize,
    #     eye3 + (2.0 / P2) * (p0 * ax2skew(p) + ax2skew_squared(p)),
    #     (p0**2 - p @ p) * eye3 + jnp.outer(p, 2.0 * p) + 2.0 * p0 * ax2skew(p),
    # )

Exp_SO3_quat_norm_batch = jit(vmap(Exp_SO3_quat_norm))

Exp_SO3_quat_P_norm = jit(jacfwd(Exp_SO3_quat_norm))
Exp_SO3_quat_P_norm_batch = jit(vmap(jacfwd(Exp_SO3_quat_norm)))


@jit
def T_SO3_quat_norm(P):
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    p0, p = P[0], P[1:]

    return (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p)))
    # return jnp.where(
    #     normalize,
    #     (2 / (P @ P)) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
    #     2 * (P @ P) * jnp.hstack((-p[:, None], p0 * eye3 - ax2skew(p))),
    # )


T_SO3_quat_norm_batch = jit(vmap(T_SO3_quat_norm))

T_SO3_quat_P_norm = jit(jacfwd(T_SO3_quat_norm))
T_SO3_quat_P_norm_batch = jit(vmap(jacfwd(T_SO3_quat_norm)))


@jit
def T_SO3_inv_quat(P):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    return 1 / (2 * (P @ P) ** 2) * jnp.vstack((-p, p0 * eye3 + ax2skew(p)))
    # return jnp.where(
    #     normalize,
    #     0.5 * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
    #     1 / (2 * (P @ P) ** 2) * jnp.vstack((-p, p0 * eye3 + ax2skew(p))),
    # )


T_SO3_inv_quat_batch = jit(vmap(T_SO3_inv_quat))

T_SO3_inv_quat_P = jit(jacfwd(T_SO3_inv_quat))
T_SO3_inv_quat_P_batch = jit(vmap(jacfwd(T_SO3_inv_quat)))
