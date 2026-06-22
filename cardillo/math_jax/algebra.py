import jax.numpy as jnp
from jax import jit, vmap


@jit
def norm(a: jnp.ndarray) -> float:
    """Euclidean norm of an array of arbitrary length."""
    return jnp.linalg.norm(a)


@jit
def ax2skew(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    # fmt: off
    return jnp.array([[0.0,    -a[2],  a[1] ],
                      [a[2],   0.0,    -a[0]],
                      [-a[1],  a[0],   0.0  ]], dtype=jnp.float64)
    # fmt: on


ax2skew_batch = jit(vmap(ax2skew))


@jit
def ax2skew_squared(a: jnp.ndarray) -> jnp.ndarray:
    """Computes the product of a skew-symmetric matrix with itself from a given axial vector."""
    a1, a2, a3 = a
    # fmt: off
    return jnp.array([
        [-a2**2 - a3**2,              a1 * a2,              a1 * a3],
        [             a2 * a1, -a1**2 - a3**2,              a2 * a3],
        [             a3 * a1,              a3 * a2, -a1**2 - a2**2],
    ], dtype=jnp.float64)
    # fmt: on


@jit
def skew2ax(A: jnp.ndarray) -> jnp.ndarray:
    """Computes the axial vector from a skew symmetric 3x3 matrix."""
    # fmt: off
    return 0.5 * jnp.array([A[2, 1] - A[1, 2], 
                            A[0, 2] - A[2, 0], 
                            A[1, 0] - A[0, 1]], dtype=jnp.float64)
    # fmt: on


@jit
def ax2skew_a() -> jnp.ndarray:
    """
    Partial derivative of the `ax2skew` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    return jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )


@jit
def skew2ax_A() -> jnp.ndarray:
    """
    Partial derivative of the `skew2ax` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    return jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
            [[0.0, -0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )


@jit
def cross3(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Vector product of two 3D vectors."""
    # fmt: off
    return jnp.array([a[1] * b[2] - a[2] * b[1], 
                      a[2] * b[0] - a[0] * b[2], 
                      a[0] * b[1] - a[1] * b[0]], dtype=jnp.float64)
    # fmt: on
