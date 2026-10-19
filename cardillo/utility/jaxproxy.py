import jax

# import functools


def _coo_matvec_jax(row, col, data, b, n_rows):
    # contribution of each nonzero: A[i, j] * b[j]
    contrib = data * b[col]
    # sum contributions that land in the same output row
    return jax.ops.segment_sum(contrib, row, num_segments=n_rows)


coo_matvec_jit = jax.jit(_coo_matvec_jax, static_argnums=4)

# def jax_proxy(fun_jax):
#     """decorate a class function, allowing access to the fun_jax via obj.method._jax."""
#     def decorator(func):
#         @functools.wraps(func)
#         def make_bound(self):
#             bound = functools.wraps(func)(functools.partial(func, self))
#             bound._jax = fun_jax.__get__(self)
#             bound.__self__ = self
#             return bound
#         return functools.cached_property(make_bound)
#     return decorator
