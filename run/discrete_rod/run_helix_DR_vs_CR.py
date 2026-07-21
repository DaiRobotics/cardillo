import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt


from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_comparison_static_helix")

from cardillo.solver import Newton, SolverOptions
from cardillo_example_systems.static_helix import gen_helix

tend = 3.0
n_load_steps = 1

##############
# discrete rod
##############
ret = gen_helix(rod="DiscreteRod")
rod1 = ret["rod"]
system = ret["system"]

solver = Newton(
    system,
    n_load_steps=n_load_steps,
    options=SolverOptions(newton_max_iter=30, newton_atol=1e-10),
)

# warm up
solver.fun(solver.x[0], system.t0)
solver.jac(solver.x[0], system.t0)

t0 = perf_counter()
sol = solver.solve()
print(f"Discrete rod time: {perf_counter() - t0:.2f} s")

t1, q1 = sol.t, sol.q[:, rod1.qDOF]


##############
# Cosserat rod
##############
ret = gen_helix(rod="CosseratRod")
rod2 = ret["rod"]
system = ret["system"]

solver = Newton(
    system,
    n_load_steps=n_load_steps,
)

t0 = perf_counter()
sol = solver.solve()
print(f"Cosserat rod time: {perf_counter() - t0:.2f} s")

t2, q2 = sol.t, sol.q[:, rod2.qDOF]


#############
# plot result
#############
r_OC1s = q1[-1, rod1.qDOF].reshape((-1, 7), order="C")

r_OC2s = q2[-1, rod2.qDOF].reshape((-1, 7), order="F")

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")

ax1.plot(r_OC1s[:, 0], r_OC1s[:, 1], r_OC1s[:, 2], "-xr", label="rod1")
ax1.plot(r_OC2s[:, 0], r_OC2s[:, 1], r_OC2s[:, 2], "-b.", label="rod2")

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

ax1.set_box_aspect([1, 1, 1])

ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(np.linalg.norm(r_OC1s - r_OC2s, axis=1), "-b.")
ax2.set_yscale("log")
ax2.grid()

plt.show(block=True)
