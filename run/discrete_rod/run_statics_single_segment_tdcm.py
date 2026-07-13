import numpy as np
from time import perf_counter

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_statics_single_segment_tdcm")

from cardillo.solver import Newton
from cardillo_example_systems.single_segment_tdcm import gen_single_segment_tdcm

ret = gen_single_segment_tdcm(live_plotter=True)
system = ret["system"]
rod = ret["rod"]
tendons = ret["tendons"]
plotter = ret["plotter"]

############
## solver ##
############
solver = Newton(
    system,
    n_load_steps=100,
    verbose=True,
)

solver.fun(solver.x[0], system.t0)
solver.jac(solver.x[0], system.t0)

# from cProfile import Profile
# prof = Profile()
# prof.enable()

t0 = perf_counter()
sol = solver.solve()
print("time:", perf_counter() - t0)

# prof.disable()
# prof.dump_stats("prof.prof")
# exit()
#################
# visualization #
#################

plotter.render_solution(sol, True)
