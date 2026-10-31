from time import perf_counter
import numpy as np

from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_statics_single_segment_tdcm")

from cardillo.solver import Newton
from cardillo_example_systems.static_tdcm_single_segment import gen_single_segment_tdcm

n_load_steps = 100

##############
# Setup system
##############
ret = gen_single_segment_tdcm(live_plotter=True)
system = ret["system"]
plotter = ret["plotter"]
assert system._jaxed
system.set_tau(lambda t, tau=np.array([50, 0, 0, 0], float): t * tau)

############
## solver ##
############
solver = Newton(
    system,
    n_load_steps=n_load_steps,
    verbose=True,
)

# warm up
solver.warmup_jax()

t0 = perf_counter()
sol = solver.solve()
print(f"simulation time: {perf_counter() - t0:.2f} s")

#################
# visualization #
#################
plotter.render_solution(sol, True)
