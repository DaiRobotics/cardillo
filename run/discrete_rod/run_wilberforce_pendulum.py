import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


from cardillo.utility.jax_cache_config import configure_cache

configure_cache("run_statics_single_segment_tdcm")

from cardillo.discrete import RigidBody
from cardillo.solver import ScipyDAE

from cardillo_example_systems.wilberforce_pendulum import gen_wilberforce_pendulum

dir_name = Path(sys.argv[0]).parent

nturns = 20  # number of coils Harsch2021
nelements_per_turn = 40

t1 = 10
ret = gen_wilberforce_pendulum(nturns=nturns, nelements_per_turn=nelements_per_turn)
system = ret["system"]
rod = ret["rod"]
bob = ret["bob"]

load_sol = False
if load_sol:
    from cardillo.solver.solution import load_solution

    sol = load_solution(dir_name / "wilberforce2p0_sol.npy")
    sol.system = system
else:

    from cProfile import Profile

    solver = ScipyDAE(system, t1=t1, dt=1e-3)
    solver.fun(system.t0, solver.y0, solver.y0)
    solver.jac(system.t0, solver.y0, solver.y0)

    prof = Profile()
    prof.enable()

    sol = solver.solve()

    prof.disable()
    prof.dump_stats("prof.prof")

    sol.system = None

    # from cardillo.solver.solution import save_solution
    # save_solution(sol, dir_name / "wilberforce2p0_sol.npy")
# exit()
q = sol.q
nt = len(q)
t = sol.t[:nt]

# ################################
# # plot characteristic quantities
# ################################
r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

ordering = "zyx"
angles = np.array(
    [
        Rotation.from_matrix(bob.A_IB(ti, qi[bob.qDOF])).as_euler(ordering)
        for (ti, qi) in zip(sol.t, sol.q)
    ]
)
angles = np.unwrap(angles, axis=0)

###############
# visualization
###############
fig, ax = plt.subplots(2, 1)

# ax[0].plot(t, r_OS[:, 0], label="x")
# ax[0].plot(t, r_OS[:, 1], label="y")
ax[0].plot(t, r_OS[:, 2], label="z")
ax[0].set_ylabel("position [m]")
ax[0].legend()
ax[0].grid()

ax[1].plot(t, np.rad2deg(angles[:, 0]), label="alpha")
# ax[1].plot(t, np.rad2deg(angles[:, 1]), label="beta")
# ax[1].plot(t, np.rad2deg(angles[:, 2]), label="gamma")
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("angle [deg]")
ax[1].legend()
ax[1].grid()

plt.show()

############
# VTK export
############
VTK_export = False
if VTK_export:
    print("exporting VTK")
    # fake second bob for export
    bob_glyph = RigidBody(1.0, np.eye(3, dtype=float), name="bob_glyph")
    bob_glyph.qDOF = bob.qDOF
    bob_glyph.uDOF = bob.uDOF
    system.add(bob_glyph)
    system.export(dir_name, f"vtk/wilberforce_pendulum", sol, fps=50)
    print("finished")
