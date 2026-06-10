import sys
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.rigid_connection import RigidConnection
from cardillo.forces import B_Moment
from cardillo.math import e1, e3
from cardillo.rods import CircularCrossSection
from cardillo.solver import Newton, SolverOptions
from cardillo.rods import CircularCrossSection
from cardillo.rods.discreteRod import DiscreteRod

if __name__ == "__main__":
    slenderness = 1e2
    n_load_steps = 10
    VTK_export = False

    name = "helix"
    n_coil = 2

    list_nnodes = []
    list_diff_r_OC = []
    for nnodes in 10 ** np.arange(2, 4):
        nelements = nnodes - 1
        list_nnodes.append(nelements)
        # handle name
        plot_name = name.replace("_", " ")
        save_name = f'{name.replace(" ", "_")}_nel{nelements}'
        print(f"Slenderness: {slenderness:1.0e}, Rod: {plot_name}, nel: {nelements}")

        # geometry of the rod
        R0 = 10  # radius of the helix
        h = 40  # height of the helix
        c = h / (2 * R0 * np.pi * n_coil)  # pitch of the helix
        length = np.sqrt(1 + c**2) * R0 * 3 * np.pi * n_coil
        cc = 1 / (np.sqrt(1 + c**2))

        alpha = lambda xi: 3 * np.pi * n_coil * xi
        alpha_xi = 3 * np.pi * n_coil

        # cross section properties
        width = length / slenderness
        radius = lambda xi: width * (xi <= 2 / 3) + width * (0.5) ** 0.25 * (xi > 2 / 3)
        cross_section = CircularCrossSection(radius=radius)

        # material model
        E = 1.0  # Young's modulus
        G = 0.5  # shear modulus

        # initialize system
        system = System()

        # initial positions and orientations at xi=0
        alpha_0 = alpha(0)

        r_OP0 = R0 * np.array([-np.sin(alpha_0), np.cos(alpha_0), c * alpha_0])

        e_x = cc * np.array([-np.cos(alpha_0), -np.sin(alpha_0), c])
        e_y = np.array([np.sin(alpha_0), -np.cos(alpha_0), 0])
        e_z = cc * np.array([c * np.cos(alpha_0), c * np.sin(alpha_0), 1])

        A_IB0 = np.vstack((e_x, e_y, e_z))
        A_IB0 = A_IB0.T

        #####
        # rod
        #####
        # generate position coordinates for straight initial configuration
        q0 = DiscreteRod.straight_configuration(
            nelements,
            length,
            r_OP0=r_OP0,
            A_IB0=A_IB0,
        )
        # create rod
        rod = DiscreteRod(
            cross_section,
            E,
            G,
            nelements,
            Q=q0,
            q0=q0,
        )
        system.add(rod)

        ##########
        # clamping
        ##########
        clamping = RigidConnection(system.origin, rod, xi2=0)
        system.add(clamping)

        ################
        # applied moment
        ################
        Fi = np.array([G, E, E]) * cross_section.second_moment(0).diagonal()
        M = lambda t: (c * e1 * Fi[0] + e3 * Fi[2]) / (R0 * (1 + c**2)) * t
        moment = B_Moment(M, rod, 1)
        system.add(moment)

        # assemble system
        system.assemble()

        # create solver
        solver = Newton(
            system,
            n_load_steps=n_load_steps,
            options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
        )

        # warm up
        solver.fun(solver.x[0], 0)
        solver.jac(solver.x[0], 0)

        t0 = perf_counter()
        sol = solver.solve()  # solve static equilibrium equations
        t_sim = perf_counter() - t0

        # read solution
        t = sol.t
        q = sol.q
        #################
        # post-processing
        #################
        # VTK export
        dir_name = Path(sys.argv[0]).parent
        if VTK_export:
            from cardillo.visualization import VisualDiscreteRod

            VisualDiscreteRod(rod)
            system.export(dir_name, f"vtk/variable_cross_section", sol)
        la_c = sol.la_c
        la_g = sol.la_g

        print(f"time jax      : {t_sim:.2f} s")

        r_OCs = q[-1, rod.qDOF].reshape((-1, 7), order="C")[:, :3]

        def r_OC_ref(xi):
            if xi <= 2 / 3:
                r = R0 * np.array(
                    [-np.sin(alpha(xi)), np.cos(alpha(xi)), c * alpha(xi)]
                )
            else:
                beta = alpha(2 * (xi - 2 / 3))
                r = np.array([0, R0 / 2, h]) + R0 / 2 * np.array(
                    [-np.sin(beta), np.cos(beta), c * beta]
                )
            return r

        r_OC_refs = np.array([r_OC_ref(xi) for xi in rod.xi_node])

        diff_r_OC = r_OC_refs - r_OCs
        assert nelements * 2 % 3 == 0
        list_diff_r_OC.append(diff_r_OC[[nelements * 2 // 3, nelements]])
    list_diff_r_OC = np.array(list_diff_r_OC)
    ######
    # plot
    ######
    plt.figure()
    plt.plot(
        list_nnodes, np.linalg.norm(list_diff_r_OC[:, 0], axis=1), "-bx", label="mid"
    )
    plt.plot(
        list_nnodes, np.linalg.norm(list_diff_r_OC[:, 1], axis=1), "-rx", label="end"
    )
    plt.xscale("log")
    plt.yscale("log")
    data = np.column_stack(
        [
            list_nnodes,
            np.linalg.norm(list_diff_r_OC[:, 0], axis=1),
            np.linalg.norm(list_diff_r_OC[:, 1], axis=1),
        ]
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")

    ax1.plot(r_OCs[:, 0], r_OCs[:, 1], r_OCs[:, 2], "-b")
    ax1.plot(r_OC_refs[:, 0], r_OC_refs[:, 1], r_OC_refs[:, 2], "-r")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # ax1.set_box_aspect([1, 1, 1])
    ax1.axis("equal")

    la_c = rod._view_element_la_c(sol.la_c[-1])

    # Export for pgfplots
    n = la_c.shape[0]
    indices = np.arange(1, n + 1)

    fig = plt.figure()
    axes = fig.subplots(2, 1, sharex=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.plot(indices, la_c[:, 3], "-m", label=r"$_B m_1$")
    ax1.set_ylabel(r"$_B m_1$")
    ax1.grid()

    ax2.plot(indices, la_c[:, 5], "-y", label=r"$_B m_3$")
    ax2.set_ylabel(r"$_B m_3$")
    ax2.set_xlabel(r"element index")
    ax2.grid()
    plt.show(block=True)
