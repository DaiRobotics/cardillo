import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from cardillo import System
from cardillo.rigid_connection import RigidConnection
from cardillo.rigid_body import RigidBody
from cardillo.forces import Force
from cardillo.math import e3
from cardillo.rods import (
    CircularCrossSection,
)
from cardillo.rods.force_line_distributed import Force_line_distributed
from cardillo.solver import (
    Newton,
    ScipyDAE,
)
from cardillo.rods.discreteRod import DiscreteRod

if __name__ == "__main__":
    dir_name = Path(sys.argv[0]).parent

    nturns = 20  # number of coils Harsch2021

    t1 = 20

    #########
    # gravity
    #########
    gravity = 9.81

    #######################
    # spring modeled as rod
    #######################

    elements_per_turn = 40
    nelements = int(elements_per_turn * nturns)

    ############
    # Harsch2021
    ############
    rho = 7850  # [kg / m^3]
    G = 81.5e9
    E = 206.0e9
    print(f"G: {G}; E: {E}")

    # 1mm cross sectional diameter
    wire_diameter = 1e-3
    wire_radius = wire_diameter / 2

    # helix parameter
    coil_diameter = 32.0e-3
    coil_radius = coil_diameter / 2
    pitch_unloaded = wire_diameter
    c = pitch_unloaded / (coil_radius * 2 * np.pi)

    # rod cross-section
    cross_section = CircularCrossSection(wire_radius)

    A_rho0 = rho * cross_section.area(0)

    # helix and derivatives
    def r(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return coil_radius * np.array(
            [np.sin(alpha + phi0), -np.cos(alpha + phi0), c * alpha]
        )

    def dr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * 2
            * np.pi
            * nturns
            * np.array([np.cos(alpha + phi0), np.sin(alpha + phi0), c])
        )

    def ddr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * (2 * np.pi * nturns) ** 2
            * np.array([-np.sin(alpha + phi0), np.cos(alpha + phi0), 0])
        )

    # definition of the parametric curve
    curve = lambda xi: r(xi, phi0=np.pi)
    dcurve = lambda xi: dr(xi, phi0=np.pi)
    ddcurve = lambda xi: ddr(xi, phi0=np.pi)

    q0 = DiscreteRod.serret_frenet_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        xi1=1,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    )

    rod = DiscreteRod(
        cross_section,
        E,
        G,
        nelements,
        Q=q0,
        q0=q0,
        # polynomial_degree=polynomial_degree,
        density=rho,
    )

    ##############
    # pendulum bob
    ##############
    R = 25e-3  # radius of the main cylinder
    h = 34e-3  # height of the main cylinder
    density = 7850  # [kg / m^3]; steel
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0 = np.concatenate((r_OS0, p0))
    # mass_bob = 0.469
    mass_bob = density * R**2 * np.pi * h
    # K_Theta_S_bob = np.diag([1.468e-4, 1.468e-4, 1.247e-4])
    mass = density * h * np.pi * R**2
    B_Theta_C = (
        np.diag(
            [
                0.25 * R**2 + 1 / 12 * h**2,
                0.25 * R**2 + 1 / 12 * h**2,
                0.5 * R**2,
            ]
        )
        * mass
    )
    bob = RigidBody(mass=mass, B_Theta_C=B_Theta_C, q0=q0)

    system = System()

    f_g_rod_statics = lambda t, xi: -t * A_rho0 * gravity * e3
    gravity_rod_statics = Force_line_distributed(f_g_rod_statics, rod)
    f_g_bob_statics = lambda t: -t * mass_bob * gravity * e3
    gravity_bob_statics = Force(f_g_bob_statics, bob)

    f_pulling = lambda t: -t * mass_bob * gravity * e3 * 0.3
    pulling_force = Force(f_pulling, bob)

    joint1 = RigidConnection(system.origin, rod, xi2=1)
    joint2 = RigidConnection(bob, rod, xi2=0)

    #####################
    # assemble the system
    #####################
    # system.add(rod, joint1, force_rod)
    system.add(
        rod,
        bob,
        joint1,
        joint2,
        gravity_rod_statics,
        gravity_bob_statics,
        pulling_force,
    )
    system.assemble()

    load_sol = False
    if load_sol:
        from cardillo.solver.solution import load_solution

        sol = load_solution(dir_name / "wilberforce2p0_sol.npy")
        sol.system = system
    else:
        #####################
        # solve static system
        #####################
        sol = Newton(
            system,
            n_load_steps=10,
        ).solve()
        q = sol.q
        nt = len(q)
        t = sol.t[:nt]

        system.set_new_initial_state(q0=sol.q[-1], u0=sol.u[-1])

        f_g_rod = lambda t, xi: -A_rho0 * gravity * e3
        gravity_rod = Force_line_distributed(f_g_rod, rod)
        f_g_bob = lambda t: -mass_bob * gravity * e3
        gravity_bob = Force(f_g_bob, bob)

        ########################
        # solve dynamical system
        ########################
        system.remove(gravity_bob_statics, gravity_rod_statics, pulling_force)
        system.add(gravity_bob, gravity_rod)
        system.assemble()
        solver = ScipyDAE(system, t1=t1, dt=1e-3)
        sol = solver.solve()
        sol.system = None
        from cardillo.solver.solution import save_solution

        save_solution(sol, dir_name / "wilberforce2p0_sol.npy")
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
    print("exporting VTK")
    if VTK_export:
        # fake second bob for export
        bob_glyph = RigidBody(1.0, np.eye(3, dtype=float), name="bob_glyph")
        bob_glyph.qDOF = bob.qDOF
        bob_glyph.uDOF = bob.uDOF
        system.add(bob_glyph)
        system.export(dir_name, f"vtk/wilberforce_pendulum", sol, fps=50)
    print("finished")
    exit()
