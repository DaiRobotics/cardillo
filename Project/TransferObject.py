import numpy as np
import sys
from pathlib import Path

from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System

if __name__ == "__main__":
    rod_nelement = 60  # number of elements for the rod discretization
    VTK_export = False

###### Run settings ######
MODE = "zero"  # "zero" / "single"
n_load_steps = 100  # load steps per solve (large forces need small increments)
visualize = True
single_force = 0.5  # [N] force applied to one tendon in "single"
single_show = 0  # which tendon (0-3) is animated after "single"

###### Parameters ######
H = 1.0  # [m] base height above ground
z_left = 0.6  
z_right = 0.8  
rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)
rod_A_IB0[0, 1] = 1   # B_ey = +e_x
rod_A_IB0[1, 2] = -1  # B_ez = -e_y (right-handed)
rod_A_IB0[2, 0] = -1  # B_ex = -e_z (rod hangs down)

rod_l = 1.0  
rod_r = 15e-3  # [m] rod radius (not sure if this is the right radius)
L1 = 0.5 
xi_end_short = L1 / rod_l
ds = 10e-3  # [m] short tendon distance from centerline
dl = 5e-3  # [m] long tendon distance from centerline
E, G = 5.93e5, 1.977e5  

###### System ######
system = System()
density = 1.41e3  # [kg/m^3] (not sure if this is the right density; unused without gravity)
radius = lambda xi: rod_r
EA = lambda xi: E * cross_section.area(xi)
EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]
GA = lambda xi: G * cross_section.area(xi)
GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]
material_model = Simo1986(
    lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),
    lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),
)
cross_section = CircularCrossSection(radius=radius)


# generate initial configuration
def r_OP(xi):
    return np.array([xi * rod_l, 0, 0], dtype=np.float64)


A_IB = lambda xi: np.eye(3, dtype=np.float64)
q0 = DiscreteRod.pose_configuration(
    rod_nelement,
    r_OP,
    A_IB,
    A_IB0=rod_A_IB0,
)
Q = q0.copy()

rod = DiscreteRod(
    cross_section,
    material_model,
    rod_nelement,
    Q=Q,
    q0=q0,
    cross_section_inertias=CrossSectionInertias(density, cross_section),
)
# ---- rigid connections ----
rc = RigidConnection(rod, system.origin, xi1=0)  # clamp the base at the origin

###### Tendons (4 straight tendons) ######
B_r_tendon_parameters = [
    (xi_end_short,  ds, 0.0),  # [0] short, right
    (xi_end_short, -ds, 0.0),  # [1] short, left
    (1.0,           dl, 0.0),  # [2] long, right
    (1.0,          -dl, 0.0),  # [3] long, left
]
tendons = []
for xi_end, x_w, y_w in B_r_tendon_parameters:
    n_path_points = int(np.ceil(xi_end * rod_nelement)) + 1  # one point per node
    B_r_CP_list = [
        rod_A_IB0.T
        @ np.array(
            [
                x_w,
                y_w,
                0,
            ]
        )  # world offset -> body frame B
        for xi in np.linspace(0, xi_end, n_path_points)
    ]
    n = len(B_r_CP_list)
    tendon = RodTendonForce(
        rod,
        xis=[i * xi_end / (n - 1) for i in range(n)],
        B_r_CPs=B_r_CP_list,
    )
    tendons.append(tendon)

# ---- add to system ----
system.add(rod)
system.add(*tendons)
system.add(rc)
system.assemble()


###### Helpers ######
def set_la(la_funcs):
    """Assign one function t -> force [N] to each tendon."""
    for tendon, f in zip(tendons, la_funcs):
        tendon.la = f


def set_la_ramp(la_vec):
    """Linear ramp 0 -> la_vec over the load steps."""
    set_la([lambda t, F=F: F * t for F in la_vec])  # F=F binds the current value


def solve(n_steps=n_load_steps):
    solver = Newton(
        system,
        n_load_steps=n_steps,
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),
    )
    return solver.solve()


def r_at(q, xi, t=1.0):
    """Position of the centerline at xi (simulation frame)."""
    q_rod = q[rod.qDOF]
    qe = q_rod[rod.local_qDOF_P(xi)]  # DOFs of the element containing xi
    return rod.r_OP(t, qe, xi)


def tip_pose(q):
    """Tip position and tip tangent (finite difference between xi=0.999 and 1)."""
    r1 = r_at(q, 1.0)
    r0 = r_at(q, 0.999)
    tangent = (r1 - r0) / np.linalg.norm(r1 - r0)
    return r1, tangent


def report(name, q):
    r, tangent = tip_pose(q)
    tilt = np.degrees(np.arccos(np.clip(-tangent[2], -1, 1)))  # angle to vertical (down)
    print(
        f"{name}: tip x={r[0]:+.4f} m, y={r[1]:+.1e} m, "
        f"height={r[2] + H:.4f} m, tilt from vertical={tilt:.3f} deg"
    )
    return r, tangent


###### Modes ######
sol = None

if MODE == "zero":
    set_la_ramp([0, 0, 0, 0])
    sol = solve(2)
    report("zero force (expect x=0, height=0)", sol.q[-1])

elif MODE == "single":
    names = ["[0] short right", "[1] short left", "[2] long right", "[3] long left"]
    for i in range(4):
        la = np.zeros(4)
        la[i] = single_force
        set_la_ramp(la)
        sol = solve(5)
        report(names[i], sol.q[-1])
        r_mid = r_at(sol.q[-1], xi_end_short)
        print(f"      junction (xi={xi_end_short}) x={r_mid[0]:+.4f} m")

    # solve the selected tendon again so that it is the one exported / animated
    la = np.zeros(4)
    la[single_show] = single_force
    set_la_ramp(la)
    sol = solve(5)


###### VTK export ######
if VTK_export and sol is not None:
    dir_name = Path(sys.argv[0]).parent
    print("exporting VTK")
    system.export(dir_name, f"vtk/transfer_{MODE}_{rod_nelement}", sol, fps=50)
    print("finished")

###### Visualization ######
if visualize and sol is not None:
    from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon

    VisualDiscreteRod(rod, subdivision=4, opacity=0.3)
    for tendon in tendons:
        VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))
    plotter = Plotter(system, (960, 540))
    # camera looks along +y at the x-z task plane
    cam = plotter.camera
    cam.parallel_projection = False
    cam.position = np.array([0, -3.0, -0.5], float)
    cam.focal_point = np.array([0, 0, -0.5], float)
    cam.view_up = np.array([0, 0, 1], float)
    cam.clipping_range = (0.01, 100)
    plotter.render_solution(sol, True, play_speed_up=0.5)