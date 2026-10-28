import numpy as np
import sys
from pathlib import Path
from scipy.optimize import minimize, least_squares

from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System

if __name__ == "__main__":
    rod_nelement = 10  # number of elements for the rod discretization
    VTK_export = False

###### Run settings ######
MODE = "inverse"  # "zero" / "single" / "inverse"
n_load_steps = 60  # load steps per solve (large forces need small increments)
visualize = True
single_force = 0.5  # [N] force applied to one tendon in "single"
single_show = 0  # which tendon (0-3) is animated after "single"

###### Inverse statics settings  ######
x_target = 0.5  
z_target = 0.8
tip_vertical = True  
forbid_above_base = True  
lambda_min = 0.0  # minimum tendon force bound 
lambda_max = np.inf  #maximum tendon force bound 
lambda_scale = 30.0  # [N] typical force, only used to scale the optimization variables
tol_c = 1e-4  # constraint violation below 0.1m

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

def A_at(q, xi, t=1.0):
    """Orientation of the cross-section at xi (simulation frame)."""
    q_rod = q[rod.qDOF]
    qe = q_rod[rod.local_qDOF_P(xi)] 
    return rod.A_IB(t, qe, xi)

def tip_pose(q):
    """Tip position and tangent in the simulation frame."""
    r_tip = r_at(q, 1.0)
    A_tip = A_at(q, 1.0)
    tangent_tip = A_tip[:, 0]  # first column of the rotation matrix is the tangent
    return r_tip, tangent_tip


def report(name, q):
    r, tangent = tip_pose(q)
    tilt = np.degrees(np.arccos(np.clip(-tangent[2], -1, 1)))  # angle to vertical (down)
    print(
        f"{name}: tip x={r[0]:+.4f} m, y={r[1]:+.1e} m, "
        f"height={r[2] + H:.4f} m, tilt from vertical={tilt:.3f} deg"
    )
    return r, tangent


forward_cache = {} ## cache for forward statics results (tip pose and q) for given tendon forces


def forward(la):
    key = tuple(np.round(np.asarray(la, float), 9))
    if key not in forward_cache:
        set_la_ramp(np.asarray(la, float))
        q = solve().q[-1]
        r, tangent = tip_pose(q)
        forward_cache[key] = (r, tangent, q)#position of the tip, tangent at the tip, and full rod configuration
    return forward_cache[key]


def shape_check(q):
    pts = np.array([r_at(q, xi) for xi in np.linspace(0, 1, rod_nelement + 1)])#get the positions of all nodes along the rod
    length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))#get the total length of the rod by summing the distances between consecutive nodes
    return {
        "max_height": pts[:, 2].max() + H, 
        "min_height": pts[:, 2].min() + H,
        "length_ratio": length / rod_l,  # check if < 1: compressed
    }


def constraints_eq(la): #  Equation 7 ,equality constraint = 0
    r, tangent, _ = forward(la)#ignore the full rod configuration, only need tip position and tangent
    c = [r[0] - x_target, (r[2] + H) - z_target]#no y constraint 
    if tip_vertical:
        c.append(tangent[0])  # [0, 0, -1] is vertical down
    return np.array(c)


def constraints_ineq(la): #  Inequality constraint >= 0: the rod stays below the base 
    """Inequality constraint >= 0: the rod stays below the base (no looped shapes)."""
    _, _, q = forward(la)
    return np.array([H - shape_check(q)["max_height"]])


# def initial_guesses():
#     """Initial tendon forces bending the rod towards the target side.
#     Left target: [1] short left + [2] long right; right target: [0] short right + [3] long left."""
#     base = [np.array([1.0, 10.0, 10.0, 1.0]), np.array([1.0, 20.0, 20.0, 1.0]), np.array([5.0, 15.0, 15.0, 5.0])]
#     if x_target >= 0:
#         base = [g[[1, 0, 3, 2]] for g in base]  # mirror left <-> right
#     return base


def inverse_statics():
    #closest = None  # best stage-1 result, reported if the target is not reachable
    #for k, la0 in enumerate(initial_guesses()):
    la0 = np.zeros(4)
    cons = [{"type": "eq", "fun": lambda u: constraints_eq(u * lambda_scale)}]#tell the SLSQP solver to satisfy the equality constraints
    if forbid_above_base:
        cons.append({"type": "ineq", "fun": lambda u: constraints_ineq(u * lambda_scale)})#no higher than the base
    res = minimize(
        lambda u: u @ u,  # objective ||lambda||^2 / lambda_scale^2
        x0=la0 / lambda_scale,
        jac=lambda u: 2 * u,  # gradient of the objective
        method="SLSQP",
        bounds=[(lambda_min / lambda_scale, lambda_max / lambda_scale)] * 4,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 200, "eps": 1e-5},#200 iterations
    )
    la = res.x * lambda_scale#the true tendon forces 
    c = np.linalg.norm(constraints_eq(la))#the norm of the equality constraints to see how close we are to satisfying them
    ok_shape = (not forbid_above_base) or constraints_ineq(la)[0] >= -1e-6

    print(f"SLSQP: iterations = {res.nit}, |c| = {c:.2e}, message: {res.message}")
    return {"la": la, "reached": bool(c < tol_c and ok_shape), "c": c, "start": None}



def print_result(result):
    la = result["la"]
    r, tangent, q = forward(la)
    tilt = np.degrees(np.arccos(np.clip(-tangent[2], -1, 1)))
    shape = shape_check(q)
    print("\n====== inverse statics result ======")
    print(f"target r* = ({x_target:+.4f}, {z_target:.4f}) m, tip vertical = {tip_vertical}")
    print(f"reached: {result['reached']}  (|c| = {result['c']:.2e})")
    print(f"lambda* = {np.round(la, 4)} N")
    print(f"||lambda*||^2 = {la @ la:.4f} N^2")
    print(f"tip: x = {r[0]:+.4f} m, height = {r[2] + H:.4f} m, tilt = {tilt:.3f} deg")
    print(f"rod: highest point {shape['max_height']:.4f} m (base at {H} m), "
          f"length ratio {shape['length_ratio']:.4f}")
    if not result["reached"]:
        print("-> target NOT reached; shown is the closest configuration (stage 1).")


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

elif MODE == "inverse":
    result = inverse_statics()
    print_result(result)
    set_la_ramp(result["la"])  # forward solve again for export / animation
    sol = solve()


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