import numpy as np
import sys
from pathlib import Path
from scipy.optimize import minimize

from cardillo.constraints import RigidConnection
from cardillo.rods.tendon import RodTendonForce

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias

from cardillo.solver import Newton, SolverOptions
from cardillo.system import System

if __name__ == "__main__":
    rod_nelement = 10  # number of elements for the rod discretization
    VTK_export = False

###### Run settings ######
MODE = "show"  # "zero" / "single" / "inverse" / "show"
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
tol_c = 1e-4  # constraint violation below 1e-4 (position error about 0.1 mm)
result_file = Path(__file__).parent / "inverse_result.npz"  # written by "inverse", read by "show"

###### Show settings ######
show_source = "manual"  # "file": forces from result_file; "manual": forces from show_forces
show_forces = [38.1471, 14.9339, 41.3797, 64.5928]  # [N] used if show_source == "manual"
tendon_offset_scale = 5.0  # tendon offsets drawn this many times larger in the 2D plot (they are only mm)

###### Parameters ######
H = 1.0  # [m] base height above ground
z_left = 0.6  
z_right = 0.8  
x_left = -0.5  # [m] left platform centre (task frame)
x_right = 0.5  # [m] right platform centre (task frame)
platform_width = 0.3  # [m] drawn width of the platforms
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
    """Rotation matrix A_IB of the cross-section at xi (from the quaternion)."""
    q_rod = q[rod.qDOF]
    qe = q_rod[rod.local_qDOF_P(xi)]  # DOFs of the element containing xi
    return rod.A_IB(t, qe, xi)


def tip_pose(q):
    """Tip position and tip direction (first column of A_IB = rod axis B_ex)."""
    r_tip = r_at(q, 1.0)
    A_tip = A_at(q, 1.0)
    tangent_tip = A_tip[:, 0]  # B_ex: points down (-e_z) when the rod hangs straight
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


def inverse_statics():
    la0 = np.zeros(4)  # initial guess: no tendon actuation (straight rod)
    cons = [{"type": "eq", "fun": lambda u: constraints_eq(u * lambda_scale)}]  # tell the SLSQP solver to satisfy the equality constraints
    if forbid_above_base:
        cons.append({"type": "ineq", "fun": lambda u: constraints_ineq(u * lambda_scale)})  # no higher than the base
    res = minimize(
        lambda u: u @ u,  # objective ||lambda||^2 / lambda_scale^2
        x0=la0 / lambda_scale,
        jac=lambda u: 2 * u,  # gradient of the objective
        method="SLSQP",
        bounds=[(lambda_min / lambda_scale, lambda_max / lambda_scale)] * 4,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 200, "eps": 1e-5},  # 200 iterations
    )
    la = res.x * lambda_scale  # the true tendon forces
    c = np.linalg.norm(constraints_eq(la))  # norm of the equality constraints to see how close we are to satisfying them
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
        print("-> target NOT reached; shown is where SLSQP stopped.")


def plot_xz(q, la, target=None):
    """2D view of the x-z task plane: rod centerline, tendon paths, tip, target, platforms."""
    import matplotlib.pyplot as plt

    xis = np.linspace(0, 1, rod_nelement + 1)
    pts = np.array([r_at(q, xi) for xi in xis])  # centerline nodes (simulation frame)
    x, h = pts[:, 0], pts[:, 2] + H  # x and height (task frame)

    # in-plane normal of the centerline: points to +x when the rod hangs straight
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n_x, n_h = -t[:, 2], t[:, 0]#rotate the tangent vector 90 degrees in the x-z plane to get the normal vector

    r_tip, tangent, _ = forward(la)
    tilt = np.degrees(np.arccos(np.clip(-tangent[2], -1, 1)))

    fig, ax = plt.subplots(figsize=(8, 7))
    # environment: ground, base plate, platforms
    ax.axhline(0, color="k", lw=1)#ground
    ax.plot([-0.3, 0.3], [H, H], color="k", lw=4)#base plate
    for xc, zc in ((x_left, z_left), (x_right, z_right)):
        ax.add_patch(plt.Rectangle((xc - platform_width / 2, 0), platform_width, zc, color="tab:blue", alpha=0.25))
    # tendon paths (offsets exaggerated by tendon_offset_scale)
    colors = ["tab:red", "tab:orange", "tab:purple", "tab:green"]
    names = ["[0] short right", "[1] short left", "[2] long right", "[3] long left"]
    for i, (xi_end, x_w, _) in enumerate(B_r_tendon_parameters):#get the tendon parameters for each tendon
        m = xis <= xi_end + 1e-12#choose the indices of the centerline nodes that are within the tendon length
        d = tendon_offset_scale * x_w #exaggerate the tendon offset for visualization
        ax.plot(x[m] + d * n_x[m], h[m] + d * n_h[m], color=colors[i], lw=1,#plot the tendon path with the exaggerated offset
                label=f"{names[i]}: {la[i]:.2f} N")
    # rod centerline, tip and tip direction
    ax.plot(x, h, color="k", lw=2.5, label="rod centerline")#centerline of the rod
    ax.plot(r_tip[0], r_tip[2] + H, "ko", ms=6)#tip position
    ax.quiver(r_tip[0], r_tip[2] + H, tangent[0], tangent[2], angles="xy", scale_units="xy",#arrow showing the tip direction
              scale=10, width=0.004, color="k")
    if target is not None:
        ax.plot(target[0], target[1], "rx", ms=12, mew=2, label="target")
    ax.set_aspect("equal")#x and z axes have the same scale
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("height [m] (task frame)")
    ax.set_title(f"tip ({r_tip[0]:+.3f}, {r_tip[2] + H:.3f}) m, tilt {tilt:.2f} deg, "
                 f"||lambda||^2 = {la @ la:.1f} N^2\n(tendon offsets drawn x{tendon_offset_scale:g})")#the title of the plot shows the tip position, tilt angle, and the squared norm of the tendon forces
                #how much the tendon offsets are exaggerated for visualization
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


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
    np.savez(  # save the result, so that "show" can display it without re-optimizing
        result_file,
        la=result["la"],
        x_target=x_target,
        z_target=z_target,
        tip_vertical=tip_vertical,
        reached=result["reached"],
        c=result["c"],
    )
    print(f"saved to {result_file}")
    set_la_ramp(result["la"])  # forward solve again for export / animation
    sol = solve()

elif MODE == "show":
    # display a given set of tendon forces: one forward solve, report, 2D plot, VTK, animation
    if show_source == "file":
        data = np.load(result_file)
        la = np.asarray(data["la"], float)
        target = (float(data["x_target"]), float(data["z_target"]))
        print(f"loaded {result_file}: target r* = ({target[0]:+.4f}, {target[1]:.4f}) m, "
              f"reached = {bool(data['reached'])}, |c| = {float(data['c']):.2e}")
    else:
        la = np.asarray(show_forces, float)
        target = None
    print(f"tendon forces lambda = {np.round(la, 4)} N")

    set_la_ramp(la)
    sol = solve()
    report("show", sol.q[-1])
    shape = shape_check(sol.q[-1])
    print(f"rod: highest point {shape['max_height']:.4f} m (base at {H} m), "
          f"length ratio {shape['length_ratio']:.4f}")

    plot_xz(sol.q[-1], la, target)

    dir_name = Path(sys.argv[0]).parent
    system.export(dir_name, f"vtk/transfer_show_{rod_nelement}", sol, fps=50)
    print(f"VTK exported to {dir_name / 'vtk'}")


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