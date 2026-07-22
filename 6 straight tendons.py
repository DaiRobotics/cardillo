
import numpy as np  # 导入 NumPy，用于数组、矩阵、三角函数和线性代数运算。
import sys  # 导入系统模块，用于读取命令行运行脚本时的路径信息。
from pathlib import Path  # 导入 Path，以跨平台方式处理文件夹路径。

from cardillo.constraints import RigidConnection  # 导入刚性连接约束，用于固定杆根部。
from cardillo.rods.tendon import RodTendonForce  # 导入杆—腱绳力模型，用于定义腱路径及其拉力。

from cardillo.rods import CircularCrossSection, Simo1986, DiscreteRod, CrossSectionInertias  # 导入圆截面、材料模型、离散杆和截面惯性类。

from cardillo.solver import Newton, SolverOptions  # 导入牛顿求解器及其收敛参数设置。
from cardillo.system import System  # 导入多体系统容器。

from matplotlib import pyplot as plt  # 导入绘图库；本脚本当前未使用，可删除而不影响仿真。

if __name__ == "__main__":  # 仅在该文件被直接运行时执行以下仿真代码。
    rod_nelement = 58  # 将杆离散为 58 个有限单元；同时产生59 个节点/路径站点。
    VTK_export = True  # 是否导出 VTK 结果；False 表示本次不导出。
    # ---- parameters ----  # 以下为杆的几何和初始姿态参数。
    rod_r0 = 30e-3  # [m] 原始杆根部半径；本新版脚本中后续未直接使用。
    rod_l0 = 95e-3  # [m] 原始杆长度；本新版脚本中后续未直接使用。
    rod_r_ratio = (  # 定义杆尖端半径与根部半径之比。
        0.533  # 尖端半径为根部半径的 53.3%，即杆沿长度逐渐变细。
    )
    rod_A_IB0 = np.zeros((3, 3), dtype=np.float64)  # 创建杆根部初始姿态矩阵 A_IB0。
    rod_A_IB0[0, 1] = rod_A_IB0[1, 2] = rod_A_IB0[2, 0] = 1  # 填入循环置换旋转矩阵，指定杆局部坐标系与世界系的初始相对方向。
    rod_l_new = 0.58  # [m] 本模型实际采用的新杆长度（580 mm）。
    rod_r_new = 15e-3  # [m] 本模型杆根部半径
    d1 = 10e-3#短腱绳距离中心线10mm
    d2 = 5e-3#长腱绳距离中心线5mm
    xi_end_base = 0.20 / rod_l_new
    ##################
    ## build system ##  # 第一部分：建立物理系统。
    ##################

    # ---- system ----
    system = System()  # 创建空的 Cardillo 系统；system.origin 是固定世界坐标系。

    # ---- rod ----  # 以下定义变半径杆的质量、几何和材料。
    density = 1.41e3  # [kg/m^3] 杆材料的体密度。
    radius = lambda xi: rod_r_new * (1 - xi * (1 - rod_r_ratio))  # 定义半径函数：xi=0 时为 rod_r_new，xi=1 时为 rod_r_new*rod_r_ratio。
    cross_section = CircularCrossSection(radius)  # 用随 xi 变化的半径函数创建变截面圆杆。
    EA = lambda xi: E * cross_section.area(xi)  # 定义位置相关的轴向拉伸刚度 EA(xi)。
    EI = lambda xi: E * cross_section.second_moment(xi)[1, 1]  # 定义位置相关的弯曲刚度 EI(xi)。
    GA = lambda xi: G * cross_section.area(xi)  # 定义位置相关的剪切刚度 GA(xi)。
    GJ = lambda xi: G * cross_section.second_moment(xi)[0, 0]  # 定义位置相关的扭转刚度 GJ(xi)。
    material_model = Simo1986(  # 创建 Simo 1986 Cosserat 杆材料模型。
        lambda xi: np.array([EA(xi), GA(xi), GA(xi)]),  # 三个平移应变方向的刚度：轴向与两个剪切方向。
        lambda xi: np.array([GJ(xi), EI(xi), EI(xi)]),  # 三个转动应变方向的刚度：扭转与两个弯曲方向。
    )
    cross_section = CircularCrossSection(radius=radius)  # 再次以关键字参数建立同一变截面对象，供离散杆及质量惯性使用。
    E, G = 5.93e5, 1.977e5  # [Pa] 设置杨氏模量 E 与剪切模量 G；上述 lambda 在实际调用时才会读取此值。

    # generate initial configuration  # 以下生成未受载时的直杆初始构型。
    def r_OP(xi):  # 定义材料坐标 xi∈[0,1] 上中心线点 P 的位置函数。
        return np.array([xi * rod_l_new, 0, 0], dtype=np.float64)  # 使中心线沿世界 x 轴从 0 延伸到 rod_l_new。

    A_IB = lambda xi: np.eye(3, dtype=np.float64)  # 定义沿杆各截面的相对姿态为单位矩阵，即初始不发生额外旋转。
    q0 = DiscreteRod.pose_configuration(  # 由中心线和姿态函数生成离散杆的实际初始广义坐标 q0。
        rod_nelement,  # 输入杆单元数。
        r_OP,  # 输入中心线位置函数。
        A_IB,  # 输入截面方向函数。
        A_IB0=rod_A_IB0,  # 输入根部初始方向，将局部构型映射到世界坐标系。
    )
    Q = q0.copy()  # 将初始构型复制为参考构型；此处假设初始杆没有预应变。

    rod = DiscreteRod(  # 创建实际参与力学计算的离散柔性杆对象。
        cross_section,  # 传入沿长度变化的圆截面。
        material_model,  # 传入位置相关的材料刚度模型。
        rod_nelement,  # 传入单元数量。
        Q=Q,  # 传入无应变/参考构型。
        q0=q0,  # 传入求解开始时的初始构型。
        cross_section_inertias=CrossSectionInertias(density, cross_section),  # 根据密度和变截面计算质量及转动惯性。
    )

    # ---- rigid connections ----
    rc = RigidConnection(rod, system.origin, xi1=0)  # 把 xi=0 的杆根部刚性固定到世界原点。
    # ---- tendons ( 6根平行走线) ----
    B_r_tendon_parameters = [
    (xi_end_base,  d1,             0.0),#六根腱绳的位置和长度
    (xi_end_base, -0.5 * d1,  np.sqrt(3) / 2 * d1),
    (xi_end_base, -0.5 * d1, -np.sqrt(3) / 2 * d1),
    (1.0,          0.5 * d2, -np.sqrt(3) / 2 * d2),
    (1.0,          0.5 * d2,  np.sqrt(3) / 2 * d2),
    (1.0,         -d2,             0.0),
    ]
    tendons = [ ]#列表
    for xi_end, y, z in B_r_tendon_parameters:#遍历上面的六组参数
        n_path_points = int(np.ceil(xi_end * rod_nelement)) + 1
        B_r_CP_list = [
        rod_A_IB0.T
        @ np.array(
            [
                y,
                z,
                0,
            ]
        )
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
    system.add(rod)  # 将柔性杆加入系统。
    system.add(*tendons)  # 将列表中的全部腱绳加入系统；* 表示将列表元素逐个传入。
    system.add(rc)  # 将根部刚性固定约束加入系统。
    system.assemble()  # 装配系统的自由度、内力、外力和约束方程。

    ############
    ## solver ##  # 第二部分：设置并运行静力平衡求解。
    ############
    F0 = [24, 6, 6, 0, 0, 15]

    tendons[0].la = lambda t: F0[0] * t
    tendons[1].la = lambda t: F0[1] * t
    tendons[2].la = lambda t: F0[2] * t
    tendons[3].la = lambda t: F0[3] * t
    tendons[4].la = lambda t: F0[4] * t
    tendons[5].la = lambda t: F0[5] * t
    solver = Newton(  # 创建 Newton-Raphson 非线性静力求解器。
        system,  # 指定待求解系统。
        n_load_steps=8,  # 将总拉力分为 8 个加载步，提高大变形问题收敛性。
        options=SolverOptions(newton_atol=1e-10, newton_rtol=1e-6),  # 设置牛顿残差的绝对与相对收敛容差。
    )

    sol = solver.solve()  # 运行分步牛顿迭代，获得杆在最终腱绳拉力下的平衡解。

    ############
    # VTK export  # 第三部分：可选地导出仿真结果，供 ParaView 等软件查看。
    ############
    if VTK_export:  # 仅当开关设为 True 时执行以下导出。
        dir_name = Path(sys.argv[0]).parent  # 获得当前 Python 脚本所在的文件夹。
        print("exporting VTK")  # 在终端提示开始导出。
        # fake second bob for export  # 原作者注释：导出所需的占位说明；本模型实际未创建第二个 bob。
        system.export(dir_name, f"vtk/tendon_robot_{rod_nelement}", sol, fps=50)  # 将结果按 50 fps 导出到 vtk/tendon_robot_100 路径。
        print("finished")  # 在终端提示导出完成。

    #################
    # visualization #  # 第四部分：建立三维可视化场景。
    #################
    # ---- visual objects ----
    from cardillo.visualization import Plotter, VisualDiscreteRod, VisualTendon  # 延迟导入绘图器、杆和腱绳视觉对象。

    VisualDiscreteRod(rod, subdivision=4, opacity=0.3)  # 绘制杆；每个单元再细分 4 段，透明度为 0.3，以便看见内部腱路径。
    for tendon in tendons:  # 逐一为所有腱绳创建可视化对象。
        VisualTendon(tendon, radius=1e-3, color=(0, 200, 50))  # 以 1 mm 半径、绿色显示腱绳；该对象不参与力学计算。
    # VisualCoordSystem(system.origin, 0.05)  # 若取消注释，可显示世界坐标系；当前未导入该类。
    # ---- plotter ----
    window_size = (960, 540)  # 设置渲染窗口像素尺寸（宽，高）。
    plotter = Plotter(system, window_size)  # 创建可视化窗口和场景管理器。
    x0, x1 = -0.2, 0.2  # 定义地面网格的 x 范围。
    y0, y1 = -0.2, 0.2  # 定义地面网格的 y 范围。
    res_x = res_y = 10  # 定义地面网格在 x、y 方向的分辨率。
    # plotter.add_ground(x0, x1, y0, y1, res_x, res_y)  # 如取消注释，将绘制地面网格；目前不显示。
    # ---- camera pose ----
    r_OC = np.array([0, -0.35, 0.35], float)*3  # 设置世界坐标系中相机位置 C。
    # r_OC = np.array([0, -0.35, 0.15], float)  # 备选的较高相机位置；当前不使用。
    r_OF = np.array([0, 0, 0.06], float)  # 设置相机注视焦点 F。
    e_x_cam = np.array([1, 0, 0], float)  # 给定相机横向参考轴方向。
    e_z_cam = r_OF - r_OC  # 计算从相机位置指向焦点的视线方向。
    e_z_cam /= np.linalg.norm(e_z_cam)  # 将视线向量归一化为单位向量。
    e_y_cam = np.cross(e_z_cam, e_x_cam)  # 用叉乘计算与视线垂直的相机上方向基向量。
    zoom = 1  # 设置视图缩放倍率；1 表示默认缩放。
    # zoom = 1.5  # 备选：将视图放大到 1.5 倍；当前不使用。
    fx = fy = 2635.5177  # 设置相机水平和垂直焦距（单位：像素）。
    px, py = 3840, 2160  # 设置相机图像分辨率（4K：宽 3840、高 2160）。
    cam_view_angle = np.rad2deg(np.arctan(min(px, py) / 2 / fx) * 2)  # 由焦距与较短图像边计算视场角，并由弧度转换为度。
    cam = plotter.camera  # 取得 Plotter 创建的相机对象。
    cam.view_angle = cam_view_angle  # 设定相机透视投影的视场角。
    cam.parallel_projection = False  # 关闭平行投影，使用透视投影。
    cam.position = r_OC  # 设置相机位置。
    cam.focal_point = r_OF  # 设置相机注视点。
    cam.view_up = -e_y_cam  # 设置画面向上方向；负号用于匹配期望的图像朝向。
    cam.clipping_range = (0.01, 100)  # 设置近、远裁剪面范围，单位为 m。
    cam.Zoom(zoom)  # 应用缩放倍率。

    # plotter.live_render()  # 如取消注释，启动实时交互渲染；当前不执行。

    plotter.render_solution(sol, True, play_speed_up=0.5)  # 渲染求解结果；True 启用交互显示，0.5 表示以较慢速度播放加载过程。
