import numpy as np
from abc import ABC, abstractmethod
from time import perf_counter, sleep

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from cardillo.interactions.n_point_interaction import nPointInteraction

from cardillo.rods import CircularCrossSection
from cardillo.rods._base import CosseratRod_PetrovGalerkin
from cardillo.solver.solution import Solution
from cardillo import math_jax


class _VisualTwinBase(ABC):
    def __init__(self, contr, xi=None):
        self.xi = xi
        self.contr = contr
        self.actors = []
        if not hasattr(contr, "visual_twins"):
            contr.visual_twins = [self]
        else:
            contr.visual_twins.append(self)

    @abstractmethod
    def update_visual_state(self, sol_i):
        pass


class VisualDiscreteRod(_VisualTwinBase):
    def __init__(
        self,
        rod,
        subdivision=3,
        color=(82, 108, 164),
        opacity=1,
    ):
        super().__init__(rod)
        nelement_visual = rod.nelement
        cross_section = rod.cross_section

        if isinstance(rod.cross_section, CircularCrossSection):
            weights = [
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
            ]
            degrees = [2, 2, 1]
            ctype = vtk.VTK_BEZIER_WEDGE
        # elif isinstance(rod.cross_section, RectangularCrossSection):
        #     npts = 16
        #     weights = [1] * 16
        #     degrees = [1, 1, 3]
        #     ctype = vtk.VTK_BEZIER_HEXAHEDRON
        else:
            raise NotImplementedError

        self._ugrid = vtk.vtkUnstructuredGrid()

        # points
        self._body_points = np.empty((6 * (nelement_visual + 1), 3), dtype=float)
        array = numpy_to_vtk(self._body_points, deep=False)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(array)
        self._ugrid.SetPoints(vtk_points)

        # cells
        self._ugrid.Allocate(nelement_visual)
        for i in range(nelement_visual):
            self._ugrid.InsertNextCell(
                ctype,
                12,
                list(range(i * 6, i * 6 + 3))
                + list(range((i + 1) * 6, (i + 1) * 6 + 3))
                + list(range(i * 6 + 3, (i + 1) * 6))
                + list(range((i + 1) * 6 + 3, (i + 2) * 6)),
            )

        # point data: RationalWeights
        pdata = self._ugrid.GetPointData()
        array = numpy_to_vtk(np.tile(weights, nelement_visual + 1))
        pdata.SetRationalWeights(array)

        # cell data: HigherOrderDegrees
        cdata = self._ugrid.GetCellData()
        array = numpy_to_vtk(np.repeat([degrees], nelement_visual, axis=0))
        cdata.SetHigherOrderDegrees(array)

        # cell data: Colors
        array = numpy_to_vtk(np.repeat([color], nelement_visual, axis=0))
        array.SetName("Colors")
        cdata.AddArray(array)

        # cell data: Strains
        self._strain = np.zeros((nelement_visual, 6), dtype=float)
        array = numpy_to_vtk(self._strain, deep=False)
        array.SetName("Strains")
        array.SetComponentName(0, "B_gamma_x")
        array.SetComponentName(1, "B_gamma_y")
        array.SetComponentName(2, "B_gamma_z")
        array.SetComponentName(3, "B_kappa_x")
        array.SetComponentName(4, "B_kappa_y")
        array.SetComponentName(5, "B_kappa_z")
        cdata.AddArray(array)

        # filter
        filter = vtk.vtkDataSetSurfaceFilter()
        filter.SetInputData(self._ugrid)
        filter.SetNonlinearSubdivisionLevel(subdivision)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

        # control points on circle
        phis = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
        phis2 = phis + (np.pi / 3.0)
        control_pts = []
        for n in range(rod.nnode):
            if cross_section._variable:
                radius = cross_section.radius(
                    rod.xi_node[n]
                )  # Assuming a simple case, adjust as needed
            else:
                radius = cross_section.radius
            # control points on circle
            xys1 = (
                np.stack([np.zeros_like(phis), np.cos(phis), np.sin(phis)], axis=1)
                * radius
            )
            # control points out of circle
            xys2 = np.stack(
                [np.zeros_like(phis2), np.cos(phis2), np.sin(phis2)], axis=1
            ) * (2.0 * radius)
            control_pts.append(np.concatenate([xys1, xys2], axis=0).T)
        self.control_pts = np.array(control_pts)

    def update_visual_state(self, sol_i):
        rod = self.contr
        q_rod = sol_i.q[rod.qDOF]
        q_nodes = rod._view_nodal_q(q_rod)
        r_OC_nodes = q_nodes[:, :3]
        A_IB_nodes = math_jax.Exp_SO3_quat_norm_batch(q_nodes[:, 3:]).__array__()
        control_pts = r_OC_nodes[:, None] + (A_IB_nodes @ self.control_pts).swapaxes(
            1, 2
        )
        control_pts = control_pts.reshape((-1, 3))

        self._body_points[:] = control_pts
        self._ugrid.Modified()

        # set stress
        _, B_gamma, B_kappa = rod._eval(q_rod)
        self._strain[:, :3] = B_gamma
        self._strain[:, 3:] = B_kappa


class _VisualvtkSource(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi,
    ):
        super().__init__(contr, xi)
        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)

    def add_vtk_source(
        self,
        source,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):

        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(source.GetOutputPort())
        tf_filter.SetTransform(_H_IM)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_visual_state(self, sol_i):
        contr = self.contr
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        xi = self.xi
        A_IB = contr.A_IB(t, q[contr.local_qDOF_P(xi)], xi)
        r_OP = contr.r_OP(t, q[contr.local_qDOF_P(xi)], xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class VisualArUco(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi=None,
        mk_size=0.04,
        mk_dis=0.045,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr, xi)
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)
        from cv2 import aruco

        n_row = 2
        n_col = 2
        x0 = -mk_size / 2 - mk_dis / 2
        y0 = -x0
        h0 = 1e-4
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        quads_black = vtk.vtkCellArray()
        quads_white = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        for row in range(n_row):
            for col in range(n_col):
                id = row * n_col + col
                qrcode = aruco_dict.generateImageMarker(id, aruco_dict.markerSize + 2)
                bit_size = mk_size / (aruco_dict.markerSize + 2)

                # Create a triangle
                n_bits = qrcode.shape[0]
                for i in range(n_bits + 1):
                    for j in range(n_bits + 1):
                        points.InsertNextPoint(
                            x0 + col * mk_dis + j * bit_size,
                            y0 - row * mk_dis - i * bit_size,
                            h0,
                        )

                for i in range(n_bits):
                    for j in range(n_bits):
                        quad = vtk.vtkQuad()
                        quad.GetPointIds().SetId(
                            0, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            1, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            2, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j + 1
                        )
                        quad.GetPointIds().SetId(
                            3, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j + 1
                        )
                        if qrcode[i, j] == 0:
                            quads_black.InsertNextCell(quad)
                        else:
                            quads_white.InsertNextCell(quad)

        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)

        # qrcode
        for triangles, color in zip(
            [quads_black, quads_white], [(0, 0, 0), (255, 255, 255)]
        ):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            filter = vtk.vtkTransformPolyDataFilter()
            filter.SetInputData(polydata)
            filter.SetTransform(_H_IM)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetColor([c / 255 for c in color])
            actor.GetProperty().SetOpacity(opacity)
            actor.SetMapper(mapper)
            self.actors.append(actor)
            # subsystem.appendfilter.AddInputConnection(filter.GetOutputPort())

    def update_visual_state(self, sol_i):
        contr = self.contr
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        xi = self.xi
        A_IB = contr.A_IB(t, q[contr.local_qDOF_P(xi)], xi)
        r_OP = contr.r_OP(t, q[contr.local_qDOF_P(xi)], xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class VisualCylinder(_VisualvtkSource):
    def __init__(
        self,
        contr,
        radius,
        height,
        xi=None,
        resolution=30,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkCylinderSource()
        source.SetRadius(radius)
        source.SetHeight(height)
        source.SetResolution(resolution)
        A_BM = A_BM @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T
        self.add_vtk_source(source, A_BM, B_r_CP, color, opacity)


class VisualSTL(_VisualvtkSource):
    def __init__(
        self,
        contr,
        stl_file,
        xi=None,
        scale=1e-3,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkSTLReader()
        source.SetFileName(stl_file)
        source.Update()
        self.add_vtk_source(source, A_BM * scale, B_r_CP, color, opacity)


class VisualCoordSystem(_VisualvtkSource):
    def __init__(
        self,
        contr,
        length,
        xi=None,
        resolution=30,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkArrowSource()
        source.SetTipResolution(resolution)
        source.SetShaftResolution(resolution)
        for i in range(3):
            if i == 0:
                color = (255, 0, 0)
            elif i == 1:
                A_BM = A_BM @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                color = (0, 255, 0)
            elif i == 2:
                A_BM = A_BM @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                color = (0, 0, 255)
            self.add_vtk_source(source, A_BM * length, B_r_CP, color, opacity)


class VisualTendon(_VisualTwinBase):
    def __init__(
        self, tendon: nPointInteraction, radius=1e-3, color=(255, 255, 255), opacity=1
    ):
        super().__init__(tendon)
        poly_data = vtk.vtkPolyData()
        # points
        npts = 2
        ncon = self.contr.n_nodes
        self.vtkpoints = vtk.vtkPoints()
        self.vtkpoints.SetNumberOfPoints(npts * ncon)
        poly_data.SetPoints(self.vtkpoints)

        # cells
        poly_data.Allocate(ncon)
        for i in range(ncon):
            poly_data.InsertNextCell(
                vtk.VTK_LINE, npts, list(range(i * npts, (i + 1) * npts))
            )

        tendon.export = (
            lambda sol_i, **kwargs: self.update_visual_state(sol_i) or poly_data
        )

        filter = vtk.vtkTubeFilter()
        filter.SetRadius(radius)
        filter.SetInputData(poly_data)
        filter.SetNumberOfSides(16)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(([c / 255 for c in color]))
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_visual_state(self, sol_i):
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        points = []
        r_OPk = [self.contr.r_OPk(t, q, k) for k in range(self.contr.n_nodes)]
        for k in range(self.contr.n_nodes - 1):
            points.append(r_OPk[k])
            points.append(r_OPk[k + 1])
        for i, p in enumerate(points):
            self.vtkpoints.SetPoint(i, p)
        self.vtkpoints.Modified()


class Plotter:
    def __init__(self, system, window_size):
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*window_size)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.window.AddRenderer(self.ren)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.window.SetInteractor(self.interactor)

        # ground
        # grid_size = 0.2
        # plane = vtk.vtkPlaneSource()
        # plane.SetOrigin(0, -grid_size, -grid_size)
        # plane.SetPoint1(0, -grid_size, grid_size,)
        # plane.SetPoint2(0, grid_size, -grid_size,)
        # plane.SetXResolution(10)
        # plane.SetYResolution(10)

        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(plane.GetOutputPort())

        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetRepresentationToWireframe()
        # actor.GetProperty().SetColor(0.6, 0.6, 0.6)
        # self.ren.AddActor(actor)

        # camera
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.ren)
        self.cam_widget.On()
        self.camera = self.ren.GetActiveCamera()
        self.camera.ParallelProjectionOff()

        self.window.AddRenderer(self.ren)
        self.__visual_twins = []
        self.system = system
        for contr in system.contributions:
            if hasattr(contr, "visual_twins"):
                for twin in contr.visual_twins:
                    self.__add_visual_twin(twin)

        self.__window_open = False

        self._live_nframe = 0
        self._live_fps = 100
        self._text_actor = vtk.vtkTextActor()
        self._text_actor.SetPosition(10, 10)
        prop = self._text_actor.GetTextProperty()
        prop.SetFontSize(20)
        prop.SetColor([i / 255 for i in (34, 136, 50)])
        self.ren.AddActor(self._text_actor)

        def decorate_step_callback(step_callback):
            def __step_callback(t, q, u):
                r = step_callback(t, q, u)
                if self.__window_open:
                    if self._live_nframe < t * self._live_fps:
                        self.step_render(Solution(self.system, t=t, q=q, u=u))
                        self._live_nframe += 1
                return r

            return __step_callback

        system.step_callback = decorate_step_callback(system.step_callback)

        def cbk(interactor, event):
            if interactor.key_code == "q":
                self.window.SetOffScreenRendering(1)
                self.__window_open = False

        self.window.SetOffScreenRendering(1)
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, cbk)

    def add_ground(
        self, x0=None, x1=None, y0=None, y1=None, subdivision_x=10, subdivision_y=10
    ):
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(x0, y0, 0)
        plane.SetPoint1(x1, y0, 0)
        plane.SetPoint2(x0, y1, 0)
        plane.SetXResolution(subdivision_x)
        plane.SetYResolution(subdivision_y)

        # mapper = vtk.vtkPolyDataMapper()
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(plane.GetOutputPort())

        converter = vtk.vtkPolyDataToUnstructuredGrid()
        converter.SetInputConnection(plane.GetOutputPort())
        converter.Update()

        self.system.origin.export = lambda sol_i, **kwargs: converter.GetOutput()

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(converter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.6, 0.6, 0.6)
        self.ren.AddActor(actor)

    def step_render(self, sol_i):
        self._text_actor.SetInput(f"t = {sol_i.t:.3f} s")
        for twin in self.__visual_twins:
            twin.update_visual_state(sol_i)
        self.window.Render()
        self.interactor.ProcessEvents()

    def __add_visual_twin(self, visual_twin: _VisualTwinBase):
        if visual_twin not in self.__visual_twins:
            self.__visual_twins.append(visual_twin)
            for actor in visual_twin.actors:
                self.ren.AddActor(actor)
        else:
            raise Exception("visual twin already added!")

    def render_solution(self, solution, repeat=False, play_speed_up=1):
        self.window.SetOffScreenRendering(0)
        self.__window_open = True
        while True:
            t0_sim = solution.t[0]
            t0_real = perf_counter()
            for i, sol_i in enumerate(solution):
                if i > 0:
                    t_real = perf_counter() - t0_real
                    t_sim = (sol_i.t - t0_sim) / play_speed_up
                    dt = t_real - t_sim
                    if dt > 0.0:
                        continue
                    else:
                        sleep(-dt)
                self.step_render(sol_i)
                if not self.__window_open:
                    return
            if not repeat:
                break
            else:
                sleep(solution.t[1] / play_speed_up)

    def live_render(self, fps=100):
        print("maximal frames per simulation time: ", fps)
        self.window.SetOffScreenRendering(0)
        self._live_fps = fps
        self.__window_open = True
