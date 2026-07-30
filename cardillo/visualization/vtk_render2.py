from abc import ABC, abstractmethod
from time import perf_counter, sleep

import vtk
import numpy as np

from cardillo.solver import Solution


class RuntimeVisualBase(ABC):
    @abstractmethod
    def to_vtk_actors(self):
        pass

    @abstractmethod
    def update_vtk_actors(self, sol_i):
        pass


class RuntimeVisualAddOn(RuntimeVisualBase):
    def __init__(self, contr, xi):
        self.contr = contr
        self.xi = xi
        if hasattr(contr, "to_vtk_actors"):
            f = contr.to_vtk_actors
            contr.to_vtk_actors = lambda f=f: self.to_vtk_actors() + f()
            f = contr.update_vtk_actors
            contr.update_vtk_actors = lambda sol_i, f=f: (
                self.update_vtk_actors(sol_i),
                f(sol_i),
            )
        else:
            contr.to_vtk_actors = lambda: self.to_vtk_actors()
            contr.update_vtk_actors = lambda sol_i: self.update_vtk_actors(sol_i)

        self._H_IB = np.eye(4)
        self._H_IB_vtk = vtk.vtkMatrix4x4()
        self._H_IB_vtk.Identity()

    def update_vtk_actors(self, sol_i):
        contr = self.contr
        t, q = sol_i.t, sol_i.q[self.contr.qDOF]
        xi = self.xi
        A_IB = contr.A_IB(t, q[contr.local_qDOF_P(xi)], xi)
        r_OP = contr.r_OP(t, q[contr.local_qDOF_P(xi)], xi)
        self._H_IB[:3, :3] = A_IB
        self._H_IB[:3, 3] = r_OP
        self._H_IB_vtk.SetData(self._H_IB.ravel())

    def vtk_source_to_actor(
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
        tf_H_IB = vtk.vtkMatrixToLinearTransform()
        tf_H_IB.SetInput(self._H_IB_vtk)
        tf_H_IM = vtk.vtkTransform()
        tf_H_IM.PostMultiply()
        tf_H_IM.SetMatrix(H_BM.flatten())
        tf_H_IM.Concatenate(tf_H_IB)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(source.GetOutputPort())
        tf_filter.SetTransform(tf_H_IM)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        return actor


class RuntimeVisualArUco(RuntimeVisualAddOn):
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
        self.mk_size = mk_size
        self.mk_dis = mk_dis
        self.A_BM = A_BM
        self.B_r_CP = B_r_CP
        self.opacity = opacity

    def to_vtk_actors(self):
        from cv2 import aruco

        mk_size = self.mk_size
        mk_dis = self.mk_dis
        A_BM = self.A_BM
        B_r_CP = self.B_r_CP
        opacity = self.opacity
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

        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        tf_H_IB = vtk.vtkMatrixToLinearTransform()
        tf_H_IB.SetInput(self._H_IB_vtk)
        tf_H_IM = vtk.vtkTransform()
        tf_H_IM.PostMultiply()
        tf_H_IM.SetMatrix(H_BM.flatten())
        tf_H_IM.Concatenate(tf_H_IB)

        # qrcode
        actors = []
        for triangles, color in zip(
            [quads_black, quads_white], [(0, 0, 0), (255, 255, 255)]
        ):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            filter = vtk.vtkTransformPolyDataFilter()
            filter.SetInputData(polydata)
            filter.SetTransform(tf_H_IM)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetColor([c / 255 for c in color])
            actor.GetProperty().SetOpacity(opacity)
            actor.SetMapper(mapper)
            actors.append(actor)
        return actors


class RuntimeVisualSTL(RuntimeVisualAddOn):
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
        self.actor = self.vtk_source_to_actor(
            source, A_BM * scale, B_r_CP, color, opacity
        )

    def to_vtk_actors(self):
        return [self.actor]


# class LiveVisualCoordSystem(LiveVisualAddOn):
#     def __init__(
#         self,
#         contr,
#         length,
#         xi=None,
#         resolution=30,
#         A_BM=np.eye(3),
#         B_r_CP=np.zeros(3),
#         opacity=1,
#     ):
#         super().__init__(contr, xi)
#         source = vtk.vtkArrowSource()
#         source.SetTipResolution(resolution)
#         source.SetShaftResolution(resolution)
#         self.actors = []
#         for i in range(3):
#             if i == 0:
#                 color = (255, 0, 0)
#             elif i == 1:
#                 A_BM = A_BM @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#                 color = (0, 255, 0)
#             elif i == 2:
#                 A_BM = A_BM @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
#                 color = (0, 0, 255)
#             actor = self.vtk_source_to_actor(
#                 source, A_BM * length, B_r_CP, color, opacity
#             )
#             self.actors.append(actor)

#     def to_vtk_actors(self):
#         return self.actors


class Plotter:
    def __init__(self, system, window_size):
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*window_size)
        self.window.SetWindowName("Cardillo")
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.window.AddRenderer(self.ren)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.window.SetInteractor(self.interactor)

        # camera
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.ren)
        self.cam_widget.On()
        self.camera = self.ren.GetActiveCamera()
        self.camera.ParallelProjectionOff()

        self.window.AddRenderer(self.ren)

        # add runtime visual contributions
        self.__runtime_visual_contrs = []
        self.system = system
        for contr in system.contributions:
            if hasattr(contr, "to_vtk_actors"):
                self.__runtime_visual_contrs.append(contr)
                for actor in contr.to_vtk_actors():
                    self.ren.AddActor(actor)
        self._window_opened = False

        self._n_live_frame = 0
        self._live_fps = 60
        self._text_actor = vtk.vtkTextActor()
        self._text_actor.SetPosition(10, 10)
        prop = self._text_actor.GetTextProperty()
        prop.SetFontSize(20)
        prop.SetColor([i / 255 for i in (34, 136, 50)])
        self.ren.AddActor(self._text_actor)

        def cbk(interactor, event):
            if interactor.key_code == "q":
                self.window.SetOffScreenRendering(1)
                self._window_opened = False

        self.window.SetOffScreenRendering(1)
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, cbk)

        # decorate the step_callback of the system to render the solution in runtime
        def decorate_step_callback(step_callback):
            def _step_callback(t, q, u):
                ret = step_callback(t, q, u)
                if self._window_opened:
                    self.interactor.ProcessEvents()
                    if t * self._live_fps >= self._n_live_frame:
                        self.step_render(Solution(self.system, t=t, q=q, u=u))
                        self._n_live_frame += 1
                return ret

            return _step_callback

        system.step_callback = decorate_step_callback(system.step_callback)

    def add_ground(
        self,
        x0=None,
        x1=None,
        y0=None,
        y1=None,
        z0=0,
        subdivision_x=10,
        subdivision_y=10,
    ):
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(x0, y0, z0)
        plane.SetPoint1(x1, y0, z0)
        plane.SetPoint2(x0, y1, z0)
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
        for contr in self.__runtime_visual_contrs:
            contr.update_vtk_actors(sol_i)
        self.window.Render()

    def render_solution(self, solution, repeat=True, speed_up=1):
        self.window.SetOffScreenRendering(0)
        self._window_opened = True
        while True:
            t_eval = solution.t
            t0_sim = t_eval[0]
            t0_real = perf_counter()
            for sol_i in solution:
                t_real = perf_counter() - t0_real
                t_sim = (sol_i.t - t0_sim) / speed_up
                if t_sim == 0 or t_real < t_sim:
                    # wait until the real time catches up with the simulation time
                    while t_real < t_sim:
                        sleep(0.001)
                        self.interactor.ProcessEvents()
                        t_real = perf_counter() - t0_real
                    self.step_render(sol_i)
                else:
                    # skip if the rendering is too slow
                    continue
                if not self._window_opened:
                    return
            if not repeat:
                break
            else:
                sleep(solution.t[1] / speed_up)

    def live_rendering_on(self, fps=60):
        print("maximal frames per simulation time: ", fps)
        self.window.SetOffScreenRendering(0)
        self._live_fps = fps
        self._window_opened = True
