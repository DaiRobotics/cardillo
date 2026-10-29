from paraview.simple import *
from paraview import catalyst

# 启用 ParaView GUI 实时连接
options = catalyst.Options()
options.EnableCatalystLive = 1
options.CatalystLiveURL = "localhost:22222"

view = CreateView("RenderView")
view.ViewSize = [900, 700]
view.Background = [0.12, 0.12, 0.16]

source = Wavelet(registrationName="input")
display = Show(source, view)
display.Representation = "Surface"
ColorBy(display, ("POINTS", "temperature"))

lut = GetColorTransferFunction("temperature")
lut.ApplyPreset("Cool to Warm", True)
lut.RescaleTransferFunction(300.0, 330.0)

display.SetScalarBarVisibility(view, True)
scalar_bar = GetScalarBar(lut, view)
scalar_bar.Title = "Temperature"
scalar_bar.ComponentTitle = ""


def catalyst_execute(info):
    view.Update()
    ResetCamera(view)
    filename = "frame_%04d.png" % info.timestep
    # SaveScreenshot(filename, view, ImageResolution=[900, 700])
    print("SAVED_SCREENSHOT:", filename)
