import numpy as np
import catalyst
from catalyst_conduit import Node


def run_catalyst_step(
    points,
    cells,
    temperature,
    velocity,
    timestep,
    time,
):
    """将 NumPy 四面体网格直接传给 Catalyst, 不写中间 VTK 文件。"""

    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int32)
    temperature = np.asarray(temperature, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    npoints = points.shape[0]

    assert points.shape == (npoints, 3)
    assert cells.ndim == 2 and cells.shape[1] == 4
    assert temperature.shape == (npoints,)
    assert velocity.shape == (npoints, 3)
    assert np.all(cells >= 0) and np.all(cells < npoints)

    mesh = Node()

    mesh["coordsets/coords/type"] = "explicit"
    mesh["coordsets/coords/values/x"] = points[:, 0].copy()
    mesh["coordsets/coords/values/y"] = points[:, 1].copy()
    mesh["coordsets/coords/values/z"] = points[:, 2].copy()

    mesh["topologies/mesh/type"] = "unstructured"
    mesh["topologies/mesh/coordset"] = "coords"
    mesh["topologies/mesh/elements/shape"] = "tet"
    mesh["topologies/mesh/elements/connectivity"] = cells.reshape(-1).copy()

    mesh["fields/temperature/association"] = "vertex"
    mesh["fields/temperature/topology"] = "mesh"
    mesh["fields/temperature/values"] = temperature.copy()

    mesh["fields/velocity/association"] = "vertex"
    mesh["fields/velocity/topology"] = "mesh"
    mesh["fields/velocity/values/x"] = velocity[:, 0].copy()
    mesh["fields/velocity/values/y"] = velocity[:, 1].copy()
    mesh["fields/velocity/values/z"] = velocity[:, 2].copy()

    node = Node()
    node["catalyst/state/cycle"] = int(timestep)
    node["catalyst/state/time"] = float(time)
    node["catalyst/channels/input/type"] = "mesh"
    node["catalyst/channels/input/data"].set(mesh)

    catalyst.execute(node)
