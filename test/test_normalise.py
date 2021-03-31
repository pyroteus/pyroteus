from pyroteus import *
from sensors import *
from utility import uniform_mesh
import pytest


def unit_mesh(name, mesh):
    """
    Given some sensor function,
    """
    if name == 'bowl':
        return mesh
    else:
        raise NotImplementedError("""
            In order to progress to the other
            sensor functions, we first need
            mesh adaptation capability.
            """)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl])
def sensor(request):
    return request.param


@pytest.fixture(params=[1, 2, 'inf'])
def degree(request):
    return request.param


def test_space_normalise(sensor, degree, target=1000.0):
    """
    Check that normalising a metric in space enables
    the attainment of a target metric complexity.

    Note that we should only expect this to be true
    if the underlying mesh is unit w.r.t. the metric.
    """
    dim = 2
    mesh = uniform_mesh(dim, 100, 2)
    coords = Function(mesh.coordinates)
    coords -= 1.0
    mesh = Mesh(coords)

    # Get a unit mesh for the sensor
    f = sensor(*mesh.coordinates)
    mesh = unit_mesh(sensor.__name__, mesh)

    # Construct a space-normalised Hessian metric
    f = sensor(*mesh.coordinates)
    H = recover_hessian(f, mesh=mesh)
    M = hessian_metric(H)
    space_normalise(M, target, degree)

    # Check that the target metric complexity is attained
    assert np.isclose(metric_complexity(M), target)


# def test_space_time_normalise(sensor):
#     pytest.xfail("TO DO")  # TODO
