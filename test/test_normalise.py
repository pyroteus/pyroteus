from pyroteus import *
from sensors import *
import pytest


def unit_mesh(name, mesh):
    """
    Given some sensor function,
    """
    if name == 'bowl':
        return mesh
    else:
        pytest.xfail("Need mesh adaptation capability")


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
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
    mesh = mesh_for_sensors(dim, 100)

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


def test_consistency(sensor, degree, target=1000.0):
    """
    Check that spatial normalisation and space-time
    normalisation on a single unit time interval with
    one timestep return the same metric.
    """
    dim = 2
    mesh = mesh_for_sensors(dim, 100)

    # Construct a Hessian metric
    f = sensor(*mesh.coordinates)
    H = recover_hessian(f, mesh=mesh)
    M = hessian_metric(H)
    M_st = M.copy(deepcopy=True)

    # Apply both normalisation strategies
    space_normalise(M, target, degree)
    space_time_normalise([M_st], 1.0, [1.0], target, degree)

    # Check that the metrics coincide
    assert np.isclose(errornorm(M, M_st), 0.0)
