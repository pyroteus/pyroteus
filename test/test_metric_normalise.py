"""
Test metric normalisation functionality.
"""
from firedrake import *
from pyroteus import *
from sensors import *
import pytest


@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=[1, 2, "inf"])
def degree(request):
    return request.param


@pytest.mark.slow
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
    pytest.skip("TO DO")  # TODO: apply normalisation to M with target and degree
    space_time_normalise([M_st], 1.0, [1.0], target, degree)

    # Check that the metrics coincide
    assert np.isclose(errornorm(M, M_st), 0.0)
