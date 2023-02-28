"""
Test metric normalisation functionality.
"""
from firedrake import *
from firedrake.meshadapt import RiemannianMetric
from pyroteus import *
from sensors import *
import pytest


@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=[1, 2, np.inf])
def degree(request):
    return request.param


@pytest.mark.slow
def test_consistency(sensor, degree, target=1000.0):
    """
    Check that spatial normalisation and space-time
    normalisation on a single unit time interval with
    one timestep return the same metric.
    """
    mesh = mesh_for_sensors(2, 100)
    metric_parameters = {
        "dm_plex_metric_p": degree,
        "dm_plex_metric_target_complexity": target,
    }

    # Construct a Hessian metric
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M = RiemannianMetric(P1_ten)
    M.compute_hessian(sensor(*mesh.coordinates))
    assert not np.isnan(M.dat.data).any()
    M_st = M.copy(deepcopy=True)

    # Apply both normalisation strategies
    M.set_parameters(metric_parameters)
    M.normalise()
    space_time_normalise([M_st], 1.0, [1.0], metric_parameters)
    assert not np.isnan(M_st.dat.data).any()

    # Check that the metrics coincide
    assert np.isclose(errornorm(M, M_st), 0.0)
