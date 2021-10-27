"""
Test metric normalisation functionality.
"""
from firedrake import *
from pyroteus import *
from sensors import *
import pytest


def unit_mesh(sensor, mesh, target, degree, num_iterations=3):
    """
    Obtain a quasi-unit mesh with respect to the
    Hessian of a given ``sensor`` function, subject
    to a target metric complexity ``target`` under
    ``degree`` for :math:`L^p` normalisation,
    starting from some input ``mesh``.

    :kwarg num_iterations: how many times should
        mesh adaptation be applied?
    """
    try:
        from firedrake import adapt
    except ImportError:
        pytest.xfail("Need mesh adaptation capability")
    for i in range(num_iterations):
        f = sensor(*mesh.coordinates)
        H = recover_hessian(f, mesh=mesh)
        M = hessian_metric(H)
        space_normalise(M, target, degree)
        mesh = adapt(mesh, M)
    return mesh


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
    mesh = unit_mesh(sensor, mesh, target, degree)

    # Construct a space-normalised Hessian metric
    f = sensor(*mesh.coordinates)
    H = recover_hessian(f, mesh=mesh)
    M = hessian_metric(H)
    space_normalise(M, target, degree)

    # Check that the target metric complexity is (approximately) attained
    if degree == 'inf':
        assert np.isclose(metric_complexity(M), target)
    else:
        assert abs(metric_complexity(M) - target) < 0.1*target


@pytest.mark.xfail("Adaptation does not currently work in parallel")
@pytest.mark.parallel
def test_space_normalise_parallel(sensor, degree, target=1000.0):
    test_space_normalise(sensor, degree, target=target)


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


@pytest.mark.parallel
def test_consistency_parallel(sensor, degree, target=1000.0):
    test_consistency(sensor, degree, target=target)
