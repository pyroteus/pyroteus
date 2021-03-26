"""
Test metric combination drivers.
"""
from pyroteus import *
import pytest


def uniform_mesh(dim, n):
    return UnitSquareMesh(n, n) if dim == 2 else UnitCubeMesh(n, n, n)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_intersection(dim):
    """
    Check that metric intersection DTRT when
    applied to two isotropic metrics.
    """
    mesh = uniform_mesh(dim, 3)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    I = Identity(dim)
    M1 = interpolate(2*I, P1_ten)
    M2 = interpolate(1*I, P1_ten)
    M = metric_intersection(M1, M2)
    assert np.isclose(norm(Function(M).assign(M - M1)), 0.0)
    M2.interpolate(2*I)
    M = metric_intersection(M1, M2)
    assert np.isclose(norm(Function(M).assign(M - M1)), 0.0)
    assert np.isclose(norm(Function(M).assign(M - M2)), 0.0)
    M2.interpolate(4*I)
    M = metric_intersection(M1, M2)
    assert np.isclose(norm(Function(M).assign(M - M2)), 0.0)
