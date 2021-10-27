"""
Test utility functions.
"""
from firedrake import *
from pyroteus import *
from pyroteus.metric import is_spd
from utility import uniform_mesh
import pytest


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_is_spd(dim):
    """
    Check `is_spd` correctly identifies an SPD
    and a non-SPD matrix.
    """
    mesh = uniform_mesh(dim, 4)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M = interpolate(Identity(dim), P1_ten)
    assert is_spd(M)
    M.sub(0).assign(-1)
    assert not is_spd(M)


def test_hessian_metric(dim):
    """
    Check that `hessian_metric` does not have
    an effect on a matrix that is already SPD.
    """
    mesh = uniform_mesh(dim, 4)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M = interpolate(Identity(dim), P1_ten)
    M -= hessian_metric(M)
    assert np.isclose(norm(M), 0.0)
