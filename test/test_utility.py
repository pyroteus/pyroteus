"""
Test utility functions.
"""
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
    mesh = uniform_mesh(dim, 4)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M = interpolate(Identity(dim), P1_ten)
    assert is_spd(M)
    M.sub(0).assign(-1)
    assert not is_spd(M)
