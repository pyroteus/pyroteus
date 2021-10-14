"""
Test interpolation schemes.
"""
from pyroteus import *
import pytest


@pytest.fixture(params=['scalar', 'vector', 'tensor'])
def shape(request):
    return request.param


def test_clement_interpolant(shape):
    """
    Check that volume averaging over two
    elements gives the average at vertices
    on the shared edge and the element-wise
    values elsewhere.

    :arg shape: choose from 'scalar', 'vector', 'tensor'
    """
    mesh = UnitSquareMesh(1, 1)
    if shape == 'scalar':
        fs = FunctionSpace
    elif shape == 'vector':
        fs = VectorFunctionSpace
    elif shape == 'tensor':
        fs = TensorFunctionSpace
    else:
        raise ValueError(f"Shape '{shape} not recognised.")
    P0 = fs(mesh, "DG", 0)
    source = Function(P0)
    source.dat.data[0] += 1.0
    target = clement_interpolant(source)
    assert np.allclose(target.dat.data[0], 0.5)
    assert np.allclose(target.dat.data[1], 1.0)
    assert np.allclose(target.dat.data[2], 0.5)
    assert np.allclose(target.dat.data[3], 0.0)
