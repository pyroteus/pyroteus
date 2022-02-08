import firedrake
from pyroteus.math import gram_schmidt
from pyroteus.metric import metric_exponential, metric_logarithm
import ufl
from utility import uniform_mesh
import numpy as np
import pytest


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False])
def normalise(request):
    return request.param


def test_gram_schmidt_numpy(dim, normalise):
    """
    Apply Gram-Schmidt to a random set of
    vectors and check that the result is
    an orthogonal/orthonormal set.
    """
    np.random.seed(0)
    v = [np.random.rand(dim) for i in range(dim)]
    u = gram_schmidt(*v, normalise=True)
    for i, ui in enumerate(u):
        for j, uj in enumerate(u):
            if not normalise and i == j:
                continue
            assert np.isclose(np.dot(ui, uj), 1.0 if i == j else 0.0)


def test_metric_math(dim):
    """
    Check that the metric exponential and
    metric logarithm are indeed inverses.
    """
    mesh = uniform_mesh(dim, 1)
    P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
    I = ufl.Identity(dim)
    M = firedrake.interpolate(2 * I, P0_ten)
    logM = metric_logarithm(M)
    expected = firedrake.interpolate(np.log(2) * I, P0_ten)
    assert np.allclose(logM.dat.data, expected.dat.data)
    M_ = metric_exponential(logM)
    assert np.allclose(M.dat.data, M_.dat.data)
    expM = metric_exponential(M)
    expected = firedrake.interpolate(np.exp(2) * I, P0_ten)
    assert np.allclose(expM.dat.data, expected.dat.data)
    M_ = metric_logarithm(expM)
    assert np.allclose(M.dat.data, M_.dat.data)
