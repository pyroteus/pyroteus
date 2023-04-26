from pyroteus.math import gram_schmidt
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
