from pyroteus.math import bessk0, bessi0, recursive_polynomial, gram_schmidt
import numpy as np
from parameterized import parameterized
import pytest
import scipy as sp
import unittest


class TestBessel(unittest.TestCase):
    """
    Unit tests for Bessel functions.
    """

    @parameterized.expand([1, 2, 3, (np.array([1, 2]),)])
    def test_recursive_polynomial0(self, a):
        x0 = 1
        p = recursive_polynomial(a, (x0,))
        self.assertEqual(p, x0)

    @parameterized.expand([1, 2, 3, (np.array([1, 2]),)])
    def test_recursive_polynomial1(self, a):
        x0, x1 = 1, -1
        p = recursive_polynomial(a, (x0, x1))
        self.assertTrue(np.allclose(p, x0 + a * x1))

    @parameterized.expand([1, 2, 3, (np.array([1, 2]),)])
    def test_recursive_polynomial2(self, a):
        x0, x1, x2 = 1, -1, 1
        p = recursive_polynomial(a, (x0, x1, x2))
        self.assertTrue(np.allclose(p, x0 + a * (x1 + a * x2)))

    def test_bessi0_numpy_divbyzero_error(self):
        with self.assertRaises(ValueError) as cm:
            bessi0(np.array([1, 0]))
        msg = "Cannot divide by zero."
        self.assertEqual(str(cm.exception), msg)

    def test_bessi0_numpy(self):
        x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(bessi0(x), sp.special.i0(x)))

    def test_bessk0_numpy_nonpositive_error(self):
        with self.assertRaises(ValueError) as cm:
            bessk0(np.array([1, -1]))
        msg = "Cannot take the logarithm of a non-positive number."
        self.assertEqual(str(cm.exception), msg)

    def test_bessk0_numpy(self):
        x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(bessk0(x), sp.special.k0(x)))


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
