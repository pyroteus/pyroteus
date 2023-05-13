from firedrake import UnitTriangleMesh
from pyroteus.math import (
    bessk0,
    bessi0,
    construct_basis,
    gram_schmidt,
    recursive_polynomial,
)
import numpy as np
from parameterized import parameterized
import scipy as sp
import ufl
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

    def test_bessi0_ufl_type_error(self):
        with self.assertRaises(TypeError) as cm:
            bessi0(UnitTriangleMesh())
        msg = "Expected UFL Expr, not '<class 'firedrake.mesh.MeshGeometry'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_bessi0_numpy(self):
        x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(bessi0(x), sp.special.i0(x)))

    def test_bessk0_numpy_nonpositive_error(self):
        with self.assertRaises(ValueError) as cm:
            bessk0(np.array([1, -1]))
        msg = "Cannot take the logarithm of a non-positive number."
        self.assertEqual(str(cm.exception), msg)

    def test_bessk0_ufl_type_error(self):
        with self.assertRaises(TypeError) as cm:
            bessk0(UnitTriangleMesh())
        msg = "Expected UFL Expr, not '<class 'firedrake.mesh.MeshGeometry'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_bessk0_numpy(self):
        x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(bessk0(x), sp.special.k0(x)))


class TestOrthogonalisation(unittest.TestCase):
    """
    Unit tests for orthogonalisation.
    """

    def setUp(self):
        np.random.seed(0)

    def test_gram_schmidt_type_error_numpy(self):
        with self.assertRaises(TypeError) as cm:
            gram_schmidt(np.ones(2), [1, 2])
        msg = (
            "Inconsistent vector types:"
            " '<class 'numpy.ndarray'>' vs. '<class 'list'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_gram_schmidt_type_error_ufl(self):
        x = ufl.SpatialCoordinate(UnitTriangleMesh())
        with self.assertRaises(TypeError) as cm:
            gram_schmidt(x, [1, 2])
        msg = (
            "Inconsistent vector types:"
            " '<class 'ufl.core.expr.Expr'>' vs. '<class 'list'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([2, 3])
    def test_gram_schmidt_orthonormal_numpy(self, dim):
        v = np.random.rand(dim, dim)
        u = np.array(gram_schmidt(*v, normalise=True))
        self.assertTrue(np.allclose(u.transpose() @ u, np.eye(dim)))

    @parameterized.expand([2, 3])
    def test_gram_schmidt_nonorthonormal_numpy(self, dim):
        v = np.random.rand(dim, dim)
        u = gram_schmidt(*v, normalise=False)
        for i, ui in enumerate(u):
            for j, uj in enumerate(u):
                if i != j:
                    self.assertAlmostEqual(np.dot(ui, uj), 0)

    def test_basis_shape_error_numpy(self):
        with self.assertRaises(ValueError) as cm:
            construct_basis(np.ones((1, 2)))
        msg = "Expected a vector, got an array of shape (1, 2)."
        self.assertEqual(str(cm.exception), msg)

    def test_basis_dim_error_numpy(self):
        with self.assertRaises(ValueError) as cm:
            construct_basis(np.ones(0))
        msg = "Dimension 0 not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_basis_ufl_type_error(self):
        with self.assertRaises(TypeError) as cm:
            construct_basis(UnitTriangleMesh())
        msg = "Expected UFL Expr, not '<class 'firedrake.mesh.MeshGeometry'>'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([2, 3])
    def test_basis_orthonormal_numpy(self, dim):
        u = np.array(construct_basis(np.random.rand(dim), normalise=True))
        self.assertTrue(np.allclose(u.transpose() @ u, np.eye(dim)))

    @parameterized.expand([2, 3])
    def test_basis_orthogonal_numpy(self, dim):
        u = construct_basis(np.random.rand(dim), normalise=False)
        for i, ui in enumerate(u):
            for j, uj in enumerate(u):
                if i != j:
                    self.assertAlmostEqual(np.dot(ui, uj), 0)
