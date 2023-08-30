from firedrake import UnitTriangleMesh
from goalie.math import *
import numpy as np
from parameterized import parameterized
import scipy as sp
import ufl
import unittest


class TestBessel(unittest.TestCase):
    """
    Unit tests for Bessel functions.
    """

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
