"""
Test utility functions.
"""
from firedrake import *
from firedrake.norms import norm as fnorm
from firedrake.norms import errornorm as ferrnorm
from pyroteus import *
from utility import uniform_mesh
import os
import pathlib
from parameterized import parameterized
import unittest


pointwise_norm_types = [["l1"], ["l2"], ["linf"]]
integral_scalar_norm_types = [["L1"], ["L2"], ["L4"], ["H1"], ["HCurl"]]
scalar_norm_types = pointwise_norm_types + integral_scalar_norm_types

# ---------------------------
# standard tests for pytest
# ---------------------------


class TestPVD(unittest.TestCase):
    """
    Test the subclass of Firedrake's :class:`File`.
    """

    def setUp(self):
        mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(mesh, "CG", 1)
        pwd = os.path.dirname(__file__)
        self.fname = os.path.join(pwd, "tmp.pvd")
        self.cleanUp()

    def cleanUp(self):
        if os.path.exists(self.fname):
            os.remove(self.fname)

    def test_adaptive(self):
        file = File(self.fname)
        self.assertTrue(os.path.exists(self.fname))
        self.assertTrue(file._adaptive)
        self.cleanUp()

    def test_different_fnames(self):
        f = Function(self.fs, name="f")
        g = Function(self.fs, name="g")
        file = File(self.fname)
        file.write(f)
        file.write(g)
        self.assertEqual("f", g.name())
        self.cleanUp()

    def test_different_lengths(self):
        f = Function(self.fs, name="f")
        g = Function(self.fs, name="g")
        file = File(self.fname)
        file.write(f)
        with self.assertRaises(ValueError) as cm:
            file.write(f, g)
        msg = "Writing different set of functions"
        self.assertEqual(str(cm.exception), msg)
        self.cleanUp()


class TestMassMatrix(unittest.TestCase):
    """
    Unit tests for :func:`assemble_mass_matrix`.
    """

    @parameterized.expand([["L2"], ["H1"]])
    def test_tiny(self, norm_type):
        mesh = uniform_mesh(2, 1)
        V = FunctionSpace(mesh, "DG", 0)
        M = assemble_mass_matrix(V, norm_type=norm_type)
        expected = np.array([[0.5, 0], [0, 0.5]])
        got = M.convert("dense").getDenseArray()
        self.assertTrue(np.allclose(expected, got))

    def test_norm_type_error(self):
        mesh = uniform_mesh(2, 1)
        V = FunctionSpace(mesh, "DG", 0)
        with self.assertRaises(ValueError) as cm:
            assemble_mass_matrix(V, norm_type="HDiv")
        msg = "Norm type 'HDiv' not recognised."
        self.assertEqual(str(cm.exception), msg)


class TestNorm(unittest.TestCase):
    """
    Unit tests for :func:`norm`.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 4)
        self.x, self.y = SpatialCoordinate(self.mesh)
        V = FunctionSpace(self.mesh, "CG", 1)
        self.f = interpolate(self.x**2 + self.y, V)

    def test_boundary_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            norm(self.f, norm_type="l1", boundary=True)
        msg = "lp errors on the boundary not yet implemented."
        self.assertEqual(str(cm.exception), msg)

    def test_l1(self):
        expected = np.sum(np.abs(self.f.dat.data))
        got = norm(self.f, norm_type="l1")
        self.assertAlmostEqual(expected, got)

    def test_l2(self):
        expected = np.sqrt(np.sum(self.f.dat.data**2))
        got = norm(self.f, norm_type="l2")
        self.assertAlmostEqual(expected, got)

    def test_linf(self):
        expected = np.max(self.f.dat.data)
        got = norm(self.f, norm_type="linf")
        self.assertAlmostEqual(expected, got)

    def test_Linf(self):
        expected = np.max(self.f.dat.data)
        got = norm(self.f, norm_type="Linf")
        self.assertAlmostEqual(expected, got)

    def test_notimplemented_lp_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            norm(self.f, norm_type="lp")
        msg = "lp norm of order p not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_L0_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="L0")
        msg = "'L0' norm does not make sense."
        self.assertEqual(str(cm.exception), msg)

    def test_notimplemented_Lp_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="Lp")
        msg = "Don't know how to interpret 'Lp' norm."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(integral_scalar_norm_types)
    def test_consistency_firedrake(self, norm_type):
        expected = fnorm(self.f, norm_type=norm_type)
        got = norm(self.f, norm_type=norm_type)
        self.assertAlmostEqual(expected, got)

    def test_consistency_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = Function(V)
        f.interpolate(as_vector([y * y, -x * x]))
        expected = fnorm(f, norm_type="HDiv")
        got = norm(f, norm_type="HDiv")
        self.assertAlmostEqual(expected, got)

    def test_invalid_norm_type_error(self):
        with self.assertRaises(ValueError) as cm:
            norm(self.f, norm_type="X")
        msg = "Unknown norm type 'X'."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([("L1", 0.25), ("L2", 0.5)])
    def test_condition_integral(self, norm_type, expected):
        self.f.assign(1.0)
        condition = conditional(And(self.x < 0.5, self.y < 0.5), 1.0, 0.0)
        val = norm(self.f, norm_type=norm_type, condition=condition)
        self.assertAlmostEqual(val, expected)


class TestErrorNorm(unittest.TestCase):
    """
    Unit tests for :func:`errornorm`.
    """

    def setUp(self):
        self.mesh = uniform_mesh(2, 4)
        self.x, self.y = SpatialCoordinate(self.mesh)
        V = FunctionSpace(self.mesh, "CG", 1)
        self.f = interpolate(self.x**2 + self.y, V)
        self.g = interpolate(self.x + self.y**2, V)

    def test_shape_error(self):
        with self.assertRaises(RuntimeError) as cm:
            errornorm(self.f, self.mesh.coordinates)
        msg = "Mismatching rank between u and uh."
        self.assertEqual(str(cm.exception), msg)

    def test_not_function_error(self):
        with self.assertRaises(TypeError) as cm:
            errornorm(self.f, Constant(1.0))
        msg = "uh should be a Function, is a Constant."
        self.assertEqual(str(cm.exception), msg)

    def test_not_function_error_lp(self):
        with self.assertRaises(TypeError) as cm:
            errornorm(Constant(1.0), self.f, norm_type="l1")
        msg = "u should be a Function, is a Constant."
        self.assertEqual(str(cm.exception), msg)

    def test_mixed_space_invalid_norm_error(self):
        V = self.f.function_space() * self.f.function_space()
        with self.assertRaises(NotImplementedError) as cm:
            errornorm(Function(V), Function(V), norm_type="L1")
        msg = "Norm type 'L1' not supported for mixed spaces."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(scalar_norm_types)
    def test_zero_scalar(self, norm_type):
        err = errornorm(self.f, self.f, norm_type=norm_type)
        self.assertAlmostEqual(err, 0.0)

    def test_zero_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = interpolate(as_vector([y * y, -x * x]), V)
        err = errornorm(f, f, norm_type="HDiv")
        self.assertAlmostEqual(err, 0.0)

    @parameterized.expand(integral_scalar_norm_types)
    def test_consistency_firedrake(self, norm_type):
        expected = ferrnorm(self.f, self.g, norm_type=norm_type)
        got = errornorm(self.f, self.g, norm_type=norm_type)
        self.assertAlmostEqual(expected, got)

    def test_consistency_hdiv(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        x, y = SpatialCoordinate(self.mesh)
        f = interpolate(as_vector([y * y, -x * x]), V)
        g = interpolate(as_vector([x * y, 1.0]), V)
        expected = ferrnorm(f, g, norm_type="HDiv")
        got = errornorm(f, g, norm_type="HDiv")
        self.assertAlmostEqual(expected, got)

    @parameterized.expand([("L1", 0.25), ("L2", 0.5)])
    def test_condition_integral(self, norm_type, expected):
        self.f.assign(1.0)
        self.g.assign(0.0)
        condition = conditional(And(self.x < 0.5, self.y < 0.5), 1.0, 0.0)
        val = errornorm(self.f, self.g, norm_type=norm_type, condition=condition)
        self.assertAlmostEqual(val, expected)


class TestEffectivityIndex(unittest.TestCase):
    """
    Unit tests for :func:`effectivity_index`.
    """

    def test_error_indicator_not_function_error(self):
        with self.assertRaises(ValueError) as cm:
            effectivity_index(1.0, 1.0)
        msg = "Error indicator must return a Function."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_degree(self):
        mesh = uniform_mesh(2, 1)
        ei = Function(FunctionSpace(mesh, "DG", 1))
        with self.assertRaises(ValueError) as cm:
            effectivity_index(ei, 1.0)
        msg = "Error indicator must be P0."
        self.assertEqual(str(cm.exception), msg)

    def test_accuracy(self):
        mesh = uniform_mesh(2, 1)  # Two elements
        ei = Function(FunctionSpace(mesh, "DG", 0))
        ei.assign(1.0)
        self.assertAlmostEqual(effectivity_index(ei, 1.0), 2.0)


def test_create_directory():
    """
    Test that :func:`create_directory` works as expected.
    """
    pwd = os.path.dirname(__file__)
    tmp = create_directory(os.path.join(pwd, "tmp"))
    assert os.path.exists(tmp)
    pathlib.Path(tmp).rmdir()


if __name__ == "__main__":
    unittest.main()
