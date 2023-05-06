from firedrake import *
from pyroteus.error_estimation import form2indicator
import unittest


class TestForm2Indicator(unittest.TestCase):
    """
    Unit tests for :func:`form2indicator`.
    """

    def setUp(self):
        self.mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(self.mesh, "CG", 1)
        self.trial = TrialFunction(self.fs)
        self.test = TestFunction(self.fs)
        self.one = Function(self.fs).assign(1)

    def test_form_type_error(self):
        with self.assertRaises(TypeError) as cm:
            form2indicator(1)
        msg = "Expected 'F' to be a Form, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_exterior_facet_integral(self):
        F = self.one * ds(1) - self.one * ds(2)
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], -2)
        self.assertAlmostEqual(indicator.dat.data[1], 2)

    def test_interior_facet_integral(self):
        F = avg(self.one) * dS
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], 2 * sqrt(2))
        self.assertAlmostEqual(indicator.dat.data[1], 2 * sqrt(2))

    def test_cell_integral(self):
        x, y = SpatialCoordinate(self.mesh)
        F = conditional(x + y < 1, 1, 0) * dx
        indicator = form2indicator(F)
        self.assertAlmostEqual(indicator.dat.data[0], 0)
        self.assertAlmostEqual(indicator.dat.data[1], 0.5)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
