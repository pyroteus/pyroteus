from firedrake import *
from pyroteus.error_estimation import form2indicator, indicators2estimator
from pyroteus.time_partition import TimeInstant, TimePartition
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


class TestIndicators2Estimator(unittest.TestCase):
    """
    Unit tests for :func:`indicators2estimator`.
    """

    def setUp(self):
        self.mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(self.mesh, "CG", 1)
        self.trial = TrialFunction(self.fs)
        self.test = TestFunction(self.fs)
        self.one = Function(self.fs).assign(1)

    def test_indicators_type_error1(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator(self.one, time_instant)
        msg = (
            "Expected 'indicators' to be a dict, not"
            " '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_indicators_type_error2(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator({"field": self.one}, time_instant)
        msg = (
            "Expected values of 'indicators' to be iterables, not"
            " '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_indicators_type_error3(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator({"field": 1}, time_instant)
        msg = "Expected values of 'indicators' to be iterables, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_indicators_type_error4(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator({"field": [self.one]}, time_instant)
        msg = (
            "Expected entries of 'indicators' to be iterables, not"
            " '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_indicators_type_error5(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator({"field": [1]}, time_instant)
        msg = "Expected entries of 'indicators' to be iterables, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_time_partition_type_error(self):
        with self.assertRaises(TypeError) as cm:
            indicators2estimator({"field": [self.one]}, 1.0)
        msg = "Expected 'time_partition' to be a TimePartition, not '<class 'float'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_time_partition_wrong_field_error(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(ValueError) as cm:
            indicators2estimator({"f": [[self.one]]}, time_instant)
        msg = "Key 'f' does not exist in the TimePartition provided."
        self.assertEqual(str(cm.exception), msg)

    def test_absolute_value_type_error(self):
        time_instant = TimeInstant("field")
        with self.assertRaises(TypeError) as cm:
            indicators2estimator(
                {"field": [[self.one]]}, time_instant, absolute_value=0
            )
        msg = "Expected 'absolute_value' to be a bool, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_unit_time_instant(self):
        time_instant = TimeInstant("field", time=1.0)
        indicator = form2indicator(self.one * dx)
        estimator = indicators2estimator({"field": [[indicator]]}, time_instant)
        self.assertAlmostEqual(estimator, 1.0)

    def test_half_time_instant(self):
        time_instant = TimeInstant("field", time=0.5)
        indicator = form2indicator(self.one * dx)
        estimator = indicators2estimator({"field": [[indicator]]}, time_instant)
        self.assertAlmostEqual(estimator, 0.5)

    def test_time_partition_same_timestep(self):
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        indicator = form2indicator(self.one * dx)
        estimator = indicators2estimator({"field": [2 * [indicator]]}, time_partition)
        self.assertAlmostEqual(estimator, 1.0)

    def test_time_partition_different_timesteps(self):
        time_partition = TimePartition(1.0, 2, [0.5, 0.25], ["field"])
        indicator = form2indicator(self.one * dx)
        estimator = indicators2estimator(
            {"field": [[indicator], 2 * [indicator]]}, time_partition
        )
        self.assertAlmostEqual(estimator, 1.0)

    def test_time_instant_multiple_fields(self):
        time_instant = TimeInstant(["field1", "field2"], time=1.0)
        indicator = form2indicator(self.one * dx)
        estimator = indicators2estimator(
            {"field1": [[indicator]], "field2": [[indicator]]}, time_instant
        )
        self.assertAlmostEqual(estimator, 2.0)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
