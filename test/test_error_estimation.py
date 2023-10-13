from firedrake import *
from goalie.error_estimation import (
    form2indicator,
    indicators2estimator,
    get_dwr_indicator,
)
from goalie.time_partition import TimeInstant, TimePartition
from parameterized import parameterized
import unittest


class ErrorEstimationTestCase(unittest.TestCase):
    """
    Base class for error estimation testing.
    """

    def setUp(self):
        self.mesh = UnitSquareMesh(1, 1)
        self.fs = FunctionSpace(self.mesh, "CG", 1)
        self.trial = TrialFunction(self.fs)
        self.test = TestFunction(self.fs)
        self.one = Function(self.fs, name="Uno")
        self.one.assign(1)


class TestForm2Indicator(ErrorEstimationTestCase):
    """
    Unit tests for :func:`form2indicator`.
    """

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
        self.assertAlmostEqual(indicator.dat.data[1], 1)


class TestIndicators2Estimator(ErrorEstimationTestCase):
    """
    Unit tests for :func:`indicators2estimator`.
    """

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

    @parameterized.expand([[False], [True]])
    def test_unit_time_instant_abs(self, absolute_value):
        time_instant = TimeInstant("field", time=1.0)
        indicator = form2indicator(-self.one * dx)
        estimator = indicators2estimator(
            {"field": [[indicator]]}, time_instant, absolute_value=absolute_value
        )
        self.assertAlmostEqual(estimator, 1.0 if absolute_value else -1.0)

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


class TestGetDWRIndicator(ErrorEstimationTestCase):
    """
    Unit tests for :func:`get_dwr_indicator`.
    """

    def setUp(self):
        super().setUp()
        self.two = Function(self.fs, name="Dos")
        self.two.assign(2)
        self.F = self.one * self.test * dx

    def test_form_type_error(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.one, self.one)
        msg = "Expected 'F' to be a Form, not '<class 'firedrake.function.Function'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_adjoint_error_type_error(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, 1)
        msg = "Expected 'adjoint_error' to be a Function or dict, not '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)

    def test_test_space_type_error1(self):
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, self.one, test_space=self.one)
        msg = (
            "Expected 'test_space' to be a FunctionSpace or dict,"
            " not '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_input_error(self):
        adjoint_error = {"field1": self.one, "field2": self.one}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=self.fs)
        msg = "Inconsistent input for 'adjoint_error' and 'test_space'."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_test_space_error(self):
        adjoint_error = {"field": self.one}
        test_space = {"f": self.fs}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        msg = "Key 'field' does not exist in the test space provided."
        self.assertEqual(str(cm.exception), msg)

    def test_test_space_type_error2(self):
        adjoint_error = {"field": self.one}
        test_space = {"field": self.one}
        with self.assertRaises(TypeError) as cm:
            get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        msg = (
            "Expected 'test_space['field']' to be a FunctionSpace,"
            " not '<class 'firedrake.function.Function'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_mesh_error1(self):
        adjoint_error = Function(FunctionSpace(UnitTriangleMesh(), "CG", 1))
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, adjoint_error)
        msg = "Meshes underlying the form and adjoint error do not match."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_mesh_error2(self):
        test_space = FunctionSpace(UnitTriangleMesh(), "CG", 1)
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, self.one, test_space=test_space)
        msg = "Meshes underlying the form and test space do not match."
        self.assertEqual(str(cm.exception), msg)

    def test_convert_neither(self):
        adjoint_error = {"field": self.two}
        test_space = {"field": self.one.function_space()}
        indicator = get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_both(self):
        test_space = self.one.function_space()
        indicator = get_dwr_indicator(self.F, self.two, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_test_space(self):
        adjoint_error = {"field": self.two}
        test_space = self.one.function_space()
        indicator = get_dwr_indicator(self.F, adjoint_error, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error(self):
        test_space = {"Dos": self.one.function_space()}
        indicator = get_dwr_indicator(self.F, self.two, test_space=test_space)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error_no_test_space(self):
        indicator = get_dwr_indicator(self.F, self.two)
        self.assertAlmostEqual(indicator.dat.data[0], 1)
        self.assertAlmostEqual(indicator.dat.data[1], 1)

    def test_convert_adjoint_error_mismatch(self):
        test_space = {"field": self.one.function_space()}
        with self.assertRaises(ValueError) as cm:
            get_dwr_indicator(self.F, self.two, test_space=test_space)
        msg = "Key 'Dos' does not exist in the test space provided."
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover
