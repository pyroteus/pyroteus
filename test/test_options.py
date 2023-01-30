from pyroteus.options import *
import unittest


class TestAdaptParameters(unittest.TestCase):
    """
    Unit tests for the base :class:`AdaptParameters` class.
    """

    def setUp(self):
        self.defaults = {"miniter": 3, "maxiter": 35, "element_rtol": 0.001}

    def test_input(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters(1)
        msg = (
            "Expected 'parameters' keyword argument to be a dictionary, not of type"
            " 'int'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_attribute_error(self):
        with self.assertRaises(AttributeError) as cm:
            AdaptParameters({"key": "value"})
        msg = "AdaptParameters does not have 'key' attribute."
        self.assertEqual(str(cm.exception), msg)

    def test_defaults(self):
        ap = AdaptParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_access(self):
        ap = AdaptParameters()
        self.assertEqual(ap.miniter, ap["miniter"])
        self.assertEqual(ap.maxiter, ap["maxiter"])
        self.assertEqual(ap.element_rtol, ap["element_rtol"])

    def test_str(self):
        ap = AdaptParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = AdaptParameters()
        expected = "AdaptParameters(miniter=3, maxiter=35, element_rtol=0.001)"
        self.assertEqual(repr(ap), expected)

    def test_miniter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"miniter": 3.0})
        msg = "Expected attribute 'miniter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_maxiter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"maxiter": 35.0})
        msg = "Expected attribute 'maxiter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_element_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"element_rtol": "0.001"})
        msg = "Expected attribute 'element_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)


class TestMetricParameters(unittest.TestCase):
    """
    Unit tests for the :class:`MetricParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "h_min": 1.0e-30,
            "h_max": 1.0e30,
            "a_max": 1.0e30,
            "p": 1.0,
            "base_complexity": 200.0,
            "target_complexity": 4000.0,
            "num_ramp_iterations": 3,
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
        }

    def test_defaults(self):
        ap = MetricParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_str(self):
        ap = MetricParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = MetricParameters()
        expected = (
            "MetricParameters(h_min=1e-30, h_max=1e+30, a_max=1e+30, p=1.0,"
            " base_complexity=200.0, target_complexity=4000.0, num_ramp_iterations=3,"
            " miniter=3, maxiter=35, element_rtol=0.001)"
        )
        self.assertEqual(repr(ap), expected)

    def test_hmin_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"h_min": "1.0e-30"})
        msg = "Expected attribute 'h_min' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_hmax_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"h_max": "1.0e30"})
        msg = "Expected attribute 'h_max' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_amax_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"a_max": "1.0e30"})
        msg = "Expected attribute 'a_max' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_p_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"p": "1.0"})
        msg = "Expected attribute 'p' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_base_complexity_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"base_complexity": "200.0"})
        msg = "Expected attribute 'base_complexity' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_target_complexity_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"target_complexity": "4000.0"})
        msg = "Expected attribute 'target_complexity' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_ramp_iter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"num_ramp_iterations": 3.0})
        msg = (
            "Expected attribute 'num_ramp_iterations' to be of type 'int', not"
            " 'float'."
        )
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
