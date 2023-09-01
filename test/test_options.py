from firedrake import TensorFunctionSpace
from firedrake.meshadapt import RiemannianMetric
from goalie.options import *
from utility import uniform_mesh
import unittest


class TestAdaptParameters(unittest.TestCase):
    """
    Unit tests for the base :class:`AdaptParameters` class.
    """

    def setUp(self):
        self.defaults = {"miniter": 3, "maxiter": 35, "element_rtol": 0.001, "rigorous": False}

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
        self.assertEqual(ap.rigorous, ap["rigorous"])

    def test_str(self):
        ap = AdaptParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = AdaptParameters()
        expected = "AdaptParameters(miniter=3, maxiter=35, element_rtol=0.001, rigorous=False)"
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

    def test_rigorous_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"rigorous": 0})
        msg = "Expected attribute 'rigorous' to be of type 'bool', not 'int'."
        self.assertEqual(str(cm.exception), msg)


class TestMetricParameters(unittest.TestCase):
    """
    Unit tests for the :class:`MetricParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "num_ramp_iterations": 3,
            "verbosity": -1,
            "p": 1.0,
            "base_complexity": 200.0,
            "target_complexity": 4000.0,
            "h_min": 1.0e-30,
            "h_max": 1.0e30,
            "a_max": 1.0e5,
            "restrict_anisotropy_first": False,
            "hausdorff_number": 0.01,
            "gradation_factor": 1.3,
            "no_insert": False,
            "no_swap": False,
            "no_move": False,
            "no_surf": False,
            "num_parmmg_iterations": 3,
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
            "rigorous": False,
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
            "MetricParameters(num_ramp_iterations=3, verbosity=-1, p=1.0,"
            " base_complexity=200.0, target_complexity=4000.0, h_min=1e-30,"
            " h_max=1e+30, a_max=100000.0, restrict_anisotropy_first=False,"
            " hausdorff_number=0.01, gradation_factor=1.3,"
            " no_insert=False, no_swap=False, no_move=False, no_surf=False,"
            " num_parmmg_iterations=3, miniter=3, maxiter=35, element_rtol=0.001,"
            " rigorous=False)"
        )
        self.assertEqual(repr(ap), expected)

    def test_ramp_iter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters({"num_ramp_iterations": 3.0})
        msg = (
            "Expected attribute 'num_ramp_iterations' to be of type 'int', not"
            " 'float'."
        )
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

    def test_export(self):
        mp = MetricParameters()
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        metric = RiemannianMetric(P1_ten)
        mp.export(metric)
        plex = metric._plex
        self.assertEqual(self.defaults["verbosity"], plex.metricGetVerbosity())
        self.assertEqual(self.defaults["p"], plex.metricGetNormalizationOrder())
        self.assertEqual(
            self.defaults["target_complexity"], plex.metricGetTargetComplexity()
        )
        self.assertEqual(self.defaults["h_min"], plex.metricGetMinimumMagnitude())
        self.assertEqual(self.defaults["h_max"], plex.metricGetMaximumMagnitude())
        self.assertEqual(self.defaults["a_max"], plex.metricGetMaximumAnisotropy())
        self.assertEqual(
            self.defaults["hausdorff_number"], plex.metricGetHausdorffNumber()
        )
        self.assertEqual(
            self.defaults["gradation_factor"], plex.metricGetGradationFactor()
        )
        self.assertEqual(
            self.defaults["restrict_anisotropy_first"],
            plex.metricRestrictAnisotropyFirst(),
        )
        self.assertEqual(self.defaults["no_insert"], plex.metricNoInsertion())
        self.assertEqual(self.defaults["no_swap"], plex.metricNoSwapping())
        self.assertEqual(self.defaults["no_move"], plex.metricNoMovement())
        self.assertEqual(self.defaults["no_surf"], plex.metricNoSurf())
        self.assertEqual(
            self.defaults["num_parmmg_iterations"], plex.metricGetNumIterations()
        )

    def test_export_type_error(self):
        with self.assertRaises(TypeError) as cm:
            MetricParameters().export(1)
        msg = (
            "<class 'goalie.options.MetricParameters'> can only be exported to"
            " RiemannianMetric, not '<class 'int'>'."
        )
        self.assertEqual(str(cm.exception), msg)


class TestGoalOrientedParameters(unittest.TestCase):
    """
    Unit tests for the :class:`GoalOrientedParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "qoi_rtol": 0.001,
            "estimator_rtol": 0.001,
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
            "rigorous": False,
        }

    def test_defaults(self):
        ap = GoalOrientedParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_str(self):
        ap = GoalOrientedParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = GoalOrientedParameters()
        expected = (
            "GoalOrientedParameters(qoi_rtol=0.001, estimator_rtol=0.001, miniter=3,"
            " maxiter=35, element_rtol=0.001, rigorous=False)"
        )
        self.assertEqual(repr(ap), expected)

    def test_qoi_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedParameters({"qoi_rtol": "0.001"})
        msg = "Expected attribute 'qoi_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_estimator_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedParameters({"estimator_rtol": "0.001"})
        msg = (
            "Expected attribute 'estimator_rtol' to be of type 'float' or 'int', not"
            " 'str'."
        )
        self.assertEqual(str(cm.exception), msg)


class TestGoalOrientedMetricParameters(unittest.TestCase):
    """
    Unit tests for the :class:`GoalOrientedMetricParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "qoi_rtol": 0.001,
            "estimator_rtol": 0.001,
            "num_ramp_iterations": 3,
            "verbosity": -1,
            "p": 1.0,
            "base_complexity": 200.0,
            "target_complexity": 4000.0,
            "h_min": 1.0e-30,
            "h_max": 1.0e30,
            "a_max": 1.0e5,
            "restrict_anisotropy_first": False,
            "hausdorff_number": 0.01,
            "gradation_factor": 1.3,
            "no_insert": False,
            "no_swap": False,
            "no_move": False,
            "no_surf": False,
            "num_parmmg_iterations": 3,
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
            "rigorous": False,
        }

    def test_defaults(self):
        ap = GoalOrientedMetricParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_str(self):
        ap = GoalOrientedMetricParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = GoalOrientedMetricParameters()
        expected = (
            "GoalOrientedMetricParameters(qoi_rtol=0.001, estimator_rtol=0.001,"
            " num_ramp_iterations=3, verbosity=-1, p=1.0, base_complexity=200.0,"
            " target_complexity=4000.0, h_min=1e-30, h_max=1e+30, a_max=100000.0,"
            " restrict_anisotropy_first=False, hausdorff_number=0.01,"
            " gradation_factor=1.3, no_insert=False, no_swap=False, no_move=False,"
            " no_surf=False, num_parmmg_iterations=3, miniter=3, maxiter=35,"
            " element_rtol=0.001, rigorous=False)"
        )
        self.assertEqual(repr(ap), expected)


if __name__ == "__main__":
    unittest.main()
