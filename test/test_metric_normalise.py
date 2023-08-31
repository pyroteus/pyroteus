"""
Test metric normalisation functionality.
"""
from firedrake import *
from goalie import *
from animate.metric import RiemannianMetric
from sensors import *
from utility import uniform_mesh
import numpy as np
from parameterized import parameterized
import pytest
import unittest


class TestMetricNormalisation(unittest.TestCase):
    """
    Unit tests for metric normalisation.
    """

    def setUp(self):
        self.time_partition = TimeInterval(1.0, 1.0, "u")

    @property
    def simple_metric(self):
        mesh = uniform_mesh(2, 1)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        return RiemannianMetric(P1_ten)

    def test_time_partition_length_error(self):
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], "u")
        mp = {"dm_plex_metric_target_complexity": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], time_partition, [mp])
        msg = "Number of metrics does not match number of subintervals: 1 vs. 2."
        self.assertEqual(str(cm.exception), msg)

    def test_metric_parameters_length_error(self):
        mp = {"dm_plex_metric_target_complexity": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, [mp, mp])
        msg = (
            "Number of metrics does not match number of sets of metric parameters:"
            " 1 vs. 2."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_metric_parameters_type_error(self):
        with self.assertRaises(TypeError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, [1])
        msg = (
            "Expected metric_parameters to consist of dictionaries,"
            " not objects of type '<class 'int'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_normalistion_order_unset_error(self):
        mp = {"dm_plex_metric_target_complexity": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, mp)
        msg = "Normalisation order 'dm_plex_metric_p' must be set."
        self.assertEqual(str(cm.exception), msg)

    def test_normalisation_order_invalid_error(self):
        mp = {"dm_plex_metric_target_complexity": 1.0, "dm_plex_metric_p": 0.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, mp)
        msg = "Normalisation order '0.0' should be one or greater or np.inf."
        self.assertEqual(str(cm.exception), msg)

    def test_target_complexity_unset_error(self):
        mp = {"dm_plex_metric_p": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, mp)
        msg = "Target complexity 'dm_plex_metric_target_complexity' must be set."
        self.assertEqual(str(cm.exception), msg)

    def test_target_complexity_zero_error(self):
        mp = {"dm_plex_metric_target_complexity": 0.0, "dm_plex_metric_p": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, mp)
        msg = "Target complexity '0.0' is not positive."
        self.assertEqual(str(cm.exception), msg)

    def test_target_complexity_negative_error(self):
        mp = {"dm_plex_metric_target_complexity": -1.0, "dm_plex_metric_p": 1.0}
        with self.assertRaises(ValueError) as cm:
            space_time_normalise([self.simple_metric], self.time_partition, mp)
        msg = "Target complexity '-1.0' is not positive."
        assert str(cm.exception) == msg

    @pytest.mark.slow
    @parameterized.expand(
        [
            (bowl, 1),
            (bowl, 2),
            (bowl, np.inf),
            (hyperbolic, 1),
            (hyperbolic, 2),
            (hyperbolic, np.inf),
            (multiscale, 1),
            (multiscale, 2),
            (multiscale, np.inf),
            (interweaved, 1),
            (interweaved, 2),
            (interweaved, np.inf),
        ]
    )
    def test_consistency(self, sensor, degree):
        """
        Check that spatial normalisation and space-time
        normalisation on a single unit time interval with
        one timestep return the same metric.
        """
        mesh = mesh_for_sensors(2, 100)
        metric_parameters = {
            "dm_plex_metric_p": degree,
            "dm_plex_metric_target_complexity": 1000.0,
        }

        # Construct a Hessian metric
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        M = RiemannianMetric(P1_ten)
        M.compute_hessian(sensor(*mesh.coordinates))
        self.assertFalse(np.isnan(M.dat.data).any())
        M_st = M.copy(deepcopy=True)

        # Apply both normalisation strategies
        M.set_parameters(metric_parameters)
        M.normalise()
        space_time_normalise([M_st], self.time_partition, metric_parameters)
        self.assertFalse(np.isnan(M_st.dat.data).any())

        # Check that the metrics coincide
        self.assertAlmostEqual(errornorm(M, M_st), 0)
