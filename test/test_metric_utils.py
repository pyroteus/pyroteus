from pyroteus import *
import unittest


class TestRampComplexity(unittest.TestCase):
    """
    Unit tests for :func:`ramp_complexity`.
    """

    def test_base_non_positive_error(self):
        with self.assertRaises(ValueError) as cm:
            ramp_complexity(0, 100, 1)
        msg = "Base complexity must be positive, not 0."
        self.assertEqual(str(cm.exception), msg)

    def test_target_non_positive_error(self):
        with self.assertRaises(ValueError) as cm:
            ramp_complexity(100, 0, 1)
        msg = "Target complexity must be positive, not 0."
        self.assertEqual(str(cm.exception), msg)

    def test_iteration_non_positive_error(self):
        with self.assertRaises(ValueError) as cm:
            ramp_complexity(100, 1000, -1)
        msg = "Current iteration must be non-negative, not -1."
        self.assertEqual(str(cm.exception), msg)

    def test_num_iterations_non_positive_error(self):
        with self.assertRaises(ValueError) as cm:
            ramp_complexity(100, 1000, 0, num_iterations=-1)
        msg = "Number of iterations must be non-negative, not -1."
        self.assertEqual(str(cm.exception), msg)

    def test_instant_ramp(self):
        C = ramp_complexity(100, 1000, 0, num_iterations=0)
        self.assertEqual(C, 1000)

    def test_ramp(self):
        base = 100
        target = 1000
        niter = 3
        for i in range(niter):
            C = ramp_complexity(base, target, i, num_iterations=niter)
            self.assertAlmostEqual(C, base + i * (target - base) / niter)
        C = ramp_complexity(base, target, niter, num_iterations=niter)
        self.assertEqual(C, target)
