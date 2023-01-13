"""
Testing for the time partition objects.
"""
from pyroteus.time_partition import TimePartition, TimeInterval, TimeInstant
import unittest


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__`` and :meth:`__repr__`` methods work as intended for
    Pyroteus' time partition objects.
    """

    def setUp(self):
        self.end_time = 1.0
        self.fields = ["field"]

        # Define time partitions
        self.time_interval = TimeInterval(self.end_time, [0.5], self.fields)
        self.time_instant = TimeInstant(self.fields, time=self.end_time)

    def get_time_partition(self, n):
        split = self.end_time / n
        timesteps = [split if i % 2 else split / 2 for i in range(n)]
        return TimePartition(self.end_time, n, timesteps, self.fields)

    def test_time_partition1_str(self):
        expected = "[(0.0, 1.0)]"
        self.assertEqual(str(self.get_time_partition(1)), expected)

    def test_time_partition2_str(self):
        expected = "[(0.0, 0.5), (0.5, 1.0)]"
        self.assertEqual(str(self.get_time_partition(2)), expected)

    def test_time_partition4_str(self):
        expected = "[(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]"
        self.assertEqual(str(self.get_time_partition(4)), expected)

    def test_time_interval_str(self):
        expected = "[(0.0, 1.0)]"
        self.assertEqual(str(self.time_interval), expected)

    def test_time_instant_str(self):
        expected = "(1.0)"
        self.assertEqual(str(self.time_instant), expected)

    def test_time_partition1_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=1,"
            " timesteps=[0.5], fields=['field'])"
        )
        self.assertEqual(repr(self.get_time_partition(1)), expected)

    def test_time_partition2_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=2,"
            " timesteps=[0.25, 0.5], fields=['field'])"
        )
        self.assertEqual(repr(self.get_time_partition(2)), expected)

    def test_time_partition4_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=4,"
            " timesteps=[0.125, 0.25, 0.125, 0.25], fields=['field'])"
        )
        self.assertEqual(repr(self.get_time_partition(4)), expected)

    def test_time_interval_repr(self):
        expected = "TimeInterval(end_time=1.0, timestep=0.5, fields=['field'])"
        self.assertEqual(repr(self.time_interval), expected)

    def test_time_instant_repr(self):
        expected = "TimeInstant(time=1.0, fields=['field'])"
        self.assertEqual(repr(self.time_instant), expected)


if __name__ == "__main__":
    unittest.main()
