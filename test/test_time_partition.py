"""
Testing for the time partition objects.
"""
from pyroteus.time_partition import TimePartition, TimeInterval, TimeInstant
import unittest


class TestSetup(unittest.TestCase):
    """
    Tests related to the construction of time partitions.
    """

    def test_time_instant_multiple_kwargs(self):
        with self.assertRaises(ValueError) as cm:
            TimeInstant("field", time=1.0, end_time=1.0)
        msg = "Both 'time' and 'end_time' are set."
        self.assertEqual(str(cm.exception), msg)

    def test_time_partition_eq_positive(self):
        time_partition1 = TimePartition(1.0, 1, [1.0], "field")
        time_partition2 = TimePartition(1.0, 1, [1.0], "field")
        self.assertTrue(time_partition1 == time_partition2)

    def test_time_partition_eq_negative(self):
        time_partition1 = TimePartition(1.0, 1, [1.0], "field")
        time_partition2 = TimePartition(2.0, 1, [1.0], "field")
        self.assertFalse(time_partition1 == time_partition2)

    def test_time_partition_ne_positive(self):
        time_partition1 = TimePartition(1.0, 2, [0.5, 0.5], "field")
        time_partition2 = TimePartition(1.0, 1, [1.0], "field")
        self.assertTrue(time_partition1 != time_partition2)

    def test_time_partition_ne_negative(self):
        time_partition1 = TimePartition(1.0, 1, [1.0], "field", num_timesteps_per_export=1)
        time_partition2 = TimePartition(1.0, 1, [1.0], "field")
        self.assertFalse(time_partition1 != time_partition2)

    def test_time_interval_eq_positive(self):
        time_interval1 = TimeInterval(1.0, 1.0, "field")
        time_interval2 = TimeInterval((0.0, 1.0), 1.0, ["field"])
        self.assertTrue(time_interval1 == time_interval2)

    def test_time_interval_eq_negative(self):
        time_interval1 = TimeInterval(1.0, 1.0, "field")
        time_interval2 = TimeInterval((0.5, 1.0), 0.5, "field")
        self.assertFalse(time_interval1 == time_interval2)

    def test_time_interval_ne_positive(self):
        time_interval1 = TimeInterval(1.0, 1.0, "field")
        time_interval2 = TimeInterval((-0.5, 0.5), 1.0, "field")
        self.assertTrue(time_interval1 != time_interval2)

    def test_time_interval_ne_negative(self):
        time_interval1 = TimeInterval(1.0, 1.0, "field")
        time_interval2 = TimeInterval((0.0, 1.0), 1.0, ["field"])
        self.assertFalse(time_interval1 != time_interval2)

    def test_time_instant_eq_positive(self):
        time_instant1 = TimeInstant("field", time=1.0)
        time_instant2 = TimeInstant(["field"], time=1.0)
        self.assertTrue(time_instant1 == time_instant2)

    def test_time_instant_eq_negative(self):
        time_instant1 = TimeInstant("field", time=1.0)
        time_instant2 = TimeInstant("f", time=1.0)
        self.assertFalse(time_instant1 == time_instant2)

    def test_time_instant_ne_positive(self):
        time_instant1 = TimeInstant("field", time=1.0)
        time_instant2 = TimeInstant("field", time=2.0)
        self.assertTrue(time_instant1 != time_instant2)

    def test_time_instant_ne_negative(self):
        time_instant1 = TimeInstant("field", time=1.0)
        time_instant2 = TimeInstant("field", end_time=1.0)
        self.assertFalse(time_instant1 != time_instant2)

    def test_time_partition_eq_interval_positive(self):
        time_partition = TimePartition(1.0, 1, [0.5], ["field"])
        time_interval = TimeInterval(1.0, 0.5, "field")
        self.assertTrue(time_partition == time_interval)

    def test_time_partition_eq_interval_negative(self):
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        time_interval = TimeInterval(1.0, 0.5, "field")
        self.assertFalse(time_partition == time_interval)

    def test_time_partition_ne_interval_positive(self):
        time_partition = TimePartition(0.5, 1, [0.5], "field")
        time_interval = TimeInterval(1.0, 0.5, "field")
        self.assertTrue(time_partition != time_interval)

    def test_time_partition_ne_interval_negative(self):
        time_partition = TimePartition(1.0, 1, 0.5, ["field"], start_time=0.0)
        time_interval = TimeInterval(1.0, 0.5, "field")
        self.assertFalse(time_partition != time_interval)

    def test_noninteger_num_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1.1, 0.5, "field")
        msg = "Non-integer number of subintervals '1.1'."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_number_of_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, 0.5, "field", subintervals=[(0.0, 0.5), (0.5, 1.0)])
        msg = "Number of subintervals provided differs from num_subintervals: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_subinterval_start(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, 0.5, "field", subintervals=[(0.1, 1.0)])
        msg = "The first subinterval does not start at the start time: 0.1 != 0.0."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 2, 0.5, "field", subintervals=[(0.0, 0.6), (0.5, 1.0)])
        msg = (
            "The end of subinterval 0 does not match the start of subinterval 1:"
            " 0.6 != 0.5."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_subinterval_end(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, 0.5, "field", subintervals=[(0.0, 1.1)])
        msg = "The final subinterval does not end at the end time: 1.1 != 1.0."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_num_timesteps(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5, 0.5], "field")
        msg = "Number of timesteps does not match num_subintervals: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_noninteger_num_timesteps_per_subinterval(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.4], "field")
        msg = "Non-integer number of timesteps on subinterval 0: 2.5."
        self.assertEqual(str(cm.exception), msg)

    def test_noninteger_num_timesteps_per_export(self):
        with self.assertRaises(TypeError) as cm:
            TimePartition(1.0, 1, [0.5], "field", num_timesteps_per_export=1.1)
        msg = (
            "Expected number of timesteps per export on subinterval 0 to be an integer,"
            " not '<class 'float'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_nonmatching_num_timesteps_per_export(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5], "field", num_timesteps_per_export=[1, 2])
        msg = "Number of timesteps per export and subinterval do not match: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_indivisible_num_timesteps_per_export(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5], "field", num_timesteps_per_export=4)
        msg = (
            "Number of timesteps per export does not divide number of timesteps per"
            " subinterval on subinterval 0: 2 | 4 != 0."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_debug_invalid_field(self):
        with self.assertRaises(AttributeError) as cm:
            TimeInstant("field").debug("blah")
        msg = "Attribute 'blah' cannot be debugged because it doesn't exist."
        self.assertEqual(str(cm.exception), msg)

    def test_field_type_error(self):
        with self.assertRaises(ValueError) as cm:
            TimeInstant("field", field_types="type")
        msg = (
            "Expected field type for field 'field' to be either 'unsteady' or"
            " 'steady', but got 'type'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_num_field_types_error(self):
        with self.assertRaises(ValueError) as cm:
            TimeInstant("field", field_types=["type1", "type2"])
        msg = "Number of fields does not match number of field types: 1 != 2."
        self.assertEqual(str(cm.exception), msg)


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__`` and :meth:`__repr__`` methods work as intended for
    Pyroteus' time partition objects.
    """

    def setUp(self):
        self.end_time = 1.0
        self.fields = ["field"]

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
        time_interval = TimeInterval(self.end_time, [0.5], self.fields)
        self.assertEqual(str(time_interval), expected)

    def test_time_instant_str(self):
        expected = "(1.0)"
        time_instant = TimeInstant(self.fields, time=self.end_time)
        self.assertEqual(str(time_instant), expected)

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
        time_interval = TimeInterval(self.end_time, [0.5], self.fields)
        self.assertEqual(repr(time_interval), expected)

    def test_time_instant_repr(self):
        expected = "TimeInstant(time=1.0, fields=['field'])"
        time_instant = TimeInstant(self.fields, time=self.end_time)
        self.assertEqual(repr(time_instant), expected)


if __name__ == "__main__":
    unittest.main()
