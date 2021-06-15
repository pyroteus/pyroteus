"""
Partitioning for the temporal domain.
"""
from .log import debug
from .utility import AttrDict
from collections.abc import Iterable
import numpy as np


__all__ = ["TimePartition", "TimeInterval"]


class TimePartition(object):
    """
    Object describing the partition of the time
    interval of interest into subintervals.

    For now, the subintervals are assumed to be
    uniform in length. However, different values
    may be used of the timestep on each.
    """
    def __init__(self, end_time, num_subintervals, timesteps, fields, **kwargs):
        """
        :arg end_time: end time of the interval
            of interest
        :arg num_subintervals: number of
            subintervals in the partition
        :arg timesteps: (list of values for the)
            timestep used on each subinterval
        :arg fields: (list of) field names ordered
            by call sequence
        :kwarg timesteps_per_export: (list of)
            timesteps per export (default 1)
        :kwarg start_time: start time of the
            interval of interest (default 0.0)
        :kwarg subinterals: user-provided sequence
            of subintervals, which need not be of
            uniform length (defaults to None)
        """
        debug('')
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        timesteps_per_export = kwargs.get('timesteps_per_export', 1)
        self.start_time = kwargs.get('start_time', 0.0)
        self.end_time = end_time
        self.num_subintervals = int(np.round(num_subintervals))
        if not np.isclose(num_subintervals, self.num_subintervals):
            raise ValueError(f"Non-integer number of subintervals {num_subintervals}")
        self.debug("num_subintervals")
        self.interval = (self.start_time, self.end_time)
        self.debug("interval")

        # Get subintervals
        self.subintervals = kwargs.get('subintervals')
        if self.subintervals is None:
            subinterval_time = (self.end_time - self.start_time)/num_subintervals
            self.subintervals = [
                (self.start_time + i*subinterval_time, self.start_time + (i+1)*subinterval_time)
                for i in range(num_subintervals)
            ]
        assert len(self.subintervals) == num_subintervals
        assert np.isclose(self.subintervals[0][0], self.start_time)
        for i in range(1, num_subintervals):
            assert np.isclose(self.subintervals[i][0], self.subintervals[i-1][1])
        assert np.isclose(self.subintervals[-1][1], self.end_time)
        self.debug("subintervals")

        # Get timestep on each subinterval
        if not isinstance(timesteps, Iterable):
            timesteps = [timesteps for subinterval in self.subintervals]
        self.timesteps = np.array(timesteps)
        self.debug("timesteps")
        if len(self.timesteps) != num_subintervals:
            raise ValueError("Number of timesteps and subintervals do not match"
                             + f" ({len(self.timesteps)} vs. {num_subintervals})")

        # Get number of timesteps on each subinterval
        _timesteps_per_subinterval = [
            (t[1] - t[0])/dt
            for t, dt in zip(self.subintervals, self.timesteps)
        ]
        self.timesteps_per_subinterval = [
            int(np.round(tsps))
            for tsps in _timesteps_per_subinterval
        ]
        if not np.allclose(self.timesteps_per_subinterval, _timesteps_per_subinterval):
            raise ValueError("Non-integer timesteps per subinterval"
                             + f" ({_timesteps_per_subinterval})")
        self.debug("timesteps_per_subinterval")

        # Get timesteps per export
        if not isinstance(timesteps_per_export, Iterable):
            if not np.isclose(timesteps_per_export, np.round(timesteps_per_export)):
                raise ValueError("Non-integer timesteps per export"
                                 + f" ({timesteps_per_export})")
            timesteps_per_export = [
                int(np.round(timesteps_per_export))
                for subinterval in self.subintervals
            ]
        self.timesteps_per_export = np.array(timesteps_per_export, dtype=np.int32)
        if len(self.timesteps_per_export) != len(self.timesteps_per_subinterval):
            raise ValueError("Number of timesteps per export and subinterval do not match"
                             + f" ({len(self.timesteps_per_export)}"
                             + f" vs. {self.timesteps_per_subinterval})")
        for i, (tspe, tsps) in enumerate(zip(self.timesteps_per_export,
                                             self.timesteps_per_subinterval)):
            if tsps % tspe != 0:
                raise ValueError("Number of timesteps per export does not divide number of"
                                 + f" timesteps per subinterval ({tspe} vs. {tsps}"
                                 + f" on subinteral {i})")
        self.debug("timesteps_per_export")

        # Get exports per subinterval
        self.exports_per_subinterval = np.array([
            tsps//tspe + 1
            for tspe, tsps in zip(self.timesteps_per_export, self.timesteps_per_subinterval)
        ], dtype=np.int32)
        self.debug("exports_per_subinterval")
        debug('')

    def debug(self, attr):
        """
        Print attribute 'msg' for debugging purposes.
        """
        try:
            val = self.__getattribute__(attr)
        except AttributeError:
            raise AttributeError(f"Attribute {attr} cannot be debuged because it doesn't exist")
        label = ' '.join(attr.split('_'))
        debug(f"TimePartition: {label:25s} {val}")

    def __len__(self):
        return self.num_subintervals

    def __getitem__(self, i):
        """
        :arg i: index
        :return: subinterval bounds and timestep
            associated with that index
        """
        return AttrDict({
            'subinterval': self.subintervals[i],
            'timestep': self.timesteps[i],
            'timesteps_per_export': self.timesteps_per_export[i],
            'num_exports': self.exports_per_subinterval[i],
            'start_time': self.subintervals[i][0],
            'end_time': self.subintervals[i][1],
        })


class TimeInterval(TimePartition):
    """
    A trivial :class:`TimePartition`.
    """
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], tuple):
            assert len(args[0]) == 2
            kwargs['start_time'] = args[0][0]
            end_time = args[0][1]
        else:
            end_time = args[0]
        timestep = args[1]
        fields = args[2]
        super(TimeInterval, self).__init__(end_time, 1, timestep, fields, **kwargs)

    def __repr__(self):
        return str(self[0])

    @property
    def timestep(self):
        return self.timesteps[0]
