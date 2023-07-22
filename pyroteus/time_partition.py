"""
Partitioning for the temporal domain.
"""
from .log import debug
from .utility import AttrDict
from collections.abc import Iterable
import numpy as np
from typing import List, Optional, Union


__all__ = ["TimePartition", "TimeInterval", "TimeInstant"]


class TimePartition:
    """
    A partition of the time interval of interest into subintervals.

    The subintervals are assumed to be uniform in length. However, different timestep
    values may be used on each subinterval.
    """

    def __init__(
        self,
        end_time: float,
        num_subintervals: int,
        timesteps: Union[List[float], float],
        fields: Union[List[str], str],
        timesteps_per_export: int = 1,
        start_time: float = 0.0,
        subintervals: Optional[List[float]] = None,
    ):
        """
        :arg end_time: end time of the interval of interest
        :arg num_subintervals: number of subintervals in the partition
        :arg timesteps: (list of values for the) timestep used on each subinterval
        :arg fields: (list of) field names ordered by call sequence
        :kwarg timesteps_per_export: (list of) timesteps per export
        :kwarg start_time: start time of the interval of interest
        :kwarg subinterals: user-provided sequence of subintervals, which need not be of
            uniform length
        """
        debug(100 * "-")
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        self.start_time = start_time
        self.end_time = end_time
        self.num_subintervals = int(np.round(num_subintervals))
        if not np.isclose(num_subintervals, self.num_subintervals):
            raise ValueError(
                f"Non-integer number of subintervals '{num_subintervals}'."
            )
        self.debug("num_subintervals")
        self.interval = (self.start_time, self.end_time)
        self.debug("interval")

        # Get subintervals
        self.subintervals = subintervals
        if self.subintervals is None:
            subinterval_time = (self.end_time - self.start_time) / num_subintervals
            self.subintervals = [
                (
                    self.start_time + i * subinterval_time,
                    self.start_time + (i + 1) * subinterval_time,
                )
                for i in range(num_subintervals)
            ]
        self._check_subintervals()
        self.debug("subintervals")

        # Get timestep on each subinterval
        if not isinstance(timesteps, Iterable):
            timesteps = [timesteps] * len(self)
        self.timesteps = timesteps
        self._check_timesteps()
        self.debug("timesteps")

        # Get number of timesteps on each subinterval
        _timesteps_per_subinterval = [
            (t[1] - t[0]) / dt for t, dt in zip(self.subintervals, self.timesteps)
        ]
        self.timesteps_per_subinterval = [
            int(np.round(tsps)) for tsps in _timesteps_per_subinterval
        ]
        if not np.allclose(self.timesteps_per_subinterval, _timesteps_per_subinterval):
            raise ValueError(
                "Non-integer timesteps per subinterval"
                f" ({_timesteps_per_subinterval})."
            )
        self.debug("timesteps_per_subinterval")

        # Get timesteps per export
        if not isinstance(timesteps_per_export, Iterable):
            if not np.isclose(timesteps_per_export, np.round(timesteps_per_export)):
                raise ValueError(
                    f"Non-integer timesteps per export ({timesteps_per_export})."
                )
            timesteps_per_export = [
                int(np.round(timesteps_per_export)) for subinterval in self.subintervals
            ]
        self.timesteps_per_export = np.array(timesteps_per_export, dtype=np.int32)
        if len(self.timesteps_per_export) != len(self.timesteps_per_subinterval):
            raise ValueError(
                "Number of timesteps per export and subinterval do not match"
                f" ({len(self.timesteps_per_export)}"
                f" vs. {len(self.timesteps_per_subinterval)})."
            )
        for i, (tspe, tsps) in enumerate(
            zip(self.timesteps_per_export, self.timesteps_per_subinterval)
        ):
            if tsps % tspe != 0:
                raise ValueError(
                    "Number of timesteps per export does not divide number of"
                    f" timesteps per subinterval ({tspe} vs. {tsps}"
                    f" on subinterval {i})."
                )
        self.debug("timesteps_per_export")

        # Get exports per subinterval
        self.exports_per_subinterval = np.array(
            [
                tsps // tspe + 1
                for tspe, tsps in zip(
                    self.timesteps_per_export, self.timesteps_per_subinterval
                )
            ],
            dtype=np.int32,
        )
        self.debug("exports_per_subinterval")
        self.steady = (
            self.num_subintervals == 1 and self.timesteps_per_subinterval[0] == 1
        )
        self.debug("steady")
        debug(100 * "-")

    def debug(self, attr: str):
        """
        Print attribute 'msg' for debugging purposes.
        """
        try:
            val = self.__getattribute__(attr)
        except AttributeError:
            raise AttributeError(
                f"Attribute '{attr}' cannot be debugged because it doesn't exist."
            )
        label = " ".join(attr.split("_"))
        debug(f"TimePartition: {label:25s} {val}")

    def __str__(self) -> str:
        return f"{self.subintervals}"

    def __repr__(self) -> str:
        timesteps = ", ".join([str(dt) for dt in self.timesteps])
        fields = ", ".join([f"'{field}'" for field in self.fields])
        return (
            f"TimePartition("
            f"end_time={self.end_time}, "
            f"num_subintervals={self.num_subintervals}, "
            f"timesteps=[{timesteps}], "
            f"fields=[{fields}])"
        )

    def __len__(self) -> int:
        return self.num_subintervals

    def __getitem__(self, i: int) -> dict:
        """
        :arg i: index
        :return: subinterval bounds and timestep
            associated with that index
        """
        return AttrDict(
            {
                "subinterval": self.subintervals[i],
                "timestep": self.timesteps[i],
                "timesteps_per_export": self.timesteps_per_export[i],
                "num_exports": self.exports_per_subinterval[i],
                "num_timesteps": self.timesteps_per_subinterval[i],
                "start_time": self.subintervals[i][0],
                "end_time": self.subintervals[i][1],
            }
        )

    def _check_subintervals(self):
        if len(self.subintervals) != self.num_subintervals:
            raise ValueError(
                "Number of subintervals provided differs from num_subintervals:"
                f" {len(self.subintervals)} != {self.num_subintervals}."
            )
        if not np.isclose(self.subintervals[0][0], self.start_time):
            raise ValueError(
                "The first subinterval does not start at the start time:"
                f" {self.subintervals[0][0]} != {self.start_time}."
            )
        for i in range(self.num_subintervals-1):
            if not np.isclose(self.subintervals[i][1], self.subintervals[i+1][0]):
                raise ValueError(
                    f"The end of subinterval {i} does not match the start of"
                    f" subinterval {i+1}: {self.subintervals[i][1]} !="
                    f" {self.subintervals[i+1][0]}."
                )
        if not np.isclose(self.subintervals[-1][1], self.end_time):
            raise ValueError(
                "The final subinterval does not end at the end time:"
                f" {self.subintervals[-1][1]} != {self.end_time}."
            )

    def _check_timesteps(self):
        if len(self.timesteps) != self.num_subintervals:
            raise ValueError(
                "Number of timesteps does not match num_subintervals:"
                f" {len(self.timesteps)} != {self.num_subintervals}."
            )

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return (
            np.allclose(self.subintervals, other.subintervals)
            and np.allclose(self.timesteps, other.timesteps)
            and np.allclose(self.exports_per_subinterval, other.exports_per_subinterval)
            and self.fields == other.fields
        )

    def __ne__(self, other):
        if len(self) != len(other):
            return True
        return (
            not np.allclose(self.subintervals, other.subintervals)
            or not np.allclose(self.timesteps, other.timesteps)
            or not np.allclose(
                self.exports_per_subinterval, other.exports_per_subinterval
            )
            or not self.fields == other.fields
        )


class TimeInterval(TimePartition):
    """
    A trivial :class:`~.TimePartition`.
    """

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], tuple):
            assert len(args[0]) == 2
            kwargs["start_time"] = args[0][0]
            end_time = args[0][1]
        else:
            end_time = args[0]
        timestep = args[1]
        fields = args[2]
        super().__init__(end_time, 1, timestep, fields, **kwargs)

    def __repr__(self) -> str:
        return (
            f"TimeInterval("
            f"end_time={self.end_time}, "
            f"timestep={self.timestep}, "
            f"fields={self.fields})"
        )

    @property
    def timestep(self) -> float:
        return self.timesteps[0]


class TimeInstant(TimeInterval):
    """
    A :class:`~.TimePartition` for steady-state problems.

    Under the hood this means dividing :math:`[0,1)` into
    a single timestep.
    """

    def __init__(self, fields: Union[List[str], str], **kwargs):
        time = kwargs.get("time", 1.0)
        if "end_time" in kwargs:
            if "time" in kwargs:
                raise ValueError("Both 'time' and 'end_time' are set.")
            time = kwargs.get("end_time")
        timestep = time
        super().__init__(time, timestep, fields)

    def __str__(self) -> str:
        return f"({self.end_time})"

    def __repr__(self) -> str:
        return f"TimeInstant(" f"time={self.end_time}, " f"fields={self.fields})"
