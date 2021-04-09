import numpy as np


__all__ = ["get_subintervals", "get_exports_per_subinterval"]


def get_subintervals(end_time, num_subintervals):
    """
    Calculate a list of subintervals.

    Note that they are assumed to be of equal length.

    :arg end_time: simulation end time
    :arg num_subintervals: number of subintervals
    """
    subinterval_time = end_time/num_subintervals  # NOTE: Assumes uniform
    return [(i*subinterval_time, (i+1)*subinterval_time) for i in range(num_subintervals)]


def get_exports_per_subinterval(subintervals, timesteps, dt_per_export):
    """
    :arg subintervals: list of tuples corresponding to subintervals
    :arg timesteps: list of floats or single float corresponding to timesteps
    :arg dt_per_export: list of ints or single in corresponding to timesteps per export
    """
    if isinstance(timesteps, float):
        timesteps = [timesteps for subinterval in subintervals]
    assert len(timesteps) == len(subintervals), f"{len(timesteps)} vs. {len(subintervals)}"
    _dt_per_mesh = [(t[1] - t[0])/dt for t, dt in zip(subintervals, timesteps)]
    dt_per_mesh = [int(dtpm) for dtpm in _dt_per_mesh]
    assert np.allclose(dt_per_mesh, _dt_per_mesh), f"{dt_per_mesh} vs. {_dt_per_mesh}"
    if isinstance(dt_per_export, int):
        dt_per_export = [dt_per_export for subinterval in subintervals]
    for dtpe, dtpm in zip(dt_per_export, dt_per_mesh):
        assert dtpm % dtpe == 0
    exports_per_subinterval = [dtpm//dtpe + 1 for dtpe, dtpm in zip(dt_per_export, dt_per_mesh)]
    return timesteps, dt_per_export, exports_per_subinterval
