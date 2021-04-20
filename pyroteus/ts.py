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
    if isinstance(timesteps, (float, int)):
        timesteps = [timesteps for subinterval in subintervals]
    timesteps = np.array(timesteps)
    assert len(timesteps) == len(subintervals), f"{len(timesteps)} vs. {len(subintervals)}"
    _dt_per_mesh = [(t[1] - t[0])/dt for t, dt in zip(subintervals, timesteps)]
    dt_per_mesh = [int(np.round(dtpm)) for dtpm in _dt_per_mesh]
    assert np.allclose(dt_per_mesh, _dt_per_mesh), f"{dt_per_mesh} vs. {_dt_per_mesh}"
    if isinstance(dt_per_export, float):
        dt_per_export = int(dt_per_export)
    if isinstance(dt_per_export, int):
        dt_per_export = [dt_per_export for subinterval in subintervals]
    dt_per_export = np.array(dt_per_export, dtype=np.int32)
    assert len(dt_per_mesh) == len(dt_per_export), f"{len(dt_per_mesh)} vs. {len(dt_per_export)}"
    for dtpe, dtpm in zip(dt_per_export, dt_per_mesh):
        assert dtpm % dtpe == 0
    export_per_subinterval = [dtpm//dtpe + 1 for dtpe, dtpm in zip(dt_per_export, dt_per_mesh)]
    export_per_subinterval = np.array(export_per_subinterval, dtype=np.int32)
    return timesteps, dt_per_export, export_per_subinterval
