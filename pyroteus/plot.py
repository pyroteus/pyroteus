"""
Driver functions for plotting solution data.
"""
from firedrake import tricontourf, triplot  # noqa
import matplotlib.pyplot as plt


__all__ = ["plot_snapshots", "plot_indicator_snapshots"]


def plot_snapshots(solutions, time_partition, field, label, **kwargs):
    """
    Plot a sequence of snapshots associated with
    ``solutions.field.label`` and :class:`TimePartition`
    ``time_partition``.

    Any keyword arguments are passed to ``tricontourf``.

    :arg solutions: :class:`AttrDict` of solutions
        computed by solving a forward or adjoint
        problem
    :arg time_partition: the :class:`TimePartition`
        object used to solve the problem
    :arg field: solution field of choice
    :arg label: choose from ``'forward'``, ``'forward_old'``
        ``'adjoint'`` and ``'adjoint_next'``
    """
    P = time_partition
    rows = P.exports_per_subinterval[0] - 1
    cols = P.num_subintervals
    fig, axes = plt.subplots(rows, cols, sharex="col", figsize=(6 * cols, 24 // cols))
    for i, sols_step in enumerate(solutions[field][label]):
        ax = axes[0] if cols == 1 else axes[0, i]
        ax.set_title(f"Mesh[{i}]")
        for j, sol in enumerate(sols_step):
            ax = axes[j] if cols == 1 else axes[j, i]
            tricontourf(sol, axes=ax, **kwargs)
            ax.annotate(
                f"t={i*P.end_time/cols + j*P.timesteps_per_export[i]*P.timesteps[i]:.2f}",
                (0.05, 0.05),
                color="white",
            )
    plt.tight_layout()
    return fig, axes


def plot_indicator_snapshots(indicators, time_partition, **kwargs):
    """
    Plot a sequence of snapshots associated with
    ``indicators`` and :class:`TimePartition`
    ``time_partition``.

    Any keyword arguments are passed to ``tricontourf``.

    :arg indicators: list of list of indicators,
        indexed by mesh sequence index, then timestep
    :arg time_partition: the :class:`TimePartition`
        object used to solve the problem
    """
    P = time_partition
    rows = P.exports_per_subinterval[0] - 1
    cols = P.num_subintervals
    fig, axes = plt.subplots(rows, cols, sharex="col", figsize=(6 * cols, 24 // cols))
    for i, indi_step in enumerate(indicators):
        ax = axes[0] if cols == 1 else axes[0, i]
        ax.set_title(f"Mesh[{i}]")
        for j, indi in enumerate(indi_step):
            ax = axes[j] if cols == 1 else axes[j, i]
            tricontourf(indi, axes=ax, **kwargs)
            ax.annotate(
                f"t={i*P.end_time/cols + j*P.timesteps_per_export[i]*P.timesteps[i]:.2f}",
                (0.05, 0.05),
                color="white",
            )
    plt.tight_layout()
    return fig, axes
