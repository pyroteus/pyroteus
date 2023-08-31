"""
Driver functions for metric-based mesh adaptation.
"""
from animate.interpolation import clement_interpolant
import animate.metric
import animate.recovery
from .log import debug
from .time_partition import TimePartition
from animate.metric import RiemannianMetric
import firedrake
from firedrake.petsc import PETSc
import numpy as np
from pyop2 import op2
from typing import List, Optional, Union
import ufl


__all__ = ["enforce_element_constraints", "space_time_normalise", "ramp_complexity"]


# TODO: Implement this on the PETSc level and port it through to Firedrake
@PETSc.Log.EventDecorator()
def enforce_element_constraints(
    metrics: List[RiemannianMetric],
    h_min: List[firedrake.Function],
    h_max: List[firedrake.Function],
    a_max: List[firedrake.Function],
    boundary_tag: Optional[Union[str, int]] = None,
    optimise: bool = False,
) -> List[firedrake.Function]:
    """
    Post-process a list of metrics to enforce minimum and
    maximum element sizes, as well as maximum anisotropy.

    :arg metrics: the metrics
    :arg h_min: minimum tolerated element size,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :arg h_max: maximum tolerated element size,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :arg a_max: maximum tolerated element anisotropy,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :kwarg boundary_tag: optional tag to enforce sizes on.
    :kwarg optimise: is this a timed run?
    """
    from collections.abc import Iterable

    if isinstance(metrics, RiemannianMetric):
        metrics = [metrics]
    if not isinstance(h_min, Iterable):
        h_min = [h_min] * len(metrics)
    if not isinstance(h_max, Iterable):
        h_max = [h_max] * len(metrics)
    if not isinstance(a_max, Iterable):
        a_max = [a_max] * len(metrics)
    for metric, hmin, hmax, amax in zip(metrics, h_min, h_max, a_max):
        fs = metric.function_space()
        mesh = fs.mesh()
        P1 = firedrake.FunctionSpace(mesh, "CG", 1)

        def interp(f):
            if isinstance(f, firedrake.Function):
                return clement_interpolant(f)
            else:
                return firedrake.Function(P1).assign(f)

        # Interpolate hmin, hmax and amax into P1
        hmin = interp(hmin)
        hmax = interp(hmax)
        amax = interp(amax)

        # Check the values are okay
        if not optimise:
            _hmin = hmin.vector().gather().min()
            if _hmin <= 0.0:
                raise ValueError(
                    f"Encountered negative non-positive hmin value: {_hmin}."
                )
            _hmax = hmax.vector().gather().min()
            if _hmax < _hmin:
                raise ValueError(
                    f"Encountered hmax value smaller than hmin: {_hmax} vs. {_hmin}."
                )
            dx = ufl.dx(domain=mesh)
            integral = firedrake.assemble(ufl.conditional(hmax < hmin, 1, 0) * dx)
            if not np.isclose(integral, 0.0):
                raise ValueError(
                    f"Encountered regions where hmax < hmin: volume {integral}."
                )
            _amax = amax.vector().gather().min()
            if _amax < 1.0:
                raise ValueError(f"Encountered amax value smaller than unity: {_amax}.")

        # Enforce constraints
        dim = fs.mesh().topological_dimension()
        if boundary_tag is None:
            node_set = fs.node_set
        else:
            node_set = firedrake.DirichletBC(fs, 0, boundary_tag).node_set
        op2.par_loop(
            animate.recovery.get_metric_kernel("postproc_metric", dim),
            node_set,
            metric.dat(op2.RW),
            hmin.dat(op2.READ),
            hmax.dat(op2.READ),
            amax.dat(op2.READ),
        )
    return metrics


@PETSc.Log.EventDecorator()
def space_time_normalise(
    metrics: List[RiemannianMetric],
    time_partition: TimePartition,
    metric_parameters: Union[dict, list],
    global_factor: Optional[float] = None,
    boundary: bool = False,
    restrict_sizes: bool = True,
    restrict_anisotropy: bool = True,
) -> List[RiemannianMetric]:
    r"""
    Apply :math:`L^p` normalisation in both space and time.

    Based on Equation (1) in :cite:`Barral:2016`.

    :arg metrics: list of :class:`RiemannianMetric`\s corresponding
        to the metric associated with each subinterval
    :arg time_partition: :class:`TimePartition` for the problem at hand
    :arg metric_parameters: dictionary containing the target *space-time* metric
        complexity under `dm_plex_metric_target_complexity` and the normalisation order
        under `dm_plex_metric_p`, or a list thereof
    :kwarg global_factor: pre-computed global normalisation factor
    :kwarg boundary: is the normalisation to be done over the boundary?
    :kwarg restrict_sizes: should minimum and maximum metric magnitudes be enforced?
    :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
    """
    if isinstance(metric_parameters, dict):
        metric_parameters = [metric_parameters for _ in range(len(time_partition))]
    d = metrics[0].function_space().mesh().topological_dimension()
    if len(metrics) != len(time_partition):
        raise ValueError(
            "Number of metrics does not match number of subintervals:"
            f" {len(metrics)} vs. {len(time_partition)}."
        )
    if len(metrics) != len(metric_parameters):
        raise ValueError(
            "Number of metrics does not match number of sets of metric parameters:"
            f" {len(metrics)} vs. {len(metric_parameters)}."
        )

    # Preparation step
    metric_parameters = metric_parameters.copy()
    for metric, mp in zip(metrics, metric_parameters):
        if not isinstance(mp, dict):
            raise TypeError(
                "Expected metric_parameters to consist of dictionaries,"
                f" not objects of type '{type(mp)}'."
            )

        # Allow concise notation
        if "dm_plex_metric" in mp and isinstance(mp["dm_plex_metric"], dict):
            for key, value in mp["dm_plex_metric"].items():
                mp[f"dm_plex_metric_{key}"] = value
            mp.pop("dm_plex_metric")

        p = mp.get("dm_plex_metric_p")
        if p is None:
            raise ValueError("Normalisation order 'dm_plex_metric_p' must be set.")
        if not (np.isinf(p) or p >= 1.0):
            raise ValueError(
                f"Normalisation order '{p}' should be one or greater or np.inf."
            )
        target = mp.get("dm_plex_metric_target_complexity")
        if target is None:
            raise ValueError(
                "Target complexity 'dm_plex_metric_target_complexity' must be set."
            )
        if target <= 0.0:
            raise ValueError(f"Target complexity '{target}' is not positive.")
        metric.set_parameters(mp)
        metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

    # Compute global normalisation factor
    if global_factor is None:
        integral = 0
        p = mp["dm_plex_metric_p"]
        exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
        for metric, S in zip(metrics, time_partition):
            dX = (ufl.ds if boundary else ufl.dx)(metric.function_space().mesh())
            scaling = pow(S.num_timesteps, 2 * exponent)
            integral += scaling * firedrake.assemble(
                pow(ufl.det(metric), exponent) * dX
            )
        target = mp["dm_plex_metric_target_complexity"] * time_partition.num_timesteps
        debug(f"space_time_normalise: target space-time complexity={target:.4e}")
        global_factor = firedrake.Constant(pow(target / integral, 2 / d))
    debug(f"space_time_normalise: global scale factor={float(global_factor):.4e}")

    for metric, S in zip(metrics, time_partition):
        # Normalise according to the global normalisation factor
        metric.normalise(
            global_factor=global_factor,
            restrict_sizes=False,
            restrict_anisotropy=False,
        )

        # Apply the separate scale factors for each metric
        if not np.isinf(p):
            metric *= pow(S.num_timesteps, -2 / (2 * p + d))
        metric.enforce_spd(
            restrict_sizes=restrict_sizes,
            restrict_anisotropy=restrict_anisotropy,
        )

    return metrics


def ramp_complexity(
    base: float, target: float, i: int, num_iterations: int = 3
) -> float:
    """
    Ramp up the target complexity over the first few iterations.

    :arg base: the base complexity to start from
    :arg target: the desired complexity
    :arg i: the current iteration
    :kwarg num_iterations: how many iterations to ramp over?
    """
    if base <= 0.0:
        raise ValueError(f"Base complexity must be positive, not {base}.")
    if target <= 0.0:
        raise ValueError(f"Target complexity must be positive, not {target}.")
    if i < 0:
        raise ValueError(f"Current iteration must be non-negative, not {i}.")
    if num_iterations < 0:
        raise ValueError(
            f"Number of iterations must be non-negative, not {num_iterations}."
        )
    alpha = 1 if num_iterations == 0 else min(i / num_iterations, 1)
    return alpha * target + (1 - alpha) * base
