"""
Driver functions for metric-based mesh adaptation.
"""
from __future__ import absolute_import
from .utility import *
from . import kernels


__all__ = ["metric_complexity", "isotropic_metric",
           "space_normalise", "space_time_normalise",
           "metric_relaxation", "metric_average", "metric_intersection", "combine_metrics"]


# --- General

def metric_complexity(metric, boundary=False):
    """
    Compute the complexity of a metric.

    This is the continuous analogue of the
    (discrete) mesh vertex count.

    :kwarg boundary: compute metric on domain
        interior or boundary?
    """
    differential = ds if boundary else dx
    return assemble(sqrt(metric)*differential)


def isotropic_metric(scalar_field, tensor_fs=None, f_min=1.0e-12):
    """
    Compute an isotropic metric from some scalar field.

    The result is a diagonal matrix whose diagonal
    entries are the absolute value of the scalar field
    at each mesh vertex.

    :arg scalar_field: field to compute metric from.
    :kwarg tensor_fs: :class:`TensorFunctionSpace` in
        which the metric will exist.
    :kwarg f_min: minimum tolerated function value.
    """
    fs = scalar_field.function_space()
    family = fs.ufl_element().family()
    degree = fs.ufl_element().degree()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)
    tensor_fs = tensor_fs or TensorFunctionSpace(mesh, "CG", 1)
    assert tensor_fs.ufl_element().family() == 'Lagrange'
    assert tensor_fs.ufl_element().degree() == 1

    # Compute metric diagonal
    if family == 'Lagrange' and degree == 1:
        M_diag = interpolate(max_value(abs(scalar_field), f_min), fs)
    else:
        M_diag = project(scalar_field, FunctionSpace(mesh, "CG", 1))
        M_diag.interpolate(max_value(abs(M_diag), f_min))

    # Assemble full metric
    return interpolate(M_diag*Identity(dim), tensor_fs)


# --- Normalisation

def space_normalise(metric, options, target, p, h_min, h_max, a_max=1000):
    """
    Apply L-p normalisation in space alone.

    :arg metric: class:`Function`s corresponding to the
        metric to be normalised.
    :arg options: :class:`ModelOptions2d` object containing
        information on timestepping.
    :arg target: target metric complexity *in space alone*.
    :arg p: normalisation order.
    :arg h_min: minimum tolerated element size.
    :arg h_max: maximum tolerated element size.
    :kwarg a_max: maximum tolerated element anisotropy (default 1000).
    """
    assert p == 'inf' or p >= 1.0, "Norm order {:} not valid".format(p)
    assert 0 < h_min < h_max, "Min/max tolerated element sizes {:}/{:} not valid".format(h_min, h_max)
    assert a_max > 0, "Max tolerated anisotropy {:} not valid".format(a_max)
    fs = metric.function_space()
    mesh = fs.mesh()
    dim = mesh.topological_dimension()

    # Apply L-p normalisation
    form = pow(det(metric), 0.5) if p == 'inf' else pow(det(metric), p/(2*p + dim))
    integral = pow(target/assemble(form*dx), 2/dim)
    determinant = 1 if p == 'inf' else pow(det(metric), -1/(2*p + dim))
    metric.interpolate(integral*determinant*metric)

    # Enforce maximum/minimum element sizes and anisotropy
    kernel = kernels.eigen_kernel(kernels.postproc_metric, dim, h_min, h_max, a_max)
    op2.par_loop(kernel, fs.node_set, metric.dat(op2.RW))
    return metric


def space_time_normalise(metrics, options, target, p, h_min, h_max, a_max=1000, timesteps=None):
    """
    Apply L-p normalisation is both space and time.

    :arg metrics: list of :class:`Function`s corresponding
        to the metric associated with each mesh iteration.
    :arg options: :class:`ModelOptions2d` object containing
        information on timestepping.
    :arg target: target metric complexity *in space alone*.
    :arg p: normalisation order.
    :arg h_min: minimum tolerated element size.
    :arg h_max: maximum tolerated element size.
    :kwarg a_max: maximum tolerated element anisotropy (default 1000).
    :kwarg timesteps: list of timesteps specified in each subinterval.
    """
    assert p == 'inf' or p >= 1.0, "Norm order {:} not valid".format(p)
    assert 0 < h_min < h_max, "Min/max tolerated element sizes {:}/{:} not valid".format(h_min, h_max)
    assert a_max > 0, "Max tolerated anisotropy {:} not valid".format(a_max)
    num_meshes = len(metrics)
    num_timesteps = int(np.ceil(options.simulation_end_time/options.timestep))
    if timesteps is None:
        dt_per_mesh = num_timesteps/num_meshes*np.ones(num_meshes)
    else:
        assert len(timesteps) == num_meshes
        dt_per_mesh = [options.simulation_end_time/num_meshes/dt for dt in timesteps]
    dim = metrics[0].function_space().mesh().topological_dimension()
    N_st = target*num_timesteps

    # Compute global normalisation factor
    integral = 0.0
    for metric, tau in zip(metrics, dt_per_mesh):
        if p == 'inf':
            integral += assemble(tau*sqrt(det(metric))*dx)
        else:
            integral += assemble(pow(tau**2*det(metric), p/(2*p + dim))*dx)
    global_norm = pow(N_st/integral, 2/dim)

    # Normalise on each subinterval
    for metric, tau in zip(metrics, dt_per_mesh):
        if p == 'inf':
            metric *= global_norm
        else:
            metric.interpolate(global_norm*metric*pow(tau**2*det(metric), -1/(2*p + dim)))

        # Enforce maximum/minimum element sizes and anisotropy
        kernel = kernels.eigen_kernel(kernels.postproc_metric, dim, h_min, h_max, a_max)
        op2.par_loop(kernel, metric.function_space().node_set, metric.dat(op2.RW))
    return metrics


# --- Combination

def metric_relaxation(*metrics, weights=None, function_space=None):
    """
    Combine a list of metrics with a weighted average.

    :args metrics: the metrics to be combined
    :kwarg weights: a list of weights
    :kwarg function_space: the :class:`FunctionSpace`
        the relaxed metric should live in
    """
    n = len(metrics)
    assert n > 0
    weights = weights or np.ones(n)/n
    assert len(weights) == n
    fs = function_space or metrics[0].function_space()
    relaxed_metric = Function(fs)
    for weight, metric in zip(weights, metrics):
        if not isinstance(metric, Function) or metric.function_space() != fs:
            metric = interpolate(metric, function_space)
        relaxed_metric += weight*metric
    return relaxed_metric


def metric_average(*metrics, function_space=None):
    """
    Combine a list of metrics by averaging.

    :args metrics: the metrics to be combined
    :kwarg function_space: the :class:`FunctionSpace`
        the averaged metric should live in
    """
    return metric_relaxation(*metrics, function_space=function_space)


def metric_intersection(*metrics, function_space=None, boundary_tag=None):
    """
    Combine a list of metrics by intersection.

    :args metrics: the metrics to be combined
    :kwarg function_space: the :class:`FunctionSpace`
        the intersected metric should live in
    :kwarg boundary_tag: boundary segment physical
        ID for boundary intersection
    """
    n = len(metrics)
    assert n > 0
    fs = function_space or metrics[0].function_space()
    for metric in metrics:
        if not isinstance(metric, Function) or metric.function_space() != fs:
            metric = interpolate(metric, function_space)
    intersected_metric = Function(metrics[0])
    node_set = fs.node_set if tag is None else DirichletBC(fs, 0, boundary_tag).node_set
    dim = fs.mesh().topological_dimension()
    assert dim in (2, 3), "Spatial dimension {:d} not supported.".format(dim)

    def intersect_pair(M1, M2):
        M12 = Function(M1)
        kernel = kernels.eigen_kernel(kernels.intersect, dim)
        op2.par_loop(kernel, node_set, M12.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ))
        return M12

    for metric in metrics[1:]:
        intersected_metric = intersect_pair(intersected_metric, metric)
    return intersected_metric


def combine_metrics(*metrics, average=True, **kwargs):
    """
    Combine a list of metrics.

    :kwarg average: combination by averaging or intersection?
    """
    if average:
        kwargs.pop('boundary_tag')
        return metric_relaxation(*metrics, **kwargs)
    else:
        kwargs.pop('weights')
        return metric_intersection(*metrics, **kwargs)
