from __future__ import absolute_import
from firedrake import *
from .kernels import eigen_kernel, postproc_metric


__all__ = ["isotropic_metric", "space_normalise"]


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
    kernel = eigen_kernel(postproc_metric, dim, h_min, h_max, a_max)
    op2.par_loop(kernel, fs.node_set, metric.dat(op2.RW))
