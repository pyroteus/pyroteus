from firedrake import *


__all__ = ["isotropic_metric"]


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
