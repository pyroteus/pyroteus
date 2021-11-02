"""
Test matrix decomposition par_loops.
"""
from firedrake import *
from pyroteus import *
from utility import uniform_mesh
import pytest


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False])
def reorder(request):
    return request.param


def test_eigendecomposition(dim, reorder):
    """
    Check decomposition of a metric into its eigenvectors
    and eigenvalues.

      * The eigenvectors should be orthonormal.
      * Applying `compute_eigendecomposition` followed by
        `set_eigendecomposition` should get back the metric.
    """
    mesh = uniform_mesh(dim, 20)

    # Recover Hessian metric for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    metric = hessian_metric(recover_hessian(f, mesh=mesh))
    P1_ten = metric.function_space()

    # Extract the eigendecomposition
    evectors, evalues = compute_eigendecomposition(metric, reorder=reorder)

    # Check eigenvectors are orthonormal
    err = Function(P1_ten)
    err.interpolate(dot(evectors, transpose(evectors)) - Identity(dim))
    if not np.isclose(norm(err), 0.0):
        raise ValueError("Eigenvectors are not orthonormal")

    # Check eigenvalues are in descending order
    if reorder:
        P1 = FunctionSpace(mesh, "CG", 1)
        for i in range(dim-1):
            f = interpolate(evalues[i], P1)
            f -= interpolate(evalues[i+1], P1)
            if f.vector().gather().min() < 0.0:
                raise ValueError("Eigenvalues are not in descending order")

    # Reassemble it and check the two match
    metric -= assemble_eigendecomposition(evectors, evalues)
    if not np.isclose(norm(metric), 0.0):
        raise ValueError("Reassembled metric does not match")


def test_density_quotients_decomposition(dim, reorder):
    """
    Check decomposition of a metric into its density
    and anisotropy quotients.

    Reassembling should get back the metric.
    """
    mesh = uniform_mesh(dim, 20)

    # Recover Hessian metric for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    metric = hessian_metric(recover_hessian(f, mesh=mesh))

    # Extract the eigendecomposition
    evectors, evalues = compute_eigendecomposition(metric, reorder=reorder)

    # Extract the density and anisotropy quotients
    density, quotients = density_and_quotients(metric, reorder=reorder)
    evalues.interpolate(as_vector([pow(density/Q, 2/dim) for Q in quotients]))

    # Reassemble the matrix and check the two match
    metric -= assemble_eigendecomposition(evectors, evalues)
    if not np.isclose(norm(metric), 0.0):
        raise ValueError("Reassembled metric does not match")
