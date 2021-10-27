"""
Test matrix decomposition par_loops.
"""
from firedrake import *
from pyroteus import *
import pyroteus.kernel as kernels
from utility import uniform_mesh
import pytest


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[
    kernels.get_eigendecomposition,
    kernels.get_reordered_eigendecomposition,
])
def eigendecomposition_kernel(request):
    return request.param


def test_eigendecomposition(dim, eigendecomposition_kernel):
    """
    Check decomposition of a metric into its eigenvectors
    and eigenvalues.

      * The eigenvectors should be orthonormal.
      * Applying `get_eigendecomposition` followed by
        `set_eigendecomposition` should get back the metric.
    """
    mesh = uniform_mesh(dim, 20)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)

    # Recover Hessian metric for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    metric = hessian_metric(recover_hessian(f, mesh=mesh))

    # Extract the eigendecomposition
    evectors, evalues = Function(P1_ten), Function(P1_vec)
    kernel = kernels.eigen_kernel(eigendecomposition_kernel, dim)
    op2.par_loop(
        kernel, P1_ten.node_set,
        evectors.dat(op2.RW), evalues.dat(op2.RW), metric.dat(op2.READ)
    )

    # Check eigenvectors are orthonormal
    VVT = interpolate(dot(evectors, transpose(evectors)), P1_ten)
    I = interpolate(Identity(dim), P1_ten)
    if not np.isclose(norm(Function(I).assign(VVT - I)), 0.0):
        raise ValueError("Eigenvectors are not orthonormal")

    # Check eigenvalues are in descending order
    if 'reorder' in eigendecomposition_kernel.__name__:
        P1 = FunctionSpace(mesh, "CG", 1)
        for i in range(dim-1):
            f = interpolate(evalues[i], P1)
            f -= interpolate(evalues[i+1], P1)
            if f.vector().gather().min() < 0.0:
                raise ValueError(
                    "Eigenvalues are not in descending order"
                )

    # Reassemble it and check the two match
    reassembled = Function(P1_ten)
    kernel = kernels.eigen_kernel(kernels.set_eigendecomposition, dim)
    op2.par_loop(
        kernel, P1_ten.node_set,
        reassembled.dat(op2.RW), evectors.dat(op2.READ), evalues.dat(op2.READ)
    )
    metric -= reassembled
    if not np.isclose(norm(metric), 0.0):
        raise ValueError("Reassembled metric does not match")


def test_density_quotients_decomposition(dim, eigendecomposition_kernel):
    """
    Check decomposition of a metric into its density
    and anisotropy quotients.

    Reassembling should get back the metric.
    """
    mesh = uniform_mesh(dim, 20)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)

    # Recover Hessian metric for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    metric = hessian_metric(recover_hessian(f, mesh=mesh))

    # Extract the eigendecomposition
    evectors, evalues = Function(P1_ten), Function(P1_vec)
    kernel = kernels.eigen_kernel(eigendecomposition_kernel, dim)
    op2.par_loop(
        kernel, P1_ten.node_set,
        evectors.dat(op2.RW), evalues.dat(op2.RW), metric.dat(op2.READ)
    )

    # Extract the density and anisotropy quotients
    reorder = 'reorder' in eigendecomposition_kernel.__name__
    d, Q = density_and_quotients(metric, reorder=reorder)
    evalues.interpolate(as_vector([pow(d/Q[i], 2/dim) for i in range(dim)]))

    # Reassemble the matrix and check the two match
    reassembled = Function(P1_ten)
    kernel = kernels.eigen_kernel(kernels.set_eigendecomposition, dim)
    op2.par_loop(
        kernel, P1_ten.node_set,
        reassembled.dat(op2.RW), evectors.dat(op2.READ), evalues.dat(op2.READ)
    )
    metric -= reassembled
    if not np.isclose(norm(metric), 0.0):
        raise ValueError("Reassembled metric does not match")
