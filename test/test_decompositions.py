from pyroteus import *
import pyroteus.kernel as kernels
import pytest


def uniform_mesh(dim, n, l):
    return SquareMesh(n, n, l) if dim == 2 else CubeMesh(n, n, n, l)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_eigendecomposition(dim):
    mesh = uniform_mesh(dim, 20, 2)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)

    # Recover Hessian metric for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    metric = hessian_metric(recover_hessian(f, mesh=mesh))

    # Extract the eigendecomposition
    evectors = Function(P1_ten)
    evalues = Function(P1_vec)
    kernel = kernels.eigen_kernel(kernels.get_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set,
                 evectors.dat(op2.RW), evalues.dat(op2.RW), metric.dat(op2.READ))

    # Check eigenvectors are orthogonal
    VVT = interpolate(dot(evectors, transpose(evectors)), P1_ten)
    I = interpolate(Identity(dim), P1_ten)
    assert np.allclose(VVT.dat.data, I.dat.data)

    # Reassemble it and check the two match
    reassembled = Function(P1_ten)
    kernel = kernels.eigen_kernel(kernels.set_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set,
                 reassembled.dat(op2.RW), evectors.dat(op2.READ), evalues.dat(op2.READ))
    assert np.allclose(metric.dat.data, reassembled.dat.data)
