"""
Test derivative recovery techniques.
"""
from pyroteus import *
from sensors import bowl, mesh_for_sensors
import pytest


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=['L2'])
def method(request):
    return request.param


@pytest.fixture(params=['L2', 'l2'])
def norm_type(request):
    return request.param


def test_recover_bowl(dim, method, norm_type, rtol=1.0e-05):
    """
    Check that the Hessian of a quadratic function is accurately
    recovered in the domain interior.
    """
    mesh = mesh_for_sensors(dim, 20)

    # Recover Hessian
    f = bowl(*mesh.coordinates)
    H = recover_hessian(f, method=method, mesh=mesh)

    # Construct analytical solution
    I = interpolate(Identity(dim), H.function_space())

    # Check that they agree
    err = errornorm(H, I, norm_type=norm_type)/norm(I, norm_type=norm_type)
    msg = "FAILED: non-zero {:s} error for method '{:s}' ({:.4e})"
    assert err < rtol, msg.format(norm_type, method, err)


@pytest.mark.parallel
def test_recover_bowl_parallel(dim, method, norm_type, rtol=1.0e-05):
    test_recover_bowl(dim, method, norm_type, rtol=rtol)


def test_recover_boundary_bowl(dim, method):
    """
    Check that the Hessian of a quadratic function is accurately
    recovered on the domain boundary.
    """
    mesh = mesh_for_sensors(dim, 20)
    tol = 0.6 if dim == 3 else 1.0e-08  # TODO: improve 3D case

    # Recover boundary Hessian
    f = bowl(*mesh.coordinates)
    tags = list(mesh.exterior_facets.unique_markers) + ['interior']
    f = {i: f for i in tags}
    H = recover_boundary_hessian(f, method=method, mesh=mesh)

    # Check its directional derivatives in boundaries are zero
    S = construct_orthonormal_basis(FacetNormal(mesh))
    for s in S:
        dHds = assemble(dot(div(H), s)*ds)
        msg = "FAILED: non-zero tangential derivative for method '{:s}' ({:.4e})"
        assert abs(dHds) < tol, msg.format(method, abs(dHds))
