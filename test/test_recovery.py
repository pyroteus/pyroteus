"""
Test derivative recovery techniques.
"""
from pyroteus import *
from sensors import bowl, mesh_for_sensors
import pytest


@pytest.mark.parametrize("dim,method,norm_type,ignore_boundary",
                         [
                             (2, "L2", "L1", False),
                             (2, "L2", "L2", False),
                             (2, "L2", "l2", False),
                             (3, "L2", "L1", False),
                             (3, "L2", "L2", False),
                             (3, "L2", "l2", False),
                             (2, "Clement", "L1", True),
                             (2, "Clement", "L2", True),
                             (2, "Clement", "l2", True),
                             (3, "Clement", "L1", True),
                             (3, "Clement", "L2", True),
                             (3, "Clement", "l2", True),
                         ],
                         ids=[
                             "double_L2_projection-2d-L1_norm",
                             "double_L2_projection-2d-L2_norm",
                             "double_L2_projection-2d-l2_norm",
                             "double_L2_projection-3d-L1_norm",
                             "double_L2_projection-3d-L2_norm",
                             "double_L2_projection-3d-l2_norm",
                             "Clement_interpolation-2d-L1_norm",
                             "Clement_interpolation-2d-L2_norm",
                             "Clement_interpolation-2d-l2_norm",
                             "Clement_interpolation-3d-L1_norm",
                             "Clement_interpolation-3d-L2_norm",
                             "Clement_interpolation-3d-l2_norm",
                         ])
def test_recover_bowl_interior(dim, method, norm_type, ignore_boundary, rtol=1.0e-05):
    """
    Check that the Hessian of a quadratic function is accurately
    recovered in the domain interior.

    :arg dim: the spatial dimension
    :arg method: choose from 'L2', 'Clement'
    :arg norm_type: choose from 'l1', 'l2', 'linf',
        'L2', 'Linf', 'H1', 'Hdiv', 'Hcurl', or any
        'Lp' with :math:`p >= 1`.
    :arg ignore_boundary: if `True`, two outer layers
        of mesh elements are ignored when computing
        error norms
    :kwarg rtol: relative tolerance for error checking
    """
    mesh = mesh_for_sensors(dim, 20)

    # Recover Hessian
    f = bowl(*mesh.coordinates)
    H = recover_hessian(f, method=method, mesh=mesh)

    # Construct analytical solution
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    I = interpolate(Identity(dim), P1_ten)

    # Check that they agree
    cond = None
    if ignore_boundary:
        x = SpatialCoordinate(mesh)
        cond = And(x[0] > -0.8, x[0] < 0.8)
        for i in range(1, dim):
            cond = And(cond, And(x[i] > -0.8, x[i] < 0.8))
        cond = conditional(cond, 1, 0)
    err = errornorm(H, I, norm_type=norm_type, condition=cond)
    err /= norm(I, norm_type=norm_type, condition=cond)
    assert err < rtol, f"FAILED: non-zero {norm_type} error for method '{method}' ({err:.4e})"
    print(f"PASS: {norm_type} error {err:.4e} for method '{method}', dimension {dim}")


@pytest.mark.parametrize("dim,method",
                         [
                             (2, "L2"),
                             (3, "L2"),
                         ],
                         ids=[
                             "double_L2_projection-2d",
                             "double_L2_projection-3d",
                         ])
def test_recover_bowl_boundary(dim, method, tol=1.0e-08):
    """
    Check that the Hessian of a quadratic function is accurately
    recovered on the domain boundary.

    :arg dim: the spatial dimension
    :arg method: choose from 'L2', 'Clement'
    :kwarg tol: absolute tolerance for checking
        that the component of the Hessian which
        is tangential to the boundary is zero
    """
    mesh = mesh_for_sensors(dim, 20)
    if dim == 3:
        pytest.xfail("FIXME: 3D case broken")  # FIXME

    # Recover boundary Hessian
    f = bowl(*mesh.coordinates)
    tags = list(mesh.exterior_facets.unique_markers) + ['interior']
    f = {i: f for i in tags}
    H = recover_boundary_hessian(f, method=method, mesh=mesh)

    # Check its directional derivatives in boundaries are zero
    S = construct_orthonormal_basis(FacetNormal(mesh))
    for s in S:
        dHds = abs(assemble(dot(div(H), s)*ds))
        assert dHds < tol, "FAILED: non-zero tangential derivative for method" \
            + f"'{method}' ({dHds:.4e})"
    print(f"PASS: method '{method}', dimension {dim}")
