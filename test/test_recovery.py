"""
Test derivative recovery techniques.
"""
from firedrake import *
from pyroteus import *
from sensors import bowl, mesh_for_sensors
import pytest
from time import perf_counter


# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.mark.parametrize(
    "dim,method,norm_type,ignore_boundary,mixed",
    [
        (2, "L2", "L1", False, False),
        (2, "L2", "L2", False, False),
        (2, "L2", "l2", False, False),
        (3, "L2", "L1", False, False),
        (3, "L2", "L2", False, False),
        (3, "L2", "l2", False, False),
        (2, "L2", "L1", False, True),
        (2, "L2", "L2", False, True),
        (2, "L2", "l2", False, True),
        (3, "L2", "L1", False, True),
        (3, "L2", "L2", False, True),
        (3, "L2", "l2", False, True),
        (2, "Clement", "L1", True, False),
        (2, "Clement", "L2", True, False),
        (2, "Clement", "l2", True, False),
        (3, "Clement", "L1", True, False),
        (3, "Clement", "L2", True, False),
        (3, "Clement", "l2", True, False),
    ],
    ids=[
        "double_L2_projection-2d-L1_norm",
        "double_L2_projection-2d-L2_norm",
        "double_L2_projection-2d-l2_norm",
        "double_L2_projection-3d-L1_norm",
        "double_L2_projection-3d-L2_norm",
        "double_L2_projection-3d-l2_norm",
        "double_L2_projection-2d-L1_norm-mixed",
        "double_L2_projection-2d-L2_norm-mixed",
        "double_L2_projection-2d-l2_norm-mixed",
        "double_L2_projection-3d-L1_norm-mixed",
        "double_L2_projection-3d-L2_norm-mixed",
        "double_L2_projection-3d-l2_norm-mixed",
        "Clement_interpolation-2d-L1_norm",
        "Clement_interpolation-2d-L2_norm",
        "Clement_interpolation-2d-l2_norm",
        "Clement_interpolation-3d-L1_norm",
        "Clement_interpolation-3d-L2_norm",
        "Clement_interpolation-3d-l2_norm",
    ],
)
def test_recover_bowl_interior(
    dim, method, norm_type, ignore_boundary, mixed, rtol=1.0e-05
):
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
    :arg mixed: should double L2 projection be done
        as a mixed system?
    :kwarg rtol: relative tolerance for error checking
    """
    mesh = mesh_for_sensors(dim, 20)

    # Recover Hessian
    f = bowl(*mesh.coordinates)
    if method == "Clement":
        P1 = FunctionSpace(mesh, "CG", 1)
        f = interpolate(f, P1)
    cpu_time = perf_counter()
    H = recover_hessian(f, method=method, mesh=mesh, mixed=mixed)
    cpu_time = perf_counter() - cpu_time

    # Construct analytical solution
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    I = interpolate(Identity(dim), P1_ten)

    # Check that they agree
    cond = Constant(1.0)
    if ignore_boundary:
        x = SpatialCoordinate(mesh)
        cond = And(x[0] > -0.8, x[0] < 0.8)
        for i in range(1, dim):
            cond = And(cond, And(x[i] > -0.8, x[i] < 0.8))
        cond = conditional(cond, 1, 0)
    err = errornorm(H, I, norm_type=norm_type, condition=cond)
    err /= norm(I, norm_type=norm_type, condition=cond)
    assert (
        err < rtol
    ), f"FAILED: non-zero {norm_type} error for method '{method}' ({err:.4e})"
    print(f"PASS: {norm_type} error {err:.4e} for method '{method}', dimension {dim}")
    return cpu_time


@pytest.mark.parametrize(
    "dim,method",
    [
        (2, "L2"),
        (2, "Clement"),
        (3, "L2"),
        (3, "Clement"),
    ],
    ids=[
        "double_L2_projection-2d",
        "Clement-2d",
        "double_L2_projection-3d",
        "Clement-3d",
    ],
)
def test_recover_bowl_boundary(dim, method, tol=2.0e-08):
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
        pytest.skip("FIXME: 3D case broken")  # FIXME

    # Recover boundary Hessian
    f = bowl(*mesh.coordinates)
    tags = list(mesh.exterior_facets.unique_markers) + ["interior"]
    f = {i: f for i in tags}
    cpu_time = perf_counter()
    H = recover_boundary_hessian(f, mesh, method=method)
    cpu_time = perf_counter() - cpu_time

    # Check its directional derivatives in boundaries are zero
    S = construct_orthonormal_basis(FacetNormal(mesh))
    for s in S:
        dHds = abs(assemble(dot(div(H), s) * ds))
        assert dHds < tol, (
            "FAILED: non-zero tangential derivative for method"
            + f" '{method}' ({dHds:.4e})"
        )
    print(f"PASS: method '{method}', dimension {dim}")
    return cpu_time


# ---------------------------
# benchmarking
# ---------------------------

if __name__ == "__main__":
    output_dir = create_directory(os.path.join(os.path.dirname(__file__), "bench"))
    dL2_2d = np.mean(
        [test_recover_bowl_interior(2, "L2", "L2", True, False) for i in range(5)]
    )
    dL2_2d_m = np.mean(
        [test_recover_bowl_interior(2, "L2", "L2", True, True) for i in range(5)]
    )
    Cle_2d = np.mean(
        [test_recover_bowl_interior(2, "Clement", "L2", True, False) for i in range(5)]
    )
    dL2_3d = np.mean(
        [test_recover_bowl_interior(3, "L2", "L2", True, False) for i in range(5)]
    )
    dL2_3d_m = np.mean(
        [test_recover_bowl_interior(3, "L2", "L2", True, True) for i in range(5)]
    )
    Cle_3d = np.mean(
        [test_recover_bowl_interior(3, "Clement", "L2", True, False) for i in range(5)]
    )
    msg = (
        "\n2D\n==\n"
        + f"mixed double L2 proj: {dL2_2d_m:8.4f}s\n"
        + f"double L2 proj:       {dL2_2d:8.4f}s\n"
        + f"Clement:              {Cle_2d:8.4f}s\n"
        + "\n3D\n==\n"
        + f"mixed double L2 proj: {dL2_3d_m:8.4f}s\n"
        + f"double L2 proj:       {dL2_3d:8.4f}s\n"
        + f"Clement:              {Cle_3d:8.4f}s"
    )
    print(msg)
    with open(os.path.join(output_dir, "recovery.log"), "w+") as log:
        log.write(msg)
