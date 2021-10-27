from firedrake import *
from pyroteus import *
import pyroteus.mesh_quality as mq
import pytest


def get_mesh(dim, n):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Dimension {dim} not considered")


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_facet_areas(dim):
    """
    Check that the computation of facet areas
    sums to the expected value for a uniform
    (isotropic) triangular or tetrahedral mesh.
    """
    mesh = get_mesh(dim, 1)
    fa = get_facet_areas(mesh)
    expected = 5.414214 if dim == 2 else 10.242641
    assert np.isclose(sum(fa.dat.data), expected)


@pytest.mark.parametrize("measure,expected",
                         [
                             ("get_min_angles2d", np.pi/4),
                             ("get_areas2d", 0.005),
                             ("get_eskews2d", 1.070796),
                             ("get_aspect_ratios2d", 1.207107),
                             ("get_scaled_jacobians2d", 0.707107),
                             ("get_skewnesses2d", 0.463648),
                             ("get_quality_metrics2d", 6.928203),
                         ],
                         ids=[
                             "minimum_angle_2d",
                             "area_2d",
                             "equiangle_skew_2d",
                             "aspect_ratio_2d",
                             "scaled_jacobian_2d",
                             "skewness_2d",
                             "quality_metric_2d",
                         ])
def test_uniform_quality_2d(measure, expected):
    """
    Check that the computation of each quality measure
    gives the expected value for a uniform (isotropic)
    2D triangular mesh.
    """
    measure = getattr(mq, measure)
    mesh = get_mesh(2, 10)
    if measure.__name__ == "get_quality_metrics2d":
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        M = interpolate(Identity(2), P1_ten)
        q = measure(mesh, M)
    else:
        q = measure(mesh)
    true_vals = np.array([expected for _ in q.dat.data])
    assert np.allclose(true_vals, q.dat.data)
    if measure.__name__ == "get_areas2d":
        assert np.isclose(sum(q.dat.data), 1)


@pytest.mark.parametrize("measure, expected",
                         [
                             ("get_min_angles3d", 0.61547971),
                             ("get_volumes3d", 0.00260417),
                             ("get_eskews3d", 0.41226017),
                             ("get_aspect_ratios3d", 1.39384685),
                             ("get_scaled_jacobians3d", 0.40824829),
                             ("get_quality_metrics3d", 1.25)
                         ],
                         ids=[
                             "minimum_angle_3d",
                             "volume_3d",
                             "eskew_3d",
                             "aspect_ratio_3d",
                             "scaled_jacobian_3d",
                             "quality_metric_3d"
                         ])
def test_uniform_quality_3d(measure, expected):
    """
    Check that the computation of each quality measure
    gives the expected value for a uniform (isotropic)
    3D tetrahedral mesh.
    """
    measure = getattr(mq, measure)
    mesh = get_mesh(3, 4)
    if measure.__name__ == "get_quality_metrics3d":
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        M = interpolate(Identity(3), P1_ten)
        q = measure(mesh, M)
    else:
        q = measure(mesh)
    true_vals = np.array([expected for _ in q.dat.data])
    assert np.allclose(true_vals, q.dat.data)
    if measure.__name__ == "get_volumes3d":
        assert np.isclose(sum(q.dat.data), 1)


@pytest.mark.parametrize("measure",
                         [
                             ("get_areas2d"),
                             ("get_aspect_ratios2d"),
                             ("get_scaled_jacobians2d"),
                         ],
                         ids=[
                             "area_2d",
                             "aspect_ratio_2d",
                             "scaled_jacobian_2d",
                         ])
def test_consistency_2d(measure):
    """
    Check that the C++ and Python implementations of the
    quality measures are consistent for a non-uniform
    2D triangular mesh.
    """
    np.random.seed(0)
    measure = getattr(mq, measure)
    mesh = get_mesh(2, 4)
    mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
    quality_cpp = measure(mesh)
    quality_py = measure(mesh, python=True)
    assert np.allclose(quality_cpp.dat.data, quality_py.dat.data)


@pytest.mark.parametrize("measure",
                         [
                             ("get_volumes3d")
                         ],
                         ids=[
                             ("volume_3d")
                         ])
def test_consistency_3d(measure):
    """
    Check that the C++ and Python implementations of the
    quality measures are consistent for a non-uniform
    3D tetrahedral mesh.
    """
    np.random.seed(0)
    measure = getattr(mq, measure)
    mesh = get_mesh(3, 4)
    mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
    quality_cpp = measure(mesh)
    quality_py = measure(mesh, python=True)
    assert np.allclose(quality_cpp.dat.data, quality_py.dat.data)
