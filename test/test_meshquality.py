from pyroteus import *
import pyroteus.mesh_quality as mq
import pytest


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
    mesh = UnitSquareMesh(10, 10)
    if measure == "get_quality_metrics2d":
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        M = interpolate(Identity(2), P1_ten)
        q = measure(mesh, M)
    else:
        q = measure(mesh)
    true_vals = np.array([expected for _ in q.dat.data])
    assert np.allclose(true_vals, q.dat.data)
    if measure == "get_areas2d":
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
    mesh = UnitSquareMesh(4, 4)
    mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
    quality_cpp = measure(mesh)
    quality_py = measure(mesh, python=True)
    assert np.allclose(quality_cpp.dat.data, quality_py.dat.data)
