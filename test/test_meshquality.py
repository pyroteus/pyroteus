from pyroteus import *
import pyroteus.mesh_quality as mq
import pytest


@pytest.fixture
def include_dirs():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    return ["%s/include/eigen3" % PETSC_ARCH]


def test_min_angle2d(include_dirs):
    """
    Check computation of minimum angle for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the minimum angle should be
    pi/4 for each cell.
    """
    mesh = UnitSquareMesh(10, 10)
    min_angles = mq.get_min_angles2d(mesh)
    true_vals = np.array([np.pi / 4 for _ in min_angles.dat.data])
    assert np.allclose(true_vals, min_angles.dat.data)


def test_area2d(include_dirs):
    """
    Check computation of areas for a 2D triangular mesh.
    Sum of areas of all cells should equal 1 on a UnitSquareMesh.
    Areas of all cells must also be equal.
    """
    mesh = UnitSquareMesh(10, 10)
    areas = mq.get_areas2d(mesh)
    true_vals = np.array([0.005 for _ in areas.dat.data])
    assert np.allclose(true_vals, areas.dat.data)
    assert (np.isclose(sum(areas.dat.data), 1))


def test_eskew2d(include_dirs):
    """
    Check computation of equiangle skew for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the equiangle skew should be
    equal for all elements.
    """
    mesh = UnitSquareMesh(10, 10)
    eskews = mq.get_eskews2d(mesh)
    true_vals = np.array([1.070796 for _ in eskews.dat.data])
    assert np.allclose(true_vals, eskews.dat.data)


def test_aspect_ratio2d(include_dirs):
    """
    Check computation of aspect ratio for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the aspect ratio should be
    equal for all elements.
    """
    mesh = UnitSquareMesh(10, 10)
    aspect_ratios = mq.get_aspect_ratios2d(mesh)
    true_vals = np.array([1.207107 for _ in aspect_ratios.dat.data])
    assert np.allclose(true_vals, aspect_ratios.dat.data)


def test_scaled_jacobian2d(include_dirs):
    """
    Check computation of scaled Jacobian for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the scaled Jacobian should be
    equal for all elements.
    """
    mesh = UnitSquareMesh(10, 10)
    scaled_jacobians = mq.get_scaled_jacobians2d(mesh)
    true_vals = np.array([0.707107 for _ in scaled_jacobians.dat.data])
    assert np.allclose(true_vals, scaled_jacobians.dat.data)


def test_skewness2d(include_dirs):
    """
    Check computation of skewness for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the skewness should be
    equal for all elements.
    """
    mesh = UnitSquareMesh(10, 10)
    skewnesses = mq.get_skewnesses2d(mesh)
    true_vals = np.array([0.463648 for _ in skewnesses.dat.data])
    assert np.allclose(true_vals, skewnesses.dat.data)


def test_metric2d(include_dirs):
    """
    Check computation of skewness for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the skewness should be
    equal for all elements.
    """
    mesh = UnitSquareMesh(10, 10)
    M = [[1, 0], [0, 1]]
    metrics = mq.get_quality_metrics2d(mesh, M)
    true_vals = np.array([6.928203 for _ in metrics.dat.data])
    assert np.allclose(true_vals, metrics.dat.data)
