from pyroteus import *
import pyroteus.kernel as kernels
import pytest

@pytest.fixture
def include_dirs():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    return ["%s/include/eigen3" % PETSC_ARCH]


@pytest.fixture
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture
def P0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture
def coords(mesh):
    return mesh.coordinates


def test_minAngle2d(include_dirs, mesh, P0, coords):
    """
    Check computation of minimum angle for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the minimum angle should be 
    pi/4 for each cell.
    """
    min_angles = Function(P0)
    kernel = op2.Kernel(kernels.get_min_angle2d(), "get_min_angle", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    true_vals = np.array([np.pi / 4 for _ in min_angles.dat.data])
    assert np.allclose(true_vals, min_angles.dat.data)


def test_area2d(include_dirs, mesh, P0, coords):
    """
    Check computation of areas for a 2D triangular mesh.
    Sum of areas of all cells should equal 1 on a UnitSquareMesh.
    Areas of all cells must also be equal.
    """
    areas = Function(P0)
    kernel = op2.Kernel(kernels.get_area2d(), "get_area", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, areas.dat(op2.WRITE, areas.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    true_vals = np.array([areas.dat.data[0] for _ in areas.dat.data])
    assert np.allclose(true_vals, areas.dat.data)
    assert (np.isclose(sum(areas.dat.data), 1))


def test_eskew2d(include_dirs, mesh, P0, coords):
    """
    Check computation of equiangle skew for a 2D triangular mesh.
    For a uniform (isotropic) mesh, the equiangle skew should be
    equal for all elements.
    """
    eskews = Function(P0)
    kernel = op2.Kernel(kernels.get_eskew2d(), "get_eskew", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, eskews.dat(op2.WRITE, eskews.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    true_vals = np.array([eskews.dat.data[0] for _ in eskews.dat.data])
    assert np.allclose(true_vals, eskews.dat.data)