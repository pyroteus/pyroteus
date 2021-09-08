from pyroteus import *


def test_minAngle2d():
    try:
        from firedrake.slate.slac.compiler import PETSC_ARCH
    except ImportError:
        import os
        PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
    include_dirs = ["%s/include/eigen3" % PETSC_ARCH]

    mesh = UnitSquareMesh(10, 10)
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    min_angles = Function(P0)
    kernel = op2.Kernel(get_min_angle2d(), "get_min_angle", cpp=True, include_dirs=include_dirs)
    op2.par_loop(kernel, mesh.cell_set, min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    true_vals = np.array([np.pi / 4 for _ in min_angles.dat.data])
    assert np.allclose(true_vals, min_angles.dat.data)
