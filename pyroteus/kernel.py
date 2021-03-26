from firedrake import op2
import os

try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except ImportError:
    PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


def eigen_kernel(kernel, *args, **kwargs):
    """
    Helper function to easily pass Eigen kernels to Firedrake via PyOP2.
    """
    return op2.Kernel(kernel(*args, **kwargs), kernel.__name__, cpp=True, include_dirs=include_dir)


def postproc_metric(d, h_min, h_max, a_max):
    """
    Post-process a metric field in order to enforce max/min element sizes and anisotropy.

    :arg d: spatial dimension.
    :arg h_min: minimum element size.
    :arg h_max: maximum element size.
    :arg a_max: maximum element anisotropy.
    """
    kernel_str = """
#include <Eigen/Dense>

using namespace Eigen;

void postproc_metric(double A_[%d])
{

  // Map input/output metric onto an Eigen object
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  // Scale eigenvalues appropriately
  int i;
  double max_eig = 0.0;
  for (i=0; i<%d; i++) {
    D(i) = fmin(pow(%f, -2), fmax(pow(%f, -2), abs(D(i))));
    max_eig = fmax(max_eig, D(i));
  }
  for (i=0; i<%d; i++) D(i) = fmax(D(i), pow(%f, -2) * max_eig);

  // Build metric from eigendecomposition
  A = Q * D.asDiagonal() * Q.transpose();
}
"""
    return kernel_str % (d*d, d, d, d, d, d, d, d, d, h_min, h_max, d, a_max)
