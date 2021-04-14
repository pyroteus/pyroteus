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
    Post-process a metric field in order to enforce
    max/min element sizes and anisotropy.

    :arg d: spatial dimension
    :arg h_min: minimum element size
    :arg h_max: maximum element size
    :arg a_max: maximum element anisotropy
    """
    return """
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
""" % (d*d, d, d, d, d, d, d, d, d, h_min, h_max, d, a_max)


def intersect(d):
    """
    Intersect two metric fields.

    :arg d: spatial dimension
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

void intersect(double M_[%d], const double * A_, const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Solve eigenvalue problem of first metric, taking square root of eigenvalues
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Matrix<double, %d, %d, RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();

  // Compute square root and inverse square root metrics
  Matrix<double, %d, %d, RowMajor> Sq = Q * D * Q.transpose();
  Matrix<double, %d, %d, RowMajor> Sqi = Q * D.inverse() * Q.transpose();

  // Solve eigenvalue problem for triple product of inverse square root metric and the second metric
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();

  // Compute metric intersection
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d)


def get_eigendecomposition(d):
    """
    Extract eigenvectors/eigenvalues from a metric field.

    :arg d: spatial dimension
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

void get_eigendecomposition(double EVecs_[%d], double EVals_[%d], const double * M_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(M);
  EVecs = eigensolver.eigenvectors();
  EVals = eigensolver.eigenvalues();
}
""" % (d*d, d, d, d, d, d, d, d, d)


def metric_from_hessian(d):
    """
    Modify the eigenvalues of a Hessian matrix so
    that it is positive-definite.

    :arg d: spatial dimension
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

void metric_from_hessian(double A_[%d], const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Compute mean diagonal and set values appropriately
  double mean_diag;
  int i,j;
  for (i=0; i<%d-1; i++) {
    for (j=i+1; i<%d; i++) {
      mean_diag = 0.5*(B(i,j) + B(j,i));
      B(i,j) = mean_diag;
      B(j,i) = mean_diag;
    }
  }

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(B);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  // Take modulus of eigenvalues
  for (i=0; i<%d; i++) D(i) = fmax(1e-10, abs(D(i)));

  // Build metric from eigendecomposition
  A += Q * D.asDiagonal() * Q.transpose();
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d, d)


def set_eigendecomposition(d):
    """
    Construct a metric from eigenvectors and eigenvalues as an
    orthogonal eigendecomposition.

    :arg d: spatial dimension
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

void set_eigendecomposition(double M_[%d], const double * EVecs_, const double * EVals_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);

  // Compute metric from eigendecomposition
  M = EVecs * EVals.asDiagonal() * EVecs.transpose();
}
""" % (d*d, d, d, d, d, d)
