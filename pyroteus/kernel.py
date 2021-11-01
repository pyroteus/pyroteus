"""
Functions which generate C++ kernels for dense numerical linear algebra.
"""
from firedrake import op2
import os

try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except ImportError:
    PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


def eigen_kernel(kernel, *args, **kwargs):
    """
    Helper function to easily pass Eigen kernels
    to Firedrake via PyOP2.

    :arg kernel: a string containing C code which
        is to be formatted.
    """
    return op2.Kernel(kernel(*args, **kwargs), kernel.__name__, cpp=True, include_dirs=include_dir)


class EigenKernelHandler(object):
    """
    Class for generating PyOP2 :class:`Kernel`
    objects from Eigen C++ code that exists in
    Pyroteus.
    """
    def __init__(self, name):
        """
        :arg name: the name of the routine
        """
        self.__name__ = name

    def __call__(self, d):
        """
        :arg d: the spatial dimension
        """
        assert d in (2, 3), f"Spatial dimension {d} not supported."
        eigen_kernels = os.path.join(os.path.dirname(__file__), "cxx/eigen_kernels{:d}d.cxx")
        return open(eigen_kernels.format(d)).read()


# Currently implemented kernels
set_eigendecomposition = EigenKernelHandler("set_eigendecomposition")
get_eigendecomposition = EigenKernelHandler("get_eigendecomposition")
get_reordered_eigendecomposition = EigenKernelHandler("get_reordered_eigendecomposition")
metric_from_hessian = EigenKernelHandler("metric_from_hessian")
postproc_metric = EigenKernelHandler("postproc_metric")
intersect = EigenKernelHandler("intersect")


def get_min_angle2d():
    """
    Compute the minimum angle of each cell
    in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_min_angle2d(double *MinAngles, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Compute angles from cosine formula
  double a1 = acos (v12.dot(v13) / (d12 * d13));
  double a2 = acos (-v12.dot(v23) / (d12 * d23));
  double a3 = acos (v23.dot(v13) / (d23 * d13));
  double aMin = std::min(a1, a2);
  MinAngles[0] = std::min(aMin, a3);
}
"""


def get_min_angle3d():
    """
    Compute the minimum angle of each cell
    in a 3D tetrahedral mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_min_angle3d(double *MinAngles, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  double angles[12];
  // Compute angles from cosine formula
  angles[0] = acos(v13.dot(v14) / (d13 * d14));
  angles[1] = acos(v12.dot(v14) / (d12 * d14));
  angles[2] = acos(v13.dot(v12) / (d13 * d12));
  angles[3] = acos(v23.dot(v24) / (d23 * d24));
  angles[4] = acos(-v12.dot(v24) / (d12 * d24));
  angles[5] = acos(-v12.dot(v23) / (d12 * d23));
  angles[6] = acos(-v23.dot(v34) / (d23 * d34));
  angles[7] = acos(-v13.dot(v34) / (d13 * d34));
  angles[8] = acos(v13.dot(v23) / (d13 * d23));
  angles[9] = acos(v24.dot(v34) / (d24 * d34));
  angles[10] = acos(v14.dot(v34) / (d14 * d34));
  angles[11] = acos(v14.dot(v24) / (d14 * d24));

  double aMin = 3.14;
  for (int i = 0; i < 12; i++) {
    aMin = std::min(aMin, angles[i]);
  }

  MinAngles[0] = aMin;
}
"""


def get_area2d():
    """
    Compute the area of each cell
    in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_area2d(double *Areas, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge lengths
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);
  double s = (d12 + d23 + d13) / 2;
  // Compute area using Heron's formula
  Areas[0] = sqrt(s * (s - d12) * (s - d23) * (s - d13));
}
"""


def get_volume3d():
    """
    Compute the volume of each cell in
    a 3D tetrahedral mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_volume3d(double *Volumes, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  Matrix3d volumeMatrix;
  for (int i = 0; i < 3; i++) {
    volumeMatrix(0, i) = v12[i];
    volumeMatrix(1, i) = v13[i];
    volumeMatrix(2, i) = v14[i];
  }
  Volumes[0] = std::abs(volumeMatrix.determinant() / 6);
}
"""


def get_eskew2d():
    """
    Compute the area of each cell
    in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_eskew2d(double *ESkews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Compute angles from cosine formula
  double a1 = acos (v12.dot(v13) / (d12 * d13));
  double a2 = acos (-v12.dot(v23) / (d12 * d23));
  double a3 = acos (v23.dot(v13) / (d23 * d13));
  double pi = 3.14159265358979323846;

  // Plug values into equiangle skew formula as per:
  // http://www.lcad.icmc.usp.br/~buscaglia/teaching/mfcpos2013/bakker_07-mesh.pdf
  double aMin = std::min(a1, a2);
  aMin = std::min(aMin, a3);
  double aMax = std::max(a1, a2);
  aMax = std::max(aMax, a3);
  double aIdeal = pi / 3;
  ESkews[0] = std::max((aMax - aIdeal / (pi - aIdeal)), (aIdeal - aMin) / aIdeal);
}
"""


def get_eskew3d():
    """
    Compute the equiangle skew of each
    cell in a 3D tetrahedral mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_eskew3d(double *ESkews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  double angles[12];
  // Compute angles from cosine formula
  angles[0] = acos(v13.dot(v14) / (d13 * d14));
  angles[1] = acos(v12.dot(v14) / (d12 * d14));
  angles[2] = acos(v13.dot(v12) / (d13 * d12));
  angles[3] = acos(v23.dot(v24) / (d23 * d24));
  angles[4] = acos(-v12.dot(v24) / (d12 * d24));
  angles[5] = acos(-v12.dot(v23) / (d12 * d23));
  angles[6] = acos(-v23.dot(v34) / (d23 * d34));
  angles[7] = acos(-v13.dot(v34) / (d13 * d34));
  angles[8] = acos(-v13.dot(-v23) / (d13 * d23));
  angles[9] = acos(-v24.dot(-v34) / (d24 * d34));
  angles[10] = acos(-v14.dot(-v34) / (d14 * d34));
  angles[11] = acos(-v14.dot(-v24) / (d14 * d24));
  double pi = 3.14159265358979323846;

  double aMin = pi;
  double aMax = 0.0;
  for (int i = 0; i < 12; i++) {
    aMin = std::min(aMin, angles[i]);
    aMax = std::max(aMax, angles[i]);
  }
  double aIdeal = pi / 3;
  ESkews[0] = std::max((aMax - aIdeal) / (pi - aIdeal), (aIdeal - aMin) / aIdeal);
}
"""


def get_aspect_ratio2d():
    """
    Compute the area of each cell
    in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_aspect_ratio2d(double *AspectRatios, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);
  double s = (d12 + d23 + d13) / 2;

  // Calculate aspect ratio based on the circumradius and inradius as per:
  // https://stackoverflow.com/questions/10289752/aspect-ratio-of-a-triangle-of-a-meshed-surface
  AspectRatios[0] = (d12 * d23 * d13) / (8 * (s - d12) * (s - d23) * (s - d13));
}
"""


def get_aspect_ratio3d():
    """
    Compute the aspect ratio of each cell
    in a 3D tetrahedral mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_aspect_ratio3d(double *AspectRatios, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d volumeMatrix;
  for (int i = 0; i < 3; i++) {
    volumeMatrix(0, i) = v12[i];
    volumeMatrix(1, i) = v13[i];
    volumeMatrix(2, i) = v14[i];
  }
  double volume = std::abs(volumeMatrix.determinant() / 6);

  // Reference for inradius and circumradius calculations on the tetrahedron
  // https://en.wikipedia.org/wiki/Tetrahedron#Inradius
  double cir_radius = sqrt((d12 * d34 + d13 * d24 + d14 * d23) *
                           (d12 * d34 + d13 * d24 - d14 * d23) *
                           (d12 * d34 - d13 * d24 + d14 * d23) *
                           (-d12 * d34 + d13 * d24 + d14 * d23)) / (24 * volume);

  double s1 = (d23 + d24 + d34) / 2;
  double s2 = (d13 + d14 + d34) / 2;
  double s3 = (d12 + d14 + d24) / 2;
  double s4 = (d12 + d13 + d23) / 2;
  double f_area1 = sqrt(s1 * (s1 - d23) * (s1 - d24) * (s1 - d34));
  double f_area2 = sqrt(s2 * (s2 - d13) * (s2 - d14) * (s2 - d34));
  double f_area3 = sqrt(s3 * (s3 - d12) * (s3 - d14) * (s3 - d24));
  double f_area4 = sqrt(s4 * (s4 - d12) * (s4 - d13) * (s4 - d23));
  double in_radius = 3 * volume / (f_area1 + f_area2 + f_area3 + f_area4);

  AspectRatios[0] = cir_radius / (3 * in_radius);
}
"""


def get_scaled_jacobian2d():
    """
    Compute the scaled jacobian of each
    cell in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_scaled_jacobian2d(double *SJacobians, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Compute edge vectors and distances
  Vector2d v12 = p2 - p1;
  Vector2d v23 = p3 - p2;
  Vector2d v13 = p3 - p1;
  double d12 = distance(p1, p2);
  double d23 = distance(p2, p3);
  double d13 = distance(p1, p3);

  // Definition and calculation reference:
  // https://cubit.sandia.gov/15.5/help_manual/WebHelp/mesh_generation/mesh_quality_assessment/triangular_metrics.htm
  // https://www.osti.gov/biblio/5009
  double sj1 = std::abs(v12[0] * v13[1] - v13[0]*v12[1]) / (d12 * d13);
  double sj2 = std::abs(v12[0] * v23[1] - v23[0]*v12[1]) / (d12 * d23);
  double sj3 = std::abs(v23[0] * v13[1] - v13[0]*v23[1]) / (d13 * d23);
  SJacobians[0] = std::min(sj1, sj2);
  SJacobians[0] = std::min(sj3, SJacobians[0]);
}
"""


def get_scaled_jacobian3d():
    """
    Compute the scaled jacobian of each cell
    in a 3D tetrahedral mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_scaled_jacobian3d(double *SJacobians, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Compute edge vectors and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d M1, M2, M3, M4;
  double sj[4];
  for (int i = 0; i < 3; i++) {
    M1(0, i) = v12[i];
    M1(1, i) = v13[i];
    M1(2, i) = v14[i];

    M2(0, i) = -v12[i];
    M2(1, i) = v23[i];
    M2(2, i) = v24[i];

    M3(0, i) = -v13[i];
    M3(1, i) = -v23[i];
    M3(2, i) = v34[i];

    M4(0, i) = -v14[i];
    M4(1, i) = -v24[i];
    M4(2, i) = -v34[i];
  }
  sj[0] = std::abs(M1.determinant()) / (d12 * d13 * d14);
  sj[1] = std::abs(M2.determinant()) / (d12 * d23 * d24);
  sj[2] = std::abs(M3.determinant()) / (d13 * d23 * d34);
  sj[3] = std::abs(M4.determinant()) / (d14 * d24 * d34);

  SJacobians[0] = std::min(sj[0], sj[1]);
  SJacobians[0] = std::min(SJacobians[0], sj[2]);
  SJacobians[0] = std::min(SJacobians[0], sj[3]);
}
"""


def get_skewness2d():
    """
    Compute the skewness of each cell
    in a 2D triangular mesh.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_skewness2d(double *Skews, double *Coords) {
  // Map coordinates onto Eigen objects
  Map<Vector2d> p1((double *) &Coords[0]);
  Map<Vector2d> p2((double *) &Coords[2]);
  Map<Vector2d> p3((double *) &Coords[4]);

  // Calculating in accordance with:
  // https://www.engmorph.com/skewness-finite-elemnt
  Vector2d midPoint1 = p2 + (p3 - p2) / 2;
  Vector2d midPoint2 = p3 + (p1 - p3) / 2;
  Vector2d midPoint3 = p1 + (p2 - p1) / 2;
  double pi = 3.14159265358979323846;

  Vector2d lineNormal1 = midPoint1 - p1;
  Vector2d lineOrth1 = midPoint3 - midPoint2;
  double t1 = acos (lineNormal1.dot(lineOrth1) / (distance(p1, midPoint1) * distance(midPoint2, midPoint3)));
  double t2 = pi - t1;
  double tMin = std::min(t1, t2);

  Vector2d lineNormal2 = midPoint2 - p2;
  Vector2d lineOrth2 = midPoint1 - midPoint3;
  double t3 = acos (lineNormal2.dot(lineOrth2) / (distance(p2, midPoint2) * distance(midPoint1, midPoint3)));
  double t4 = std::min(t3, pi - t3);
  tMin = std::min(tMin, t4);

  Vector2d lineNormal3 = midPoint3 - p3;
  Vector2d lineOrth3 = midPoint2 - midPoint1;
  double t5 = acos (lineNormal3.dot(lineOrth3) / (distance(p3, midPoint3) * distance(midPoint1, midPoint2)));
  double t6 = std::min(t3, pi - t5);
  tMin = std::min(tMin, t6);

  Skews[0] = pi/2 - tMin;
}
"""


def get_metric2d():
    """
    Given a matrix M, a linear function in 2 dimensions,
    this function outputs the value of the Quality metric Q_M
    based on the transformation encoded in M.
    The suggested use case is to create the matrix M,
    interpolate to all vertices of the mesh and pass it with
    its corresponding cell_node_map() to this kernel.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector2d p1, Vector2d p2)  {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

void get_metric2d(double *Metrics, const double *T_, double *Coords) {
    // Map coordinates onto Eigen objects
    Map<Vector2d> p1((double *) &Coords[0]);
    Map<Vector2d> p2((double *) &Coords[2]);
    Map<Vector2d> p3((double *) &Coords[4]);

    // Compute edge vectors and distances
    Vector2d v12 = p2 - p1;
    Vector2d v23 = p3 - p2;
    Vector2d v13 = p3 - p1;
    double d12 = distance(p1, p2);
    double d23 = distance(p2, p3);
    double d13 = distance(p1, p3);
    double s = (d12 + d23 + d13) / 2;
    double area = sqrt(s * (s-d12) * (s-d13) * (s-d23));

    // Map tensor  function as 2x2 Matrices
    Map<Matrix2d> M1((double *) &T_[0]);
    Map<Matrix2d> M2((double *) &T_[4]);
    Map<Matrix2d> M3((double *) &T_[8]);

    // Compute M(x, y) at centroid x_c to get area_M
    Matrix2d Mxc = (M1 + M2 + M3) / 3;
    double areaM = area * sqrt(Mxc.determinant());

    // Compute (squared) edge lengths in metric space
    double L1 = v23.dot(((M2 + M3)/2) * v23);
    double L2 = v13.dot(((M1 + M3)/2) * v13);
    double L3 = v12.dot(((M1 + M2)/2) * v12);

    // Calculated using Q_M formula in 2D, reference:
    // https://epubs.siam.org/doi/10.1137/090754078
    Metrics[0] = sqrt(3) * (L1 + L2 + L3) / (2 * areaM);
}
"""


def get_metric3d():
    """
    Given a matrix M, a linear function in 3 dimensions,
    this function outputs the value of the Quality metric Q_M
    based on the transformation encoded in M.
    The suggested use case is to create the matrix M,
    interpolate to all vertices of the mesh and pass it with
    its corresponding cell_node_map() to this kernel.
    """
    return """
#include <Eigen/Dense>

using namespace Eigen;

double distance(Vector3d p1, Vector3d p2) {
  return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2));
}

void get_metric3d(double *Metrics, const double *T_, double *Coords) {
  // Map vertices as vectors
  Map<Vector3d> p1((double *) &Coords[0]);
  Map<Vector3d> p2((double *) &Coords[3]);
  Map<Vector3d> p3((double *) &Coords[6]);
  Map<Vector3d> p4((double *) &Coords[9]);

  // Precompute some vectors, and distances
  Vector3d v12 = p2 - p1;
  Vector3d v13 = p3 - p1;
  Vector3d v14 = p4 - p1;
  Vector3d v23 = p3 - p2;
  Vector3d v24 = p4 - p2;
  Vector3d v34 = p4 - p3;

  double d12 = distance(p1, p2);
  double d13 = distance(p1, p3);
  double d14 = distance(p1, p4);
  double d23 = distance(p2, p3);
  double d24 = distance(p2, p4);
  double d34 = distance(p3, p4);

  Matrix3d volMatrix;
  for (int i = 0; i < 3; i++) {
    volMatrix(0, i) = v12[i];
    volMatrix(1, i) = v13[i];
    volMatrix(2, i) = v14[i];
  }

  double volume = std::abs(volMatrix.determinant()) / 6;

  // Map tensor as 3x3 Matrices
  Map<Matrix3d> M1((double *) &T_[0]);
  Map<Matrix3d> M2((double *) &T_[9]);
  Map<Matrix3d> M3((double *) &T_[18]);
  Map<Matrix3d> M4((double *) &T_[27]);

  // Compute M(x, y) at centroid x_c to get area_M
  Matrix3d Mxc = (M1 + M2 + M3 + M4) / 3;
  double volumeM = volume * sqrt(Mxc.determinant());

  // Compute (squared) edge lengths in metric
  double L1 = v12.dot(((M1 + M2)/2) * v12);
  double L2 = v13.dot(((M1 + M3)/2) * v13);
  double L3 = v14.dot(((M1 + M4)/2) * v14);
  double L4 = v23.dot(((M2 + M3)/2) * v23);
  double L5 = v24.dot(((M2 + M4)/2) * v24);
  double L6 = v34.dot(((M3 + M4)/2) * v34);

  Metrics[0] = sqrt(3) * (L1 + L2 + L3 + L4 + L5 + L6) / (216 * volumeM);
}
"""
