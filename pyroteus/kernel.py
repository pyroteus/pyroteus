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


class QualityKernelHandler(object):
    """
    Class for generating PyOP2 :class:`Kernel`
    objects from Eigen C++ code that exists in
    Pyroteus.
    """
    def __init__(self, name, d_restrict=None):
        """
        :arg name: the name of the routine
        :arg d_restrict: dimension restriction (e.g [2] for get_area)
        """
        self.__name__ = name
        if d_restrict is not None:
            self.d_restrict = tuple(d_restrict)
        else:
            self.d_restrict = None

    def __call__(self, d):
        """
        :arg d: the spatial dimension
        """
        assert d in (2, 3), f"Spatial dimension {d} not supported."
        if self.d_restrict is not None:
            assert d in self.d_restrict, f"Spatial dimension {d} not supported for {self.__name__}"

        qual_kernels = os.path.join(os.path.dirname(__file__), "cxx/qual_kernels{}d.cxx".format(d))
        return open(qual_kernels).read()


get_min_angle = QualityKernelHandler("get_min_angle")
get_area = QualityKernelHandler("get_area", d_restrict=[2])
get_volume = QualityKernelHandler("get_volume", d_restrict=[3])
get_eskew = QualityKernelHandler("get_eskew")
get_aspect_ratio = QualityKernelHandler("get_aspect_ratio")
get_scaled_jacobian = QualityKernelHandler("get_scaled_jacobian")
get_skewness = QualityKernelHandler("get_skewness", d_restrict=[2])
get_metric = QualityKernelHandler("get_metric")
