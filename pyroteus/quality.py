import os
import firedrake
from firedrake import op2
from firedrake.petsc import PETSc
import ufl


try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except ImportError:
    PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


class QualityKernelHandler():
    """
    Class for generating PyOP2 :class:`Kernel`
    objects from Mesh Quality C++ code that exists in
    Pyroteus. Requires Eigen C++ library.
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

        qual_kernels = os.path.join(os.path.dirname(__file__), "cxx/quality{:d}d.cxx")
        return open(qual_kernels.format(d)).read()

    @staticmethod
    def get_pyop2_kernel(kernel, *args, **kwargs):
        """
        Helper function to easily pass Eigen kernels
        to Firedrake via PyOP2.

        :arg kernel: a string containing C code which
            is to be formatted.
        """
        return op2.Kernel(kernel(*args, **kwargs), kernel.__name__, cpp=True, 
                          include_dirs=include_dir)


get_min_angle = QualityKernelHandler("get_min_angle")
get_area = QualityKernelHandler("get_area", d_restrict=[2])
get_volume = QualityKernelHandler("get_volume", d_restrict=[3])
get_eskew = QualityKernelHandler("get_eskew")
get_aspect_ratio = QualityKernelHandler("get_aspect_ratio")
get_scaled_jacobian = QualityKernelHandler("get_scaled_jacobian")
get_skewness = QualityKernelHandler("get_skewness", d_restrict=[2])
get_metric = QualityKernelHandler("get_metric")


@PETSc.Log.EventDecorator("pyroteus.get_min_angles2d")
def get_min_angles2d(mesh, python=False):
    """
    Computes the minimum angle of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function min_angles with
        minimum angle data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        min_angles = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_min_angle, 2)
        op2.par_loop(kernel, mesh.cell_set, min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return min_angles


@PETSc.Log.EventDecorator("pyroteus.get_min_angles3d")
def get_min_angles3d(mesh, python=False):
    """
    Computes the minimum angle of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function min_angles with
        minimum angle data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        min_angles = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_min_angle, 3)
        op2.par_loop(kernel, mesh.cell_set, min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return min_angles


@PETSc.Log.EventDecorator("pyroteus.get_areas2d")
def get_areas2d(mesh, python=False):
    """
    Computes the area of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function areas with
        area data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        areas = firedrake.interpolate(ufl.CellVolume(mesh), P0)
    else:
        coords = mesh.coordinates
        areas = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_area, 2)
        op2.par_loop(kernel, mesh.cell_set, areas.dat(op2.WRITE, areas.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return areas


@PETSc.Log.EventDecorator("pyroteus.get_volumes3d")
def get_volumes3d(mesh, python=False):
    """
    Computes the volume of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function volumes with
        volume data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        volumes = firedrake.interpolate(ufl.CellVolume(mesh), P0)
    else:
        coords = mesh.coordinates
        volumes = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_volume, 3)
        op2.par_loop(kernel, mesh.cell_set, volumes.dat(op2.WRITE, volumes.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return volumes


def get_facet_areas(mesh):
    """
    Compute area of each facet of `mesh`.

    The facet areas are stored as a HDiv
    trace field.

    Note that the plus sign is arbitrary
    and could equally well be chosen as minus.

    :arg mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function facet_areas with
        facet area data
    """
    HDivTrace = firedrake.FunctionSpace(mesh, "HDiv Trace", 0)
    v = firedrake.TestFunction(HDivTrace)
    u = firedrake.TrialFunction(HDivTrace)
    facet_areas = firedrake.Function(HDivTrace, name="Facet areas")
    mass_term = v('+')*u('+')*ufl.dS + v*u*ufl.ds
    rhs = v('+')*ufl.FacetArea(mesh)*ufl.dS + v*ufl.FacetArea(mesh)*ufl.ds
    sp = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
    firedrake.solve(mass_term == rhs, facet_areas, solver_parameters=sp)
    return facet_areas


@PETSc.Log.EventDecorator("pyroteus.get_facet_areas2d")
def get_facet_areas2d(mesh, python=True):
    """
    Computes the area of each facet in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function facet_areas with
        facet area data
    """
    if python:
        return get_facet_areas(mesh)
    else:
        raise NotImplementedError


@PETSc.Log.EventDecorator("pyroteus.get_facet_areas3d")
def get_facet_areas3d(mesh, python=True):
    """
    Computes the area of each facet in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function facet_areas with
        facet area data
    """
    if python:
        return get_facet_areas(mesh)
    else:
        raise NotImplementedError


@PETSc.Log.EventDecorator("pyroteus.get_aspect_ratios2d")
def get_aspect_ratios2d(mesh, python=False):
    """
    Computes the aspect ratio of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function aspect_ratios with
        aspect ratio data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        J = firedrake.interpolate(ufl.Jacobian(mesh), P0_ten)
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        a = ufl.sqrt(ufl.dot(edge1, edge1))
        b = ufl.sqrt(ufl.dot(edge2, edge2))
        c = ufl.sqrt(ufl.dot(edge3, edge3))
        aspect_ratios = firedrake.interpolate(a*b*c/((a+b-c)*(b+c-a)*(c+a-b)), P0)
    else:
        coords = mesh.coordinates
        aspect_ratios = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_aspect_ratio, 2)
        op2.par_loop(kernel, mesh.cell_set, 
                     aspect_ratios.dat(op2.WRITE, aspect_ratios.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return aspect_ratios


@PETSc.Log.EventDecorator("pyroteus.get_aspect_ratios3d")
def get_aspect_ratios3d(mesh, python=False):
    """
    Computes the aspect ratio of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function aspect_ratios with
        aspect ratio data
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        aspect_ratios = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_aspect_ratio, 3)
        op2.par_loop(kernel, mesh.cell_set, 
                     aspect_ratios.dat(op2.WRITE, aspect_ratios.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return aspect_ratios


@PETSc.Log.EventDecorator("pyroteus.get_eskews2d")
def get_eskews2d(mesh, python=False):
    """
    Computes the equiangle skew of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function eskews with equiangle skew
        data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        eskews = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_eskew, 2)
        op2.par_loop(kernel, mesh.cell_set, eskews.dat(op2.WRITE, eskews.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return eskews


@PETSc.Log.EventDecorator("pyroteus.get_eskews3d")
def get_eskews3d(mesh, python=False):
    """
    Computes the equiangle skew of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function eskews with equiangle skew
        data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        eskews = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_eskew, 3)
        op2.par_loop(kernel, mesh.cell_set, eskews.dat(op2.WRITE, eskews.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return eskews


@PETSc.Log.EventDecorator("pyroteus.get_skewnesses2d")
def get_skewnesses2d(mesh, python=False):
    """
    Computes the skewness of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function skews with skewness
        data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        skews = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_skewness, 2)
        op2.par_loop(kernel, mesh.cell_set, skews.dat(op2.WRITE, skews.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return skews


@PETSc.Log.EventDecorator("pyroteus.get_skewnesses3d")
def get_skewnesses3d(mesh, python=False):
    """
    Computes the skewness of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function skews with skewness
        data.
    """
    raise NotImplementedError


@PETSc.Log.EventDecorator("pyroteus.get_scaled_jacobians2d")
def get_scaled_jacobians2d(mesh, python=False):
    """
    Computes the scaled Jacobian of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function scaled_jacobians with scaled
        jacobian data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        J = firedrake.interpolate(ufl.Jacobian(mesh), P0_ten)
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        a = ufl.sqrt(ufl.dot(edge1, edge1))
        b = ufl.sqrt(ufl.dot(edge2, edge2))
        c = ufl.sqrt(ufl.dot(edge3, edge3))
        detJ = ufl.JacobianDeterminant(mesh)
        jacobian_sign = ufl.sign(detJ)
        max_product = ufl.Max(
            ufl.Max(ufl.Max(a*b, a*c), ufl.Max(b*c, b*a)), ufl.Max(c*a, c*b))
        scaled_jacobians = firedrake.interpolate(detJ/max_product*jacobian_sign, P0)
    else:
        coords = mesh.coordinates
        scaled_jacobians = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_scaled_jacobian, 2)
        op2.par_loop(kernel, mesh.cell_set, 
                     scaled_jacobians.dat(op2.WRITE, scaled_jacobians.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return scaled_jacobians


@PETSc.Log.EventDecorator("pyroteus.get_scaled_jacobians3d")
def get_scaled_jacobians3d(mesh, python=False):
    """
    Computes the scaled Jacobian of each cell in a 3D tetrahedral

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function scaled_jacobians with scaled
        jacobian data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        scaled_jacobians = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_scaled_jacobian, 3)
        op2.par_loop(kernel, mesh.cell_set, 
                     scaled_jacobians.dat(op2.WRITE, scaled_jacobians.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return scaled_jacobians


@PETSc.Log.EventDecorator("pyroteus.get_quality_metrics2d")
def get_quality_metrics2d(mesh, metric, python=False):
    """
    Given a Riemannian metric, M, this function
    outputs the value of the Quality metric Q_M based on the
    transformation encoded in M.

    :arg mesh: the input mesh to do computations on
    :arg M: the (2 x 2) matrix field representing the metric space transformation
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function metrics with metric data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        quality = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_metric, 2)
        op2.par_loop(kernel, mesh.cell_set, quality.dat(op2.WRITE, quality.cell_node_map()),
                     metric.dat(op2.READ, metric.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return quality


@PETSc.Log.EventDecorator("pyroteus.get_quality_metrics3d")
def get_quality_metrics3d(mesh, metric, python=False):
    """
    Given a Riemannian metric, M, this function
    outputs the value of the Quality metric Q_M based on the
    transformation encoded in M.

    :arg mesh: the input mesh to do computations on
    :arg M: the (3 x 3) matrix field representing the metric space transformation
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function metrics with metric data.
    """
    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        quality = firedrake.Function(P0)
        kernel = QualityKernelHandler.get_pyop2_kernel(get_metric, 3)
        op2.par_loop(kernel, mesh.cell_set, quality.dat(op2.WRITE, quality.cell_node_map()),
                     metric.dat(op2.READ, metric.cell_node_map()),
                     coords.dat(op2.READ, coords.cell_node_map()))
    return quality
