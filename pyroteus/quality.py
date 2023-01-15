"""
Functions for computing mesh quality measures.
"""
import os
import firedrake
from firedrake import Function, FunctionSpace, interpolate
from firedrake.mesh import MeshGeometry
from firedrake.petsc import PETSc
from pyop2 import op2
from pyop2.utils import get_petsc_dir
import ufl


PETSC_DIR, PETSC_ARCH = get_petsc_dir()
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


def get_pyop2_kernel(func: Function, dim: int) -> op2.Kernel:
    """
    Helper function to easily pass Eigen kernels
    to Firedrake via PyOP2.

    :arg func: function name
    :arg dim: spatial dimension
    """
    fname = os.path.join(os.path.dirname(__file__), "cxx", f"quality{dim}d.cxx")
    code = open(fname, "r").read()
    return op2.Kernel(code, func, cpp=True, include_dirs=include_dir)


@PETSc.Log.EventDecorator("pyroteus.get_min_angles2d")
def get_min_angles2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the minimum angle of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function min_angles with
        minimum angle data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        min_angles = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_min_angle", 2),
            mesh.cell_set,
            min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return min_angles


@PETSc.Log.EventDecorator("pyroteus.get_min_angles3d")
def get_min_angles3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the minimum angle of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function min_angles with
        minimum angle data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        min_angles = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_min_angle", 3),
            mesh.cell_set,
            min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return min_angles


@PETSc.Log.EventDecorator("pyroteus.get_areas2d")
def get_areas2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the area of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function areas with
        area data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        areas = interpolate(ufl.CellVolume(mesh), P0)
    else:
        coords = mesh.coordinates
        areas = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_area", 2),
            mesh.cell_set,
            areas.dat(op2.WRITE, areas.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return areas


@PETSc.Log.EventDecorator("pyroteus.get_volumes3d")
def get_volumes3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the volume of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function volumes with
        volume data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        volumes = interpolate(ufl.CellVolume(mesh), P0)
    else:
        coords = mesh.coordinates
        volumes = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_volume", 3),
            mesh.cell_set,
            volumes.dat(op2.WRITE, volumes.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return volumes


def get_facet_areas(mesh: MeshGeometry) -> Function:
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
    HDivTrace = FunctionSpace(mesh, "HDiv Trace", 0)
    v = firedrake.TestFunction(HDivTrace)
    u = firedrake.TrialFunction(HDivTrace)
    facet_areas = Function(HDivTrace, name="Facet areas")
    mass_term = v("+") * u("+") * ufl.dS + v * u * ufl.ds
    rhs = v("+") * ufl.FacetArea(mesh) * ufl.dS + v * ufl.FacetArea(mesh) * ufl.ds
    sp = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    }
    firedrake.solve(mass_term == rhs, facet_areas, solver_parameters=sp)
    return facet_areas


@PETSc.Log.EventDecorator("pyroteus.get_facet_areas2d")
def get_facet_areas2d(mesh: MeshGeometry, python: bool = True) -> Function:
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
def get_facet_areas3d(mesh: MeshGeometry, python: bool = True) -> Function:
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
def get_aspect_ratios2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the aspect ratio of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function aspect_ratios with
        aspect ratio data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        J = interpolate(ufl.Jacobian(mesh), P0_ten)
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        a = ufl.sqrt(ufl.dot(edge1, edge1))
        b = ufl.sqrt(ufl.dot(edge2, edge2))
        c = ufl.sqrt(ufl.dot(edge3, edge3))
        aspect_ratios = interpolate(
            a * b * c / ((a + b - c) * (b + c - a) * (c + a - b)), P0
        )
    else:
        coords = mesh.coordinates
        aspect_ratios = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_aspect_ratio", 2),
            mesh.cell_set,
            aspect_ratios.dat(op2.WRITE, aspect_ratios.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return aspect_ratios


@PETSc.Log.EventDecorator("pyroteus.get_aspect_ratios3d")
def get_aspect_ratios3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the aspect ratio of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function aspect_ratios with
        aspect ratio data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        aspect_ratios = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_aspect_ratio", 3),
            mesh.cell_set,
            aspect_ratios.dat(op2.WRITE, aspect_ratios.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return aspect_ratios


@PETSc.Log.EventDecorator("pyroteus.get_eskews2d")
def get_eskews2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the equiangle skew of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function eskews with equiangle skew
        data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        eskews = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_eskew", 2),
            mesh.cell_set,
            eskews.dat(op2.WRITE, eskews.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return eskews


@PETSc.Log.EventDecorator("pyroteus.get_eskews3d")
def get_eskews3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the equiangle skew of each cell in a 3D tetrahedral mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function eskews with equiangle skew
        data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        eskews = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_eskew", 3),
            mesh.cell_set,
            eskews.dat(op2.WRITE, eskews.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return eskews


@PETSc.Log.EventDecorator("pyroteus.get_skewnesses2d")
def get_skewnesses2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the skewness of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function skews with skewness
        data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        skews = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_skewness", 2),
            mesh.cell_set,
            skews.dat(op2.WRITE, skews.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return skews


@PETSc.Log.EventDecorator("pyroteus.get_skewnesses3d")
def get_skewnesses3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the skewness of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function skews with skewness
        data.
    """
    raise NotImplementedError


@PETSc.Log.EventDecorator("pyroteus.get_scaled_jacobians2d")
def get_scaled_jacobians2d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the scaled Jacobian of each cell in a 2D triangular mesh

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function scaled_jacobians with scaled
        jacobian data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        J = interpolate(ufl.Jacobian(mesh), P0_ten)
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        a = ufl.sqrt(ufl.dot(edge1, edge1))
        b = ufl.sqrt(ufl.dot(edge2, edge2))
        c = ufl.sqrt(ufl.dot(edge3, edge3))
        detJ = ufl.JacobianDeterminant(mesh)
        jacobian_sign = ufl.sign(detJ)
        max_product = ufl.Max(
            ufl.Max(ufl.Max(a * b, a * c), ufl.Max(b * c, b * a)), ufl.Max(c * a, c * b)
        )
        scaled_jacobians = interpolate(detJ / max_product * jacobian_sign, P0)
    else:
        coords = mesh.coordinates
        scaled_jacobians = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_scaled_jacobian", 2),
            mesh.cell_set,
            scaled_jacobians.dat(op2.WRITE, scaled_jacobians.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return scaled_jacobians


@PETSc.Log.EventDecorator("pyroteus.get_scaled_jacobians3d")
def get_scaled_jacobians3d(mesh: MeshGeometry, python: bool = False) -> Function:
    """
    Computes the scaled Jacobian of each cell in a 3D tetrahedral

    :arg mesh: the input mesh to do computations on
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function scaled_jacobians with scaled
        jacobian data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        scaled_jacobians = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_scaled_jacobian", 3),
            mesh.cell_set,
            scaled_jacobians.dat(op2.WRITE, scaled_jacobians.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return scaled_jacobians


@PETSc.Log.EventDecorator("pyroteus.get_quality_metrics2d")
def get_quality_metrics2d(
    mesh: MeshGeometry, metric: Function, python: bool = False
) -> Function:
    """
    Given a Riemannian metric, M, this function
    outputs the value of the Quality metric Q_M based on the
    transformation encoded in M.

    :arg mesh: the input mesh to do computations on
    :arg M: the (2 x 2) matrix field representing the metric space transformation
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function metrics with metric data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        quality = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_metric", 2),
            mesh.cell_set,
            quality.dat(op2.WRITE, quality.cell_node_map()),
            metric.dat(op2.READ, metric.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return quality


@PETSc.Log.EventDecorator("pyroteus.get_quality_metrics3d")
def get_quality_metrics3d(
    mesh: MeshGeometry, metric: Function, python: bool = False
) -> Function:
    """
    Given a Riemannian metric, M, this function
    outputs the value of the Quality metric Q_M based on the
    transformation encoded in M.

    :arg mesh: the input mesh to do computations on
    :arg M: the (3 x 3) matrix field representing the metric space transformation
    :kwarg python: compute the measure using Python?

    :rtype: firedrake.function.Function metrics with metric data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    if python:
        raise NotImplementedError
    else:
        coords = mesh.coordinates
        quality = Function(P0)
        op2.par_loop(
            get_pyop2_kernel("get_metric", 3),
            mesh.cell_set,
            quality.dat(op2.WRITE, quality.cell_node_map()),
            metric.dat(op2.READ, metric.cell_node_map()),
            coords.dat(op2.READ, coords.cell_node_map()),
        )
    return quality
