from .kernel import *
from firedrake import *


def get_min_angles2d(mesh):
    """
    Computes the minimum angle of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function min_angles with
    minimum angle data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    min_angles = Function(P0)
    kernel = eigen_kernel(get_min_angle2d)
    op2.par_loop(kernel, mesh.cell_set, min_angles.dat(op2.WRITE, min_angles.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return min_angles


def get_areas2d(mesh):
    """
    Computes the area of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function areas with
    area data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    areas = Function(P0)
    kernel = eigen_kernel(get_area2d)
    op2.par_loop(kernel, mesh.cell_set, areas.dat(op2.WRITE, areas.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return areas


def get_aspect_ratios2d(mesh):
    """
    Computes the aspect ratio of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function aspect_ratios with
    aspect ratio data
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    aspect_ratios = Function(P0)
    kernel = eigen_kernel(get_aspect_ratio2d)
    op2.par_loop(kernel, mesh.cell_set, aspect_ratios.dat(op2.WRITE, aspect_ratios.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return aspect_ratios


def get_eskews2d(mesh):
    """
    Computes the equiangle skew of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function eskews with equiangle skew
    data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    eskews = Function(P0)
    kernel = eigen_kernel(get_eskew2d)
    op2.par_loop(kernel, mesh.cell_set, eskews.dat(op2.WRITE, eskews.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return eskews


def get_skewnesses2d(mesh):
    """
    Computes the skewness of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function skews with skewness
    data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    skews = Function(P0)
    kernel = eigen_kernel(get_skewness2d)
    op2.par_loop(kernel, mesh.cell_set, skews.dat(op2.WRITE, skews.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return skews


def get_scaled_jacobians2d(mesh):
    """
    Computes the scaled Jacobian of each cell in a 2D triangular mesh

    :param mesh: the input mesh to do computations on

    :rtype: firedrake.function.Function scaled_jacobians with scaled
    jacobian data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    coords = mesh.coordinates
    scaled_jacobians = Function(P0)
    kernel = eigen_kernel(get_scaled_jacobian2d)
    op2.par_loop(kernel, mesh.cell_set, scaled_jacobians.dat(op2.WRITE, scaled_jacobians.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return scaled_jacobians


def get_quality_metrics2d(mesh, M):
    """
    Given a matrix M, a linear function in 2 dimensions, this function
    outputs the value of the Quality metric Q_M based on the
    transformation encoded in M.

    :param mesh: the input mesh to do computations on
    :param M: the (2 x 2) matrix representing the metric space transformation

    :rtype: firedrake.function.Function metrics with metric data.
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    coords = mesh.coordinates
    tensor = interpolate(as_matrix(M), P1_ten)
    metrics = Function(P0)
    kernel = eigen_kernel(get_metric2d)
    op2.par_loop(kernel, mesh.cell_set, metrics.dat(op2.WRITE, metrics.cell_node_map()),
                 tensor.dat(op2.READ, tensor.cell_node_map()),
                 coords.dat(op2.READ, coords.cell_node_map()))
    return metrics
