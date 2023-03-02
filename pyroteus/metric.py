"""
Driver functions for metric-based mesh adaptation.
"""
from .utility import *
from firedrake.meshadapt import RiemannianMetric
from .interpolation import clement_interpolant
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union


__all__ = [
    "compute_eigendecomposition",
    "assemble_eigendecomposition",
    "isotropic_metric",
    "anisotropic_metric",
    "hessian_metric",
    "enforce_element_constraints",
    "space_time_normalise",
    "metric_intersection",
    "combine_metrics",
    "determine_metric_complexity",
    "density_and_quotients",
    "ramp_complexity",
]


def get_metric_kernel(func: str, dim: int) -> op2.Kernel:
    """
    Helper function to easily pass Eigen kernels
    for metric utilities to Firedrake via PyOP2.

    :arg func: function name
    :arg dim: spatial dimension
    """
    code = open(os.path.join(os.path.dirname(__file__), f"cxx/metric{dim}d.cxx")).read()
    return op2.Kernel(code, func, cpp=True, include_dirs=include_dir)


@PETSc.Log.EventDecorator()
def compute_eigendecomposition(
    metric: RiemannianMetric, reorder: bool = False
) -> Tuple[Function, Function]:
    """
    Compute the eigenvectors and eigenvalues of
    a matrix-valued function.

    :arg M: a :class:`firedrake.meshadapt.RiemannianMetric`
    :kwarg reorder: should the eigendecomposition
        be reordered in order of *descending*
        eigenvalue magnitude?
    :return: eigenvector :class:`firedrake.function.Function` and
        eigenvalue :class:`firedrake.function.Function` from the
        :func:`firedrake.functionspace.TensorFunctionSpace`
        underpinning the metric
    """
    V_ten = metric.function_space()
    if len(V_ten.ufl_element().value_shape()) != 2:
        raise ValueError(
            "Can only compute eigendecompositions of matrix-valued functions."
        )
    mesh = V_ten.mesh()
    fe = (V_ten.ufl_element().family(), V_ten.ufl_element().degree())
    V_vec = firedrake.VectorFunctionSpace(mesh, *fe)
    dim = mesh.topological_dimension()
    evectors, evalues = Function(V_ten), Function(V_vec)
    if reorder:
        kernel = get_metric_kernel("get_reordered_eigendecomposition", dim)
    else:
        kernel = get_metric_kernel("get_eigendecomposition", dim)
    op2.par_loop(
        kernel,
        V_ten.node_set,
        evectors.dat(op2.RW),
        evalues.dat(op2.RW),
        metric.dat(op2.READ),
    )
    return evectors, evalues


@PETSc.Log.EventDecorator()
def assemble_eigendecomposition(
    evectors: Function, evalues: Function
) -> RiemannianMetric:
    """
    Assemble a matrix from its eigenvectors and
    eigenvalues.

    :arg evectors: eigenvector
        :class:`firedrake.function.Function`
    :arg evalues: eigenvalue
        :class:`firedrake.function.Function`
    :return: the assembled matrix as a
        :class:`firedrake.meshadapt.RiemannianMetric`
    """
    V_ten = evectors.function_space()
    fe_ten = V_ten.ufl_element()
    if len(fe_ten.value_shape()) != 2:
        raise ValueError("Eigenvector Function should be rank-2")
    V_vec = evalues.function_space()
    fe_vec = V_vec.ufl_element()
    if len(fe_vec.value_shape()) != 1:
        raise ValueError("Eigenvector Function should be rank-1")
    if fe_ten.family() != fe_vec.family():
        raise ValueError("Mismatching finite element families")
    if fe_ten.degree() != fe_vec.degree():
        raise ValueError("Mismatching finite element space degrees")
    dim = V_ten.mesh().topological_dimension()
    M = RiemannianMetric(V_ten)
    op2.par_loop(
        get_metric_kernel("set_eigendecomposition", dim),
        V_ten.node_set,
        M.dat(op2.RW),
        evectors.dat(op2.READ),
        evalues.dat(op2.READ),
    )
    return M


# --- General


@PETSc.Log.EventDecorator()
def isotropic_metric(
    error_indicator: Function, interpolant: str = "Clement", **kwargs
) -> RiemannianMetric:
    r"""
    Compute an isotropic metric from some error indicator.

    The result is a :math:`\mathbb P1` diagonal tensor
    field whose entries are projections of the error
    indicator in modulus.

    :arg error_indicator: the error indicator
    :kwarg target_space:
        :func:`firedrake.functionspace.TensorFunctionSpace`
        in which the metric will exist
    :kwarg interpolant: choose from 'Clement', 'L2',
        'interpolate', 'project'
    :return: the isotropic
        :class:`firedrake.meshadapt.RiemannianMetric`
    """
    target_space = kwargs.get("target_space")
    mesh = error_indicator.ufl_domain()
    dim = mesh.topological_dimension()
    assert dim in (2, 3), f"Spatial dimension {dim:d} not supported."
    target_space = target_space or firedrake.TensorFunctionSpace(mesh, "CG", 1)
    assert target_space.ufl_element().family() == "Lagrange"
    assert target_space.ufl_element().degree() == 1
    metric = RiemannianMetric(target_space)

    # Interpolate P0 indicators into P1 space
    if interpolant == "Clement":
        if not isinstance(error_indicator, Function):
            raise NotImplementedError("Can only apply Clement interpolant to Functions")
        fs = error_indicator.function_space()
        family = fs.ufl_element().family()
        degree = fs.ufl_element().degree()
        if family != "Lagrange" or degree != 1:
            if family != "Discontinuous Lagrange" or degree != 0:
                raise ValueError(
                    "Was expecting an error indicator in P0 or P1"
                    f" space, not {family} of degree {degree}"
                )
            error_indicator = clement_interpolant(error_indicator)
    if interpolant in ("Clement", "interpolate"):
        metric.interpolate(abs(error_indicator) * ufl.Identity(dim), target_space)
    elif interpolant in ("L2", "project"):
        error_indicator = firedrake.project(
            error_indicator, FunctionSpace(mesh, "CG", 1)
        )
        metric.interpolate(abs(error_indicator) * ufl.Identity(dim), target_space)
    else:
        raise ValueError(f"Interpolant {interpolant} not recognised")
    return metric


@PETSc.Log.EventDecorator("pyroteus.hessian_metric")
def hessian_metric(hessian: Function) -> Function:
    """
    Modify the eigenvalues of a Hessian matrix so
    that it is positive-definite.

    :arg hessian: the Hessian matrix
    """
    fs = hessian.function_space()
    shape = fs.ufl_element().value_shape()
    if len(shape) != 2:
        raise ValueError(f"Expected a rank 2 tensor, got rank {len(shape)}.")
    if shape[0] != shape[1]:
        raise ValueError(
            f"Expected a square tensor field, got {fs.ufl_element().value_shape()}."
        )
    metric = Function(fs)
    op2.par_loop(
        get_metric_kernel("metric_from_hessian", shape[0]),
        fs.node_set,
        metric.dat(op2.RW),
        hessian.dat(op2.READ),
    )
    return metric


def anisotropic_metric(
    error_indicator: Function, hessian: RiemannianMetric, **kwargs
) -> RiemannianMetric:
    r"""
    Compute an element-wise anisotropic metric from some
    error indicator, given a Hessian field.

    Two formulations are currently available:
      * the element-wise formulation presented
        in :cite:`CPB:13`; and
      * the vertex-wise formulation presented in
        :cite:`PPP+:06`.

    In both cases, a :math:`\mathbb P1` metric is
    returned by default.

    :arg error_indicator: (list of) error indicator(s)
    :arg hessian: (list of) Hessian(s)
    :kwarg target_space:
        :func:`firedrake.functionspace.TensorFunctionSpace`
        in which the metric will exist
    :kwarg interpolant: choose from 'Clement', 'L2',
        'interpolate', 'project'
    """
    approach = kwargs.pop("approach", "anisotropic_dwr")
    if approach == "anisotropic_dwr":
        return anisotropic_dwr_metric(error_indicator, hessian=hessian, **kwargs)
    elif approach == "weighted_hessian":
        return weighted_hessian_metric(error_indicator, hessian=hessian, **kwargs)
    else:
        raise ValueError(f"Anisotropic metric approach {approach} not recognised.")


@PETSc.Log.EventDecorator()
def anisotropic_dwr_metric(
    error_indicator: Function, interpolant: str = "Clement", **kwargs
) -> Function:
    r"""
    Compute an anisotropic metric from some error
    indicator, given a Hessian field.

    The formulation used is based on that presented
    in :cite:`CPB:13`.

    Whilst an element-based formulation is used to
    derive the metric, the result is projected into
    :math:`\mathbb P1` space, by default.

    Note that normalisation is implicit in the metric
    construction and involves the `convergence_rate`
    parameter, named :math:`alpha` in :cite:`CPB:13`.

    If a Hessian is not provided then an isotropic
    formulation is used.

    :arg error_indicator: the error indicator
    :kwarg hessian: the Hessian
    :kwarg target_space:
        :func:`firedrake.functionspace.TensorFunctionSpace`
        in which the metric will exist
    :kwarg target_complexity: target metric complexity
    :kwarg convergence rate: normalisation parameter
    :kwarg interpolant: choose from 'Clement', 'L2',
        'interpolate', 'project'
    """
    hessian = kwargs.get("hessian")
    target_space = kwargs.get("target_space")
    if isinstance(error_indicator, list):  # FIXME: This is hacky
        error_indicator = error_indicator[0]
    if isinstance(hessian, list):  # FIXME: This is hacky
        hessian = hessian[0]
    target_complexity = kwargs.get("target_complexity", None)
    # min_eigenvalue = kwargs.get("min_eigenvalue", 1.0e-05)
    # TODO: why was this ^^^ here?
    if target_complexity <= 0.0:
        raise ValueError(
            f"Target complexity must be positive, not {target_complexity}."
        )
    mesh = error_indicator.ufl_domain()
    dim = mesh.topological_dimension()
    if dim not in (2, 3):
        raise ValueError(f"Spatial dimension {dim} not supported. Must be 2 or 3.")
    convergence_rate = kwargs.get("convergence_rate", 1.0)
    if convergence_rate < 1.0:
        raise ValueError(
            f"Convergence rate must be at least one, not {convergence_rate}."
        )
    P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
    P0_metric = RiemannianMetric(P0_ten)

    # Get reference element volume
    K_hat = 1 / 2 if dim == 2 else 1 / 6

    # Get current element volume
    K = K_hat * abs(ufl.JacobianDeterminant(mesh))

    # Get optimal element volume
    P0 = FunctionSpace(mesh, "DG", 0)
    K_opt = firedrake.interpolate(pow(error_indicator, 1 / (convergence_rate + 1)), P0)
    K_opt.interpolate(K / target_complexity * K_opt.vector().gather().sum() / K_opt)
    K_ratio = pow(abs(K_hat / K_opt), 2 / dim)
    if hessian is None:
        P0_metric.interpolate(K_ratio * ufl.Identity(dim))
        P0_metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
    else:
        # Interpolate from P1 to P0 by averaging the vertex-wise values
        P0_metric.project(hessian)
        P0_metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

        # Compute stretching factors, in ascending order
        evectors, evalues = compute_eigendecomposition(P0_metric, reorder=True)
        # lmin = max(evalues.vector().gather().min(), min_eigenvalue)
        # TODO: why was this ^^^ here?
        S = abs(evalues / pow(np.prod(evalues), 1 / dim))

        # Assemble metric with modified eigenvalues
        evalues.interpolate(K_ratio * S)
        v = evalues.vector().gather()
        # lmin = max(v.min(), min_eigenvalue)
        # TODO: why was this ^^^ here?
        if np.isnan(v).any():
            raise ValueError("At least one modified stretching factor is not finite.")
        P0_metric.assign(assemble_eigendecomposition(evectors, evalues))

    # Interpolate the metric into target space and ensure SPD
    target_space = target_space or firedrake.TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(target_space)
    if interpolant == "Clement":
        metric.assign(clement_interpolant(P0_metric, target_space=target_space))
    elif interpolant == "interpolate":
        metric.interpolate(P0_metric)
    elif interpolant in ("project", "L2"):
        metric.project(P0_metric)
    else:
        raise ValueError(f"Interpolant {interpolant} not recognised")
    metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
    return metric


@PETSc.Log.EventDecorator()
def weighted_hessian_metric(
    error_indicators: List[Function],
    hessians: List[RiemannianMetric],
    interpolant: str = "Clement",
    **kwargs,
) -> Function:
    r"""
    Compute a vertex-wise anisotropic metric from a (list
    of) error indicator(s), given a (list of) Hessian
    field(s).

    The formulation used is based on that presented
    in :cite:`PPP+:06`.

    :arg error_indicators: (list of) error indicator(s)
    :arg hessians: (list of) Hessian(s)
    :kwarg target_space:
        :func:`firedrake.functionspace.TensorFunctionSpace`
        in which the metric will exist
    :kwarg average: should metric components be averaged
        or intersected?
    :kwarg interpolant: choose from 'Clement', 'L2',
        'interpolate', 'project'
    """
    target_space = kwargs.get("target_space")
    if not isinstance(error_indicators, Iterable):
        error_indicators = [error_indicators]
    if not isinstance(hessians, Iterable):
        hessians = [hessians]
    mesh = hessians[0].ufl_domain()
    dim = mesh.topological_dimension()
    if dim not in (2, 3):
        raise ValueError(f"Spatial dimension {dim:d} not supported. Must be 2 or 3.")

    # Project error indicators into P1 space and use them to weight Hessians
    target_space = target_space or firedrake.TensorFunctionSpace(mesh, "CG", 1)
    metrics = [RiemannianMetric(target_space) for hessian in hessians]
    for error_indicator, hessian, metric in zip(error_indicators, hessians, metrics):
        if interpolant == "Clement":
            error_indicator = clement_interpolant(error_indicator)
        else:
            P1 = FunctionSpace(mesh, "CG", 1)
            if interpolant == "interpolate":
                error_indicator = firedrake.interpolate(error_indicator, P1)
            elif interpolant in ("L2", "project"):
                error_indicator = firedrake.project(error_indicator, P1)
            else:
                raise ValueError(f"Interpolant {interpolant} not recognised")
        metric.interpolate(abs(error_indicator) * hessian)

    # Combine the components
    return combine_metrics(*metrics, average=kwargs.get("average", False))


# TODO: Implement this on the PETSc level and port it through to Firedrake
@PETSc.Log.EventDecorator()
def enforce_element_constraints(
    metrics: List[RiemannianMetric],
    h_min: List[Function],
    h_max: List[Function],
    a_max: List[Function],
    boundary_tag: Optional[Union[str, int]] = None,
    optimise: bool = False,
) -> List[Function]:
    """
    Post-process a list of metrics to enforce minimum and
    maximum element sizes, as well as maximum anisotropy.

    :arg metrics: the metrics
    :arg h_min: minimum tolerated element size,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :arg h_max: maximum tolerated element size,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :arg a_max: maximum tolerated element anisotropy,
        which could be a :class:`firedrake.function.Function`
        or a number.
    :kwarg boundary_tag: optional tag to enforce sizes on.
    :kwarg optimise: is this a timed run?
    """
    from collections.abc import Iterable
    from firedrake import Function

    if isinstance(metrics, RiemannianMetric):
        metrics = [metrics]
    if not isinstance(h_min, Iterable):
        h_min = [h_min] * len(metrics)
    if not isinstance(h_max, Iterable):
        h_max = [h_max] * len(metrics)
    if not isinstance(a_max, Iterable):
        a_max = [a_max] * len(metrics)
    for metric, hmin, hmax, amax in zip(metrics, h_min, h_max, a_max):
        fs = metric.function_space()
        mesh = fs.mesh()
        P1 = FunctionSpace(mesh, "CG", 1)

        def interp(f):
            if isinstance(f, Function):
                return Function(P1).assign(f)
            else:
                return clement_interpolant(f)

        # Interpolate hmin, hmax and amax into P1
        hmin = interp(hmin)
        hmax = interp(hmax)
        amax = interp(amax)

        # Check the values are okay
        if not optimise:
            _hmin = hmin.vector().gather().min()
            if _hmin <= 0.0:
                raise ValueError(
                    f"Encountered negative non-positive hmin value: {_hmin}."
                )
            if hmax.vector().gather().min() < _hmin:
                raise ValueError(
                    f"Encountered hmax value smaller than hmin: {_hmax} vs. {_hmin}."
                )
            dx = ufl.dx(domain=mesh)
            integral = firedrake.assemble(ufl.conditional(hmax < hmin, 1, 0) * dx)
            if not np.isclose(integral, 0.0):
                raise ValueError(
                    f"Encountered regions where hmax < hmin: volume {integral}."
                )
            _amax = amax.vector().gather().min()
            if _amax < 1.0:
                raise ValueError(f"Encountered amax value smaller than unity: {_amax}.")

        # Enforce constraints
        dim = fs.mesh().topological_dimension()
        if boundary_tag is None:
            node_set = fs.node_set
        else:
            node_set = firedrake.DirichletBC(fs, 0, boundary_tag).node_set
        op2.par_loop(
            get_metric_kernel("postproc_metric", dim),
            node_set,
            metric.dat(op2.RW),
            hmin.dat(op2.READ),
            hmax.dat(op2.READ),
            amax.dat(op2.READ),
        )
    return metrics


# --- Normalisation


@PETSc.Log.EventDecorator()
def space_time_normalise(
    metrics: List[RiemannianMetric],
    end_time: float,
    timesteps: List[float],
    metric_parameters: dict,
    global_factor: Optional[float] = None,
    boundary: bool = False,
    **kwargs,
) -> List[RiemannianMetric]:
    r"""
    Apply :math:`L^p` normalisation in both space and time.

    :arg metrics: list of :class:`firedrake.meshadapt.RiemannianMetric`\s
        corresponding to the metric associated with
        each subinterval
    :arg end_time: end time of simulation
    :arg timesteps: list of timesteps specified in each
        subinterval
    :arg metric_parameters: dictionary containing the target *space-time*
        metric complexity under `dm_plex_metric_target_complexity` and the
        normalisation order under `dm_plex_metric_p`
    :kwarg global_factor: pre-computed global normalisation factor
    :kwarg boundary: is the normalisation to be done over the boundary?
    :kwarg restrict_sizes: should minimum and maximum metric magnitudes
        be enforced?
    :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
    """
    p = metric_parameters.get("dm_plex_metric_p")
    if p is None:
        raise ValueError("Normalisation order 'dm_plex_metric_p' must be set.")
    if not (np.isinf(p) or p >= 1.0):
        raise ValueError(f"Normalisation order {p} not valid.")
    target = metric_parameters.get("dm_plex_metric_target_complexity")
    if target is None:
        raise ValueError(
            "Target complexity 'dm_plex_metric_target_complexity' must be set."
        )
    if target < 0.0:
        raise ValueError("Target complexity must be positive.")

    # TODO: Avoid assuming uniform subinterval lengths
    assert len(metrics) == len(timesteps)
    dt_per_mesh = [firedrake.Constant(end_time / len(metrics) / dt) for dt in timesteps]
    d = metrics[0].function_space().mesh().topological_dimension()

    # Enforce that the metric is SPD
    for metric in metrics:
        metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

    # Compute global normalisation factor
    if global_factor is None:
        integral = 0
        for metric, tau in zip(metrics, dt_per_mesh):
            detM = ufl.det(metric)
            dX = (ufl.ds if boundary else ufl.dx)(metric.function_space().mesh())
            exponent = 0.5 if np.isinf(p) else (p / (2 * p + d))
            integral += firedrake.assemble(pow(tau**2 * detM, exponent) * dX)
        global_factor = firedrake.Constant(pow(target / integral, 2 / d))

    # Normalise on each subinterval
    for metric, tau in zip(metrics, dt_per_mesh):
        metric.set_parameters(metric_parameters)
        metric.normalise(global_factor=pow(tau, -2 / (2 * p + d)) * global_factor)
        metric.enforce_spd(**kwargs)
    return metrics


@PETSc.Log.EventDecorator("pyroteus.determine_metric_complexity")
def determine_metric_complexity(
    H_interior: Function, H_boundary: Function, target: float, p: float, **kwargs
) -> float:
    """
    Solve an algebraic problem to obtain coefficients
    for the interior and boundary metrics to obtain a
    given metric complexity.

    See :cite:`LDA:10` for details. Note that we use a
    slightly different formulation here.

    :arg H_interior: Hessian component from domain interior
    :arg H_boundary: Hessian component from domain boundary
    :arg target: target metric complexity
    :arg p: normalisation order
    :kwarg H_interior_scaling: optional scaling for interior component
    :kwarg H_boundary_scaling: optional scaling for boundary component
    """
    import sympy

    d = H_interior.function_space().mesh().topological_dimension()
    if d not in (2, 3):
        raise ValueError(f"Spatial dimension {d} not supported.")
    if np.isinf(p):
        raise NotImplementedError(
            "Metric complexity cannot be determined in the L-infinity case."
        )
    g = kwargs.get("H_interior_scaling", firedrake.Constant(1.0))
    gbar = kwargs.get("H_boundary_scaling", firedrake.Constant(1.0))
    g = pow(g, d / (2 * p + d))
    gbar = pow(gbar, d / (2 * p + d - 1))

    # Compute coefficients for the algebraic problem
    a = firedrake.assemble(g * pow(ufl.det(H_interior), p / (2 * p + d)) * ufl.dx)
    b = firedrake.assemble(
        gbar * pow(ufl.det(H_boundary), p / (2 * p + d - 1)) * ufl.ds
    )

    # Solve algebraic problem
    c = sympy.Symbol("c")
    c = sympy.solve(a * pow(c, d / 2) + b * pow(c, (d - 1) / 2) - target, c)
    eq = f"{a}*c^{d/2} + {b}*c^{(d-1)/2} = {target}"
    if len(c) == 0:
        raise ValueError(f"Could not find any solutions for equation {eq}.")
    elif len(c) > 1:
        raise ValueError(f"Could not find a unique solution for equation {eq}.")
    elif not np.isclose(float(sympy.im(c[0])), 0.0):
        raise ValueError(f"Could not find any real solutions for equation {eq}.")
    else:
        return float(sympy.re(c[0]))


# --- Combination


@PETSc.Log.EventDecorator("pyroteus.metric_intersection")
def metric_intersection(
    *metrics: Tuple[Function],
    function_space: Optional[FunctionSpace] = None,
    boundary_tag: Optional[Union[str, int]] = None,
) -> Function:
    """
    Combine a list of metrics by intersection.

    :arg metrics: the metrics to be combined
    :kwarg function_space: the
        :class:`firedrake.functionspaceimpl.FunctionSpace`
        the intersected metric should live in
    :kwarg boundary_tag: optional boundary segment physical
        ID for boundary intersection. Otherwise, the
        intersection is over the whole domain
    """
    n = len(metrics)
    assert n > 0, "Nothing to combine"
    fs = function_space or metrics[0].function_space()
    for metric in metrics:
        if not isinstance(metric, Function) or metric.function_space() != fs:
            metric = firedrake.interpolate(metric, function_space)
    intersected_metric = Function(metrics[0])
    if boundary_tag is None:
        node_set = fs.node_set
    elif boundary_tag == []:
        raise ValueError(
            "It is unclear what to do with an empty list of boundary tags."
        )
    else:
        node_set = firedrake.DirichletBC(fs, 0, boundary_tag).node_set
    dim = fs.mesh().topological_dimension()
    assert dim in (2, 3), f"Spatial dimension {dim:d} not supported."

    def intersect_pair(M1, M2):
        M12 = Function(M1)
        op2.par_loop(
            get_metric_kernel("intersect", dim),
            node_set,
            M12.dat(op2.RW),
            M1.dat(op2.READ),
            M2.dat(op2.READ),
        )
        return M12

    for metric in metrics[1:]:
        intersected_metric = intersect_pair(intersected_metric, metric)
    return intersected_metric


def combine_metrics(
    *metrics: Tuple[RiemannianMetric], average: bool = True, **kwargs
) -> RiemannianMetric:
    """
    Combine a list of metrics.

    :kwarg average: combination by averaging or intersection?
    """
    metric = RiemannianMetric(metrics[0].function_space())
    if average:
        metric.average(*metrics, **kwargs)
    else:
        metric.intersect(*metrics, **kwargs)
    return metric


# --- Metric decompositions and properties


@PETSc.Log.EventDecorator()
def density_and_quotients(
    metric: RiemannianMetric, reorder: bool = False
) -> List[Function]:
    r"""
    Extract the density and anisotropy quotients from a
    metric.

    By symmetry, Riemannian metrics admit an orthogonal
    eigendecomposition,

    .. math::
        \underline{\mathbf M}(\mathbf x)
        = \underline{\mathbf V}(\mathbf x)\:
        \underline{\boldsymbol\Lambda}(\mathbf x)\:
        \underline{\mathbf V}(\mathbf x)^T,

    at each point :math:`\mathbf x\in\Omega`, where
    :math:`\underline{\mathbf V}` and
    :math:`\underline{\boldsymbol\Sigma}` are matrices
    holding the eigenvectors and eigenvalues, respectively.
    By positive-definiteness, entries of
    :math:`\underline{\boldsymbol\Lambda}` are all positive.

    An alternative decomposition,

    .. math::
        \underline{\mathbf M}(\mathbf x)
        = d(\mathbf x)^\frac2n
        \underline{\mathbf V}(\mathbf x)\:
        \underline{\mathbf R}(\mathbf x)^{-\frac2n}
        \underline{\mathbf V}(\mathbf x)^T

    can also be deduced, in terms of the `metric density`
    and `anisotropy quotients`,

    .. math::
        d = \prod_{i=1}^n h_i,\qquad
        r_i = \frac{h_i^n}d,\qquad \forall i=1:n,

    where :math:`h_i := \frac1{\sqrt{\lambda_i}}`.

    :arg metric: the metric to extract density and
        quotients from
    :kwarg reorder: should the eigendecomposition be
        reordered?
    :return: metric density, anisotropy quotients and
        eigenvector matrix
    """
    fs_ten = metric.function_space()
    mesh = fs_ten.mesh()
    fe = (fs_ten.ufl_element().family(), fs_ten.ufl_element().degree())
    fs_vec = firedrake.VectorFunctionSpace(mesh, *fe)
    fs = FunctionSpace(mesh, *fe)
    dim = mesh.topological_dimension()
    density = Function(fs, name="Metric density")
    quotients = Function(fs_vec, name="Anisotropic quotients")

    # Compute eigendecomposition
    evectors, evalues = compute_eigendecomposition(metric, reorder=reorder)

    # Extract density and quotients
    magnitudes = [1 / ufl.sqrt(evalues[i]) for i in range(dim)]
    density.interpolate(1 / np.prod(magnitudes))
    quotients.interpolate(ufl.as_vector([density * h**dim for h in magnitudes]))
    return density, quotients, evectors


def ramp_complexity(
    base: float, target: float, i: int, num_iterations: int = 3
) -> float:
    """
    Ramp up the target complexity over the first few iterations.

    :arg base: the base complexity to start from
    :arg target: the desired complexity
    :arg i: the current iteration
    :kwarg num_iterations: how many iterations to ramp over?
    """
    if base <= 0.0:
        raise ValueError(f"Base complexity must be positive, not {base}.")
    if target <= 0.0:
        raise ValueError(f"Target complexity must be positive, not {target}.")
    if i < 0:
        raise ValueError(f"Current iteration must be non-negative, not {i}.")
    if num_iterations < 0:
        raise ValueError(
            f"Number of iterations must be non-negative, not {num_iterations}."
        )
    alpha = min(i / num_iterations, 1)
    return alpha * target + (1 - alpha) * base
