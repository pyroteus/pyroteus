"""
Driver functions for metric-based mesh adaptation.
"""
import firedrake.meshadapt
from .log import debug
from .time_partition import TimePartition
from .recovery import *
from typing import List, Optional, Tuple, Union
import ufl


__all__ = [
    "RiemannianMetric",
    "enforce_element_constraints",
    "space_time_normalise",
    "intersect_on_boundary",
    "determine_metric_complexity",
    "ramp_complexity",
]


def get_metric_kernel(func: str, dim: int) -> op2.Kernel:
    """
    Helper function to easily pass Eigen kernels
    for metric utilities to Firedrake via PyOP2.

    :arg func: function name
    :arg dim: spatial dimension
    """
    pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "cxx"))
    with open(os.path.join(pwd, f"metric{dim}d.cxx"), "r") as code:
        return op2.Kernel(code.read(), func, cpp=True, include_dirs=include_dir)


class RiemannianMetric(firedrake.meshadapt.RiemannianMetric):
    """
    Subclass of :class:`firedrake.meshadapt.RiemannianMetric` to allow adding methods.
    """

    @PETSc.Log.EventDecorator()
    def compute_hessian(self, f: Function, method: str = "mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field.

        :arg f: the scalar field whose Hessian we seek to recover
        :kwarg method: recovery method

        All other keyword arguments are passed to the chosen recovery routine.

        In the case of the `'L2'` method, the `target_space` keyword argument is used
        for the gradient recovery. The target space for the Hessian recovery is
        inherited from the metric itself.
        """
        if method == "L2":
            g = recover_gradient_l2(f, target_space=kwargs.get("target_space"))
            return self.assign(recover_gradient_l2(g))
        elif method == "mixed_L2":
            return super().compute_hessian(f, **kwargs)
        elif method == "Clement":
            return self.assign(recover_hessian_clement(f, **kwargs)[1])
        elif method == "ZZ":
            raise NotImplementedError(
                "Zienkiewicz-Zhu recovery not yet implemented."
            )  # TODO
        else:
            raise ValueError(f"Recovery method '{method}' not recognised.")

    @PETSc.Log.EventDecorator()
    def compute_boundary_hessian(self, f: Function, method: str = "mixed_L2", **kwargs):
        """
        Recover the Hessian of a scalar field on the domain boundary.

        :arg f: field to recover over the domain boundary
        :kwarg method: choose from 'mixed_L2' and 'Clement'
        """
        return self.assign(recover_boundary_hessian(f, method=method, **kwargs))

    @PETSc.Log.EventDecorator()
    def compute_eigendecomposition(
        self, reorder: bool = False
    ) -> Tuple[Function, Function]:
        """
        Compute the eigenvectors and eigenvalues of a matrix-valued function.

        :kwarg reorder: should the eigendecomposition be reordered in order of
            *descending* eigenvalue magnitude?
        :return: eigenvector :class:`firedrake.function.Function` and eigenvalue
            :class:`firedrake.function.Function` from the
            :func:`firedrake.functionspace.TensorFunctionSpace` underpinning the metric
        """
        V_ten = self.function_space()
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
            self.dat(op2.READ),
        )
        return evectors, evalues

    @PETSc.Log.EventDecorator()
    def assemble_eigendecomposition(self, evectors: Function, evalues: Function):
        """
        Assemble a matrix from its eigenvectors and eigenvalues.

        :arg evectors: eigenvector :class:`firedrake.function.Function`
        :arg evalues: eigenvalue :class:`firedrake.function.Function`
        """
        V_ten = evectors.function_space()
        fe_ten = V_ten.ufl_element()
        if len(fe_ten.value_shape()) != 2:
            raise ValueError(
                "Eigenvector Function should be rank-2,"
                f" not rank-{len(fe_ten.value_shape())}."
            )
        V_vec = evalues.function_space()
        fe_vec = V_vec.ufl_element()
        if len(fe_vec.value_shape()) != 1:
            raise ValueError(
                "Eigenvalue Function should be rank-1,"
                f" not rank-{len(fe_vec.value_shape())}."
            )
        if fe_ten.family() != fe_vec.family():
            raise ValueError(
                "Mismatching finite element families:"
                f" '{fe_ten.family()}' vs. '{fe_vec.family()}'."
            )
        if fe_ten.degree() != fe_vec.degree():
            raise ValueError(
                "Mismatching finite element space degrees:"
                f" {fe_ten.degree()} vs. {fe_vec.degree()}."
            )
        dim = V_ten.mesh().topological_dimension()
        op2.par_loop(
            get_metric_kernel("set_eigendecomposition", dim),
            V_ten.node_set,
            self.dat(op2.RW),
            evectors.dat(op2.READ),
            evalues.dat(op2.READ),
        )
        return self

    @PETSc.Log.EventDecorator()
    def density_and_quotients(self, reorder: bool = False) -> List[Function]:
        r"""
        Extract the density and anisotropy quotients from a metric.

        By symmetry, Riemannian metrics admit an orthogonal eigendecomposition,

        .. math::
            \underline{\mathbf M}(\mathbf x)
            = \underline{\mathbf V}(\mathbf x)\:
            \underline{\boldsymbol\Lambda}(\mathbf x)\:
            \underline{\mathbf V}(\mathbf x)^T,

        at each point :math:`\mathbf x\in\Omega`, where
        :math:`\underline{\mathbf V}` and :math:`\underline{\boldsymbol\Sigma}` are
        matrices holding the eigenvectors and eigenvalues, respectively. By
        positive-definiteness, entries of :math:`\underline{\boldsymbol\Lambda}` are all
        positive.

        An alternative decomposition,

        .. math::
            \underline{\mathbf M}(\mathbf x)
            = d(\mathbf x)^\frac2n
            \underline{\mathbf V}(\mathbf x)\:
            \underline{\mathbf R}(\mathbf x)^{-\frac2n}
            \underline{\mathbf V}(\mathbf x)^T

        can also be deduced, in terms of the `metric density` and
        `anisotropy quotients`,

        .. math::
            d = \prod_{i=1}^n h_i,\qquad
            r_i = h_i^n d,\qquad \forall i=1:n,

        where :math:`h_i := \frac1{\sqrt{\lambda_i}}`.

        :kwarg reorder: should the eigendecomposition be reordered?
        :return: metric density, anisotropy quotients and eigenvector matrix
        """
        fs_ten = self.function_space()
        mesh = fs_ten.mesh()
        fe = (fs_ten.ufl_element().family(), fs_ten.ufl_element().degree())
        dim = mesh.topological_dimension()
        evectors, evalues = self.compute_eigendecomposition(reorder=reorder)

        # Extract density and quotients
        density = Function(FunctionSpace(mesh, *fe), name="Metric density")
        density.interpolate(np.prod([ufl.sqrt(e) for e in evalues]))
        quotients = Function(
            firedrake.VectorFunctionSpace(mesh, *fe), name="Anisotropic quotients"
        )
        quotients.interpolate(
            ufl.as_vector([density / ufl.sqrt(e) ** dim for e in evalues])
        )
        return density, quotients, evectors

    def combine(self, *metrics, average: bool = True, **kwargs):
        """
        Combine metrics using either averaging or intersection.

        :arg metrics: the list of metrics to combine with
        :kwarg average: toggle between averaging and intersection

        All other keyword arguments are passed to the relevant method.
        """
        return (self.average if average else self.intersect)(*metrics, **kwargs)

    @PETSc.Log.EventDecorator()
    def compute_isotropic_metric(
        self, error_indicator: Function, interpolant: str = "Clement", **kwargs
    ):
        r"""
        Compute an isotropic metric from some error indicator.

        The result is a :math:`\mathbb P1` diagonal tensor field whose entries are
        projections of the error indicator in modulus.

        :arg error_indicator: the error indicator
        :kwarg interpolant: choose from 'Clement' or 'L2'
        """
        mesh = ufl.domain.extract_unique_domain(error_indicator)
        if mesh != self.function_space().mesh():
            raise ValueError("Cannot use an error indicator from a different mesh.")
        dim = mesh.topological_dimension()

        # Interpolate P0 indicators into P1 space
        if interpolant == "Clement":
            P1_indicator = clement_interpolant(error_indicator)
        elif interpolant == "L2":
            P1_indicator = firedrake.project(
                error_indicator, FunctionSpace(mesh, "CG", 1)
            )
        else:
            raise ValueError(f"Interpolant '{interpolant}' not recognised.")
        return self.interpolate(abs(P1_indicator) * ufl.Identity(dim))

    def compute_isotropic_dwr_metric(
        self,
        error_indicator: Function,
        convergence_rate: float = 1.0,
        min_eigenvalue: float = 1.0e-05,
        interpolant: str = "Clement",
    ):
        r"""
        Compute an isotropic metric from some error indicator using an element-based
        formulation.

        The formulation is based on that presented in :cite:`CPB:13`. Note that
        normalisation is implicit in the metric construction and involves the
        `convergence_rate` parameter, named :math:`alpha` in :cite:`CPB:13`.

        Whilst an element-based formulation is used to derive the metric, the result is
        projected into :math:`\mathbb P1` space, by default.

        :arg error_indicator: the error indicator
        :kwarg convergence_rate: normalisation parameter
        :kwarg min_eigenvalue: minimum tolerated eigenvalue
        :kwarg interpolant: choose from 'Clement' or 'L2'
        """
        return self.compute_anisotropic_dwr_metric(
            error_indicator=error_indicator,
            convergence_rate=convergence_rate,
            min_eigenvalue=min_eigenvalue,
            interpolant=interpolant,
        )

    def _any_inf(self, f):
        arr = f.vector().gather()
        return np.isinf(arr).any() or np.isnan(arr).any()

    @PETSc.Log.EventDecorator()
    def compute_anisotropic_dwr_metric(
        self,
        error_indicator: Function,
        hessian: Optional[Function] = None,
        convergence_rate: float = 1.0,
        min_eigenvalue: float = 1.0e-05,
        interpolant: str = "Clement",
    ):
        r"""
        Compute an anisotropic metric from some error indicator, given a Hessian field.

        The formulation used is based on that presented in :cite:`CPB:13`. Note that
        normalisation is implicit in the metric construction and involves the
        `convergence_rate` parameter, named :math:`alpha` in :cite:`CPB:13`.

        If a Hessian is not provided then an isotropic formulation is used.

        Whilst an element-based formulation is used to derive the metric, the result is
        projected into :math:`\mathbb P1` space, by default.

        :arg error_indicator: the error indicator
        :kwarg hessian: the Hessian
        :kwarg convergence_rate: normalisation parameter
        :kwarg min_eigenvalue: minimum tolerated eigenvalue
        :kwarg interpolant: choose from 'Clement' or 'L2'
        """
        mp = self.metric_parameters.copy()
        target_complexity = mp.get("dm_plex_metric_target_complexity")
        if target_complexity is None:
            raise ValueError("Target complexity must be set.")
        mesh = ufl.domain.extract_unique_domain(error_indicator)
        if mesh != self.function_space().mesh():
            raise ValueError("Cannot use an error indicator from a different mesh.")
        dim = mesh.topological_dimension()
        if convergence_rate < 1.0:
            raise ValueError(
                f"Convergence rate must be at least one, not {convergence_rate}."
            )
        if min_eigenvalue <= 0.0:
            raise ValueError(
                f"Minimum eigenvalue must be positive, not {min_eigenvalue}."
            )
        if interpolant not in ("Clement", "L2"):
            raise ValueError(f"Interpolant '{interpolant}' not recognised.")
        P0_ten = firedrake.TensorFunctionSpace(mesh, "DG", 0)
        P0_metric = P0Metric(P0_ten)

        # Get reference element volume
        K_hat = 1 / 2 if dim == 2 else 1 / 6

        # Get current element volume
        K = K_hat * abs(ufl.JacobianDeterminant(mesh))

        # Get optimal element volume
        P0 = FunctionSpace(mesh, "DG", 0)
        K_opt = pow(error_indicator, 1 / (convergence_rate + 1))
        K_opt_av = K_opt / interpolate(K_opt, P0).vector().gather().sum()
        K_ratio = target_complexity * pow(abs(K_opt_av * K_hat / K), 2 / dim)

        if self._any_inf(interpolate(K_ratio, P0)):
            raise ValueError("K_ratio contains non-finite values.")

        # Interpolate from P1 to P0
        #   Note that this shouldn't affect symmetric positive-definiteness.
        if hessian is not None:
            hessian.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)
        P0_metric.project(hessian or ufl.Identity(dim))

        # Compute stretching factors (in ascending order)
        evectors, evalues = P0_metric.compute_eigendecomposition(reorder=True)
        divisor = pow(np.prod(evalues), 1 / dim)
        modified_evalues = [
            abs(ufl.max_value(e, min_eigenvalue) / divisor) for e in evalues
        ]

        # Assemble metric with modified eigenvalues
        evalues.interpolate(K_ratio * ufl.as_vector(modified_evalues))
        if self._any_inf(evalues):
            raise ValueError(
                "At least one modified stretching factor contains non-finite values."
            )
        P0_metric.assemble_eigendecomposition(evectors, evalues)

        # Interpolate the metric into the target space
        fs = self.function_space()
        metric = RiemannianMetric(fs)
        if interpolant == "Clement":
            metric.assign(clement_interpolant(P0_metric, target_space=fs))
        else:
            metric.project(P0_metric)

        # Rescale to enforce that the target complexity is met
        #   Note that we use the L-infinity norm so that the metric is just scaled to the
        #   target metric complexity, as opposed to being redistributed spatially.
        mp["dm_plex_metric_p"] = np.inf
        metric.set_parameters(mp)
        metric.normalise()
        return self.assign(metric)

    @PETSc.Log.EventDecorator()
    def compute_weighted_hessian_metric(
        self,
        error_indicators: List[Function],
        hessians: List[Function],
        average: bool = False,
        interpolant: str = "Clement",
    ):
        r"""
        Compute a vertex-wise anisotropic metric from a list of error indicators, given
        a list of corresponding Hessian fields.

        The formulation used is based on that presented in :cite:`PPP+:06`. It is
        assumed that the error indicators have been constructed in the appropriate way.

        :arg error_indicators: list of error indicators
        :arg hessians: list of Hessians
        :kwarg average: should metric components be averaged or intersected?
        :kwarg interpolant: choose from 'Clement' or 'L2'
        """
        if isinstance(error_indicators, Function):
            error_indicators = [error_indicators]
        if isinstance(hessians, Function):
            hessians = [hessians]
        mesh = self.function_space().mesh()
        P1 = FunctionSpace(mesh, "CG", 1)
        for error_indicator, hessian in zip(error_indicators, hessians):
            if mesh != error_indicator.function_space().mesh():
                raise ValueError("Cannot use an error indicator from a different mesh.")
            if mesh != hessian.function_space().mesh():
                raise ValueError("Cannot use a Hessian from a different mesh.")
            if not isinstance(hessian, RiemannianMetric):
                raise TypeError(
                    f"Expected Hessian to be a RiemannianMetric, not {type(hessian)}."
                )
            if interpolant == "Clement":
                error_indicator = clement_interpolant(error_indicator, target_space=P1)
            elif interpolant == "L2":
                error_indicator = firedrake.project(error_indicator, P1)
            else:
                raise ValueError(f"Interpolant '{interpolant}' not recognised.")
            hessian.interpolate(abs(error_indicator) * hessian)
        return self.combine(*hessians, average=average)


class P0Metric(RiemannianMetric):
    r"""
    Subclass of :class:`firedrake.meshadapt.RiemannianMetric` which allows use of
    :math:`\mathbb P0` space.
    """

    def _check_space(self):
        el = self.function_space().ufl_element()
        if (el.family(), el.degree()) != ("Discontinuous Lagrange", 0):
            raise ValueError(f"P0 metric should be in P0 space, not '{el}'.")


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
                return clement_interpolant(f)
            else:
                return Function(P1).assign(f)

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
            integral = assemble(ufl.conditional(hmax < hmin, 1, 0) * dx)
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


@PETSc.Log.EventDecorator()
def space_time_normalise(
    metrics: List[RiemannianMetric],
    time_partition: TimePartition,
    metric_parameters: Union[dict, list],
    global_factor: Optional[float] = None,
    boundary: bool = False,
    restrict_sizes: bool = True,
    restrict_anisotropy: bool = True,
) -> List[RiemannianMetric]:
    r"""
    Apply :math:`L^p` normalisation in both space and time.

    Based on Equation (1) in :cite:`Barral:2016`.

    :arg metrics: list of :class:`firedrake.meshadapt.RiemannianMetric`\s corresponding
        to the metric associated with each subinterval
    :arg time_partition: :class:`TimePartition` for the problem at hand
    :arg metric_parameters: dictionary containing the target *space-time* metric
        complexity under `dm_plex_metric_target_complexity` and the normalisation order
        under `dm_plex_metric_p`, or a list thereof
    :kwarg global_factor: pre-computed global normalisation factor
    :kwarg boundary: is the normalisation to be done over the boundary?
    :kwarg restrict_sizes: should minimum and maximum metric magnitudes be enforced?
    :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
    """
    if isinstance(metric_parameters, dict):
        metric_parameters = [metric_parameters for _ in range(len(time_partition))]
    d = metrics[0].function_space().mesh().topological_dimension()
    if len(metrics) != len(time_partition):
        raise ValueError(
            "Number of metrics does not match number of subintervals:"
            f" {len(metrics)} vs. {len(time_partition)}."
        )
    if len(metrics) != len(metric_parameters):
        raise ValueError(
            "Number of metrics does not match number of sets of metric parameters:"
            f" {len(metrics)} vs. {len(metric_parameters)}."
        )

    # Preparation step
    metric_parameters = metric_parameters.copy()
    for metric, mp in zip(metrics, metric_parameters):
        if not isinstance(mp, dict):
            raise TypeError(
                "Expected metric_parameters to consist of dictionaries,"
                f" not objects of type '{type(mp)}'."
            )

        # Allow concise notation
        if "dm_plex_metric" in mp and isinstance(mp["dm_plex_metric"], dict):
            for key, value in mp["dm_plex_metric"].items():
                mp[f"dm_plex_metric_{key}"] = value
            mp.pop("dm_plex_metric")

        p = mp.get("dm_plex_metric_p")
        if p is None:
            raise ValueError("Normalisation order 'dm_plex_metric_p' must be set.")
        if not (np.isinf(p) or p >= 1.0):
            raise ValueError(
                f"Normalisation order '{p}' should be one or greater or np.inf."
            )
        target = mp.get("dm_plex_metric_target_complexity")
        if target is None:
            raise ValueError(
                "Target complexity 'dm_plex_metric_target_complexity' must be set."
            )
        if target <= 0.0:
            raise ValueError(f"Target complexity '{target}' is not positive.")
        metric.set_parameters(mp)
        metric.enforce_spd(restrict_sizes=False, restrict_anisotropy=False)

    # Compute global normalisation factor
    if global_factor is None:
        integral = 0
        p = mp["dm_plex_metric_p"]
        exponent = 0.5 if np.isinf(p) else p / (2 * p + d)
        for metric, S in zip(metrics, time_partition):
            dX = (ufl.ds if boundary else ufl.dx)(metric.function_space().mesh())
            scaling = pow(S.length / S.timestep, 2 * exponent)
            integral += scaling * assemble(pow(ufl.det(metric), exponent) * dX)
        target = mp["dm_plex_metric_target_complexity"] * time_partition.num_timesteps
        debug(f"space_time_normalise: target space-time complexity={target:.4e}")
        global_factor = firedrake.Constant(pow(target / integral, 2 / d))
    debug(f"space_time_normalise: global scale factor={float(global_factor):.4e}")

    for metric, S in zip(metrics, time_partition):

        # Normalise according to the global normalisation factor
        metric.normalise(
            global_factor=global_factor,
            restrict_sizes=False,
            restrict_anisotropy=False,
        )

        # Apply the separate scale factors for each metric
        if not np.isinf(p):
            metric *= pow(S.length / S.timestep, -2 / (2 * p + d))
        metric.enforce_spd(
            restrict_sizes=restrict_sizes,
            restrict_anisotropy=restrict_anisotropy,
        )

    return metrics


@PETSc.Log.EventDecorator()
def determine_metric_complexity(
    H_interior: Function, H_boundary: Function, target: float, p: float, **kwargs
) -> float:
    """
    Solve an algebraic problem to obtain coefficients for the interior and boundary
    metrics to obtain a given metric complexity.

    See :cite:`LDA:10` for details. Note that we use a slightly different formulation
    here.

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
    a = assemble(g * pow(ufl.det(H_interior), p / (2 * p + d)) * ufl.dx)
    b = assemble(gbar * pow(ufl.det(H_boundary), p / (2 * p + d - 1)) * ufl.ds)

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


# TODO: Use the intersection functionality in PETSc
@PETSc.Log.EventDecorator()
def intersect_on_boundary(
    *metrics: Tuple[RiemannianMetric],
    boundary_tag: Union[str, int] = "on_boundary",
) -> Function:
    """
    Combine a list of metrics by intersection.

    :arg metrics: the metrics to be combined
    :kwarg boundary_tag: optional boundary segment physical
        ID for boundary intersection. Otherwise, the
        intersection is over the whole domain ('on_boundary')
    """
    n = len(metrics)
    assert n > 0, "Nothing to combine"
    fs = metrics[0].function_space()
    dim = fs.mesh().topological_dimension()
    if dim not in (2, 3):
        raise ValueError(
            f"Spatial dimension {dim} not supported." " Must be either 2 or 3."
        )
    for i, metric in enumerate(metrics):
        if not isinstance(metric, RiemannianMetric):
            raise ValueError(
                f"Metric {i} should be of type 'RiemannianMetric',"
                f" but is of type '{type(metric)}'."
            )
        fsi = metric.function_space()
        if fs != fsi:
            raise ValueError(
                f"Function space of metric {i} does not match that"
                f" of metric 0: {fsi} vs. {fs}."
            )

    # Create the metric to be returned
    intersected_metric = RiemannianMetric(fs)
    intersected_metric.assign(metrics[0])

    # Establish the boundary node set
    if isinstance(boundary_tag, (list, tuple)) and len(boundary_tag) == 0:
        raise ValueError(
            "It is unclear what to do with an empty"
            f" {type(boundary_tag)} of boundary tags."
        )
    node_set = firedrake.DirichletBC(fs, 0, boundary_tag).node_set

    # Compute the intersection
    Mtmp = RiemannianMetric(fs)
    for metric in metrics[1:]:
        Mtmp.assign(intersected_metric)
        op2.par_loop(
            get_metric_kernel("intersect", dim),
            node_set,
            intersected_metric.dat(op2.RW),
            Mtmp.dat(op2.READ),
            metric.dat(op2.READ),
        )
    return intersected_metric


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
    alpha = 1 if num_iterations == 0 else min(i / num_iterations, 1)
    return alpha * target + (1 - alpha) * base
