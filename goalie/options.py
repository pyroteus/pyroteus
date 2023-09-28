from .utility import AttrDict
from animate.adapt import RiemannianMetric


__all__ = [
    "AdaptParameters",
    "MetricParameters",
    "GoalOrientedParameters",
    "GoalOrientedMetricParameters",
]


class AdaptParameters(AttrDict):
    """
    A class for holding parameters associated with adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters: dict = {}):
        """
        :arg parameters: dictionary of parameters to set
        """
        self["miniter"] = 3  # Minimum iteration count
        self["maxiter"] = 35  # Maximum iteration count
        self["element_rtol"] = 0.001  # Relative tolerance for element count
        self["drop_out_converged"] = False  # Drop out converged subintervals?

        if not isinstance(parameters, dict):
            raise TypeError(
                "Expected 'parameters' keyword argument to be a dictionary, not of"
                f" type '{parameters.__class__.__name__}'."
            )
        for key, value in parameters.items():
            if key not in self:
                raise AttributeError(
                    f"{self.__class__.__name__} does not have '{key}' attribute."
                )
        super().__init__(parameters)
        self._check_type("miniter", int)
        self._check_type("maxiter", int)
        self._check_type("element_rtol", (float, int))
        self._check_type("drop_out_converged", bool)

    def _check_type(self, key, expected):
        if not isinstance(self[key], expected):
            if isinstance(expected, tuple):
                name = "' or '".join([e.__name__ for e in expected])
            else:
                name = expected.__name__
            raise TypeError(
                f"Expected attribute '{key}' to be of type '{name}', not"
                f" '{type(self[key]).__name__}'."
            )

    def _check_value(self, key, possibilities):
        value = self[key]
        if value not in possibilities:
            raise ValueError(
                f"Unsupported value '{value}' for '{key}'. Choose from {possibilities}."
            )

    def __str__(self) -> str:
        return str({key: value for key, value in self.items()})

    def __repr__(self) -> str:
        d = ", ".join([f"{key}={value}" for key, value in self.items()])
        return f"{self.__class__.__name__}({d})"


class MetricParameters(AdaptParameters):
    """
    A class for holding parameters associated with
    metric-based adaptive mesh fixed point iteration loops.
    """

    def __init__(self, parameters: dict = {}):
        """
        :arg parameters: dictionary of parameters to set
        """
        self["num_ramp_iterations"] = 3  # Number of iterations to ramp over
        self["verbosity"] = -1  # -1 = silent, 10 = maximum

        # --- Normalisation

        self["p"] = 1.0  # Metric normalisation order
        self["base_complexity"] = 200.0  # Base metric complexity
        self["target_complexity"] = 4000.0  # Target metric complexity

        # --- Post-processing
        self["h_min"] = 1.0e-30  # Minimum metric magnitude
        self["h_max"] = 1.0e30  # Maximum metric magnitude
        self["a_max"] = 1.0e5  # Maximum anisotropy
        self["restrict_anisotropy_first"] = False
        self["hausdorff_number"] = 0.01  # Controls length scales
        self["gradation_factor"] = 1.3  # Controls ratio between adjacent edge lengths

        # --- Adaptation
        self["no_insert"] = False  # Turn off node insertion
        self["no_swap"] = False  # Turn off edge and face swapping
        self["no_move"] = False  # Turn off node movement
        self["no_surf"] = False  # Turn off surface meshing
        self["num_parmmg_iterations"] = 3

        super().__init__(parameters=parameters)

        self._check_type("num_ramp_iterations", int)
        self._check_type("verbosity", int)
        self._check_type("p", (float, int))
        self._check_type("base_complexity", (float, int))
        self._check_type("target_complexity", (float, int))
        self._check_type("h_min", (float, int))
        self._check_type("h_max", (float, int))
        self._check_type("a_max", (float, int))
        self._check_type("hausdorff_number", (float, int))
        self._check_type("gradation_factor", (float, int))
        self._check_type("restrict_anisotropy_first", bool)
        self._check_type("no_insert", bool)
        self._check_type("no_swap", bool)
        self._check_type("no_move", bool)
        self._check_type("no_surf", bool)
        self._check_type("num_parmmg_iterations", int)

    def export(self, metric):
        """
        Set parameters appropriate to a given :class:`RiemannianMetric`.

        :arg metric: the :class:`RiemannianMetric` to apply parameters to
        """
        if not isinstance(metric, RiemannianMetric):
            raise TypeError(
                f"{type(self)} can only be exported to RiemannianMetric,"
                f" not '{type(metric)}'."
            )
        petsc_specific = (
            "verbosity",
            "p",
            "target_complexity",
            "no_insert",
            "no_swap",
            "no_move",
            "no_surf",
            "h_min",
            "h_max",
            "a_max",
            "restrict_anisotropy_first",
            "hausdorff_number",
            "gradation_factor",
        )
        metric_parameters = {
            "dm_plex_metric": {key: self[key] for key in petsc_specific}
        }
        metric_parameters["num_iterations"] = self["num_parmmg_iterations"]
        metric.set_parameters(metric_parameters)


class GoalOrientedParameters(AdaptParameters):
    """
    A class for holding parameters associated with
    goal-oriented adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters: dict = {}):
        self["qoi_rtol"] = 0.001  # Relative tolerance for QoI
        self["estimator_rtol"] = 0.001  # Relative tolerance for estimator
        self["convergence_criteria"] = "any"  # Mode for convergence checking

        super().__init__(parameters=parameters)

        self._check_type("qoi_rtol", (float, int))
        self._check_type("estimator_rtol", (float, int))
        self._check_type("convergence_criteria", str)
        self._check_value("convergence_criteria", ["all", "any"])


class GoalOrientedMetricParameters(GoalOrientedParameters, MetricParameters):
    """
    A class for holding parameters associated with
    metric-based, goal-oriented adaptive mesh fixed
    point iteration loops.
    """
