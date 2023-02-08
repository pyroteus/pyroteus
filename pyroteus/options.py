from .utility import AttrDict


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
        self["h_min"] = 1.0e-30  # Minimum metric magnitude
        self["h_max"] = 1.0e30  # Maximum metric magnitude
        self["a_max"] = 1.0e30  # Maximum anisotropy
        self["p"] = 1.0  # Metric normalisation order
        self["base_complexity"] = 200.0  # Base metric complexity
        self["target_complexity"] = 4000.0  # Target metric complexity
        self["num_ramp_iterations"] = 3  # Number of iterations to ramp over

        super().__init__(parameters=parameters)

        self._check_type("h_min", (float, int))
        self._check_type("h_max", (float, int))
        self._check_type("a_max", (float, int))
        self._check_type("p", (float, int))
        self._check_type("base_complexity", (float, int))
        self._check_type("target_complexity", (float, int))
        self._check_type("num_ramp_iterations", int)


class GoalOrientedParameters(AdaptParameters):
    """
    A class for holding parameters associated with
    goal-oriented adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters: dict = {}):
        self["qoi_rtol"] = 0.001  # Relative tolerance for QoI
        self["estimator_rtol"] = 0.001  # Relative tolerance for estimator

        super().__init__(parameters=parameters)

        self._check_type("qoi_rtol", (float, int))
        self._check_type("estimator_rtol", (float, int))


class GoalOrientedMetricParameters(GoalOrientedParameters, MetricParameters):
    """
    A class for holding parameters associated with
    metric-based, goal-oriented adaptive mesh fixed
    point iteration loops.
    """
