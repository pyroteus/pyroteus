import numpy as np
import ufl
import ufl.core.expr


__all__ = ["bessi0", "bessk0"]


def bessi0(x):
    """
    Modified Bessel function of the first kind.

    Code taken from :cite:`VVP+:92`.
    """
    if isinstance(x, np.ndarray):
        if np.isclose(x, 0).any():
            raise ValueError("Cannot divide by zero.")
        exp = np.exp
        sqrt = np.sqrt
        ax = np.abs(x)
        where = np.where
    else:
        if not isinstance(x, ufl.core.expr.Expr):
            raise TypeError(f"Expected UFL Expr, not '{type(x)}'.")
        exp = ufl.exp
        sqrt = ufl.sqrt
        ax = ufl.as_ufl(abs(x))
        where = ufl.conditional

    x1 = (x / 3.75) ** 2
    coeffs1 = (
        1.0,
        3.5156229,
        3.0899424,
        1.2067492,
        2.659732e-1,
        3.60768e-2,
        4.5813e-3,
    )
    x2 = 3.75 / ax
    coeffs2 = (
        0.39894228,
        1.328592e-2,
        2.25319e-3,
        -1.57565e-3,
        9.16281e-3,
        -2.057706e-2,
        2.635537e-2,
        -1.647633e-2,
        3.92377e-3,
    )

    expr1 = np.polyval(np.flip(coeffs1), x1)
    expr2 = exp(ax) / sqrt(ax) * np.polyval(np.flip(coeffs2), x2)

    return where(ax < 3.75, expr1, expr2)


def bessk0(x):
    """
    Modified Bessel function of the second kind.

    Code taken from :cite:`VVP+:92`.
    """
    if isinstance(x, np.ndarray):
        if (x <= 0).any():
            raise ValueError("Cannot take the logarithm of a non-positive number.")
        ln = np.log
        exp = np.exp
        sqrt = np.sqrt
        where = np.where
    else:
        if not isinstance(x, ufl.core.expr.Expr):
            raise TypeError(f"Expected UFL Expr, not '{type(x)}'.")
        ln = ufl.ln
        exp = ufl.exp
        sqrt = ufl.sqrt
        where = ufl.conditional

    x1 = x * x / 4.0
    coeffs1 = (
        -0.57721566,
        0.42278420,
        0.23069756,
        3.488590e-2,
        2.62698e-3,
        1.0750e-4,
        7.4e-6,
    )
    x2 = 2.0 / x
    coeffs2 = (
        1.25331414,
        -7.832358e-2,
        2.189568e-2,
        -1.062446e-2,
        5.87872e-3,
        -2.51540e-3,
        5.3208e-4,
    )

    expr1 = -ln(0.5 * x) * bessi0(x) + np.polyval(np.flip(coeffs1), x1)
    expr2 = exp(-x) / sqrt(x) * np.polyval(np.flip(coeffs2), x2)

    return where(x <= 2, expr1, expr2)
