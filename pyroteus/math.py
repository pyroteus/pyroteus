import numpy as np
import ufl
import ufl.core.expr


__all__ = ["bessi0", "bessk0", "gram_schmidt", "construct_basis"]


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


def gram_schmidt(*vectors, normalise=False):
    """
    Given some vectors, construct an orthogonal basis
    using Gram-Schmidt orthogonalisation.

    :args vectors: the vectors to orthogonalise
    :kwargs normalise: do we want an orthonormal basis?
    """
    if isinstance(vectors[0], np.ndarray):
        expected = np.ndarray
        dot = np.dot
        sqrt = np.sqrt
    else:
        expected = ufl.core.expr.Expr
        dot = ufl.dot
        sqrt = ufl.sqrt

    # Check that vector types match
    for i, vi in enumerate(vectors[1:]):
        if not isinstance(vi, expected):
            raise TypeError(
                f"Inconsistent vector types: '{expected}' vs. '{type(vi)}'."
            )

        # TODO: Check that valid UFL types are used

    def proj(x, y):
        return dot(x, y) / dot(x, x) * x

    # Apply Gram-Schmidt algorithm
    u = []
    for i, vi in enumerate(vectors):
        if i > 0:
            vi -= sum([proj(uj, vi) for uj in u])
        u.append(vi / sqrt(dot(vi, vi)) if normalise else vi)

    # Ensure consistency of outputs
    if isinstance(vectors[0], np.ndarray):
        u = [np.array(ui) for ui in u]

    return u


def construct_basis(vector, normalise=True):
    """
    Construct a basis from a given vector.

    :arg vector: the starting vector
    :kwargs normalise: do we want an orthonormal basis?
    """
    is_numpy = isinstance(vector, np.ndarray)
    if is_numpy:
        if len(vector.shape) > 1:
            raise ValueError(
                f"Expected a vector, got an array of shape {vector.shape}."
            )
        as_vector = np.array
        dim = vector.shape[0]
    else:
        if not isinstance(vector, ufl.core.expr.Expr):
            raise TypeError(f"Expected UFL Expr, not '{type(vector)}'.")
        as_vector = ufl.as_vector
        dim = ufl.domain.extract_unique_domain(vector).topological_dimension()

    if dim not in (2, 3):
        raise ValueError(f"Dimension {dim} not supported.")
    vectors = [vector]

    # Generate some arbitrary vectors and apply Gram-Schmidt
    if dim == 2:
        vectors.append(as_vector((-vector[1], vector[0])))
    else:
        vectors.append(as_vector((vector[1], vector[2], vector[0])))
        vectors.append(as_vector((vector[2], vector[0], vector[1])))
        # TODO: Account for the case where all three components match
    return gram_schmidt(*vectors, normalise=normalise)
