import numpy as np
import ufl


__all__ = ["bessi0", "bessk0", "gram_schmidt", "construct_basis"]


def recursive_polynomial(x, coeffs):
    r"""
    Compute the polynomial

    ..math::
        a_0 + x (a_1 + x (a_2 + \dots + x a_n)),

    with coefficients :math:`a_0,\dots,a_n` in a recursive manner.

    :arg x: value at which to evaluate the polynomial
    :arg coeffs: tuple of coefficients defining the polynomial
    """
    p = coeffs[0]
    if len(coeffs) > 1:
        p += x * recursive_polynomial(x, coeffs[1:])
    return p


def bessi0(x):
    """
    Modified Bessel function of the first kind.

    Code taken from :cite:`VVP+:92`.
    """
    if isinstance(x, np.ndarray):
        from numpy import abs, exp, sqrt

        if np.isclose(x, 0).any():
            raise ValueError("Cannot divide by zero.")
    else:
        from ufl import abs, exp, sqrt

        # TODO: Check that valid UFL types are used

    ax = abs(x)
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

    expr1 = recursive_polynomial(x1, coeffs1)
    expr2 = exp(ax) / sqrt(ax) * recursive_polynomial(x2, coeffs2)

    if isinstance(x, np.ndarray):
        return np.where(ax < 3.75, expr1, expr2)
    else:
        return ufl.conditional(ax < 3.75, expr1, expr2)


def bessk0(x):
    """
    Modified Bessel function of the second kind.

    Code taken from :cite:`VVP+:92`.
    """
    if isinstance(x, np.ndarray):
        from numpy import log as ln
        from numpy import exp, sqrt, where

        if (x <= 0).any():
            raise ValueError("Cannot take the logarithm of a non-positive number.")
    else:
        from ufl import exp, ln, sqrt
        from ufl import conditional as where

        # TODO: Check that valid UFL types are used

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
    expr1 = -ln(x / 2.0) * bessi0(x) + recursive_polynomial(x1, coeffs1)
    expr2 = exp(-x) / sqrt(x) * recursive_polynomial(x2, coeffs2)
    return where(x <= 2, expr1, expr2)


def gram_schmidt(*vectors, normalise=False):
    """
    Given some vectors, construct an orthogonal basis
    using Gram-Schmidt orthogonalisation.

    :args vectors: the vectors to orthogonalise
    :kwargs normalise: do we want an orthonormal basis?
    """
    if isinstance(vectors[0], np.ndarray):
        from numpy import dot, sqrt

        # Check that vector types match
        for i, vi in enumerate(vectors[1:]):
            if not isinstance(vi, type(vectors[0])):
                raise TypeError(
                    f"Inconsistent vector types: '{type(vectors[0])}' vs. '{type(vi)}'."
                )
    else:
        from ufl import dot, sqrt

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


def construct_basis(vector, normalise=True, seed=0):
    """
    Construct a basis from a given vector.

    :arg vector: the starting vector
    :kwargs normalise: do we want an orthonormal basis?
    :kwarg seed: seed for random number generator
    """
    np.random.seed(seed)
    is_numpy = isinstance(vector, np.ndarray)
    if is_numpy:
        from numpy import dot

        if len(vector.shape) > 1:
            raise ValueError(
                f"Expected a vector, got an array of shape {vector.shape}."
            )
        as_vector = np.array
        dim = vector.shape[0]
    else:
        from ufl import as_vector, dot

        # TODO: Check that valid UFL types are used
        dim = ufl.domain.extract_unique_domain(vector).topological_dimension()

    if dim == 0:
        raise ValueError("Dimension 0 not supported.")
    vectors = [vector]
    while len(vectors) < dim:
        v = as_vector(np.random.rand(dim))
        if is_numpy and np.allclose(vector / dot(vector, vector), v / dot(v, v)):
            np.random.seed(seed + 100)
            continue
            # TODO: Add a similar check in the UFL case
        vectors.append(v)

    # Apply Gram-Schmidt to the set of vectors generated
    return gram_schmidt(*vectors, normalise=normalise)
