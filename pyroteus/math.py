import numpy as np
import ufl


__all__ = ["bessi0", "bessk0", "gram_schmidt", "construct_orthonormal_basis"]


def _recurse(a, *val):
    out = val[0]
    if len(val) > 1:
        out += a * _recurse(*val[1:])
    return out


def bessi0(x):
    """
    Modified Bessel function of the first kind.

    Code taken from :cite:`VVP+:92`.
    """
    ax = abs(x)
    expr1 = _recurse(
        (x / 3.75) ** 2,
        1.0,
        3.5156229,
        3.0899424,
        1.2067492,
        2.659732e-1,
        3.60768e-2,
        4.5813e-3,
    )
    expr2 = (
        ufl.exp(ax)
        / ufl.sqrt(ax)
        * _recurse(
            3.75 / ax,
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
    )
    return ufl.conditional(ax < 3.75, expr1, expr2)


def bessk0(x):
    """
    Modified Bessel function of the second kind.

    Code taken from :cite:`VVP+:92`.
    """
    expr1 = -ufl.ln(x / 2.0) * bessi0(x)
    expr1 += _recurse(
        x * x / 4.0,
        -0.57721566,
        0.42278420,
        0.23069756,
        3.488590e-2,
        2.62698e-3,
        1.0750e-4,
        7.4e-6,
    )
    expr2 = (
        ufl.exp(-x)
        / ufl.sqrt(x)
        * _recurse(
            2.0 / x,
            1.25331414,
            -7.832358e-2,
            2.189568e-2,
            -1.062446e-2,
            5.87872e-3,
            -2.51540e-3,
            5.3208e-4,
        )
    )
    return ufl.conditional(x > 2, expr2, expr1)


def gram_schmidt(*v, normalise=False):
    """
    Given some vectors, construct an orthogonal basis
    using Gram-Schmidt orthogonalisation.

    :args v: the vectors to orthogonalise
    :kwargs normalise: do we want an orthonormal basis?
    """
    if isinstance(v[0], np.ndarray):
        from numpy import dot, sqrt
    else:
        from ufl import dot, sqrt
    u = []

    def proj(x, y):
        return dot(x, y) / dot(x, x) * x

    for i, vi in enumerate(v):
        if i > 0:
            vi -= sum([proj(uj, vi) for uj in u])
        u.append(vi / sqrt(dot(vi, vi)) if normalise else vi)
    if isinstance(v[0], np.ndarray):
        u = [np.array(ui) for ui in u]
    return u


def construct_orthonormal_basis(v, dim=None, seed=0):
    """
    Starting from a single vector in UFL, construct
    a set of vectors which are orthonormal w.r.t. it.

    :arg v: the vector
    :kwarg dim: its dimension
    :kwarg seed: seed for random number generator
    """
    np.random.seed(seed)
    dim = dim or ufl.domain.extract_unique_domain(v).topological_dimension()
    if dim == 2:
        return [ufl.perp(v)]
    elif dim > 2:
        vectors = [
            ufl.as_vector(np.random.rand(dim)) for i in range(dim - 1)
        ]  # (arbitrary)
        return gram_schmidt(v, *vectors, normalise=True)[1:]  # (orthonormal)
    else:
        raise ValueError(f"Dimension {dim} not supported.")
