import numpy as np
import ufl


__all__ = ["bessi0", "bessk0", "gram_schmidt", "construct_orthonormal_basis"]


def bessi0(x):
    """
    Modified Bessel function of the first kind.

    Code taken from

    [Flannery et al. 1992] B.P. Flannery,
    W.H. Press, S.A. Teukolsky, W. Vetterling,
    "Numerical recipes in C", Press Syndicate
    of the University of Cambridge, New York
    (1992).
    """
    ax = abs(x)
    y1 = x / 3.75
    y1 *= y1
    expr1 = 1.0 + y1 * (
        3.5156229
        + y1
        * (
            3.0899424
            + y1 * (1.2067492 + y1 * (0.2659732 + y1 * (0.360768e-1 + y1 * 0.45813e-2)))
        )
    )
    y2 = 3.75 / ax
    expr2 = (
        ufl.exp(ax)
        / ufl.sqrt(ax)
        * (
            0.39894228
            + y2
            * (
                0.1328592e-1
                + y2
                * (
                    0.225319e-2
                    + y2
                    * (
                        -0.157565e-2
                        + y2
                        * (
                            0.916281e-2
                            + y2
                            * (
                                -0.2057706e-1
                                + y2
                                * (
                                    0.2635537e-1
                                    + y2 * (-0.1647633e-1 + y2 * 0.392377e-2)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return ufl.conditional(ax < 3.75, expr1, expr2)


def bessk0(x):
    """
    Modified Bessel function of the second kind.

    Code taken from

    [Flannery et al. 1992] B.P. Flannery,
    W.H. Press, S.A. Teukolsky, W. Vetterling,
    "Numerical recipes in C", Press Syndicate
    of the University of Cambridge, New York
    (1992).
    """
    y1 = x * x / 4.0
    expr1 = -ufl.ln(x / 2.0) * bessi0(x) + (
        -0.57721566
        + y1
        * (
            0.42278420
            + y1
            * (
                0.23069756
                + y1
                * (0.3488590e-1 + y1 * (0.262698e-2 + y1 * (0.10750e-3 + y1 * 0.74e-5)))
            )
        )
    )
    y2 = 2.0 / x
    expr2 = (
        ufl.exp(-x)
        / ufl.sqrt(x)
        * (
            1.25331414
            + y2
            * (
                -0.7832358e-1
                + y2
                * (
                    0.2189568e-1
                    + y2
                    * (
                        -0.1062446e-1
                        + y2 * (0.587872e-2 + y2 * (-0.251540e-2 + y2 * 0.53208e-3))
                    )
                )
            )
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
    proj = lambda x, y: dot(x, y) / dot(x, x) * x
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
    dim = dim or v.ufl_domain().topological_dimension()
    if dim == 2:
        return [ufl.perp(v)]
    elif dim > 2:
        vectors = [
            ufl.as_vector(np.random.rand(dim)) for i in range(dim - 1)
        ]  # (arbitrary)
        return gram_schmidt(v, *vectors, normalise=True)[1:]  # (orthonormal)
    else:
        raise ValueError(f"Dimension {dim} not supported.")
