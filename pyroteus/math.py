from firedrake import *


__all__ = ["bessi0", "bessk0"]


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
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(
        0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = exp(ax)/sqrt(ax)*(0.39894228 + y2*(0.1328592e-1 + y2*(
        0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(
            -0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)


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
    y1 = x*x/4.0
    expr1 = -ln(x/2.0)*bessi0(x) + (-0.57721566 + y1*(0.42278420 + y1*(
        0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = exp(-x)/sqrt(x)*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(
        -0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)
