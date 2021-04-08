"""
Problem specification for a simple Burgers equation demo.

Code here is based on that found at
    https://firedrakeproject.org/demos/burgers.py.html
"""
from firedrake import *


__all__ = ["setup", "solver", "initial_condition", "end_time_qoi", "time_integrated_qoi"]


def setup():
    """
    Get default :class:`FunctionSpace`, simulation
    end time and timestep.
    """
    n = 32
    mesh = UnitSquareMesh(n, n, diagonal='left')
    fs = VectorFunctionSpace(mesh, "CG", 2)
    end_time = 0.5
    dt = 1/n
    return fs, end_time, dt


def solver(ic, t_start, t_end, dt, J=0, qoi=None):
    """
    Solve Burgers' equation on an interval
    (t_start, t_end), given viscosity `nu` and some
    initial condition `ic`.
    """
    fs = ic.function_space()
    dtc = Constant(dt)
    nu = Constant(0.0001)

    # Set initial condition
    u_ = Function(fs)
    u_.assign(ic)

    # Setup variational problem
    v = TestFunction(fs)
    u = Function(fs)
    F = inner((u - u_)/dtc, v)*dx \
        + inner(dot(u, nabla_grad(u)), v)*dx \
        + nu*inner(grad(u), grad(v))*dx

    # Time integrate from t_start to t_end
    t = t_start
    while t < t_end - 1.0e-05:
        solve(F == 0, u)
        if qoi is not None:
            J += qoi(u, t)
        u_.assign(u)
        t += dt
    return u_, J


def initial_condition(fs):
    """
    Initial condition for Burgers' equation
    which is sinusoidal in the x-direction.

    :arg fs: :class:`FunctionSpace` which
        the initial condition will live in
    """
    x, y = SpatialCoordinate(fs.mesh())
    return interpolate(as_vector([sin(pi*x), 0]), fs)


def end_time_qoi(sol):
    """
    Quantity of interest for Burgers' equation
    which computes the square L2 norm over the
    right hand boundary segment at the final
    time.

    :arg sol: the solution :class:`Function`
    """
    return inner(sol, sol)*ds(2)


def time_integrated_qoi(sol, t):
    """
    Quantity of interest for Burgers' equation
    which computes the integrand for a time
    integral given by the square L2 norm over the
    right hand boundary segment.

    :arg sol: the solution :class:`Function`
    :arg t: time level
    """
    return inner(sol, sol)*ds(2)
