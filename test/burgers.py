"""
Problem specification for a simple Burgers
equation test case.

The test case is notable for Pyroteus
because the prognostic equation is
nonlinear.

Code here is based on that found at
    https://firedrakeproject.org/demos/burgers.py.html
"""
from firedrake import *
import pyadjoint


# Problem setup
n = 32
mesh = UnitSquareMesh(n, n, diagonal='left')
fields = ['uv_2d']
function_space = {'uv_2d': VectorFunctionSpace(mesh, "CG", 2)}
solves_per_dt = [1]
end_time = 0.5
dt = 1/n
dt_per_export = 2


def solver(ic, t_start, t_end, dt, J=0, qoi=None):
    """
    Solve Burgers' equation on a subinterval
    (t_start, t_end), given some initial
    conditions `ic` and a timestep `dt`.
    """
    fs = ic['uv_2d'].function_space()
    dtc = Constant(dt)
    nu = Constant(0.0001)

    # Set initial condition
    u_ = Function(fs)
    u_.assign(ic['uv_2d'])

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
            J += qoi({'uv_2d': u}, t)
        u_.assign(u)
        t += dt
    return {'uv_2d': u_}, J


@pyadjoint.no_annotations
def initial_condition(fs):
    """
    Initial condition for Burgers' equation
    which is sinusoidal in the x-direction.

    :arg fs: :class:`FunctionSpace` which
        the initial condition will live in
    """
    init_fs = fs['uv_2d'][0]
    x, y = SpatialCoordinate(init_fs.mesh())
    return {'uv_2d': interpolate(as_vector([sin(pi*x), 0]), init_fs)}


def time_integrated_qoi(sol, t):
    """
    Quantity of interest which
    integrates the square L2
    norm over the right hand
    boundary in time.

    :arg sol: the solution :class:`Function`
    :arg t: time level
    """
    u = sol['uv_2d']
    return inner(u, u)*ds(2)


def end_time_qoi(sol):
    """
    Quantity of interest which
    evaluates the square L2 norm
    over the right hand boundary
    segment at the final time.

    :arg sol: the solution :class:`Function`
    """
    u = sol['uv_2d']
    return inner(u, u)*ds(2)


if __name__ == "__main__":
    outfile = File('outputs/burgers/solution.pvd')

    def qoi(sol, t):
        outfile.write(sol['uv_2d'])
        return assemble(time_integrated_qoi(sol, t))

    ic = initial_condition(function_space)
    sol, J = solver(ic, 0, end_time, dt, qoi=qoi)
    print(f"Quantity of interest: {J:.4e}")
