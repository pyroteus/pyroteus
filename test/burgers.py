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


# Problem setup
n = 32
mesh = UnitSquareMesh(n, n, diagonal='left')
fields = ['uv_2d']
solves_per_dt = [1]
end_time = 0.5
dt = 1/n
dt_per_export = 2


def get_function_spaces(mesh):
    """
    Construct a sequence of P2 function
    spaces based on the meshes of the
    :class:`MeshSeq`.
    """
    return {'uv_2d': VectorFunctionSpace(mesh, "CG", 2)}


def get_solver(self):
    """
    Burgers equation is solved using
    Backward Euler timestepping.
    """
    def solver(i, ic):
        t_start, t_end = self.time_partition[i].subintervals[i]
        dt = self.time_partition.timesteps[i]
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
        qoi = self.qoi
        while t < t_end - 1.0e-05:
            solve(F == 0, u)
            if self.qoi_type == 'time_integrated':
                self.J += qoi({'uv_2d': u}, t)
            u_.assign(u)
            t += dt
        return {'uv_2d': u_}
    return solver


def get_initial_condition(self):
    """
    Initial condition for Burgers' equation
    which is sinusoidal in the x-direction.
    """
    init_fs = self.function_spaces['uv_2d'][0]
    x, y = SpatialCoordinate(self.meshes[0])
    return {'uv_2d': interpolate(as_vector([sin(pi*x), 0]), init_fs)}


def get_qoi(self):
    """
    Quantity of interest which
    computes the square L2
    norm over the right hand
    boundary.
    """
    def end_time_qoi(sol):
        u = sol['uv_2d']
        return inner(u, u)*ds(2)

    def time_integrated_qoi(sol, t):
        u = sol['uv_2d']
        return inner(u, u)*ds(2)

    if self.qoi_type == 'end_time':
        return end_time_qoi
    else:
        return time_integrated_qoi
