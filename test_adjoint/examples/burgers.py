"""
Problem specification for a simple Burgers
equation test case.

The test case is notable for Goalie
because the prognostic equation is
nonlinear.

Code here is based on that found at
    https://firedrakeproject.org/demos/burgers.py.html
"""
from firedrake import *


# Problem setup
n = 32
mesh = UnitSquareMesh(n, n, diagonal="left")
fields = ["uv_2d"]
end_time = 0.5
dt = 1 / n
dt_per_export = 2
steady = False
get_bcs = None


def get_function_spaces(mesh):
    r"""
    :math:`\mathbb P2` space.
    """
    return {"uv_2d": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(self):
    """
    Burgers equation weak form.
    """

    def form(i, sols):
        u, u_ = sols["uv_2d"]
        dt = self.time_partition[i].timestep
        fs = self.function_spaces["uv_2d"][i]
        R = FunctionSpace(self[i], "R", 0)
        dtc = Function(R).assign(dt)
        nu = Function(R).assign(0.0001)
        v = TestFunction(fs)
        F = (
            inner((u - u_) / dtc, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return F

    return form


def get_solver(self):
    """
    Burgers equation solved using
    a direct method and backward
    Euler timestepping.
    """

    def solver(i, ic):
        t_start, t_end = self.time_partition[i].subinterval
        dt = self.time_partition[i].timestep
        fs = self.function_spaces["uv_2d"][i]
        u = Function(fs, name="uv_2d")
        solution_map = {"uv_2d": u}

        # Set initial condition
        u_ = Function(fs, name="uv_2d_old")
        u_.assign(ic["uv_2d"])

        # Setup variational problem
        F = self.form(i, {"uv_2d": (u, u_)})

        # Time integrate from t_start to t_end
        t = t_start
        qoi = self.get_qoi(solution_map, i)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="uv_2d")
            if self.qoi_type == "time_integrated":
                self.J += qoi(t)
            u_.assign(u)
            t += dt
        return solution_map

    return solver


def get_initial_condition(self):
    """
    Initial condition which is
    sinusoidal in the x-direction.
    """
    init_fs = self.function_spaces["uv_2d"][0]
    x, y = SpatialCoordinate(self.meshes[0])
    return {"uv_2d": interpolate(as_vector([sin(pi * x), 0]), init_fs)}


def get_qoi(self, sol, i):
    """
    Quantity of interest which
    computes the square L2
    norm over the right hand
    boundary.
    """
    R = FunctionSpace(self[i], "R", 0)
    dtc = Function(R).assign(self.time_partition[i].timestep)

    def time_integrated_qoi(t):
        u = sol["uv_2d"]
        return dtc * inner(u, u) * ds(2)

    def end_time_qoi():
        return time_integrated_qoi(end_time)

    if self.qoi_type == "end_time":
        dtc.assign(1.0)
        return end_time_qoi
    else:
        return time_integrated_qoi
