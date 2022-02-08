"""
Problem specification for a simple DG
advection test case.

Code here is based on that found at
    https://firedrakeproject.org/demos/DG-advection.py.html

It is concerned with solid body
rotation of a collection of shapes,
as first considered in [LeVeque 1996].

[LeVeque 1996] R. LeVeque, 'High
    -resolution conservative algorithms
    for advection in incompressible
    flow' (1996).
"""
from firedrake import *
from pyroteus.utility import rotate


# Problem setup
mesh = UnitSquareMesh(40, 40)
coords = mesh.coordinates.copy(deepcopy=True)
coords.interpolate(coords - as_vector([0.5, 0.5]))
mesh = Mesh(coords)
fields = ["tracer_2d"]
full_rotation = 2 * pi
end_time = full_rotation
dt = pi / 300
dt_per_export = 25
steady = False
wq = Constant(1.0)


def get_function_spaces(mesh):
    r"""
    :math:`\mathbb P1` space.
    """
    return {"tracer_2d": FunctionSpace(mesh, "CG", 1)}


def get_solver(self):
    """
    Advection equation solved using an
    iterative method and time integrated
    using Crank-Nicolson with implicitness
    one half.
    """

    def solver(i, ic, field="tracer_2d"):
        t_start, t_end = self.time_partition[i].subinterval
        dt = self.time_partition[i].timestep
        V = ic[field].function_space()
        mesh = V.mesh()
        x, y = SpatialCoordinate(mesh)
        W = VectorFunctionSpace(mesh, "CG", 1)
        u = interpolate(as_vector([-y, x]), W)
        nu = Constant(0.0)
        dtc = Constant(dt)
        theta = Constant(0.5)

        # Set initial condition
        q = Function(V, name=field)
        q_old = Function(V, name=field + "_old")
        q_old.assign(ic[field])

        # Setup variational problem
        psi = TrialFunction(V)
        phi = TestFunction(V)
        a = (
            psi * phi * dx
            + dtc * theta * dot(u, grad(psi)) * phi * dx
            + dtc * theta * nu * inner(grad(psi), grad(phi)) * dx
        )
        L = (
            q_old * phi * dx
            - dtc * (1 - theta) * dot(u, grad(q_old)) * phi * dx
            - dtc * (1 - theta) * nu * inner(grad(q_old), grad(phi)) * dx
        )

        # Setup Crank-Nicolson time integrator
        sp = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "pc_factor_levels": 0,
        }
        problem = LinearVariationalProblem(
            a, L, q, bcs=DirichletBC(V, 0, "on_boundary")
        )
        solver = LinearVariationalSolver(
            problem, solver_parameters=sp, ad_block_tag=field
        )

        # Time integrate from t_start to t_end
        t = t_start
        qoi = self.get_qoi(i)
        solutions = {field: q}
        while t < t_end - 0.5 * dt:
            solver.solve()
            if self.qoi_type == "time_integrated":
                self.J += qoi(solutions, t)
            q_old.assign(q)
            t += dt
        return solutions

    return solver


def bell_initial_condition(x, y, fs):
    bell_r0, bell_x0, bell_y0 = 0.15, -0.25, 0.0
    return 0.25 * (
        1
        + cos(
            pi
            * min_value(sqrt(pow(x - bell_x0, 2) + pow(y - bell_y0, 2)) / bell_r0, 1.0)
        )
    )


def cone_initial_condition(x, y, fs):
    cone_r0, cone_x0, cone_y0 = 0.15, 0.0, -0.25
    return 1.0 - min_value(
        sqrt(pow(x - cone_x0, 2) + pow(y - cone_y0, 2)) / cone_r0, 1.0
    )


def slot_cyl_initial_condition(x, y, fs):
    cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.0, 0.25
    slot_left, slot_right, slot_top = -0.025, 0.025, 0.35
    return conditional(
        sqrt(pow(x - cyl_x0, 2) + pow(y - cyl_y0, 2)) < cyl_r0,
        conditional(And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0),
        0.0,
    )


def get_initial_condition(self, coordinates=None):
    """
    Initial condition consisting of
    the sum of a bell, cone and
    slotted cylinder.
    """
    init_fs = self.function_spaces["tracer_2d"][0]
    if coordinates is not None:
        assert init_fs.mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(init_fs.mesh())
    bell = bell_initial_condition(x, y, init_fs)
    cone = cone_initial_condition(x, y, init_fs)
    slot_cyl = slot_cyl_initial_condition(x, y, init_fs)
    return {"tracer_2d": interpolate(bell + cone + slot_cyl, init_fs)}


def get_qoi(self, i, exact=get_initial_condition, linear=True):
    """
    Quantity of interest which
    computes square L2 error of the
    advected slotted cylinder (or
    specified shape).
    """
    dtc = Constant(self.time_partition[i].timestep)

    def time_integrated_qoi(sol, t):
        assert len(list(sol.keys())) == 1
        field = list(sol.keys())[0]
        q = sol[field]
        V = q.function_space()
        mesh = V.mesh()
        x = SpatialCoordinate(mesh)
        W = VectorFunctionSpace(mesh, "CG", 1)
        angle = -2 * pi * t / full_rotation
        X = interpolate(rotate(x, angle), W)
        q_exact = exact(self, X)
        r0 = 0.15
        if field in ("tracer_2d", "slot_cyl_2d"):
            x0, y0 = 0.0, 0.25
        elif field == "bell_2d":
            x0, y0 = -0.25, 0.0
        elif field == "cone_2d":
            x0, y0 = 0.0, -0.25
        else:
            raise ValueError(f"Tracer field {field} not recognised")
        x0, y0 = interpolate(rotate(as_vector([x0, y0]), angle), W)
        ball = conditional((x[0] - x0) ** 2 + (x[1] - y0) ** 2 < r0**2, 1.0, 0.0)
        if linear:
            return wq * dtc * ball * q * dx
        else:
            return wq * dtc * ball * (q - q_exact[field]) ** 2 * dx

    def end_time_qoi(sol):
        return sum(
            time_integrated_qoi({key: value}, end_time) for key, value in sol.items()
        )

    if self.qoi_type == "end_time":
        dtc.assign(1.0)
        return end_time_qoi
    else:
        return time_integrated_qoi
