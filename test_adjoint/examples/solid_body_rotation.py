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
import numpy as np
import ufl


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


def rotation_matrix_2d(angle):
    """
    Rotation matrix associated with some
    angle, as a UFL matrix.

    :arg angle: the angle
    """
    return ufl.as_matrix(
        [[ufl.cos(angle), -ufl.sin(angle)], [ufl.sin(angle), ufl.cos(angle)]]
    )


def rotate(v, angle, origin=None):
    """
    Rotate a UFL :class:`ufl.tensors.as_vector`
    by some angle.

    :arg v: the vector to rotate
    :arg angle: the angle to rotate by
    :kwarg origin: origin of rotation
    """
    dim = len(v)
    origin = origin or ufl.as_vector(np.zeros(dim))
    assert len(origin) == dim, "Origin does not match dimension"
    if dim == 2:
        R = rotation_matrix_2d(angle)
    else:
        raise NotImplementedError
    return ufl.dot(R, v - origin) + origin


def get_function_spaces(mesh, field="tracer_2d"):
    r"""
    :math:`\mathbb P1` space.
    """
    return {field: FunctionSpace(mesh, "CG", 1)}


def get_form(self):
    """
    Weak form for advection-diffusion
    using Crank-Nicolson with implicitness
    one half.
    """

    def form(i, sols, field="tracer_2d"):
        q, q_ = sols[field]
        dt = self.time_partition[i].timestep
        V = q_.function_space()
        mesh = V.mesh()
        x, y = SpatialCoordinate(mesh)
        W = VectorFunctionSpace(mesh, "CG", 1)
        u = interpolate(as_vector([-y, x]), W)
        dtc = Constant(dt)
        theta = Constant(0.5)
        psi = TrialFunction(V)
        phi = TestFunction(V)
        a = psi * phi * dx + dtc * theta * dot(u, grad(psi)) * phi * dx
        L = q_ * phi * dx - dtc * (1 - theta) * dot(u, grad(q_)) * phi * dx
        return a, L

    return form


def get_bcs(self):
    """
    Zero Dirichlet condition on all boundaries.
    """

    def bcs(i, field="tracer_2d"):
        fs = self.function_spaces[field][i]
        return [DirichletBC(fs, 0, "on_boundary")]

    return bcs


def get_solver(self):
    """
    Advection equation solved using an
    iterative method.
    """

    def solver(i, ic, field="tracer_2d"):
        t_start, t_end = self.time_partition[i].subinterval
        dt = self.time_partition[i].timestep
        V = ic[field].function_space()
        q = Function(V, name=field)
        solutions = {field: q}

        # Set initial condition
        q_ = Function(V, name=field + "_old")
        q_.assign(ic[field])

        # Setup variational problem
        a, L = self.form(i, {field: (q, q_)}, field=field)
        bc = self.bcs(i, field=field)

        # Setup Crank-Nicolson time integrator
        sp = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "pc_factor_levels": 0,
        }
        problem = LinearVariationalProblem(a, L, q, bcs=bc)
        solver = LinearVariationalSolver(
            problem, solver_parameters=sp, ad_block_tag=field
        )

        # Time integrate from t_start to t_end
        t = t_start
        qoi = self.get_qoi(solutions, i)
        while t < t_end - 0.5 * dt:
            solver.solve()
            if self.qoi_type == "time_integrated":
                self.J += qoi(t)
            q_.assign(q)
            t += dt
        return solutions

    return solver


def bell_initial_condition(x, y, fs):
    bell_r0, bell_x0, bell_y0 = 0.15, -0.25, 0.0
    r = sqrt(pow(x - bell_x0, 2) + pow(y - bell_y0, 2))
    return 0.25 * (1 + cos(pi * min_value(r / bell_r0, 1.0)))


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


def get_initial_condition(self, coordinates=None, field="tracer_2d"):
    """
    Initial condition consisting of
    the sum of a bell, cone and
    slotted cylinder.
    """
    init_fs = self.function_spaces[field][0]
    if coordinates is not None:
        assert init_fs.mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(init_fs.mesh())
    bell = bell_initial_condition(x, y, init_fs)
    cone = cone_initial_condition(x, y, init_fs)
    slot_cyl = slot_cyl_initial_condition(x, y, init_fs)
    return {field: interpolate(bell + cone + slot_cyl, init_fs)}


def get_qoi(self, sol, i, exact=get_initial_condition, linear=True):
    """
    Quantity of interest which
    computes square L2 error of the
    advected slotted cylinder (or
    specified shape).
    """
    dtc = Constant(self.time_partition[i].timestep)

    def get_sub_qoi(field):
        def qoi(t):
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

        return qoi

    def time_integrated_qoi(t):
        return sum(get_sub_qoi(key)(t) for key in sol)

    def end_time_qoi():
        return time_integrated_qoi(end_time)

    if self.qoi_type == "end_time":
        dtc.assign(1.0)
        return end_time_qoi
    else:
        return time_integrated_qoi
