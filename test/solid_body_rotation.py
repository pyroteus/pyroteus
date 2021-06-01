"""
Problem specification for a simple DG
advection test case.

Code here is based on that found at
    https://firedrakeproject.org/demos/DG-advection.py.html

It it concerned with solid body
rotation of a collection of shapes,
as first considered in [LeVeque 1996].

The test case is notable for Pyroteus
because more than one linear solve is
required per timestep.

[LeVeque 1996] R. LeVeque, 'High
    -resolution conservative algorithms
    for advection in incompressible
    flow' (1996).
"""
from firedrake import *
from pyroteus.utility import rotate
import pyadjoint


# Problem setup
mesh = UnitSquareMesh(40, 40, quadrilateral=True)
coords = mesh.coordinates.copy(deepcopy=True)
coords.interpolate(coords - as_vector([0.5, 0.5]))
mesh = Mesh(coords)
fields = ['tracer_2d']
function_space = {'tracer_2d': FunctionSpace(mesh, "DQ", 1)}
solves_per_dt = [3]
full_rotation = 2*pi
end_time = full_rotation
dt = pi/300
dt_per_export = 75


def solver(ic, t_start, t_end, dt, J=0, qoi=None, label='tracer_2d'):
    """
    Solve an advection-diffusion equation on a
    subinterval (t_start, t_end), given some
    initial conditions `ic` and a timestep `dt`.
    """
    V = ic[label].function_space()
    mesh = V.mesh()
    x, y = SpatialCoordinate(mesh)
    W = VectorFunctionSpace(mesh, "CG", 1)
    u = interpolate(as_vector([-y, x]), W)
    dtc = Constant(dt)
    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))

    # Set initial condition
    q = Function(V)
    q.assign(ic[label])

    # Set inflow condition value
    q_in = Constant(1.0)

    # Setup variational problem
    dq_trial = TrialFunction(V)
    phi = TestFunction(V)
    a = phi*dq_trial*dx
    L1 = dtc*q*div(phi*u)*dx \
        - dtc*conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds \
        - dtc*conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds \
        - dtc*(phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS
    q1, q2 = Function(V), Function(V)
    L2, L3 = replace(L1, {q: q1}), replace(L1, {q: q2})
    dq = Function(V)

    # Setup SSPRK33 time integrator
    sp = {
        "ksp_type": "preonly",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    }
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=sp)
    prob2 = LinearVariationalProblem(a, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=sp)
    prob3 = LinearVariationalProblem(a, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=sp)

    # Time integrate from t_start to t_end
    t = t_start
    while t < t_end - 0.5*dt:
        solv1.solve()
        q1.assign(q + dq)
        solv2.solve()
        q2.assign(0.75*q + 0.25*(q1 + dq))
        solv3.solve()
        q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))
        if qoi is not None:
            J += qoi({label: q}, t)
        t += dt
    return {label: q}, J


def bell_initial_condition(x, y, fs):
    bell_r0, bell_x0, bell_y0 = 0.15, -0.25, 0.0
    return 0.25*(1 + cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))


def cone_initial_condition(x, y, fs):
    cone_r0, cone_x0, cone_y0 = 0.15, 0.0, -0.25
    return 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)


def slot_cyl_initial_condition(x, y, fs):
    cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.0, 0.25
    slot_left, slot_right, slot_top = -0.025, 0.025, 0.35
    return conditional(
        sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0, conditional(
            And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0), 0.0)


@pyadjoint.no_annotations
def initial_condition(fs, coordinates=None):
    """
    Initial condition consisting of
    the sum of a bell, cone and
    slotted cylinder.
    """
    init_fs = fs['tracer_2d'][0]
    if coordinates is not None:
        assert init_fs.mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(init_fs.mesh())
    bell = bell_initial_condition(x, y, init_fs)
    cone = cone_initial_condition(x, y, init_fs)
    slot_cyl = slot_cyl_initial_condition(x, y, init_fs)
    return {'tracer_2d': interpolate(1.0 + bell + cone + slot_cyl, init_fs)}


def time_integrated_qoi(sol, t, exact=initial_condition):
    """
    Quantity of interest which
    integrates the square L2 error
    of the advected slotted cylinder
    (or a specified shape) in time.
    """
    labels = list(sol.keys())
    assert len(labels) == 1
    label = labels[0]
    q = sol[label]
    V = q.function_space()
    mesh = V.mesh()
    x = SpatialCoordinate(mesh)
    W = VectorFunctionSpace(mesh, "CG", 1)
    theta = -2*pi*t/full_rotation
    X = interpolate(rotate(x, theta), W)
    q_exact = exact({label: V}, X)
    r0 = 0.15
    if label in ('tracer_2d', 'slot_cyl_2d'):
        x0, y0 = 0.0, 0.25
    elif label == 'bell_2d':
        x0, y0 = -0.25, 0.0
    elif label == 'cone_2d':
        x0, y0 = 0.0, -0.25
    else:
        raise ValueError(f"Tracer field {label} not recognised")
    x0, y0 = interpolate(rotate(as_vector([x0, y0]), theta), W)
    ball = conditional((x[0] - x0)**2 + (x[1] - y0)**2 < r0**2, 1.0, 0.0)
    return ball*(q-q_exact[label])**2*dx


def end_time_qoi(sol, exact=initial_condition):
    """
    Quantity of interest which
    computes square L2 error of the
    advected slotted cylinder (or
    specified shape) at the
    simulation end time.
    """
    ret = 0
    for key, value in sol.items():
        ret += time_integrated_qoi({key: value}, end_time, exact=exact)
    return ret


# ---------------------------
# plotting and debugging
# ---------------------------

if __name__ == "__main__":

    # Plot analytical solution
    outfile = File("outputs/solid_body_rotation/analytical.pvd")
    outfile.step = 0
    x = SpatialCoordinate(mesh)
    W = VectorFunctionSpace(mesh, "CG", 1)
    V = function_space
    solution = Function(V, name="Analytical solution")
    t = 0.0
    while t < end_time + 0.5*dt:
        if outfile.step % dt_per_export == 0:
            theta = -2*pi*t/end_time
            X = interpolate(rotate(x, theta), W)
            solution.interpolate(initial_condition(V, X))
            outfile.write(solution)
        outfile.step += 1
        t += dt

    # Plot finite element solution
    outfile = File("outputs/solid_body_rotation/solution.pvd")
    outfile.step = 0

    def qoi(sol, t):
        if outfile.step % dt_per_export != 0:
            sol.rename("Finite element solution")
            outfile.write(sol['tracer_2d'])
            outfile.step += 1
        return assemble(time_integrated_qoi(sol, t))

    ic = initial_condition({'tracer_2d': V})
    sol, J = solver(ic, 0, end_time, dt, qoi=qoi)
    outfile.write(sol['tracer_2d'])
    print(f"Quantity of interest: {J:.4e}")
