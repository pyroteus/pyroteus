from firedrake import *
from pyroteus.utility import rotate


mesh = UnitSquareMesh(40, 40, quadrilateral=True)
coords = mesh.coordinates.copy(deepcopy=True)
coords -= 0.5
mesh = Mesh(coords)
function_space = FunctionSpace(mesh, "DQ", 1)
end_time = 2*pi
dt = pi/300
dt_per_export = 75
solves_per_dt = 3


def solver(ic, t_start, t_end, dt, J=0, qoi=None):
    """
    Solve the tracer transport problem in DQ1 space
    on an interval (t_start, t_end), given some
    initial condition `ic` and timestep `dt`.
    """
    V = ic.function_space()
    mesh = V.mesh()
    x, y = SpatialCoordinate(mesh)
    W = VectorFunctionSpace(mesh, "CG", 1)
    u = interpolate(as_vector([-y, x]), W)
    dtc = Constant(dt)
    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))

    # Set initial condition
    q = Function(V)
    q.assign(ic)

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
            J += qoi(q, t)
        t += dt
    return q, J


def initial_condition(fs, coordinates=None):
    """
    Initial condition for tracer
    transport problem consisting of a
    bell, cone and slotted cylinder.
    """
    if coordinates is not None:
        assert fs.mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(fs.mesh())

    bell_r0, bell_x0, bell_y0 = 0.15, -0.25, 0.0
    cone_r0, cone_x0, cone_y0 = 0.15, 0.0, -0.25
    cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.0, 0.25
    slot_left, slot_right, slot_top = -0.025, 0.025, 0.35

    bell = 0.25*(1 + cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
    slot_cyl = conditional(
        sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0, conditional(
            And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0), 0.0)

    return interpolate(1.0 + bell + cone + slot_cyl, fs)


def end_time_qoi(q):
    """
    Quantity of interest for the
    tracer transport problem which
    computes square L2 error of the
    advected slotted cylinder.
    """
    V = q.function_space()
    q_exact = initial_condition(V)

    x, y = SpatialCoordinate(V.mesh())
    cyl_x, cyl_y, cyl_r = 0.0, 0.25, 0.15
    ball = conditional((x - cyl_x)**2 + (y - cyl_y)**2 < cyl_r**2, 1.0, 0.0)

    return ball*(q-q_exact)**2*dx


def time_integrated_qoi(q, t):
    """
    Quantity of interest for the
    tracer transport problem which
    computes the time-integrated
    square L2 error of the
    advected slotted cylinder.
    """
    V = q.function_space()
    mesh = V.mesh()
    x = SpatialCoordinate(mesh)
    W = VectorFunctionSpace(mesh, "CG", 1)
    theta = -2*pi*t/end_time
    X = interpolate(rotate(x, theta), W)
    q_exact = initial_condition(V, X)

    cyl_x, cyl_y, cyl_r = 0.0, 0.25, 0.15
    cyl_x, cyl_y = interpolate(rotate(as_vector([cyl_x, cyl_y]), theta), W)
    ball = conditional((x[0] - cyl_x)**2 + (x[1] - cyl_y)**2 < cyl_r**2, 1.0, 0.0)

    return ball*(q-q_exact)**2*dx


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
            outfile.write(sol)
            outfile.step += 1
        return assemble(time_integrated_qoi(sol, t))

    ic = initial_condition(V)
    sol, J = solver(ic, 0, end_time, dt, qoi=qoi)
    outfile.write(sol)
    print(f"Quantity of interest: {J:.4e}")
