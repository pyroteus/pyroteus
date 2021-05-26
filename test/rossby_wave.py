"""
"Rossby equatorial soliton" test case, as described in [Huang et al.].

This was added as a Thetis 2D shallow water test by Joe Wallwork. It
can be found at
    thetis/test/swe2d/test_rossby_wave.py

[Huang et al.] H. Huang, C. Chen, G.W. Cowles, C.D. Winant,
    R.C. Beardsley, K.S. Hedstrom and D.B. Haidvogel, "FVCOM
    validation experiments: Comparisons with ROMS for three
    idealized barotropic test problems", Journal of Geophysical
    Research: Oceans, 113(C7). (2008), pp. 3-6.
"""
try:
    from thetis import *
except ImportError:
    import pytest
    pytest.xfail("Thetis is not installed")
import pyadjoint


# Problem setup
n = 24
lx, ly = 48, 24
mesh = PeriodicRectangleMesh(2*n, n, lx, ly, direction='x')
x, y = SpatialCoordinate(mesh)
W = mesh.coordinates.function_space()
mesh = Mesh(interpolate(as_vector([x-lx/2, y-ly/2]), W))
U_2d = VectorFunctionSpace(mesh, "DG", 1, name="U_2d")
H_2d = get_functionspace(mesh, "DG", 1, name="H_2d")
function_space = MixedFunctionSpace([U_2d, H_2d])
dt = 9.6/n
end_time = 20.0
dt_per_export = int(10.0/dt)
solves_per_dt = [1]
order = 1
soliton_amplitude = 0.395
fields = ['velocity_elevation']


def solver(ic, t_start, t_end, dt, J=0, qoi=None, **model_options):
    """
    The same P1DG-P1DG discretisation is used as
    in the Thetis 2D shallow water test. See the
    file cited above for further details.
    """
    mesh2d = ic.function_space().mesh()
    P1_2d = FunctionSpace(mesh2d, "CG", 1)
    bathymetry2d = Function(P1_2d).assign(1.0)

    # Stash default gravitational acceleration
    g = physical_constants['g_grav'].values()[0]
    physical_constants['g_grav'].assign(1.0)

    # Setup solver and stash QoI
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
    options = solver_obj.options
    options._isfrozen = False
    options.J = J

    # Setup problem
    options.timestepper_type = 'CrankNicolson'
    options.timestep = 9.6/n
    options.simulation_export_time = 10.0
    options.simulation_end_time = t_end
    if qoi is not None and np.isclose(t_end, end_time):
        options.simulation_end_time += 0.5*dt
    options.use_grad_div_viscosity_term = False
    options.use_grad_depth_viscosity_term = False
    options.horizontal_viscosity = None
    options.output_directory = 'outputs/rossby_wave'
    solver_obj.create_function_spaces()
    options.coriolis_frequency = SpatialCoordinate(mesh2d)[1]
    model_options.setdefault('no_exports', True)
    options.update(model_options)

    # Apply no-slip boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {
        'on_boundary': {'uv': Constant(as_vector([0, 0]))}
    }

    # Apply initial conditions
    uv_a, elev_a = ic.split()
    solver_obj.assign_initial_conditions(uv=uv_a, elev=elev_a)

    def update_forcings(t):
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t)

    # Correct counters and iterate
    i_export = int(t_start/dt/dt_per_export)
    solver_obj.simulation_time = t_start
    solver_obj.i_export = i_export
    solver_obj.next_export_t = t_start
    solver_obj.iteration = int(t_start/dt)
    solver_obj.export_initial_state = np.isclose(t_start, 0.0)
    if not options.no_exports and len(options.fields_to_export) > 0:
        for e in solver_obj.exporters['vtk'].exporters:
            solver_obj.exporters['vtk'].exporters[e].set_next_export_ix(i_export)
    solver_obj.iterate(update_forcings=update_forcings)

    # Revert gravitational acceleration
    physical_constants['g_grav'].assign(g)
    return solver_obj.fields.solution_2d, options.J


def initial_condition(fs):
    """
    The initial condition is a modon with
    two peaks of equal size, equally spaced
    to the North and South.
    """
    return asymptotic_expansion(fs, time=0.0)


@pyadjoint.no_annotations
def asymptotic_expansion(fs, time=0.0):
    """
    The test case admits an asymptotic
    expansion solution in terms of
    Hermite series.
    """
    x, y = SpatialCoordinate(fs.mesh())
    q_a = Function(fs)
    uv_a, elev_a = q_a.split()

    # Variables for asymptotic expansion
    t = Constant(time)
    B = Constant(soliton_amplitude)
    modon_propagation_speed = -1.0/3.0
    if order != 0:
        assert order == 1
        modon_propagation_speed -= 0.395*B*B
    c = Constant(modon_propagation_speed)
    xi = x - c*t
    psi = exp(-0.5*y*y)
    phi = 0.771*(B/cosh(B*xi))**2
    dphidx = -2*B*phi*tanh(B*xi)
    C = -0.395*B*B

    # Zeroth order terms
    u_terms = phi*0.25*(-9 + 6*y*y)*psi
    v_terms = 2*y*dphidx*psi
    eta_terms = phi*0.25*(3 + 6*y*y)*psi
    if order == 0:
        uv_a.interpolate(as_vector([u_terms, v_terms]))
        elev_a.interpolate(eta_terms)
        return q_a

    # Unnormalised Hermite series coefficients for u
    u = np.zeros(28)
    u[0] = 1.7892760e+00
    u[2] = 0.1164146e+00
    u[4] = -0.3266961e-03
    u[6] = -0.1274022e-02
    u[8] = 0.4762876e-04
    u[10] = -0.1120652e-05
    u[12] = 0.1996333e-07
    u[14] = -0.2891698e-09
    u[16] = 0.3543594e-11
    u[18] = -0.3770130e-13
    u[20] = 0.3547600e-15
    u[22] = -0.2994113e-17
    u[24] = 0.2291658e-19
    u[26] = -0.1178252e-21

    # Unnormalised Hermite series coefficients for v
    v = np.zeros(28)
    v[3] = -0.6697824e-01
    v[5] = -0.2266569e-02
    v[7] = 0.9228703e-04
    v[9] = -0.1954691e-05
    v[11] = 0.2925271e-07
    v[13] = -0.3332983e-09
    v[15] = 0.2916586e-11
    v[17] = -0.1824357e-13
    v[19] = 0.4920951e-16
    v[21] = 0.6302640e-18
    v[23] = -0.1289167e-19
    v[25] = 0.1471189e-21

    # Unnormalised Hermite series coefficients for eta
    eta = np.zeros(28)
    eta[0] = -3.0714300e+00
    eta[2] = -0.3508384e-01
    eta[4] = -0.1861060e-01
    eta[6] = -0.2496364e-03
    eta[8] = 0.1639537e-04
    eta[10] = -0.4410177e-06
    eta[12] = 0.8354759e-09
    eta[14] = -0.1254222e-09
    eta[16] = 0.1573519e-11
    eta[18] = -0.1702300e-13
    eta[20] = 0.1621976e-15
    eta[22] = -0.1382304e-17
    eta[24] = 0.1066277e-19
    eta[26] = -0.1178252e-21

    # Hermite polynomials
    polynomials = [Constant(1.0), 2*y]
    for i in range(2, 28):
        polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])

    # First order terms
    u_terms += C*phi*0.5625*(3 + 2*y*y)*psi
    u_terms += phi*phi*psi*sum(u[i]*polynomials[i] for i in range(28))
    v_terms += dphidx*phi*psi*sum(v[i]*polynomials[i] for i in range(28))
    eta_terms += C*phi*0.5625*(-5 + 2*y*y)*psi
    eta_terms += phi*phi*psi*sum(eta[i]*polynomials[i] for i in range(28))

    uv_a.interpolate(as_vector([u_terms, v_terms]))
    elev_a.interpolate(eta_terms)
    return q_a


def time_integrated_qoi(sol, t):
    """
    Quantity of interest for the Rossby
    wave propagation problem which computes
    the time-integrated square L2 error of
    the advected Rossby soliton.
    """
    q_a = asymptotic_expansion(sol.function_space(), t)
    return inner(sol - q_a, sol - q_a)*dx


def end_time_qoi(sol):
    """
    Quantity of interest for the Rossby
    wave propagation problem which computes
    the square L2 error of the advected
    Rossby soliton.
    """
    return time_integrated_qoi(sol, end_time)


if __name__ == "__main__":

    # Plot analytical solution
    outfile = File("outputs/rossby_wave/analytical.pvd")
    V = function_space
    solution = Function(V, name="Analytical solution")
    u_a, eta_a = solution.split()
    u_a.rename("Analytical velocity solution")
    eta_a.rename("Analytical elevation solution")
    t = 0.0
    step = 0
    while t < end_time + 0.5*dt:
        if step % dt_per_export == 0:
            solution.assign(asymptotic_expansion(V, t))
            outfile.write(u_a, eta_a)
        t += dt
        step += 1

    def qoi(sol, t):
        return assemble(time_integrated_qoi(sol, t))

    # Plot finite element solution
    ic = initial_condition(function_space)
    sol, J = solver(ic, 0, end_time, dt, qoi=qoi, no_exports=False)
    print_output(f"Quantity of interest: {J:.4e}")
