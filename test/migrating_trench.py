"""
'Migrating trench' test case, as
described in [Clare et al.].

This was added as a Thetis 2D
coupled model test by Mariana
Clare. It can be found at
    thetis/test/sediment/test_migrating_trench.py

The test case is notable for Pyroteus
because the model is comprised of a
system of coupled equations which
are solved sequentially.

[Clare et al.] M.C.A. Clare et al.,
    'Hydro-morphodynamics 2D
    modelling using a discontinuous
    Galerkin discretisaton', (2020)
    Computers & Geosciences, 104658.
"""
try:
    from thetis import *
except ImportError:
    import pytest
    pytest.xfail("Thetis is not installed")
import pyadjoint


# Problem setup
lx, ly = 16, 1.1
nx, ny = lx*5, 5
mesh = RectangleMesh(nx, ny, lx, ly)
x, y = SpatialCoordinate(mesh)
fields = ['uv-elev', 'sediment', 'bed']
P1_2d = get_functionspace(mesh, "CG", 1)
U_2d = VectorFunctionSpace(mesh, "DG", 1, name="U_2d")
H_2d = get_functionspace(mesh, "DG", 1, name="H_2d")
function_space = {
    'uv-elev': MixedFunctionSpace([U_2d, H_2d]),
    'sediment': H_2d,
    'bed': P1_2d,
}
solves_per_dt = [1, 1, 1]
morfac = 300
end_time = 1.5*3600/morfac  # TODO: reduce?
dt = 0.3
dt_per_export = 6
morfac = 300


def solver(ic, t_start, t_end, dt, J=0, qoi=None, **model_options):
    """
    Solve the coupled hydro-morphodynamics
    system on a subinterval (t_start, t_end),
    given some initial conditions `ic` and
    a timestep `dt`.
    """
    bathymetry2d = ic['bed']
    mesh2d = bathymetry2d.function_space().mesh()

    # Setup solver and stash QoI
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
    options = solver_obj.options
    options._isfrozen = False
    options.J = J

    # Setup sediment model
    options.sediment_model_options.solve_suspended_sediment = True
    options.sediment_model_options.use_bedload = True
    options.sediment_model_options.solve_exner = True
    options.sediment_model_options.use_sediment_conservative_form = True
    options.sediment_model_options.average_sediment_size = Constant(1.6e-04)
    options.sediment_model_options.bed_reference_height = Constant(0.025)
    options.sediment_model_options.morphological_acceleration_factor = Constant(morfac)

    # Setup problem
    options.timestepper_type = 'CrankNicolson'
    options.timestepper_options.implicitness_theta = 1.0
    options.norm_smoother = Constant(0.1)
    options.timestep = dt
    options.simulation_export_time = 6*0.3
    options.simulation_end_time = t_end
    # if qoi is not None and np.isclose(t_end, end_time):
    #     options.simulation_end_time += 0.5*dt
    options.horizontal_viscosity = Constant(1.0e-06)
    options.horizontal_diffusivity = Constant(0.15)
    options.nikuradse_bed_roughness = Constant(3*options.sediment_model_options.average_sediment_size)
    model_options.setdefault('no_exports', True)
    options.update(model_options)

    # Apply boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {
        1: {'flux': Constant(-0.22)},
        2: {'elev': Constant(0.397)},
    }
    solver_obj.bnd_functions['sediment'] = {
        1: {'flux': Constant(-0.22), 'equilibrium': None},
        2: {'elev': Constant(0.397)},
    }

    # Apply initial conditions
    uv, elev = ic['uv-elev'].split()
    solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment=ic['sediment'])
    solutions = {
        'uv-elev': solver_obj.fields.solution_2d,
        'sediment': solver_obj.fields.sediment_2d,
        'bed': solver_obj.fields.bathymetry_2d,
    }

    def update_forcings(t):
        if qoi is not None:
            options.J += qoi(solutions, t)

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
    return solutions, options.J


@pyadjoint.no_annotations
def initial_condition(fs):
    """
    The initial bed is given by the trench
    profile, sediment is initialised to zero
    and velocity and elevation are given
    constant values.
    """
    q_init = Function(fs['uv-elev'][0])
    sediment_init = Function(fs['sediment'][0])
    bed_init = Function(fs['bed'][0])

    uv_init, elev_init = q_init.split()
    uv_init.interpolate(as_vector([0.51, 0.0]))
    elev_init.assign(0.4)

    initial_depth = Constant(0.397)
    depth_riv = Constant(initial_depth - 0.397)
    depth_trench = Constant(depth_riv - 0.15)
    depth_diff = depth_trench - depth_riv
    bed_init.interpolate(
        -conditional(
            le(x, 5),
            depth_riv,
            conditional(
                le(x, 6.5),
                (1/1.5)*depth_diff*(x - 6.5) + depth_trench,
                conditional(
                    le(x, 9.5),
                    depth_trench,
                    conditional(
                        le(x, 11),
                        -(1/1.5)*depth_diff*(x - 11) + depth_riv,
                        depth_riv
                    )
                )
            )
        )
    )

    return {
        'uv-elev': q_init,
        'sediment': sediment_init,
        'bed': bed_init,
    }


def time_integrated_qoi(sol, t):
    """
    Quantity of interest which integrates
    sediment over the domain.
    """
    s = sol['sediment']
    return s*dx


def end_time_qoi(sol):
    """
    Quantity of interest which integrates
    sediment over the domain at the final
    time.
    """
    return time_integrated_qoi(sol, end_time)
