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
    import thetis  # noqa
except ImportError:
    import pytest
    pytest.xfail("Thetis is not installed")
from pyroteus.thetis_compat import *


# Problem setup
lx, ly = 16, 1.1
nx, ny = lx*5, 5
mesh = RectangleMesh(nx, ny, lx, ly)
x, y = SpatialCoordinate(mesh)
fields = ['swe2d', 'sediment', 'exner']
morfac = 300
end_time = 0.75*3600/morfac
dt = 0.3
dt_per_export = 5
morfac = 300
steady = False


def get_function_spaces(mesh):
    """
    An equal order P1DG-P1DG element pair
    is used for the shallow water equations.
    P1DG is also used for sediment, but
    the Exner equation is solved in P1 space.
    """
    return {
        'swe2d':
            MixedFunctionSpace([
                VectorFunctionSpace(mesh, "DG", 1, name="U_2d"),
                get_functionspace(mesh, "DG", 1, name="H_2d"),
            ]),
        'sediment':
            get_functionspace(mesh, "DG", 1, name="Q_2d"),
        'exner':
            get_functionspace(mesh, "CG", 1, name="P1_2d"),
    }


def get_solver(self):
    def solver(i, ic, **model_options):
        """
        Solve the coupled hydro-morphodynamics
        system on a subinterval (t_start, t_end),
        given some initial conditions `ic` and
        a timestep `dt`.
        """
        t_start, t_end = self.time_partition[i].subinterval
        dt = self.time_partition[i].timestep
        bathymetry2d = Function(self.function_spaces['exner'][i])
        bathymetry2d.assign(ic['exner'])
        mesh2d = bathymetry2d.function_space().mesh()

        # Setup solver
        solver_obj = FlowSolver2d(mesh2d, bathymetry2d)
        options = solver_obj.options

        # Setup sediment model
        sed_options = options.sediment_model_options
        sed_options.solve_suspended_sediment = True
        sed_options.use_bedload = True
        sed_options.solve_exner = True
        sed_options.use_sediment_conservative_form = True
        sed_options.average_sediment_size = Constant(1.6e-04)
        sed_options.bed_reference_height = Constant(0.025)
        sed_options.morphological_acceleration_factor = Constant(morfac)

        # Setup problem
        options.timestepper_type = 'CrankNicolson'
        options.timestepper_options.implicitness_theta = 1.0
        options.norm_smoother = Constant(0.1)
        options.timestep = dt
        options.simulation_export_time = dt*self.time_partition[i].timesteps_per_export
        options.simulation_end_time = t_end
        options.horizontal_viscosity = Constant(1.0e-06)
        options.horizontal_diffusivity = Constant(0.15)
        options.nikuradse_bed_roughness = Constant(3*sed_options.average_sediment_size)
        options.output_directory = 'outputs/migrating_trench'
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
        uv, elev = ic['swe2d'].split()
        solver_obj.assign_initial_conditions(uv=uv, elev=elev, sediment=ic['sediment'])

        # Setup QoI
        qoi = self.get_qoi(i)
        solutions = {
            'swe2d': solver_obj.fields.solution_2d,
            'sediment': solver_obj.fields.sediment_2d,
            'exner': solver_obj.fields.bathymetry_2d,
        }

        def update_forcings(t):
            if self.qoi_type == 'time_integrated':
                self.J += qoi(solutions, t - dt)
                # TODO: Use a callback instead

        # Correct counters and iterate
        solver_obj.correct_counters(self.time_partition[i])
        solver_obj.iterate(update_forcings=update_forcings)

        if self.qoi_type == 'time_integrated':
            for solution in solutions:  # FIXME: HACK
                self.J += qoi(solutions, t_end)
        return solutions

    return solver


def get_initial_condition(self):
    """
    The initial bed is given by the trench
    profile, sediment is initialised to zero
    and velocity and elevation are given
    constant values.
    """
    fs = self.function_spaces
    q_init = Function(fs['swe2d'][0])
    sediment_init = Function(fs['sediment'][0])
    bed_init = Function(fs['exner'][0])

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
        'swe2d': q_init,
        'sediment': sediment_init,
        'exner': bed_init,
    }


def get_qoi(self, i):
    """
    Quantity of interest which integrates
    sediment over the domain.
    """
    dtc = Constant(self.time_partition[i].timestep)
    wq = Constant(1.0)
    self.counter = 0

    def time_integrated_qoi(sol, t):
        t_start, t_end = self.time_partition[i].subinterval
        wq.assign(0.0 if np.isclose(t, t_start) or self.counter % 3 != 2 else 1.0)  # Backward Euler
        b = sol['exner']
        self.counter += 1
        return wq*dtc*inner(grad(b), grad(b))*dx

    def end_time_qoi(sol):
        b = sol['exner']
        return inner(grad(b), grad(b))*dx

    if self.qoi_type == 'end_time':
        return end_time_qoi
    else:
        return time_integrated_qoi
