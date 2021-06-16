# TODO: doc
from pyroteus.compat.thwaites import *


mesh = RectangleMesh(50, 400, 0.5, 4)
end_time = 2.0
dt = 0.01
dt_per_export = 50
At = 0.5
rho_min = 1.0
rho_max = rho_min*(1.0 + At)/(1.0 - At)


def get_function_space(mesh):
    r"""
    :math:`\mathbb P1_{DG}-\mathbb P2`
    velocity-pressure pair and
    :math:`\mathbb P1_{DG}` density.
    """
    return {
        'up': MixedFunctionSpace([
            VectorFunctionSpace(mesh, "DG", 1),
            FunctionSpace(mesh, "CG", 2),
        ]),
        'rho': FunctionSpace(mesh, "DG", 1),
    }


def get_solver(self):
    # TODO: doc
    def solver(i, ic):
        t = self.time_partition[i].start_time
        dt = self.time_partition[i].timestep
        dt_per_export = self.time_partition[i].timesteps_per_export

        # Get FunctionSpaces
        Z = self.function_spaces['up'][i]
        Q = self.function_spaces['rho'][i]

        # Apply initial conditions
        z = Function(self.function_spaces['up'][0])
        z.assign(ic['up'])
        u, p = z.split()
        rho = Function(self.function_spaces['rho'][0])
        rho.assign(ic['rho'])

        # Create equations
        mom_eq = MomentumEquation(Z.sub(0), Z.sub(0))
        cty_eq = ContinuityEquation(Z.sub(1), Z.sub(1))
        rho_eq = ScalarAdvectionDiffusionEquation(Q, Q)

        # Specify physical parameters
        kappa = Constant(0.0)  # No diffusion of density
        Re = Constant(1000.0)  # Reynolds number
        mu = Constant(1.0/Re)
        mom_source = as_vector([0.0, -1.0])*(rho - rho_min)  # Buoyancy term Boussinesq approximation
        up_fields = {'viscosity': mu, 'source': mom_source}
        rho_fields = {'diffusivity': kappa, 'velocity': u}

        # Define boundary conditions
        rho_bcs = {}
        no_normal_flow = {'un': 0.0}            # TODO: Use Constant?
        no_slip = {'u': as_vector([0.0, 0.0])}  # TODO: Use Constant?
        up_bcs = {
            1: no_slip,
            2: no_normal_flow,
            3: no_slip,
            4: no_normal_flow,
        }

        # Specify solver parameters
        mumps_solver_parameters = {
            'mat_type': 'aij',
            'snes_monitor': None,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        up_solver_parameters = {
            'mat_type': 'matfree',
            'snes_type': 'newtonls',
            'ksp_type': 'preonly',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'full',
            # Velocity block
            'fieldsplit_0': {
                'ksp_type': 'gmres',
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'ksp_converged_reason': None,
                'assembled_ksp_type': 'preonly',
                'assembled_pc_type': 'bjacobi',
                'assembled_sub_pc_type': 'ilu',
            },
            # Pressure block
            'fieldsplit_1': {
                'ksp_type': 'preonly',
                'pc_type': 'python',
                'pc_python_type': 'thwaites.AssembledSchurPC',
                'schur_ksp_type': 'cg',
                'schur_ksp_max_it': 100,
                'schur_ksp_converged_reason': None,
                'schur_pc_type': 'gamg',
            }
        }
        pressure_nullspace = VectorSpaceBasis(constant=True)
        up_coupling = [{'pressure': 1}, {'velocity': 0}]

        # Create timesteppers
        up_timestepper = PressureProjectionTimeIntegrator(
            [mom_eq, cty_eq], z, up_fields, up_coupling, dt, up_bcs,
            solver_parameters=up_solver_parameters,
            predictor_solver_parameters=mumps_solver_parameters,
            picard_iterations=1, pressure_nullspace=pressure_nullspace,
        )
        rho_timestepper = DIRK33(
            rho_eq, rho, rho_fields, dt, rho_bcs,
            solver_parameters=rho_solver_parameters,
        )

        # Modify options prefixes
        up_timestepper.solver.options_prefix = 'up'
        up_timestepper.solution_old.rename('up_old')
        # FIXME: This is not the lagged solution! It is the last tendency
        rho_timestepper.solver.options_prefix = 'rho'
        rho_timestepper.solution_old.rename('rho_old')
        # FIXME: This is not the lagged solution! It is the last tendency

        # Setup for limiters
        # rho_limiter = VertexBasedLimiter(Q)
        # u_comp = Function(Q)
        # v_comp = Function(Q)
        # z_ = Function(Z)
        # u_, p_ = z_.split()

        # Setup QoI
        qoi = self.qoi
        solutions = {'up': z, 'rho': rho}

        # Enter timeloop
        step = 0
        u_file = File('outputs/rayleigh_taylor/velocity.pvd')
        p_file = File('outputs/rayleigh_taylor/pressure.pvd')
        rho_file = File('outputs/rayleigh_taylor/density.pvd')
        while t < self.time_partition[i].end_time - 0.5*dt:

            # Take a timestep
            up_timestepper.advance(t)
            rho_timestepper.advance(t)

            # Evalute QoI
            if self.qoi_type == 'time_integrated':
                self.J += qoi(solutions, t)

            # # Apply limiters
            # rho_limiter.apply(rho)
            # u_comp.interpolate(u[0])
            # rho_limiter.apply(u_comp)
            # v_comp.interpolate(v[0])
            # rho_limiter.apply(v_comp)
            # u_.interpolate(as_vector([u_comp, v_comp]))
            u_, p_ = u, p

            # Export
            if step % dt_per_export == 0:
                u_file.write(u_)
                p_file.write(p_)
                rho_file.write(rho)
            step += 1
            t += dt

        return solutions

    return solver


def get_initial_condition(self):
    """
    Zero initial velocity and pressure
    and a sharp, curved interface
    between density values.
    """
    z = Function(self.function_spaces['up'][0])
    rho = Function(self.function_spaces['rho'][0])
    x, y = SpatialCoordinate(self[0])
    y = y-2
    d = 1
    rho.interpolate(
        0.5*(
            (rho_max + rho_min)
            + (rho_max - rho_min)*tanh(y + 0.1*d*cos(2*pi*x/d))/(0.01*d)
        )
    )
    return {'up': z, 'rho': rho}


def get_qoi(self):
    r"""
    Quantity of interest which computes
    the :math:`H^1` norm of density.
    """
    def end_time_qoi(sol):
        rho = sol['rho']
        return inner(grad(rho), grad(rho))*dx

    def time_integrated_qoi(sol, t):
        rho = sol['rho']
        return inner(grad(rho), grad(rho))*dx

    if self.qoi_type == 'end_time':
        return end_time_qoi
    else:
        return time_integrated_qoi
