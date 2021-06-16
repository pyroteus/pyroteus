try:
    import thwaites  # noqa
except ImportError:
    raise ImportError("Thwaites is not installed")
from thwaites import *


class DIRK33(thwaites.time_stepper.DIRK33):
    def update_solver(self):
        self.solver = []
        for i in range(self.n_stages):
            p = NonlinearVariationalProblem(self.F[i], self.k[i])
            self.solver.append(
                NonlinearVariationalSolver(p,
                    solver_parameters=self.solver_parameters,
                    options_prefix="rho" if i == self.n_stages-1 else None))
        self.solution_old.rename("rho_old")
        # FIXME: This is not the lagged solution! It is the last tendency


class PressureProjectionTimeIntegrator(thwaites.coupled_integrators.PressureProjectionTimeIntegrator):
    def initialize(self, *args):
        self.name = 'up'
        self.solution_old.rename('up_old')
        # FIXME: This is not the lagged solution! It is the last tendency
        super(PressureProjectionTimeIntegrator, self).initialize(*args)
