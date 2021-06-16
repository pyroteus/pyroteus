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
