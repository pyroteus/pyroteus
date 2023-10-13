# Object-oriented Burgers equation
# ================================

# You may have noticed that the functions :func:`get_form`,
# :func:`get_solver`, :func:`get_initial_condition` and
# :func:`get_qoi` all take a :class:`MeshSeq` as input and return
# a function. If this all feels a lot like writing methods for a
# :class:`MeshSeq` subclass, that's because this is exactly what
# we are doing. The constructors for :class:`MeshSeq` and
# :class:`AdjointMeshSeq` simply take these functions and adopt
# them as methods. A more natural way to write the subclass yourself.
#
# In the following, we mostly copy the contents from the previous
# demos and combine the methods into a subclass called
# :class:`BurgersMeshSeq`. The main difference to note is that
# :meth:`get_qoi` should get the :func:`annotate_qoi` decorator
# so that it gets modified internally to account for annotation.
# ::

from firedrake import *
from goalie_adjoint import *


set_log_level(DEBUG)


class BurgersMeshSeq(AdjointMeshSeq):
    @staticmethod
    def get_function_spaces(mesh):
        return {"u": VectorFunctionSpace(mesh, "CG", 2)}

    def get_form(self):
        def form(index, solutions):
            u, u_ = solutions["u"]
            P = self.time_partition

            # Define constants
            R = FunctionSpace(mesh_seq[index], "R", 0)
            dt = Function(R).assign(P.timesteps[index])
            nu = Function(R).assign(0.0001)

            # Setup variational problem
            v = TestFunction(u.function_space())
            F = (
                inner((u - u_) / dt, v) * dx
                + inner(dot(u, nabla_grad(u)), v) * dx
                + nu * inner(grad(u), grad(v)) * dx
            )
            return {"u": F}

        return form

    def get_solver(self):
        def solver(index, ic):
            function_space = self.function_spaces["u"][index]
            u = Function(function_space, name="u")
            solution_map = {"u": u}

            # Initialise 'lagged' solution
            u_ = Function(function_space, name="u_old")
            u_.assign(ic["u"])

            # Define form
            F = self.form(index, {"u": (u, u_)})["u"]

            # Time integrate from t_start to t_end
            t_start, t_end = self.subintervals[index]
            dt = self.time_partition.timesteps[index]
            t = t_start
            qoi = self.get_qoi(solution_map, index)
            while t < t_end - 1.0e-05:
                solve(F == 0, u, ad_block_tag="u")
                if self.qoi_type == "time_integrated":
                    self.J += qoi(t)
                u_.assign(u)
                t += dt
            return solution_map

        return solver

    def get_initial_condition(self):
        fs = self.function_spaces["u"][0]
        x, y = SpatialCoordinate(self[0])
        return {"u": interpolate(as_vector([sin(pi * x), 0]), fs)}

    @annotate_qoi
    def get_qoi(self, solutions, i):
        R = FunctionSpace(self[i], "R", 0)
        dt = Function(R).assign(self.time_partition[i].timestep)

        def end_time_qoi():
            u = solutions["u"]
            return inner(u, u) * ds(2)

        def time_integrated_qoi(t):
            u = solutions["u"]
            return dt * inner(u, u) * ds(2)

        if self.qoi_type == "end_time":
            return end_time_qoi
        else:
            return time_integrated_qoi


# Notice that the :meth:`get_solver` and :meth:`get_qoi_function`
# methods have been modified to account for both ``"end_time"`` and
# ``"time_integrated"`` QoIs.
#
# We apply exactly the same setup as before, except that the
# :class:`BurgersMeshSeq` class is used. ::

n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n
num_subintervals = len(meshes)
P = TimePartition(
    end_time, num_subintervals, dt, ["u"], num_timesteps_per_export=2
)
mesh_seq = BurgersMeshSeq(P, meshes, qoi_type="end_time")
solutions = mesh_seq.solve_adjoint()

# Plotting this, we find that the results are identical to those generated previously. ::

fig, axes, tcs = plot_snapshots(
    solutions, P, "u", "adjoint", levels=np.linspace(0, 0.8, 9)
)
fig.savefig("burgers-oo.jpg")

# .. figure:: burgers-oo.jpg
#    :figwidth: 90%
#    :align: center

# In the `next demo <./solid_body_rotation.py.html>`__, we move on from Burgers equation
# to consider a linear advection example with a rotational velocity field.
#
# This demo can also be accessed as a `Python script <burgers_oo.py>`__.
