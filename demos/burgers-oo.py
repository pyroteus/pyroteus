# Object-oriented Burgers equation
# ================================

# You may have noticed that the functions `get_form`, `get_solver`,
# `get_initial_condition` and `get_qoi` all take a :class:`MeshSeq`
# as input and return a function. If this all feels a lot like
# writing methods for a :class:`MeshSeq` subclass, that's because
# this is exactly what we are doing. The constructors for
# :class:`MeshSeq` and :class:`AdjointMeshSeq` simply take these
# functions and adopt them as methods. A more natural way to write
# the subclass yourself.
#
# In the following, we mostly copy the contents from the previous
# demos and combine the methods into a subclass called
# :class:`BurgersMeshSeq`. The main difference to note is that
# `get_qoi` should named as :meth:`get_qoi_function`. This is
# because it gets modified internally to account for annotation.
# ::

from firedrake import *
from pyroteus_adjoint import *


class BurgersMeshSeq(AdjointMeshSeq):
    def get_function_spaces(self, mesh):
        return {"u": VectorFunctionSpace(mesh, "CG", 2)}

    def get_form(self):
        def form(index, solutions):
            u, u_ = solutions["u"]
            P = self.time_partition
            dt = Constant(P.timesteps[index])

            # Specify viscosity coefficient
            nu = Constant(0.0001)

            # Setup variational problem
            v = TestFunction(u.function_space())
            F = (
                inner((u - u_) / dt, v) * dx
                + inner(dot(u, nabla_grad(u)), v) * dx
                + nu * inner(grad(u), grad(v)) * dx
            )
            return F

        return form

    def get_solver(self):
        def solver(index, ic):
            function_space = self.function_spaces["u"][index]
            u = Function(function_space)

            # Initialise 'lagged' solution
            u_ = Function(function_space, name="u_old")
            u_.assign(ic["u"])

            # Define form
            F = self.form(index, {"u": (u, u_)})

            # Time integrate from t_start to t_end
            P = self.time_partition
            t_start, t_end = P.subintervals[index]
            dt = P.timesteps[index]
            t = t_start
            while t < t_end - 1.0e-05:
                solve(F == 0, u, ad_block_tag="u")
                u_.assign(u)
                t += dt
            return {"u": u}

        return solver

    def get_initial_condition(self):
        fs = self.function_spaces["u"][0]
        x, y = SpatialCoordinate(self[0])
        return {"u": interpolate(as_vector([sin(pi * x), 0]), fs)}

    def get_qoi_function(self, solutions, i):
        def end_time_qoi():
            u = solutions["u"]
            return inner(u, u) * ds(2)

        return end_time_qoi


# We apply exactly the same setup as before, except that the :class:`BurgersMeshSeq`
# class is used. ::

n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
end_time = 0.5
dt = 1 / n
num_subintervals = 2
P = TimePartition(
    end_time, num_subintervals, dt, "u", timesteps_per_export=2, debug=True
)
mesh_seq = BurgersMeshSeq(P, meshes, qoi_type="end_time")
solutions = mesh_seq.solve_adjoint()

# Plotting this, we find that the results are identical to those generated previously. ::

fig, axes = plot_snapshots(solutions, P, "u", "adjoint", levels=np.linspace(0, 0.8, 9))
fig.savefig("burgers-oo.jpg")

# .. figure:: burgers-oo.jpg
#    :figwidth: 90%
#    :align: center
#
# This demo can also be accessed as a `Python script <burgers-oo.py>`__.
