# Solid body rotation with multiple prognostic variables
# ======================================================

# So far, we have only considered PDEs with a single
# prognostic variable. The Pyroteus API can sometimes be
# cumbersome for such equations, since solutions are
# passed between methods in dictionaries, which are
# indexed by the field name. This approach becomes
# useful when we solve PDEs with multiple prognostic variables.
#
# In the the `previous demo
# <./solid_body_rotation.py.html>`__, we solved the solid
# body rotation problem with a single tracer concentration
# field. Here, we instead use different fields for each
# component of the initial condition, supposing that they
# correspond to different chemical compounds. Let's call
# them ``"bell"``, ``"cone"`` and ``"slot_cyl"``. ::

from solid_body_rotation import *


fields = ["bell", "cone", "slot_cyl"]


# Given that we previously included keyword arguments for
# field name, we are able to easily duplicate the various
# functions for the "split" case. ::

def get_function_spaces_split(mesh):
    ret = {}
    for f in fields:
        ret.update(get_function_spaces(mesh, field=f))
    return ret


def get_solver_split(mesh_seq):
    solver = get_solver(mesh_seq)

    def split_solver(*args):
        ret = {}
        for f in fields:
            ret.update(solver(*args, field=f))
        return ret

    return split_solver


def get_initial_condition_split(mesh_seq):
    x, y = SpatialCoordinate(mesh_seq[0])
    init = {
        "bell": bell_initial_condition,
        "cone": cone_initial_condition,
        "slot_cyl": slot_cyl_initial_condition,
    }
    return {
        f: interpolate(init[f](x, y), fs[0])
        for f, fs in mesh_seq.function_spaces.items()
    }


def get_qoi_split(mesh_seq, sols, index):
    return get_qoi(mesh_seq, sols, index, field="slot_cyl")


# Then we can set up the :class:`AdjointMeshSeq` much
# as before and plot the outputs in the same way. ::

end_time = 2 * pi
dt = pi / 300
time_partition = TimeInterval(
    end_time,
    dt,
    fields,
    timesteps_per_export=20,
)
mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_function_spaces=get_function_spaces_split,
    get_initial_condition=get_initial_condition_split,
    get_form=get_form,
    get_bcs=get_bcs,
    get_solver=get_solver_split,
    get_qoi=get_qoi_split,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

for field, sols in solutions.items():
    fwd_outfile = File(f"solid_body_rotation_split/{field}_forward.pvd")
    adj_outfile = File(f"solid_body_rotation_split/{field}_adjoint.pvd")
    for i, mesh in enumerate(mesh_seq):
        for sol in sols["forward"][i]:
            fwd_outfile.write(sol)
        for sol in sols["adjoint"][i]:
            adj_outfile.write(sol)

# Looking at the Paraview files, you should find that the
# ``slot_cyl_adjoint`` solution fields match the ``c_adjoint``
# fields from the previous demo. You should also find that the
# ``bell_adjoint`` and ``cone_adjoint`` solution fields are zero
# for all time. Convince yourself that these observations are to
# be expected.
#
# .. rubric:: Exercise
#
# Change the QoI so that we integrate over a region where the
# final cone is expected to be non-zero. Run the example again
# to see how the adjoint solutions differ for each field.
#
# This tutorial can be dowloaded as a
# `Python script <solid_body_rotation_split.py>`__.
