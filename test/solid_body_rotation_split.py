"""
Problem specification for a version of
the DG advection test case which treats
the solution as three separate tracer
fields.

See
    ./solid_body_rotation.py

for details on the problem itself.

The test case is notable for Pyroteus
because: (a) the model is comprised of a
system of equations which are solved
sequentially; and (b) more than one
linear solve is required per field per
timestep.
"""
from solid_body_rotation import *
import solid_body_rotation as sbr


fields = ['bell_2d', 'cone_2d', 'slot_cyl_2d']
DQ = FunctionSpace(mesh, "DQ", 1)
function_space = {
    'bell_2d': DQ,
    'cone_2d': DQ,
    'slot_cyl_2d': DQ,
}
solves_per_dt = [3, 3, 3]


def solver(*args, qoi=None, J=0):
    """
    Apply the same solver from the original
    version to each tracer field simultaneously
    on a subinterval (t_start, t_end), given
    some initial conditions `ic` and a
    timestep `dt`.
    """
    sols = {}
    for label in fields:
        sol, j = sbr.solver(*args, J=0, qoi=qoi, label=label)
        sols.update(sol)
        J += j
    return sols, J


@pyadjoint.no_annotations
def initial_condition(fs, coordinates=None):
    """
    Initial conditions consisting of a
    bell, cone and slotted cylinder.
    """
    if coordinates is not None:
        for key, value in fs.items():
            assert value[0].mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(fs[list(fs.keys())[0]][0].mesh())
    init = {
        'bell_2d': bell_initial_condition,
        'cone_2d': cone_initial_condition,
        'slot_cyl_2d': slot_cyl_initial_condition,
    }
    return {
        field: interpolate(1.0 + init[field](x, y, fs[field][0]), fs[field][0])
        for field in fs
    }


def time_integrated_qoi(q, t):
    """
    Quantity of interest which
    integrates the square L2 error
    of a specified shape in time.
    """
    return sbr.time_integrated_qoi(q, t, exact=initial_condition)


def end_time_qoi(q):
    """
    Quantity of interest which
    computes square L2 error of a
    specified shape at the
    simulation end time.
    """
    return sbr.end_time_qoi(q, exact=initial_condition)
