"""
Problem specification for a version of
the DG advection test case which treats
the solution as three separate tracer
fields.

See
    ./solid_body_rotation.py

for details on the problem itself.

The test case is notable for Pyroteus
because the model is comprised of a
system of equations which are solved
sequentially.
"""
from solid_body_rotation import *
import solid_body_rotation as sbr


fields = ["bell_2d", "cone_2d", "slot_cyl_2d"]
end_time /= 3


def get_function_spaces(mesh):
    return {field: FunctionSpace(mesh, "CG", 1) for field in fields}


def get_solver(self):
    """
    The same solver as in the original version,
    but with the tracer fields solved for
    sequentially.
    """
    solver_ = sbr.get_solver(self)

    def solver(*args):
        ret = {}
        for field in self.fields:
            ret.update(solver_(*args, field=field))
        return ret

    return solver


def get_initial_condition(self, coordinates=None):
    """
    The same initial conditions as in the
    original version, but with the three
    shapes treated as separate tracer fields.
    """
    fs = self.function_spaces
    if coordinates is not None:
        for key, value in fs.items():
            assert value[0].mesh() == coordinates.function_space().mesh()
        x, y = coordinates
    else:
        x, y = SpatialCoordinate(self.meshes[0])
    init = {
        "bell_2d": bell_initial_condition,
        "cone_2d": cone_initial_condition,
        "slot_cyl_2d": slot_cyl_initial_condition,
    }
    return {
        field: interpolate(1.0 + init[field](x, y, fs[field][0]), fs[field][0])
        for field in fs
    }


def get_qoi(self, i):
    """
    Unlike in the original version, the QoI
    is the squared L2 error for each shape,
    rather than just the slotted cylinder.
    """
    return sbr.get_qoi(self, i, exact=get_initial_condition)
