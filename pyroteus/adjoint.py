"""
Driver functions for solving adjoint problems on multiple meshes.
"""
import firedrake
from firedrake_adjoint import Control
import pyadjoint
from pyroteus.interpolation import project
from pyroteus.utility import AttrDict, norm
from functools import wraps
import numpy as np


__all__ = ["get_checkpoints", "solve_adjoint"]


def wrap_solver(solver):
    """
    Wrapper for a solver which returns a :class:`Control` associated
    with its initial condition.

    The solver should take four arguments: an initial condition
    :class:`Function`, a start time, an end time and a timestep.
    It should return the final solution :class:`Function`.
    """

    @wraps(solver)
    def wrapper(ic, t_start, t_end, dt, **kwargs):
        init = {key: value.copy(deepcopy=True) for key, value in ic.items()}
        control = [Control(value) for key, value in init.items()]  # TODO: Check order
        out, J = solver(init, t_start, t_end, dt, **kwargs)
        msg = "Solver should return a dictionary of Functions" \
              + "corresponding to the final solution"
        for field in ic:
            assert isinstance(out[field], firedrake.Function), msg
        return out, control

    return wrapper


def wrap_qoi(qoi):
    """
    Wrapper for contributions to a quantity of interest which sets
    the corresponding adj_value to unity.

    The QoI can have either one or two arguments. In the former case,
    it depends only on the solution (at the final time)
    """

    @wraps(qoi)
    def wrapper(*args, **kwargs):
        j = firedrake.assemble(qoi(*args, **kwargs))
        j.adj_value = 1.0
        return j

    return wrapper


def solve_adjoint(solver, initial_condition, qoi, function_spaces, time_partition, **kwargs):
    """
    Solve an adjoint problem on a sequence of subintervals.

    As well as the quantity of interest value, a dictionary
    of solution fields is returned, the contents of which
    give values at all exported timesteps, indexed first by
    the field label and then by type. The contents of these
    nested dictionaries are lists which are indexed first by
    subinterval and then by export. For a given exported
    timestep, the solution types are:

    * ``'forward'``: the forward solution after taking the
        timestep;
    * ``'forward_old'``: the forward solution before taking
        the timestep;
    * ``'adjoint'``: the adjoint solution after taking the
        timestep;
    * ``'adjoint_next'``: the adjoint solution before taking
        the timestep (backwards).

    :arg solver: a function which takes an initial condition
        :class:`Function`, a start time and an end time as
        arguments and returns the solution value at the final
        time
    :arg initial_condition: a function which maps a list of
        :class:`FunctionSpace` s to a list of
        :class:`Function` s
    :arg qoi: a function which maps a :class:`Function` (and
        possibly a time level) to a UFL 1-form
    :arg function_spaces: dictionary of lists of
        :class:`FunctionSpace` s for each field, indexed
        by subinterval
    :arg time_partition: :class:`TimePartition` object
        containing the subintervals
    :kwarg solver_kwargs: a dictionary providing parameters
        to the solver

    :return J: quantity of interest value
    :return solution: an :class:`AttrDict` containing
        solution fields and their lagged versions.
    """
    import inspect
    nargs = len(inspect.getfullargspec(qoi).args)
    # assert nargs in (1, 2), f"QoI has more arguments than expected ({nargs})"
    assert nargs >= 1, "QoI should have at least one argument"  # FIXME: kwargs are counted as args
    fields = time_partition.fields

    # Solve forward to get checkpoints and evaluate QoI
    J, checkpoints = get_checkpoints(
        solver, initial_condition, qoi, function_spaces, time_partition, **kwargs,
    )

    # Create arrays to hold exported foward and adjoint solutions
    solutions = AttrDict({
        field: AttrDict({
            label: [
                [
                    firedrake.Function(fs)
                    for j in range(time_partition.exports_per_subinterval[i]-1)
                ] for i, fs in enumerate(function_spaces[field])
            ] for label in ('forward', 'forward_old', 'adjoint', 'adjoint_next')
        }) for field in fields
    })

    # Wrap solver and QoI
    wrapped_solver = wrap_solver(solver)
    wrapped_qoi = wrap_qoi(qoi)
    solver_kwargs = kwargs.get('solver_kwargs', {})
    if nargs > 1:
        solver_kwargs['qoi'] = wrapped_qoi

    # Clear tape
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()

    # Loop over subintervals in reverse
    seeds = None
    for i in reversed(range(time_partition.num_subintervals)):

        # Annotate tape on current subinterval
        sols, controls = wrapped_solver(checkpoints[i], *time_partition[i], **solver_kwargs)

        # Get seed vector for reverse mode propagation
        if i == time_partition.num_subintervals-1:
            if nargs == 1:
                J = wrapped_qoi(sols)  # TODO: What about kwargs?
        else:
            with pyadjoint.stop_annotating():
                for field, fs in function_spaces.items():
                    sols[field].adj_value = project(seeds[field], fs[i], adjoint=True)

        # Solve adjoint problem
        m = pyadjoint.enlisting.Enlist(controls)
        with pyadjoint.stop_annotating():
            with tape.marked_nodes(m):
                tape.evaluate_adj(markings=True)

        # Loop over prognostic variables
        for field in fields:

            # Get old solution dependency index
            solve_blocks = time_partition.get_solve_blocks(field, subinterval=i)
            for dep_index, dep in enumerate(solve_blocks[0].get_dependencies()):
                if hasattr(dep.output, 'function_space'):
                    if dep.output.function_space() == solve_blocks[0].function_space:
                        break
            # FIXME: What if other dependencies are in the prognostic space?

            # Extract solution data
            sols = solutions[field]
            for j, block in enumerate(solve_blocks[::time_partition.timesteps_per_export[i]]):
                sols.forward_old[i][j].assign(block.get_dependencies()[dep_index].saved_output)
                sols.forward[i][j].assign(block.get_outputs()[0].saved_output)
                sols.adjoint[i][j].assign(block.adj_sol)
                sols.adjoint_next[i][j].assign(solve_blocks[j+1].adj_sol)
            if np.isclose(norm(solutions[field].adjoint[i][0]), 0.0):
                raise ValueError(f"Adjoint solution on subinterval {i} is zero")

        # Get adjoint action
        seeds = {
            field: firedrake.Function(function_spaces[field][i],
                                      val=control.block_variable.adj_value)
            for field, control in zip(fields, controls)
        }
        for field, seed in seeds.items():
            if np.isclose(norm(seed), 0.0):
                raise ValueError(f"Adjoint action for field {field} on subinterval {i} is zero")
        tape.clear_tape()

    return J, solutions


@pyadjoint.no_annotations
def get_checkpoints(solver, initial_condition, qoi, function_spaces, time_partition, **kwargs):
    """
    Just solve forward to get checkpoints and evaluate the QoI.

    :arg solver: a function which takes an initial condition
        :class:`Function`, a start time and an end time as
        arguments and returns the solution value at the final
        time
    :arg initial_condition: a function which maps a list of
        :class:`FunctionSpace` s to a list of
        :class:`Function` s
    :arg qoi: a function which maps a :class:`Function` (and
        possibly a time level) to a UFL 1-form
    :arg function_spaces: dictionary of lists of
        :class:`FunctionSpace` s for each field, indexed
        by subinterval
    :arg time_partition: :class:`TimePartition` object
        containing the subintervals
    :kwarg solver_kwargs: a dictionary providing parameters
        to the solver

    :return J: quantity of interest value
    :return checkpoints: forward solution data at the beginning
        of each subinterval
    """
    import inspect
    nargs = len(inspect.getfullargspec(qoi).args)
    # assert nargs in (1, 2), f"QoI has more arguments than expected ({nargs})"
    assert nargs >= 1, "QoI should have at least one argument"  # FIXME: kwargs are counted as args
    fields = time_partition.fields

    # Solve forward to get checkpoints and evaluate QoI
    solver_kwargs = kwargs.get('solver_kwargs', {})
    if nargs > 1:
        solver_kwargs['qoi'] = lambda *args, **kwargs: firedrake.assemble(qoi(*args, **kwargs))
    J = 0
    checkpoints = [initial_condition(function_spaces)]
    for i in range(time_partition.num_subintervals):
        sols, J = solver(checkpoints[i], *time_partition[i], J=J, **solver_kwargs)
        if i < time_partition.num_subintervals-1:
            checkpoints.append({
                field: project(sols[field], fs[i+1])
                for field, fs in function_spaces.items()
            })
    if nargs == 1:
        J = firedrake.assemble(qoi(sols))  # TODO: What about kwargs?
    return J, checkpoints
