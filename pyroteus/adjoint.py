import firedrake
from firedrake.adjoint.blocks import GenericSolveBlock
import numpy as np
import pyadjoint
from functools import wraps


__all__ = ["solve_adjoint"]


def _get_initial_condition_control(solver):
    """
    Wrapper for a solver which returns a :class:`Control` associated
    with its initial condition.

    The solver should take three arguments: an initial condition
    :class:`Function`, a start time, an end time and a timestep.
    It should return the final solution :class:`Function`.
    """

    @wraps(solver)
    def wrapper(ic, t_start, t_end, dt, **kwargs):
        init = ic.copy(deepcopy=True)
        control = pyadjoint.Control(init)
        out = solver(init, t_start, t_end, dt, **kwargs)
        msg = "Solver should return the Function corresponding to the final solution"
        assert isinstance(out, firedrake.Function), msg
        return out, control

    return wrapper


def _get_exports_per_subinterval(subintervals, timesteps, dt_per_export):
    """
    :arg subintervals: list of tuples corresponding to subintervals
    :arg timesteps: list of floats or single float corresponding to timesteps
    :arg dt_per_export: list of ints or single in corresponding to timesteps per export
    """
    if isinstance(timesteps, float):
        timesteps = [timesteps for subinterval in subintervals]
    assert len(timesteps) == len(subintervals)
    _dt_per_mesh = [(t[1] - t[0])/dt for t, dt in zip(subintervals, timesteps)]
    dt_per_mesh = [int(dtpm) for dtpm in _dt_per_mesh]
    assert np.allclose(dt_per_mesh, _dt_per_mesh)
    if isinstance(dt_per_export, int):
        dt_per_export = [dt_per_export for subinterval in subintervals]
    for dtpe, dtpm in zip(dt_per_export, dt_per_mesh):
        assert dtpm % dtpe == 0
    exports_per_subinterval = [dtpm//dtpe + 1 for dtpe, dtpm in zip(dt_per_export, dt_per_mesh)]
    return timesteps, dt_per_export, exports_per_subinterval


def solve_adjoint(solver, initial_condition, qoi, function_spaces, end_time, timesteps, **kwargs):
    """
    Solve an adjoint problem on a sequence of subintervals.

    :arg solver: a function which takes an initial condition :class:`Function`, a start time and
        an end time as arguments and returns the solution value at the final time
    :arg initial_condition: a function which maps a function space to a :class:`Function`
    :arg qoi: a function which maps a terminal value :class:`Function` to a UFL 1-form
    :arg end_time: the simulation end time
    :arg timesteps: a list of floats or single float corresponding to the timestep on each
        subinterval
    :kwarg timesteps_per_export: a list of ints or single int correspondioing to the number of
        timesteps per export on each subinterval
    :kwarg solver_kwargs: a list of dictionaries or single dictionary providing parameters to
        the solver
    :return: a list of lists containing the adjoint solution at all exported timesteps, indexed
        by subinterval and then export
    """
    # TODO: Account for QoIs involving solution during simulation
    num_subintervals = len(function_spaces)
    subinterval_time = end_time/num_subintervals  # NOTE: Assumes uniform
    subintervals = [(i*subinterval_time, (i+1)*subinterval_time) for i in range(num_subintervals)]
    dt_per_export = kwargs.get('timesteps_per_export', 1)
    timesteps, dt_per_export, export_per_mesh = \
        _get_exports_per_subinterval(subintervals, timesteps, dt_per_export)

    # Clear tape
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()

    # Solve forward to get checkpoints
    solver_kwargs = kwargs.get('solver_kwargs', {})
    if isinstance(solver_kwargs, dict):
        solver_kwargs = [solver_kwargs for subinterval in subintervals]
    with pyadjoint.stop_annotating():
        checkpoints = [initial_condition(function_spaces[0])]
        for i, subinterval in enumerate(subintervals[:-1]):
            sol = solver(checkpoints[i], *subinterval, timesteps[i], **solver_kwargs[i])
            checkpoints.append(firedrake.project(sol, function_spaces[i+1]))

    # Create an array to hold exported adjoint solutions
    adj_sols = [[
        firedrake.Function(fs)
        for j in range(export_per_mesh[i])]
        for i, fs in enumerate(function_spaces)
    ]

    # Loop over subintervals in reverse
    adj_sol = None
    seed = None
    for i, irev in enumerate(reversed(range(num_subintervals))):
        subinterval = subintervals[irev]
        wrapped_solver = _get_initial_condition_control(solver)

        # Annotate tape on current subinterval
        sol, control = wrapped_solver(
            checkpoints[irev],
            *subinterval,
            timesteps[irev],
            **solver_kwargs[i],
        )

        # Store terminal condition and get seed vector for reverse mode propagation
        if i == 0:
            out = firedrake.assemble(qoi(sol))
            seed = 1.0
            tc = firedrake.assemble(firedrake.derivative(qoi(sol), sol))
        else:
            out = sol
            seed = firedrake.project(seed, function_spaces[irev], annotate=False)   # TODO: Adjoint
            tc = firedrake.project(adj_sol, function_spaces[irev], annotate=False)  # TODO: Adjoint
        adj_sols[i][0].assign(tc)

        # Solve adjoint problem
        pyadjoint.compute_gradient(out, control, adj_value=seed, tape=tape)

        # Store adjoint solutions
        solve_blocks = [
            block
            for block in tape.get_blocks()
            if issubclass(block.__class__, GenericSolveBlock)
            and block.adj_sol is not None
        ]
        for j, jj in enumerate(reversed(range(0, len(solve_blocks), dt_per_export[i]))):
            adj_sol = solve_blocks[jj].adj_sol
            adj_sols[i][j+1].assign(adj_sol)

        # Get adjoint action
        seed = firedrake.Function(function_spaces[i], val=control.block_variable.adj_value)
        tape.clear_tape()

    return adj_sols
