import firedrake
from firedrake.adjoint.blocks import GenericSolveBlock, ProjectBlock
import pyadjoint
from pyroteus.interpolation import mesh2mesh_project_adjoint
from pyroteus.ts import get_subintervals, get_exports_per_subinterval
from functools import wraps


__all__ = ["solve_adjoint", "get_subintervals", "get_exports_per_subinterval"]


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
        init = ic.copy(deepcopy=True)
        control = pyadjoint.Control(init)
        out, J = solver(init, t_start, t_end, dt, **kwargs)
        msg = "Solver should return the Function corresponding to the final solution"
        assert isinstance(out, firedrake.Function), msg
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
    def wrapper(*args):
        j = firedrake.assemble(qoi(*args))
        j.adj_value = 1.0
        return j

    return wrapper


def solve_adjoint(solver, initial_condition, qoi, function_spaces, end_time, timesteps, **kwargs):
    """
    Solve an adjoint problem on a sequence of subintervals.

    The current implementation assumes that the quantity of interest is a function of the
    terminal solution alone.

    :arg solver: a function which takes an initial condition :class:`Function`, a start time and
        an end time as arguments and returns the solution value at the final time
    :arg initial_condition: a function which maps a function space to a :class:`Function`
    :arg qoi: a function which maps a :class:`Function` (and possibly a time level) to a UFL 1-form
    :arg end_time: the simulation end time
    :arg timesteps: a list of floats or single float corresponding to the timestep on each
        subinterval
    :kwarg timesteps_per_export: a list of ints or single int correspondioing to the number of
        timesteps per export on each subinterval
    :kwarg solver_kwargs: a dictionary providing parameters to the solver

    :return J: quantity of interest value
    :return adj_sols: a list of lists containing the adjoint solution at all exported timesteps,
        indexed by subinterval and then export
    """
    import inspect
    nargs = len(inspect.getfullargspec(qoi).args)
    assert nargs in (1, 2), f"QoI has more arguments than expected ({nargs})"

    # Handle timestepping
    num_subintervals = len(function_spaces)
    subintervals = get_subintervals(end_time, num_subintervals)
    dt_per_export = kwargs.get('timesteps_per_export', 1)
    solves_per_dt = kwargs.get('solves_per_timestep', 1)
    timesteps, dt_per_export, export_per_mesh = \
        get_exports_per_subinterval(subintervals, timesteps, dt_per_export)
    dt_per_mesh = [dtpe*(epm-1) for dtpe, epm in zip(dt_per_export, export_per_mesh)]

    # Solve forward to get checkpoints and evaluate QoI
    solver_kwargs = kwargs.get('solver_kwargs', {})
    if nargs == 2:
        solver_kwargs['qoi'] = lambda *args: firedrake.assemble(qoi(*args))
    J = 0
    with pyadjoint.stop_annotating():
        checkpoints = [initial_condition(function_spaces[0])]
        for i, subinterval in enumerate(subintervals):
            sol, J = solver(checkpoints[i], *subinterval, timesteps[i], J=J, **solver_kwargs)
            if i < num_subintervals-1:
                checkpoints.append(firedrake.project(sol, function_spaces[i+1]))
    if nargs == 2:
        solver_kwargs.pop('qoi')

    # Create an array to hold exported adjoint solutions
    adj_sols = [[
        firedrake.Function(fs)
        for j in range(export_per_mesh[i])]
        for i, fs in enumerate(function_spaces)
    ]

    # Wrap solver and QoI
    wrapped_solver = wrap_solver(solver)
    wrapped_qoi = wrap_qoi(qoi)
    if nargs == 2:
        solver_kwargs['qoi'] = wrapped_qoi

    # Clear tape
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()

    # Loop over subintervals in reverse
    adj_sol = None
    seed = None
    for i, irev in enumerate(reversed(range(num_subintervals))):
        subinterval = subintervals[irev]

        # Annotate tape on current subinterval
        sol, control = wrapped_solver(
            checkpoints[irev],
            *subinterval,
            timesteps[irev],
            **solver_kwargs,
        )

        # Store terminal condition and get seed vector for reverse mode propagation
        if i == 0:
            if nargs == 1:
                J = wrapped_qoi(sol)
            tc = firedrake.Function(sol.function_space())  # Zero terminal condition
        else:
            with pyadjoint.stop_annotating():
                sol.adj_value = mesh2mesh_project_adjoint(seed, function_spaces[irev])
                tc = mesh2mesh_project_adjoint(adj_sol, function_spaces[irev])
        assert function_spaces[irev] == tc.function_space(), "FunctionSpaces do not match"
        adj_sols[i][0].assign(tc, annotate=False)

        # Solve adjoint problem
        m = pyadjoint.enlisting.Enlist(control)
        with pyadjoint.stop_annotating():
            with tape.marked_nodes(m):
                tape.evaluate_adj(markings=True)

        # Store adjoint solutions
        solve_blocks = [
            block
            for block in tape.get_blocks()
            if issubclass(block.__class__, GenericSolveBlock)
            and not issubclass(block.__class__, ProjectBlock)
            and block.adj_sol is not None
        ][-dt_per_mesh[i]*solves_per_dt::solves_per_dt]
        for j, jj in enumerate(reversed(range(0, len(solve_blocks), dt_per_export[i]))):
            adj_sol = solve_blocks[jj].adj_sol
            adj_sols[i][j+1].assign(adj_sol)

        # Get adjoint action
        seed = firedrake.Function(function_spaces[i], val=control.block_variable.adj_value)
        tape.clear_tape()

    return J, adj_sols
