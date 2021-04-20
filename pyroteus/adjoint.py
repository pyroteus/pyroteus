import firedrake
from firedrake_adjoint import Control
from firedrake.adjoint.blocks import GenericSolveBlock, ProjectBlock
import pyadjoint
from pyroteus.interpolation import mesh2mesh_project
from pyroteus.ts import get_subintervals, get_exports_per_subinterval
from pyroteus.utility import norm
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
        control = Control(init)
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

    As well as the quantity of interest value, a dictionary of solution fields is returned,
    the contents of which give values at all exported timesteps, indexed by subinterval and
    then export. For a given export timestep, the solution fields are:
        * 'forward': the forward solution after taking the timestep;
        * 'forward_old': the forward solution before taking the timestep;
        * 'adjoint': the adjoint solution after taking the timestep;
        * 'adjoint_next': the adjoint solution before taking the timestep (backwards).

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
    :kwarg adjoint_projection: if `False`, conservative projection is applied when transferring
        data between meshes in the adjoint solve, rather than the corresponding adjoint operator

    :return J: quantity of interest value
    :return solution: a dictionary containing solution fields and their lagged versions
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

    # Create arrays to hold exported foward and adjoint solutions
    solutions = {
        label: [[
            firedrake.Function(fs)
            for j in range(export_per_mesh[i])]
            for i, fs in enumerate(function_spaces)
        ]
        for label in ('forward', 'forward_old', 'adjoint', 'adjoint_next')
    }
    adj_sols = solutions['adjoint']

    # Wrap solver and QoI
    wrapped_solver = wrap_solver(solver)
    wrapped_qoi = wrap_qoi(qoi)
    if nargs == 2:
        solver_kwargs['qoi'] = wrapped_qoi

    # Clear tape
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()

    # Loop over subintervals in reverse
    seed = None
    adj_proj = kwargs.get('adjoint_projection', True)
    for i in reversed(range(num_subintervals)):
        subinterval = subintervals[i]

        # Annotate tape on current subinterval
        sol, control = wrapped_solver(
            checkpoints[i],
            *subinterval,
            timesteps[i],
            **solver_kwargs,
        )

        # Store terminal condition and get seed vector for reverse mode propagation
        if i == num_subintervals-1:
            if nargs == 1:
                J = wrapped_qoi(sol)
            tc = firedrake.Function(sol.function_space())  # Zero terminal condition
        else:
            with pyadjoint.stop_annotating():
                sol.adj_value = mesh2mesh_project(seed, function_spaces[i], adjoint=adj_proj)
                tc = mesh2mesh_project(adj_sols[i+1][0], function_spaces[i], adjoint=adj_proj)
        assert function_spaces[i] == tc.function_space(), "FunctionSpaces do not match"
        adj_sols[i][-1].assign(tc, annotate=False)  # TODO: What about forward, forward_old, adjoint_next?

        # Solve adjoint problem
        m = pyadjoint.enlisting.Enlist(control)
        with pyadjoint.stop_annotating():
            with tape.marked_nodes(m):
                tape.evaluate_adj(markings=True)

        # Get solve blocks
        solve_blocks = [
            block
            for block in tape.get_blocks()
            if issubclass(block.__class__, GenericSolveBlock)
            and not issubclass(block.__class__, ProjectBlock)
            and block.adj_sol is not None
        ][-dt_per_mesh[i]*solves_per_dt::solves_per_dt]
        N = len(solve_blocks)

        # Get old solution dependency index
        for dep_index, dep in enumerate(solve_blocks[0].get_dependencies()):
            if hasattr(dep.output, 'function_space'):
                if dep.output.function_space() == solve_blocks[0].function_space:
                    break
        # FIXME: What if other dependencies are in the prognostic space?

        # Extract solution data
        for j, block in enumerate(solve_blocks[::dt_per_export[i]]):
            solutions['forward_old'][i][j].assign(block.get_dependencies()[dep_index].saved_output)
            solutions['forward'][i][j].assign(block.get_outputs()[0].saved_output)
            solutions['adjoint'][i][j].assign(block.adj_sol)
            solutions['adjoint_next'][i][j].assign(tc if j >= N else solve_blocks[j+1].adj_sol)
        assert norm(adj_sols[i][0]) > 0.0, f"Adjoint solution on subinterval {i} is zero"

        # Get adjoint action
        seed = firedrake.Function(function_spaces[i], val=control.block_variable.adj_value)
        assert norm(seed) > 0.0, f"Adjoint action on subinterval {i} is zero"
        tape.clear_tape()

    return J, solutions
