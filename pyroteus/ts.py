__all__ = ["get_subintervals"]


def get_subintervals(end_time, num_subintervals):
    """
    Calculate a list of subintervals.

    Note that they are assumed to be of equal length.

    :arg end_time: simulation end time
    :arg num_subintervals: number of subintervals
    """
    step_length = end_time/num_subintervals
    t_start = 0.0
    t_end = step_length
    subintervals = []
    for subinterval in range(num_subintervals):
        subintervals.append((t_start, t_end))
        t_start = t_end
        t_end += step_length
    return subintervals
