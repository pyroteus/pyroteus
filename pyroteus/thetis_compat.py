try:
    from thetis import *
except ImportError:
    raise ImportError("Thetis is not installed!")


class FlowSolver2d(thetis.solver2d.FlowSolver2d):
    """
    Augmented version of Thetis' ``FlowSolver2d``
    class which accounts for restarting the
    simulation on a new mesh and modifying
    options prefixes.
    """

    def update_options_prefixes(self):
        if not hasattr(self, 'timestepper'):
            self.create_timesteppers()
        if hasattr(self.timestepper, 'timesteppers'):
            for field, ts in self.timestepper.timesteppers.items():
                self.timestepper.timesteppers[field].name = field
                self.timestepper.timesteppers[field].update_solver()
        else:
            self.timestepper.name = 'swe2d'
            self.timestepper.update_solver()

    def iterate(self, **kwargs):
        self.update_options_prefixes()
        super(FlowSolver2d, self).iterate(**kwargs)

    def correct_counters(self, ts_data):
        i_export = int(ts_data.start_time/ts_data.timestep/ts_data.timesteps_per_export)
        self.simulation_time = ts_data.start_time
        self.i_export = i_export
        self.next_export_t = ts_data.start_time
        self.iteration = int(ts_data.start_time/ts_data.timestep)
        self.export_initial_state = np.isclose(ts_data.start_time, 0.0)
        if not self.options.no_exports and len(self.options.fields_to_export) > 0:
            for e in self.exporters['vtk'].exporters:
                self.exporters['vtk'].exporters[e].set_next_export_ix(i_export)
