from .mesh_seq import MeshSeq
from .time_partition import TimePartition
from .utility import AttrDict, File, Function, FunctionSpace, tricontourf
from collections.abc import Iterable
import os
import matplotlib
import matplotlib.pyplot as plt


__all__ = ["SolutionData", "FieldDataDict", "FieldDataSeq", "FieldDataList"]


class FieldDataList:
    """
    Class to hold solution data for a particular field, mesh sequence index, and export.
    """

    def __init__(
        self,
        field: str,
        label: str,
        index: int,
        tp_info: AttrDict,
        function_space: FunctionSpace,
    ):
        """
        :arg field: the field under consideration
        :arg label: the label under consideration
        :arg index: the mesh sequence index to extract data for
        :arg tp_info: the time parition information relevant to this index
        :arg function_space: the corresponding function space
        """
        self.field = field
        self.label = label
        self.index = index
        self.tp_info = tp_info
        self._functions = []
        for i in range(tp_info.num_exports):
            self._functions.append(Function(function_space, name=f"{field}_{label}"))

    def __getitem__(self, index: int) -> Function:
        return self._functions[index]

    def __len__(self) -> int:
        return len(self._functions)

    def write_vtu(self, outfile: File):
        r"""
        Write all :class:`Function`\s to VTU.

        :arg outfile: ParaView file object to write to
        """
        for f in self:
            outfile.write(*f.subfunctions)

    def plot(self, axes: matplotlib.axes.Axes, **kwargs):
        """
        Plot all field data using matplotlib.

        :arg axes: the matplotlib Axes object
        """
        if not isinstance(axes, Iterable):
            axes = [axes]
        plots = []
        for j, sol in enumerate(self):
            plots.append(tricontourf(sol, axes=axes[j], **kwargs))
            # TODO: Account for vector and mixed cases
        return plots


class FieldDataSeq:
    """
    Class to hold solution data for a particular field and mesh sequence index.
    """

    def __init__(
        self,
        field: str,
        label: str,
        time_partition: TimePartition,
        function_spaces: list,
    ):
        """
        :arg field: the field under consideration
        :arg label: the label to extract data for
        :arg time_partition: the :class:`TimePartition` associated with the field
        :arg function_spaces: the corresponding list of function spaces
        """
        self.field = field
        self.label = label
        self._field_data = []
        for index, tp_info in enumerate(time_partition):
            self._field_data.append(
                FieldDataList(
                    field,
                    label,
                    index,
                    tp_info,
                    function_spaces[index],
                )
            )

    def __getitem__(self, index: int) -> FieldDataList:
        return self._field_data[index]

    def __len__(self) -> int:
        return len(self._field_data)

    def write_vtu(self, outfile: File):
        """
        Write all field data to VTU files.

        :arg outfile: ParaView file object to write to
        """
        for fdl in self:
            fdl.write_vtu(outfile)

    def plot(self, axes: matplotlib.axes.Axes, **kwargs):
        """
        Plot all field data using matplotlib.

        :arg axes: the matplotlib Axes object
        """
        if not isinstance(axes, Iterable):
            axes = [axes]
        if not isinstance(axes[0], Iterable):
            axes = [axes]
        plots = []
        for i, sol in enumerate(self):
            axes[0, i].set_title(f"Mesh[{i}]")
            plots.append(self[i].plot(axes=axes[i, :], **kwargs))
        return plots


class FieldDataDict:
    """
    Class to hold solution data for a particular field.
    """

    def __init__(
        self,
        field: str,
        labels: list,
        time_partition: TimePartition,
        function_spaces: list,
    ):
        """
        :arg field: the field under consideration
        :arg labels: a tuple of labels to extract data for
        :arg time_partition: the :class:`TimePartition` associated with the field
        :arg function_spaces: the corresponding list of function spaces
        """
        self.field = field
        self.labels = labels
        self._field_data = AttrDict()
        for label in labels:
            self._field_data[label] = FieldDataSeq(
                field,
                label,
                time_partition,
                function_spaces,
            )

    def __getitem__(self, key: str) -> FieldDataSeq:
        return self._field_data[key]

    def __len__(self) -> int:
        return len(self._field_data)

    def write_vtu(self, output_dir: str):
        """
        Write all field data to VTU files.

        :arg output_dir: top-level directory to store the files in
        """
        for label in self.labels:
            o = File(
                os.path.join(output_dir, f"{self.field}_{label}.pvd"), adaptive=True
            )
            self[label].write_vtu(o)


class SolutionData:
    """
    Class to hold solution data from :meth:`MeshSeq.solve_forward`, or
    :meth:`AdjointMeshSeq.solve_adjoint`.

    The data can be accessed by field name, label ('forward', 'forward_old', 'adjoint',
    etc.), mesh sequence index, or export index.
    """

    def __init__(self, mesh_seq: MeshSeq, labels: tuple):
        """
        :arg mesh_seq: the :class:`MeshSeq` whose solver method is being called
        :arg labels: a tuple of labels to extract data for
        """
        self.fields = mesh_seq.fields
        self.labels = labels
        self._solution_data = AttrDict()
        for field in self.fields:
            self._solution_data[field] = FieldDataDict(
                field,
                labels,
                mesh_seq.time_partition,
                mesh_seq.function_spaces[field],
            )

    # TODO: implement extraction by label
    # TODO: implement extraction by index
    # TODO: implement extraction by export

    def __getitem__(self, key: str) -> FieldDataDict:
        return self._solution_data[key]

    def __len__(self) -> int:
        return len(self._solution_data)

    def write_vtu(self, output_dir: str):
        """
        Write all solution data to VTU files.

        :arg output_dir: top-level directory to store the files in
        """
        for fdd in self:
            fdd.write_vtu(output_dir)

    def plot(self, field: str, label: str, **kwargs):
        """
        Plot all field data using matplotlib.

        :arg field: solution field to plot
        :arg label: type of solution field, e.g., 'forward' or 'adjoint'
        """
        fig = kwargs.pop("fig", None)
        axes = kwargs.pop("axes", None)
        if fig is None or axes is None:
            fig, axes = plt.subplots()
        return fig, axes, self[field][label].plot(axes, **kwargs)
