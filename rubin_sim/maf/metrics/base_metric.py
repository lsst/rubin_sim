# Base class for metrics - defines methods which must be implemented.

__all__ = ("MetricRegistry", "BaseMetric", "ColRegistry")

import inspect
import warnings

import numpy as np

from rubin_sim.maf.stackers.get_col_info import ColInfo


class MetricRegistry(type):
    """
    Meta class for metrics, to build a registry of metric classes.
    """

    def __init__(cls, name, bases, dict):
        super(MetricRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, "registry"):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__
        if modname.startswith("rubin_sim.maf.metrics"):
            modname = ""
        else:
            if len(modname.split(".")) > 1:
                modname = ".".join(modname.split(".")[:-1]) + "."
            else:
                modname = modname + "."
        metricname = modname + name
        if metricname in cls.registry:
            warnings.warn("Redefining metric %s! (there are >1 metrics with the same name)" % (metricname))
        if metricname not in ["BaseMetric", "SimpleScalarMetric"]:
            cls.registry[metricname] = cls

    def get_class(cls, metricname):
        return cls.registry[metricname]

    def help(cls, doc=False):
        for metricname in sorted(cls.registry):
            if not doc:
                print(metricname)
            if doc:
                print("---- ", metricname, " ----")
                print(inspect.getdoc(cls.registry[metricname]))

    def help_metric(cls, metricname):
        print(metricname)
        print(inspect.getdoc(cls.registry[metricname]))
        k = inspect.signature(cls.registry[metricname])
        print(" Metric __init__ keyword args and defaults: ")
        print(k)


class ColRegistry:
    """
    ColRegistry tracks the columns needed for all metric objects
    (kept internally in a set).

    ColRegistry.col_set : a set of all unique columns required for metrics.
    ColRegistry.dbCols : the subset of these which come from the database.
    ColRegistry.stackerCols : the dictionary of [columns: stacker class].
    """

    col_info = ColInfo()

    def __init__(self):
        self.col_set = set()
        self.db_set = set()
        self.stacker_dict = {}

    def clear_reg(self):
        """Clear the registry"""
        self.__init__()

    def add_cols(self, col_array):
        """Add the columns in ColArray into the ColRegistry.

        Add the columns in col_array into the ColRegistry set (self.col_set)
        and identifies their source, using ColInfo
        (rubin_sim.maf.stackers.getColInfo).

        Parameters
        ----------
        col_array : `list`
            list of columns used in a metric.
        """
        for col in col_array:
            if col is not None:
                self.col_set.add(col)
                source = self.col_info.get_data_source(col)
                if source == self.col_info.default_data_source:
                    self.db_set.add(col)
                else:
                    if col not in self.stacker_dict:
                        self.stacker_dict[col] = source


class BaseMetric(metaclass=MetricRegistry):
    """
    Base class for the metrics.
    Sets up some basic functionality for the MAF framework:
    after __init__ every metric will record the columns (and stackers)
    it requires into the column registry, and the metric_name,
    metric_dtype, and units for the metric will be set.

    Parameters
    ----------
    col : `str` or `list` [`str`]
        Names of the data columns that the metric will use.
        The columns required for each metric is tracked in the ColRegistry,
        and used to retrieve data from the opsim database.
        Can be a single string or a list.
    metric_name : `str`
        Name to use for the metric (optional - if not set, will be derived).
    maps : `list` [`rubin_sim.maf.maps`]
        The maps that the metric will need (passed from the slicer).
    units : `str`
        The units for the value returned by the metric (optional - if not set,
        will be derived from the ColInfo).
    metric_dtype : `str`
        The type of value returned by the metric - 'int', 'float', 'object'.
        If not set, will be derived by introspection.
    badval : `float`
        The value indicating "bad" values calculated by the metric.
    """

    col_registry = ColRegistry()
    col_info = ColInfo()

    def __init__(
        self,
        col=None,
        metric_name=None,
        maps=None,
        units=None,
        metric_dtype=None,
        badval=-666,
        mask_val=None,
    ):
        # Turn cols into numpy array so we know
        # we can iterate over the columns.
        self.col_name_arr = np.array(col, copy=True, ndmin=1)
        # To support simple metrics operating on a single column,
        # set self.colname
        if len(self.col_name_arr) == 1:
            self.colname = self.col_name_arr[0]
        # Add the columns to the colRegistry.
        self.col_registry.add_cols(self.col_name_arr)
        # Set the maps that are needed:
        if maps is None:
            maps = []
        self.maps = maps
        # Value to return if the metric can't be computed
        self.badval = badval
        if mask_val is not None:
            self.mask_val = mask_val
        # Save a unique name for the metric.
        self.name = metric_name
        if self.name is None:
            # If none provided, construct our own from the class name
            # and the data columns.
            self.name = (
                self.__class__.__name__.replace("Metric", "", 1)
                + " "
                + ", ".join(map(str, self.col_name_arr))
            )
        # Set up dictionary of reduce functions (may be empty).
        self.reduce_funcs = {}
        self.reduce_order = {}
        for i, r in enumerate(inspect.getmembers(self, predicate=inspect.ismethod)):
            if r[0].startswith("reduce"):
                # Remove the "reduce_" part of the reduce functions names.
                reducename = r[0].replace("reduce_", "", 1)
                self.reduce_funcs[reducename] = r[1]
                self.reduce_order[reducename] = i
        # Identify type of metric return value.
        if metric_dtype is not None:
            self.metric_dtype = metric_dtype
        elif len(list(self.reduce_funcs.keys())) > 0:
            self.metric_dtype = "object"
        else:
            self.metric_dtype = "float"
        # Set physical units, for plotting purposes.
        if units is None:
            units = " ".join([self.col_info.get_units(col_name) for col_name in self.col_name_arr])
            if len(units.replace(" ", "")) == 0:
                units = ""
        self.units = units
        # Add the ability to set a comment
        # (that could be propagated automatically to a benchmark's
        # display caption).
        self.comment = None

        # Default to only return one metric value per slice
        self.shape = 1

    def run(self, data_slice, slice_point):
        """Calculate metric values.

        Parameters
        ----------
        data_slice : `numpy.ndarray`, (N,)
           Values passed to metric by the slicer, which the metric will
           use to calculate metric values at each slice_point.
        slice_point : `dict` or None
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        metricValue : `int` `float` or `object`
            The metric value at each slice_point.
        """
        raise NotImplementedError("Please implement your metric calculation.")
