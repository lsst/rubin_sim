# Base class for metrics - defines methods which must be implemented.
# If a metric calculates a vector or list at each gridpoint, then there
#  should be additional 'reduce_*' functions defined, to convert the vector
#  into scalar (and thus plottable) values at each gridpoint.
# The philosophy behind keeping the vector instead of the scalar at each gridpoint
#  is that these vectors may be expensive to compute; by keeping/writing the full
#  vector we permit multiple 'reduce' functions to be executed on the same data.

import numpy as np
import inspect
from rubin_sim.maf.stackers.getColInfo import ColInfo
from six import with_metaclass
import warnings

__all__ = ['MetricRegistry', 'BaseMetric']


class MetricRegistry(type):
    """
    Meta class for metrics, to build a registry of metric classes.
    """
    def __init__(cls, name, bases, dict):
        super(MetricRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        modname = inspect.getmodule(cls).__name__
        if modname.startswith('rubin_sim.maf.metrics'):
            modname = ''
        else:
            if len(modname.split('.')) > 1:
                modname = '.'.join(modname.split('.')[:-1]) + '.'
            else:
                modname = modname + '.'
        metricname = modname + name
        if metricname in cls.registry:
            warnings.warn('Redefining metric %s! (there are >1 metrics with the same name)' % (metricname))
        if metricname not in ['BaseMetric', 'SimpleScalarMetric']:
            cls.registry[metricname] = cls

    def getClass(cls, metricname):
        return cls.registry[metricname]

    def help(cls, doc=False):
        for metricname in sorted(cls.registry):
            if not doc:
                print(metricname)
            if doc:
                print('---- ', metricname, ' ----')
                print(inspect.getdoc(cls.registry[metricname]))

    def help_metric(cls, metricname):
        print(metricname)
        print(inspect.getdoc(cls.registry[metricname]))
        k = inspect.signature(cls.registry[metricname])
        print(' Metric __init__ keyword args and defaults: ')
        print(k)


class ColRegistry(object):
    """
    ColRegistry tracks the columns needed for all metric objects (kept internally in a set).

    ColRegistry.colSet : a set of all unique columns required for metrics.
    ColRegistry.dbCols : the subset of these which come from the database.
    ColRegistry.stackerCols : the dictionary of [columns: stacker class].
    """
    colInfo = ColInfo()

    def __init__(self):
        self.colSet = set()
        self.dbSet = set()
        self.stackerDict = {}

    def addCols(self, colArray):
        """Add the columns in ColArray into the ColRegistry.

        Add the columns in colArray into the ColRegistry set (self.colSet) and identifies their source,
         using ColInfo (rubin_sim.maf.stackers.getColInfo).

        Parameters
        ----------
        colArray : list
            list of columns used in a metric.
        """
        for col in colArray:
            if col is not None:
                self.colSet.add(col)
                source = self.colInfo.getDataSource(col)
                if source == self.colInfo.defaultDataSource:
                    self.dbSet.add(col)
                else:
                    if col not in self.stackerDict:
                        self.stackerDict[col] = source


class BaseMetric(with_metaclass(MetricRegistry, object)):
    """
    Base class for the metrics.
    Sets up some basic functionality for the MAF framework: after __init__ every metric will
    record the columns (and stackers) it requires into the column registry, and the metricName,
    metricDtype, and units for the metric will be set.

    Parameters
    ----------
    col : `str` or `list` [`str`]
        Names of the data columns that the metric will use.
        The columns required for each metric is tracked in the ColRegistry, and used to retrieve data
        from the opsim database. Can be a single string or a list.
    metricName : `str`
        Name to use for the metric (optional - if not set, will be derived).
    maps : `list` [`rubin_sim.maf.maps`]
        The maps that the metric will need (passed from the slicer).
    units : `str`
        The units for the value returned by the metric (optional - if not set,
        will be derived from the ColInfo).
    metricDtype : `str`
        The type of value returned by the metric - 'int', 'float', 'object'.
        If not set, will be derived by introspection.
    badval : `float`
        The value indicating "bad" values calculated by the metric.
    """
    colRegistry = ColRegistry()
    colInfo = ColInfo()

    def __init__(self, col=None, metricName=None, maps=None, units=None,
                 metricDtype=None, badval=-666, maskVal=None):
        # Turn cols into numpy array so we know we can iterate over the columns.
        self.colNameArr = np.array(col, copy=False, ndmin=1)
        # To support simple metrics operating on a single column, set self.colname
        if len(self.colNameArr) == 1:
            self.colname = self.colNameArr[0]
        # Add the columns to the colRegistry.
        self.colRegistry.addCols(self.colNameArr)
        # Set the maps that are needed:
        if maps is None:
            maps = []
        self.maps = maps
        # Value to return if the metric can't be computed
        self.badval = badval
        if maskVal is not None:
            self.maskVal = maskVal
        # Save a unique name for the metric.
        self.name = metricName
        if self.name is None:
            # If none provided, construct our own from the class name and the data columns.
            self.name = (self.__class__.__name__.replace('Metric', '', 1) + ' ' +
                         ', '.join(map(str, self.colNameArr)))
        # Set up dictionary of reduce functions (may be empty).
        self.reduceFuncs = {}
        self.reduceOrder = {}
        for i, r in enumerate(inspect.getmembers(self, predicate=inspect.ismethod)):
            if r[0].startswith('reduce'):
                reducename = r[0].replace('reduce', '', 1)
                self.reduceFuncs[reducename] = r[1]
                self.reduceOrder[reducename] = i
        # Identify type of metric return value.
        if metricDtype is not None:
            self.metricDtype = metricDtype
        elif len(list(self.reduceFuncs.keys())) > 0:
            self.metricDtype = 'object'
        else:
            self.metricDtype = 'float'
        # Set physical units, for plotting purposes.
        if units is None:
            units = ' '.join([self.colInfo.getUnits(colName) for colName in self.colNameArr])
            if len(units.replace(' ', '')) == 0:
                units = ''
        self.units = units
        # Add the ability to set a comment
        # (that could be propagated automatically to a benchmark's display caption).
        self.comment = None

        # Default to only return one metric value per slice
        self.shape = 1

    def run(self, dataSlice, slicePoint=None):
        """Calculate metric values.

        Parameters
        ----------
        dataSlice : `numpy.ndarray`
           Values passed to metric by the slicer, which the metric will use to calculate
           metric values at each slicePoint.
        slicePoint : `dict`
           Dictionary of slicePoint metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel or opsim fieldId.

        Returns
        -------
        metricValue: `int` `float` or `object`
            The metric value at each slicePoint.
        """
        raise NotImplementedError('Please implement your metric calculation.')
