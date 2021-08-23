# oneDSlicer - slices based on values in one data column in simData.

import numpy as np
from functools import wraps
import warnings
from rubin_sim.maf.utils import optimalBins
from rubin_sim.maf.stackers import ColInfo
from rubin_sim.maf.plots.onedPlotters import OneDBinnedData

from .baseSlicer import BaseSlicer

__all__ = ['OneDSlicer']

class OneDSlicer(BaseSlicer):
    """OneD Slicer allows the 'slicing' of data into bins in a single dimension.

    Parameters
    ----------
    sliceColName : `str`
        The name of the data column to base slicing on (i.e. 'airmass', etc.)
    sliceColUnits : `str`, optional
        Set a name for the units of the sliceCol. Used for plotting labels. Default None.
    bins : np.ndarray, optional
        The data will be sliced into 'bins': this can be defined as an array here. Default None.
    binMin : `float`, optional
    binMax : `float`, optional
    binsize : `float`, optional
        If bins is not defined, then binMin/binMax/binsize can be chosen to anchor the slice points.
        Default None.
        Priority goes: bins >> binMin/binMax/binsize >> data values (if none of the above are chosen).

    The bins act like numpy histogram bins: the last bin value is the end value of the last bin.
    All bins except for the last bin are half-open ([a, b)) while the last bin is ([a, b]).
    """
    def __init__(self, sliceColName=None, sliceColUnits=None,
                 bins=None, binMin=None, binMax=None, binsize=None,
                 verbose=True, badval=0):
        super().__init__(verbose=verbose, badval=badval)
        if sliceColName is None:
            raise ValueError('sliceColName cannot be left None - choose a data column to group data by')
        self.sliceColName = sliceColName
        self.columnsNeeded = [sliceColName]
        self.bins = bins
        # Forget binmin/max/stepsize if bins was set
        if self.bins is not None:
            if binMin is not None or binMax is not None or binsize is not None:
                warnings.warning(f'Both bins and one of the binMin/binMax/binsize was specified. '
                                 f'Using bins ({self.bins} values only.')
            self.binMin = self.bins.min()
            self.binMax = self.bins.max()
            self.binsize = np.diff(self.bins)
            if len(np.unique(self.binsize)) == 1:
                self.binsize = np.unique(self.binsize)
        else:
            self.binMin = binMin
            self.binMax = binMax
            self.binsize = binsize
        # Set the column units
        if sliceColUnits is not None:
            self.sliceColUnits = sliceColUnits
        # Try to determine the column units
        else:
            co = ColInfo()
            self.sliceColUnits = co.getUnits(self.sliceColName)
        # Set slicer re-initialize values and default plotFunction
        self.slicer_init = {'sliceColName':self.sliceColName, 'sliceColUnits':sliceColUnits,
                            'badval':badval,
                            'binMin': self.binMin, 'binMax': self.binMax, 'binsize': self.binsize}
        self.plotFuncs = [OneDBinnedData,]

    def setupSlicer(self, simData, maps=None):
        """
        Set up bins in slicer.
        This happens AFTER simData is defined, thus typically in the MetricBundleGroup.
        This maps data into the bins; it's not a good idea to reuse a OneDSlicer as a result.
        """
        if 'bins' in self.slicePoints:
            warning_msg = 'Warning: this OneDSlicer was already set up once. '
            warning_msg += 'Re-setting up a OneDSlicer is unpredictable; at the very least, it ' \
                           'will change the mapping of the simulated data into the data slices, ' \
                           'and may result in poor binsize choices (although these may potentially be ok). '
            warning_msg += 'A safer choice is to use a separate OneDSlicer for each MetricBundle.'
            warnings.warn(warning_msg)
        sliceCol = simData[self.sliceColName]
        # Set bins from data or specified values, if they were previously defined.
        if self.bins is None:
            # Set bin min/max values (could have been set in __init__)
            if self.binMin is None:
                self.binMin = np.nanmin(sliceCol)
            if self.binMax is None:
                self.binMax = np.nanmax(sliceCol)
            # Give warning if binMin = binMax, and do something at least slightly reasonable.
            if self.binMin == self.binMax:
                warnings.warn('binMin = binMax (maybe your data is single-valued?). '
                              'Increasing binMax by 1 (or 2*binsize, if binsize was set).')
                if self.binsize is not None:
                    self.binMax = self.binMax + 2 * self.binsize
                else:
                    self.binMax = self.binMax + 1
            if self.binsize is None:
                bins = optimalBins(sliceCol, self.binMin, self.binMax)
                nbins = np.round(bins)
                self.binsize = (self.binMax - self.binMin) / float(nbins)
            # Set bins
            self.bins = np.arange(self.binMin, self.binMax + self.binsize / 2.0, self.binsize, 'float')
        # Set nbins to be one less than # of bins because last binvalue is RH edge only
        self.nslice = len(self.bins) - 1
        self.shape = self.nslice
        # Set slicePoint metadata.
        self.slicePoints['sid'] = np.arange(self.nslice)
        self.slicePoints['bins'] = self.bins
        # Add metadata from map if needed.
        self._runMaps(maps)
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.sliceColName])
        simFieldsSorted = np.sort(simData[self.sliceColName])
        # "left" values are location where simdata == bin value
        self.left = np.searchsorted(simFieldsSorted, self.bins[:-1], 'left')
        self.left = np.concatenate((self.left, np.array([len(self.simIdxs),])))
        # Set up _sliceSimData method for this class.
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Slice simData on oneD sliceCol, to return relevant indexes for slicepoint."""
            idxs = self.simIdxs[self.left[islice]:self.left[islice+1]]
            return {'idxs':idxs,
                    'slicePoint':{'sid':islice, 'binLeft':self.bins[islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        result = False
        if isinstance(otherSlicer, OneDSlicer):
            if self.sliceColName == otherSlicer.sliceColName:
                # If slicer restored from disk or setup, then 'bins' in slicePoints dict.
                # This is preferred method to see if slicers are equal.
                if ('bins' in self.slicePoints) & ('bins' in otherSlicer.slicePoints):
                    result = np.array_equal(otherSlicer.slicePoints['bins'], self.slicePoints['bins'])
                # However, before we 'setup' the slicer with data, the slicers could be equivalent.
                else:
                    if (self.bins is not None) and (otherSlicer.bins is not None):
                        result = np.array_equal(self.bins, otherSlicer.bins)
                    elif ((self.binsize is not None) and
                          (self.binMin is not None) & (self.binMax is not None) and
                          (otherSlicer.binsize is not None) and
                          (otherSlicer.binMin is not None) and
                          (otherSlicer.binMax is not None)):
                          if ((self.binsize == otherSlicer.binsize) and
                              (self.binMin == otherSlicer.binMin) and
                              (self.binMax == otherSlicer.binMax)):
                              result = True
        return result
