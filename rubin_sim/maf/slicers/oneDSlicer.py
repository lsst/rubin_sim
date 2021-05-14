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
    """oneD Slicer."""
    def __init__(self, sliceColName=None, sliceColUnits=None,
                 bins=None, binMin=None, binMax=None, binsize=None,
                 verbose=True, badval=0):
        """
        'sliceColName' is the name of the data column to use for slicing.
        'sliceColUnits' lets the user set the units (for plotting purposes) of the slice column.
        'bins' can be a numpy array with the binpoints for sliceCol or a single integer value
        (if a single value, this will be used as the number of bins, together with data min/max or binMin/Max),
        as in numpy's histogram function.
        If 'binsize' is used, this will override the bins value and will be used together with the data min/max
        or binMin/Max to set the binpoint values.

        Bins work like numpy histogram bins: the last 'bin' value is end value of last bin;
          all bins except for last bin are half-open ([a, b>), the last one is ([a, b]).
        """
        super(OneDSlicer, self).__init__(verbose=verbose, badval=badval)
        self.sliceColName = sliceColName
        self.columnsNeeded = [sliceColName]
        self.bins = bins
        self.binMin = binMin
        self.binMax = binMax
        self.binsize = binsize
        if sliceColUnits is None:
            co = ColInfo()
            self.sliceColUnits = co.getUnits(self.sliceColName)
        else:
            self.sliceColUnits = sliceColUnits
        self.slicer_init = {'sliceColName':self.sliceColName, 'sliceColUnits':sliceColUnits,
                            'badval':badval}
        self.plotFuncs = [OneDBinnedData,]

    def setupSlicer(self, simData, maps=None):
        """
        Set up bins in slicer.
        """
        if self.sliceColName is None:
            raise Exception('sliceColName was not defined when slicer instantiated.')
        sliceCol = simData[self.sliceColName]
        # Set bin min/max values.
        if self.binMin is None:
            self.binMin = np.nanmin(sliceCol)
        if self.binMax is None:
            self.binMax = np.nanmax(sliceCol)
        # Give warning if binMin = binMax, and do something at least slightly reasonable.
        if self.binMin == self.binMax:
            warnings.warn('binMin = binMax (maybe your data is single-valued?). '
                          'Increasing binMax by 1 (or 2*binsize, if binsize set).')
            if self.binsize is not None:
                self.binMax = self.binMax + 2 * self.binsize
            else:
                self.binMax = self.binMax + 1
        # Set bins.
        # Using binsize.
        if self.binsize is not None:
            # Add an extra 'bin' to the edge values of the bins (makes plots much prettier).
            self.binMin -= self.binsize
            self.binMax += self.binsize
            if self.bins is not None:
                warnings.warn('Both binsize and bins have been set; Using binsize %f only.' %(self.binsize))
            self.bins = np.arange(self.binMin, self.binMax+self.binsize/2.0, self.binsize, 'float')
        # Using bins value.
        else:
            # Bins was a sequence (np array or list)
            if hasattr(self.bins, '__iter__'):
                self.bins = np.sort(self.bins)
                self.binMin = self.bins[0]
                self.binMax = self.bins[-1]
            # Or bins was a single value.
            else:
                if self.bins is None:
                    self.bins = optimalBins(sliceCol, self.binMin, self.binMax)
                nbins = np.round(self.bins)
                self.binsize = (self.binMax - self.binMin) / float(nbins)
                self.bins = np.arange(self.binMin, self.binMax+self.binsize/2.0, self.binsize, 'float')
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
                    result = np.all(otherSlicer.slicePoints['bins'] == self.slicePoints['bins'])
                # However, even before we 'setup' the slicer with data, the slicers could be equivalent.
                else:
                    if (self.bins is not None) and (otherSlicer.bins is not None):
                        result = np.all(self.bins == otherSlicer.bins)
                    elif ((self.binsize is not None) and (self.binMin is not None) & (self.binMax is not None) and
                          (otherSlicer.binsize is not None) and (otherSlicer.binMin is not None) and (otherSlicer.binMax is not None)):
                          if ((self.binsize == otherSlicer.binsize) and
                              (self.binMin == otherSlicer.binMin) and
                              (self.binMax == otherSlicer.binMax)):
                              result = True
        return result
