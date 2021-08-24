from builtins import zip
from builtins import map
from builtins import range
# nd Slicer slices data on N columns in simData

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
from functools import wraps

from rubin_sim.maf.plots.ndPlotters import TwoDSubsetData, OneDSubsetData
from .baseSlicer import BaseSlicer

__all__ = ['NDSlicer']

class NDSlicer(BaseSlicer):
    """Nd slicer (N dimensions)"""
    def __init__(self, sliceColList=None, verbose=True, binsList=100):
        """Instantiate object.
        binsList can be a list of numpy arrays with the respective slicepoints for sliceColList,
         or a list of integers (one per column in sliceColList) or a single value
            (repeated for all columns, default=100)."""
        super(NDSlicer, self).__init__(verbose=verbose)
        self.bins = None
        self.nslice = None
        self.sliceColList = sliceColList
        self.columnsNeeded = self.sliceColList
        if self.sliceColList is not None:
            self.nD = len(self.sliceColList)
        else:
            self.nD = None
        self.binsList = binsList
        if not (isinstance(binsList, float) or isinstance(binsList, int)):
            if len(self.binsList) != self.nD:
                raise Exception('BinsList must be same length as sliceColNames, unless it is a single value')
        self.slicer_init={'sliceColList':sliceColList}
        self.plotFuncs = [TwoDSubsetData, OneDSubsetData]

    def setupSlicer(self, simData, maps=None):
        """Set up bins. """
        # Parse input bins choices.
        self.bins = []
        # If we were given a single number for the binsList, convert to list.
        if isinstance(self.binsList, float) or isinstance(self.binsList, int):
            self.binsList = [self.binsList for c in self.sliceColList]
        # And then build bins.
        for bl, col in zip(self.binsList, self.sliceColList):
            if isinstance(bl, float) or isinstance(bl, int):
                sliceCol = simData[col]
                binMin = sliceCol.min()
                binMax = sliceCol.max()
                if binMin == binMax:
                    warnings.warn('BinMin=BinMax for column %s: increasing binMax by 1.' %(col))
                    binMax = binMax + 1
                binsize = (binMax - binMin) / float(bl)
                bins = np.arange(binMin, binMax + binsize/2.0, binsize, 'float')
                self.bins.append(bins)
            else:
                self.bins.append(np.sort(bl))
        # Count how many bins we have total (not counting last 'RHS' bin values, as in oneDSlicer).
        self.nslice = (np.array(list(map(len, self.bins)))-1).prod()
        # Set up slice metadata.
        self.slicePoints['sid'] = np.arange(self.nslice)
        # Including multi-D 'leftmost' bin values
        binsForIteration = []
        for b in self.bins:
            binsForIteration.append(b[:-1])
        biniterator = itertools.product(*binsForIteration)
        self.slicePoints['bins'] = []
        for b in biniterator:
            self.slicePoints['bins'].append(b)
        # and multi-D 'leftmost' bin indexes corresponding to each sid
        self.slicePoints['binIdxs'] = []
        binIdsForIteration = []
        for b in self.bins:
            binIdsForIteration.append(np.arange(len(b[:-1])))
        binIdIterator = itertools.product(*binIdsForIteration)
        for bidx in binIdIterator:
            self.slicePoints['binIdxs'].append(bidx)
        # Add metadata from maps.
        self._runMaps(maps)
        # Set up indexing for data slicing.
        self.simIdxs = []
        self.lefts = []
        for sliceColName, bins in zip(self.sliceColList, self.bins):
            simIdxs = np.argsort(simData[sliceColName])
            simFieldsSorted = np.sort(simData[sliceColName])
            # "left" values are location where simdata == bin value
            left = np.searchsorted(simFieldsSorted, bins[:-1], 'left')
            left = np.concatenate((left, np.array([len(simIdxs),])))
            # Add these calculated values into the class lists of simIdxs and lefts.
            self.simIdxs.append(simIdxs)
            self.lefts.append(left)
        @wraps (self._sliceSimData)
        def _sliceSimData(islice):
            """Slice simData to return relevant indexes for slicepoint."""
            # Identify relevant pointings in each dimension.
            simIdxsList = []
            # Translate islice into indexes in each bin dimension
            binIdxs = self.slicePoints['binIdxs'][islice]
            for d, i in zip(list(range(self.nD)), binIdxs):
                simIdxsList.append(set(self.simIdxs[d][self.lefts[d][i]:self.lefts[d][i+1]]))
            idxs = list(set.intersection(*simIdxsList))
            return {'idxs':idxs,
                    'slicePoint':{'sid':islice,
                                  'binLeft':self.slicePoints['bins'][islice],
                                  'binIdx':self.slicePoints['binIdxs'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if grids are equivalent."""
        if isinstance(otherSlicer, NDSlicer):
            if otherSlicer.nD != self.nD:
                return False
            for i in range(self.nD):
                if not np.array_equal(otherSlicer.slicePoints['bins'][i], self.slicePoints['bins'][i]):
                    return False
            return True
        else:
            return False
