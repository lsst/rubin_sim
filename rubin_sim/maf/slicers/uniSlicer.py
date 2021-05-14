# UniSlicer class.
# This slicer simply returns the indexes of all data points. No slicing done at all.

import numpy as np
from functools import wraps

from .baseSlicer import BaseSlicer

__all__ = ['UniSlicer']

class UniSlicer(BaseSlicer):
    """UniSlicer."""
    def __init__(self, verbose=True, badval=-666):
        """Instantiate unislicer. """
        super(UniSlicer, self).__init__(verbose=verbose, badval=badval)
        self.nslice = 1
        self.shape = self.nslice
        self.slicePoints['sid'] = np.array([0,], int)
        self.plotFuncs = []

    def setupSlicer(self, simData, maps=None):
        """Use simData to set indexes to return."""
        self._runMaps(maps)
        simDataCol = simData.dtype.names[0]
        self.indices = np.ones(len(simData[simDataCol]),  dtype='bool')
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return all indexes in simData. """
            idxs = self.indices
            return {'idxs':idxs,
                    'slicePoint':{'sid':islice}}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if slicers are equivalent."""
        if isinstance(otherSlicer, UniSlicer):
            return True
        else:
            return False
