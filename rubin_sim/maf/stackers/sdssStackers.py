from builtins import zip
import numpy as np
from .baseStacker import BaseStacker
from .ditherStackers import wrapRA

__all__ = ['SdssRADecStacker']

class SdssRADecStacker(BaseStacker):
    """convert the p1,p2,p3... columns to radians and wrap them """
    colsAdded = ['RA1', 'Dec1', 'RA2', 'Dec2', 'RA3', 'Dec3', 'RA4', 'Dec4']

    def __init__(self, pcols = ['p1','p2','p3','p4','p5','p6','p7','p8']):
        """ The p1,p2 columns represent the corners of chips.  Could generalize this a bit."""
        self.units = ['rad']*8
        self.colsReq = pcols

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Assume this is unusual enough to run that you really mean it.
            pass
        for pcol, newcol in zip(self.colsReq, self.colsAdded):
            if newcol[0:2] == 'RA':
                simData[newcol] = wrapRA(np.radians(simData[pcol]))
            else:
                simData[newcol] = np.radians(simData[pcol])
        return simData
