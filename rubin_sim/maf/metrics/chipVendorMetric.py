import numpy as np
from .baseMetric import BaseMetric

__all__ = ['ChipVendorMetric']

class ChipVendorMetric(BaseMetric):
    """
    See what happens if we have chips from different vendors
    """

    def __init__(self, cols=None, **kwargs):
        if cols is None:
            cols = []
        super(ChipVendorMetric,self).__init__(col=cols, metricDtype=float,
                                              units='1,2,3:v1,v2,both', **kwargs)

    def _chipNames2vendorID(self, chipName):
        """
        given a list of chipnames, convert to 1 or 2, representing
        different vendors
        """
        vendors=[]
        for chip in chipName:
            # Parse the chipName string.
            if int(chip[2]) % 2 == 0:
                vendors.append(1)
            else:
                vendors.append(2)
        return vendors

    def run(self, dataSlice, slicePoint=None):

        if 'chipNames' not in list(slicePoint.keys()):
            raise ValueError('No chipname info, need to set useCamera=True with a spatial slicer.')

        uvendorIDs = np.unique(self._chipNames2vendorID(slicePoint['chipNames']))
        if np.size(uvendorIDs) == 1:
            result = uvendorIDs
        else:
            result = 3
        return result
