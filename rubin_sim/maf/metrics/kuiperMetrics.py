import numpy as np
from .baseMetric import BaseMetric

__all__ = ['KuiperMetric']


class KuiperMetric(BaseMetric):
    """Find the Kuiper V statistic for a distribution, useful for angles.

    Value of 0 means perfecty uniform, 1 means delta function
    """
    def run(self, dataSlice, slicePoint=None):
        """
        """
        # Assume input in degrees
        values = np.sort(dataSlice[self.colname])

        dist_1 = (np.arange(values.size)+1)/values.size
        uniform = values/(360.)

        d_plus = np.max(uniform - dist_1)
        d_minus = np.max(dist_1-uniform)
        result = d_plus + d_minus

        return result
