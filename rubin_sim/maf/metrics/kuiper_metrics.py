__all__ = ("KuiperMetric",)

import numpy as np

from .base_metric import BaseMetric


class KuiperMetric(BaseMetric):
    """Find the Kuiper V statistic for a distribution, useful for angles.

    Value of 0 means perfecty uniform, 1 means delta function
    """

    def run(self, data_slice, slice_point=None):
        """"""
        # Assume input in degrees
        values = np.sort(data_slice[self.colname] % 360)

        dist_1 = (np.arange(values.size) + 1) / values.size
        uniform = values / (360.0)

        d_plus = np.max(uniform - dist_1)
        d_minus = np.max(dist_1 - uniform)
        result = d_plus + d_minus

        return result
