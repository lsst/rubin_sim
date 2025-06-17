__all__ = ("KuiperMetric",)

import numpy as np

from .metrics import BaseMetric


class KuiperMetric(BaseMetric):
    """Find the Kuiper V statistic for a distribution, useful for angles.

    Value of 0 means perfecty uniform, 1 means delta function
    """
    def __init__(self, col="rotTelPos", unit="Kuiper V (0=uniform, 1=delta)", name="name"):
        super().__init__(col=col, unit=unit, name=name)

    def __call__(self, data_slice, slice_point=None):
        """"""
        # Assume input in degrees
        values = np.sort(data_slice[self.col] % 360)

        dist_1 = (np.arange(values.size) + 1) / values.size
        uniform = values / (360.0)

        d_plus = np.max(uniform - dist_1)
        d_minus = np.max(dist_1 - uniform)
        result = d_plus + d_minus

        return result
