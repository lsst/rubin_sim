__all__ = ("SnrWeightedMetric",)

import numpy as np

from rubin_sim.maf.utils import m52snr

from .base_metric import BaseMetric


class SnrWeightedMetric(BaseMetric):
    """Take the SNR weighted average of a column."""

    def __init__(self, col, m5_col="fiveSigmaDepth", metric_name=None, **kwargs):
        if metric_name is None:
            metric_name = "SNR Weighted %s" % col
        super(SnrWeightedMetric, self).__init__(col=[m5_col, col], metric_name=metric_name, **kwargs)
        self.m5_col = m5_col
        self.col = col
        self.star_mag = 20.0  # Arbitrary reference, value doesn't matter

    def run(self, data_slice, slice_point=None):
        snr = m52snr(self.star_mag, data_slice[self.m5_col])
        result = np.average(data_slice[self.col], weights=snr)
        return result
