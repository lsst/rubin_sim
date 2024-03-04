# Example for PeriodicMetric
# Mike Lund - Vanderbilt University
# mike.lund@gmail.com
# Last edited 3/10/2015
# Motivation: The detection of periodic signals can be examined by
# using canonical signals and attempted to recover these.
# However, a more general approach would be to examine the strength in
# signal that is lost as a result of poor phase coverage.
# This metric calculates the spectral window function for a set of
# scheduled observations. The largest peak at a nonzero frequency is
# used as a proxy to quantify how much power is
# lost to other frequencies. Full phase coverage will result in a value of 1.
# We refer to this as the Periodic Purity Function.

__all__ = ("PeriodicMetric",)

import numpy as np

from rubin_sim.maf.metrics import BaseMetric


class PeriodicMetric(BaseMetric):
    """From a set of observation times, uses code provided by Robert Siverd
    (LCOGT) to calculate the spectral window function.
    """

    def __init__(self, time_col="expMJD", **kwargs):
        self.time_col = time_col
        super(PeriodicMetric, self).__init__(col=[self.time_col], **kwargs)

    def run(self, data_slice, slice_point=None):
        frq_pts = 30000.0
        max_frq = 25.0
        times = data_slice[self.time_col]
        times = times - times[0]  # change times to smaller numbers
        use_jd = np.array(times)
        window_frq = np.arange(frq_pts) * max_frq / frq_pts
        window_val = np.zeros_like(window_frq, dtype="float")
        for x, frq in enumerate(window_frq):
            window_val[x] = np.sum(np.cos(-2.0 * np.pi * frq * use_jd))
        window_val /= np.float(use_jd.size)
        secondpeak = np.sort(window_val)[-2]
        totalsum = (np.sum(window_val) - np.sort(window_val)[-1]) / (frq_pts - 1)
        data = np.asarray([secondpeak, totalsum])
        return data

    def reduce_peak(self, data):
        return 1.0 - data[0]

    def reduce_sum(self, data):
        return data[1]
