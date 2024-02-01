__all__ = ("SFUncertMetric",)

import warnings

import numpy as np
from astropy.stats import mad_std

from rubin_sim.maf.utils import m52snr
from rubin_sim.phot_utils import DustValues

from .base_metric import BaseMetric


class SFUncertMetric(BaseMetric):
    """Structure Function (SF) Uncertainty Metric.
    Developed on top of LogTGaps

    Adapted from Weixiang Yu & Gordon Richards at:
    https://github.com/RichardsGroup/
    LSST_SF_Metric/blob/main/notebooks/00_SFErrorMetric.ipynb

    Parameters
    ----------
    mag : `float`
        The magnitude of the fiducial object. Default 22.
    times_col : `str`
        Time column name. Defaults to "observationStartMJD".
    all_gaps : `bool`
         Whether to use all gaps (between any two pairs of observations).
         If False, only use consecutive paris. Defaults to True.
    units : `str`
        Unit of this metric. Defaults to "mag".
    bins : `object`
        An array of bin edges.
        Defaults to "np.logspace(0, np.log10(3650), 16)" for a
        total of 15 (final) bins.
    weight : `object`
        The weight assigned to each delta_t bin for deriving the final metric.
        Defaults to flat weighting with sum of 1.
        Should have length 1 less than bins.
    snr_cut : `float`
        Ignore observations below an SNR limit, default 5.
    dust : `bool`
        Apply dust extinction to the fiducial object magnitude. Default True.
    """

    def __init__(
        self,
        mag=22,
        times_col="observationStartMJD",
        m5_col="fiveSigmaDepth",
        all_gaps=True,
        units="mag",
        bins=np.logspace(0, np.log10(3650), 16),
        weight=None,
        metric_name="Structure Function Uncert",
        snr_cut=5,
        filter_col="filter",
        dust=True,
        **kwargs,
    ):
        # Assign metric parameters to instance object
        self.times_col = times_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.all_gaps = all_gaps
        self.bins = bins
        if weight is None:
            # If weight is none, set weight so that sum over bins = 1
            self.weight = np.ones(len(self.bins) - 1)
            self.weight /= self.weight.sum()
        self.metric_name = metric_name
        self.mag = mag
        self.snr_cut = snr_cut
        self.dust = dust

        maps = ["DustMap"]
        super(SFUncertMetric, self).__init__(
            col=[self.times_col, m5_col, filter_col],
            metric_name=self.metric_name,
            units=units,
            maps=maps,
            **kwargs,
        )
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

    def run(self, data_slice, slice_point=None):
        """Code executed at each healpix pixel to compute the metric"""

        df = np.unique(data_slice[self.filter_col])
        if np.size(df) > 1:
            msg = """Running structure function on multiple filters simultaneously.
                     Should probably change your SQL query to limit to a single filter."""
            warnings.warn(msg)
        if self.dust:
            a_x = self.ax1[data_slice[self.filter_col][0]] * slice_point["ebv"]
            extincted_mag = self.mag + a_x
        else:
            extincted_mag = self.mag
        snr = m52snr(extincted_mag, data_slice[self.m5_col])
        bright_enough = np.where(snr > self.snr_cut)[0]

        # If the total number of visits < 2, mask as bad pixel
        if data_slice[bright_enough].size < 2:
            return self.badval

        # sort data by time column
        order = np.argsort(data_slice[self.times_col][bright_enough])
        times = data_slice[self.times_col][bright_enough][order]
        # Using the simple Gaussian approximation for magnitude uncertainty.
        mag_err = 2.5 * np.log10(1.0 + 1.0 / snr[bright_enough][order])

        # check if use all gaps (between any pairs of observations)
        if self.all_gaps:
            # use the vectorized method
            dt_matrix = times.reshape((1, times.size)) - times.reshape((times.size, 1))
            dts = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)
        else:
            dts = np.diff(times)

        # bin delta_t using provided bins;
        # if zero pair found at any delta_t bin,
        # replace 0 with 0.01 to avoid the exploding 1/sqrt(n) term
        # in this metric
        result, bins = np.histogram(dts, self.bins)
        new_result = np.where(result > 0, result, 0.01)

        # compute photometric_error^2 population variance and population mean
        # note that variance is replaced by median_absolute_deviate^2
        # mean is replaced by median in this implementation to make it robust
        # to outliers in simulations (e.g., dcr simulations)
        err_var = mag_err**2
        err_var_mu = np.median(err_var)
        err_var_std = mad_std(err_var)

        # compute SF error
        sf_var_dt = 2 * (err_var_mu + err_var_std / np.sqrt(new_result))
        sf_var_metric = np.sum(sf_var_dt * self.weight)

        return sf_var_metric
