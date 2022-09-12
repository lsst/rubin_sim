import numpy as np
from astropy.stats import mad_std
from .base_metric import BaseMetric
from rubin_sim.maf.utils import m52snr
import warnings
from rubin_sim.phot_utils import Dust_values


__all__ = ["SFUncertMetric"]


class SFUncertMetric(BaseMetric):
    """Structure Function (SF) Uncertainty Metric. Developed on top of LogTGaps

    Adapted from Weixiang Yu & Gordon Richards at:
    https://github.com/RichardsGroup/LSST_SF_Metric/blob/main/notebooks/00_SFErrorMetric.ipynb

    Parameters
    ----------
    mag: `float` (22)
        The magnitude of the fiducial object. Default 22.
    timesCol: `str`  ('observationStartMJD')
        Time column name. Defaults to "observationStartMJD".
    allGaps: `bool` (True)
         Whether to use all gaps (between any two pairs of observations).
         If False, only use consecutive paris. Defaults to True.
    units: `str` ('mag')
        Unit of this metric. Defaults to "mag".
    bins: `object`
        An array of bin edges. Defaults to "np.logspace(0, np.log10(3650), 11)" for a
        total of 10 bins.
    weight: `object
        The weight assigned to each delta_t bin for deriving the final metric.
        Defaluts to "weight=np.full(10, 0.1)".
    snr_cut : float (5)
        Ignore observations below an SNR limit, default 5.
    dust : `bool` (True)
        Apply dust extinction to the fiducial object magnitude. Default True.
    """

    def __init__(
        self,
        mag=22,
        timesCol="observationStartMJD",
        m5Col="fiveSigmaDepth",
        allGaps=True,
        units="mag",
        bins=np.logspace(0, np.log10(3650), 11),
        weight=np.full(10, 0.1),
        metricName="Structure Function Uncert",
        snr_cut=5,
        filterCol="filter",
        dust=True,
        **kwargs
    ):
        # Assign metric parameters to instance object
        self.timesCol = timesCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.allGaps = allGaps
        self.bins = bins
        self.weight = weight
        self.metricName = metricName
        self.mag = mag
        self.snr_cut = snr_cut
        self.dust = dust

        maps = ["DustMap"]
        super(SFUncertMetric, self).__init__(
            col=[self.timesCol, m5Col, filterCol],
            metricName=self.metricName,
            units=units,
            maps=maps,
            **kwargs
        )
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

    def run(self, dataSlice, slicePoint=None):
        """Code executed at each healpix pixel to compute the metric"""

        df = np.unique(dataSlice[self.filterCol])
        if np.size(df) > 1:
            msg = """Running structure function on multiple filters simultaneously. 
                     Should probably change your SQL query to limit to a single filter."""
            warnings.warn(msg)
        if self.dust:
            A_x = self.Ax1[dataSlice[self.filterCol][0]] * slicePoint["ebv"]
            extincted_mag = self.mag + A_x
        else:
            extincted_mag = self.mag
        snr = m52snr(extincted_mag, dataSlice[self.m5Col])
        bright_enough = np.where(snr > self.snr_cut)[0]

        # If the total number of visits < 2, mask as bad pixel
        if dataSlice[bright_enough].size < 2:
            return self.badval

        # sort data by time column
        order = np.argsort(dataSlice[self.timesCol][bright_enough])
        times = dataSlice[self.timesCol][bright_enough][order]
        # Using the simple Gaussian approximation for magnitude uncertainty.
        mag_err = 2.5 * np.log10(1.0 + 1.0 / snr[bright_enough][order])

        # check if use all gaps (between any pairs of observations)
        if self.allGaps:
            # use the vectorized method
            dt_matrix = times.reshape((1, times.size)) - times.reshape((times.size, 1))
            dts = dt_matrix[dt_matrix > 0].flatten().astype(np.float16)
        else:
            dts = np.diff(times)

        # bin delta_t using provided bins; if zero pair found at any delta_t bin,
        # replace 0 with 0.01 to avoid the exploding 1/sqrt(n) term in this metric
        result, bins = np.histogram(dts, self.bins)
        new_result = np.where(result > 0, result, 0.01)

        # compute photometric_error^2 population variance and population mean
        # note that variance is replaced by median_absolute_deviate^2
        # mean is replaced by median in this implementation to make it robust to
        # outliers in simulations (e.g., dcr simulations)
        err_var = mag_err**2
        err_var_mu = np.median(err_var)
        err_var_std = mad_std(err_var)

        # compute SF error
        sf_var_dt = 2 * (err_var_mu + err_var_std / np.sqrt(new_result))
        sf_var_metric = np.sum(sf_var_dt * self.weight)

        return sf_var_metric
