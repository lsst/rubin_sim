__all__ = ("AgnTimeLagMetric",)

import numpy as np

from rubin_sim.phot_utils import DustValues

from .base_metric import BaseMetric


class AgnTimeLagMetric(BaseMetric):
    def __init__(
        self,
        lag=100,
        z=1,
        log=False,
        threshold=2.2,
        calc_type="mean",
        mjd_col="observationStartMJD",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        dust=True,
        g_cutoff=22.0,
        r_cutoff=21.8,
        metric_name=None,
        **kwargs,
    ):
        self.lag = lag
        self.z = z
        self.log = log
        self.threshold = threshold
        self.calc_type = calc_type
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.m5_col = m5_col
        if metric_name is None:
            metric_name = f"AGN_TimeLag_{lag}_days"
        self.dust = dust
        self.g_cutoff = g_cutoff
        self.r_cutoff = r_cutoff
        if dust:
            maps = ["DustMap"]
            dust_properties = DustValues()
            self.ax1 = dust_properties.ax1
        else:
            maps = []
        super().__init__(
            col=[self.mjd_col, self.filter_col, self.m5_col],
            metric_name=metric_name,
            maps=maps,
            **kwargs,
        )

    # Calculate NQUIST value for time-lag and sampling time
    # (redshift is included in formula if desired)
    def _get_nquist_value(self, caden, lag, z):
        return lag / ((1 + z) * caden)

    def run(self, data_slice, slice_point=None):
        # Dust extinction
        filterlist = np.unique(data_slice[self.filter_col])
        if self.dust:
            m5 = np.zeros(len(data_slice))
            for filtername in filterlist:
                in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
                a_x = self.ax1[data_slice[self.filter_col][0]] * slice_point["ebv"]
                m5[in_filt] = data_slice[self.m5_col][in_filt] - a_x
        else:
            m5 = data_slice[self.m5_col]

        # Identify times which pass magnitude cuts (chosen by AGN contributors)
        mjds = np.zeros(len(data_slice))
        for filtername in filterlist:
            in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
            if filtername in ("u", "i", "z", "y", "r", "g"):
                mjds[in_filt] = data_slice[self.mjd_col][in_filt]
            elif filtername == "g":
                faint = np.where(m5[in_filt] > self.g_cutoff)
                mjds[in_filt][faint] = data_slice[self.mjd_col][in_filt][faint]
            elif filtername == "r":
                faint = np.where(m5[in_filt] > self.r_cutoff)
                mjds[in_filt][faint] = data_slice[self.mjd_col][in_filt][faint]
        # Remove the visits which were not faint enough
        mjds = mjds[np.where(mjds > 0)]

        # Calculate differences in time between visits
        mv = np.sort(mjds)
        val = np.diff(mv)
        # If there was only one visit; bail out now.
        if len(val) == 0:
            return self.badval

        # Otherwise summarize the time differences as:
        if self.calc_type == "mean":
            val = np.mean(val)
        elif self.calc_type == "min":
            val = np.min(val)
        elif self.calc_type == "max":
            val = np.max(val)
        else:
            # find the greatest common divisor
            val = np.rint(val).astype(int)
            val = np.gcd.reduce(val)

        # Will always have a value at this point
        nquist = self._get_nquist_value(val, self.lag, self.z)
        if self.log:
            nquist = np.log(nquist)

        # Threshold nquist value is 2.2,
        # hence we are aiming to show values higher than threshold (2.2) value
        threshold = self.threshold
        if self.log:
            threshold = np.log(threshold)

        if nquist < threshold:
            nquist = self.badval

        return nquist
