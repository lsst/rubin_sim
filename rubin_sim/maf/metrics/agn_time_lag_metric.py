import numpy as np
from rubin_sim.phot_utils import DustValues
from .base_metric import BaseMetric

__all__ = ["AGN_TimeLagMetric"]


class AGN_TimeLagMetric(BaseMetric):
    def __init__(
        self,
        lag=100,
        z=1,
        log=False,
        threshold=2.2,
        calcType="mean",
        mjdCol="observationStartMJD",
        filterCol="filter",
        m5Col="fiveSigmaDepth",
        dust=True,
        g_cutoff=22.0,
        r_cutoff=21.8,
        metricName=None,
        **kwargs,
    ):
        self.lag = lag
        self.z = z
        self.log = log
        self.threshold = threshold
        self.calcType = calcType
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.m5Col = m5Col
        if metricName is None:
            metricName = f"AGN_TimeLag_{lag}_days"
        self.dust = dust
        self.g_cutoff = g_cutoff
        self.r_cutoff = r_cutoff
        if dust:
            maps = ["DustMap"]
            dust_properties = DustValues()
            self.Ax1 = dust_properties.ax1
        else:
            maps = []
        super().__init__(
            col=[self.mjdCol, self.filterCol, self.m5Col],
            metricName=metricName,
            maps=maps,
            **kwargs,
        )

    # Calculate NQUIST value for time-lag and sampling time (redshift is included in formula if desired)
    def _getNquistValue(self, caden, lag, z):
        return lag / ((1 + z) * caden)

    def run(self, dataSlice, slicePoint=None):
        # Dust extinction
        filterlist = np.unique(dataSlice[self.filterCol])
        if self.dust:
            m5 = np.zeros(len(dataSlice))
            for filtername in filterlist:
                in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
                A_x = self.Ax1[dataSlice[self.filterCol][0]] * slicePoint["ebv"]
                m5[in_filt] = dataSlice[self.m5Col][in_filt] - A_x
        else:
            m5 = dataSlice[self.m5Col]

        # Identify times which pass magnitude cuts (chosen by AGN contributors)
        mjds = np.zeros(len(dataSlice))
        for filtername in filterlist:
            in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
            if filtername in ("u", "i", "z", "y", "r", "g"):
                mjds[in_filt] = dataSlice[self.mjdCol][in_filt]
            elif filtername == "g":
                faint = np.where(m5[in_filt] > self.g_cutoff)
                mjds[in_filt][faint] = dataSlice[self.mjdCol][in_filt][faint]
            elif filtername == "r":
                faint = np.where(m5[in_filt] > self.r_cutoff)
                mjds[in_filt][faint] = dataSlice[self.mjdCol][in_filt][faint]
        # Remove the visits which were not faint enough
        mjds = mjds[np.where(mjds > 0)]

        # Calculate differences in time between visits
        mv = np.sort(mjds)
        val = np.diff(mv)
        # If there was only one visit; bail out now.
        if len(val) == 0:
            return self.badval

        # Otherwise summarize the time differences as:
        if self.calcType == "mean":
            val = np.mean(val)
        elif self.calcType == "min":
            val = np.min(val)
        elif self.calcType == "max":
            val = np.max(val)
        else:
            # find the greatest common divisor
            val = np.rint(val).astype(int)
            val = np.gcd.reduce(val)

        # Will always have a value at this point
        nquist = self._getNquistValue(val, self.lag, self.z)
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
