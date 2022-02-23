import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.photUtils import Bandpass, PhotometricParameters
from rubin_sim.data import get_data_dir
import os


__all__ = ["SkySaturationMetric"]


class SkySaturationMetric(BaseMetric):
    """Check if the sky would saturate a visit in an exposure"""

    def __init__(
        self,
        exptimeCol="visitExposureTime",
        metricName="SkySaturation",
        skybrightnessCol="skyBrightness",
        nexpCol="numExposures",
        filterCol="filter",
        zp_inst=None,
        units="#",
        saturation_e=150e3,
        pixscale=0.2,
        **kwargs
    ):

        self.pixscale = pixscale
        self.A_pix = pixscale ** 2
        self.max_n_photons = saturation_e

        self.filterCol = filterCol
        self.nexpCol = nexpCol
        self.skybrightnessCol = skybrightnessCol
        self.exptimeCol = exptimeCol

        super().__init__(
            col=[filterCol, exptimeCol, skybrightnessCol, nexpCol],
            units=units,
            metricName=metricName,
            **kwargs
        )

        if zp_inst is None:
            # Load the default expected instumental zeropoints
            zp_inst = {}
            datadir = get_data_dir()
            for filtername in "ugrizy":
                # set gain and exptime to 1 so the instrumental zeropoint will be in photoelectrons and per second
                phot_params = PhotometricParameters(
                    nexp=1, gain=1, exptime=1, bandpass=filtername
                )
                bp = Bandpass()
                bp.readThroughput(
                    os.path.join(
                        datadir, "throughputs/baseline/", "total_%s.dat" % filtername
                    )
                )
                zp_inst[filtername] = bp.calcZP_t(phot_params)
            self.zpt = zp_inst
        else:
            self.zpt = zp_inst

    def run(self, dataSlice, slicePoint):

        result = 0
        filters = np.unique(dataSlice[self.filterCol])
        for filtername in filters:
            n_photons = (
                10.0
                ** ((dataSlice[self.skybrightnessCol] - self.zpt[filtername]) / -2.5)
                * self.A_pix
                * dataSlice[self.exptimeCol]
                / dataSlice[self.nexpCol]
            )
            over = np.where(n_photons > self.max_n_photons)[0]
            result += np.size(over)

        return result
