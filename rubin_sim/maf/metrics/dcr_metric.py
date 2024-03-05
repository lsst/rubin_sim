__all__ = ("DcrPrecisionMetric",)

import numpy as np
import rubin_scheduler.utils as utils

import rubin_sim.maf.utils as mafUtils

from .base_metric import BaseMetric


class DcrPrecisionMetric(BaseMetric):
    """Determine how precise a DCR correction could be made

    Parameters
    ----------
    atm_err : `float`
        Minimum error in photometry centroids introduced by the atmosphere
        (arcseconds). Default 0.01.
    """

    def __init__(
        self,
        metric_name="DCRprecision",
        seeing_col="seeingFwhmGeom",
        m5_col="fiveSigmaDepth",
        ha_col="HA",
        pa_col="paraAngle",
        filter_col="filter",
        atm_err=0.01,
        sed_template="flat",
        rmag=20.0,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.pa_col = pa_col
        self.seeing_col = seeing_col
        self.mags = {}
        self.filters = ["u", "g", "r", "i", "z", "y"]
        if sed_template == "flat":
            for f in self.filters:
                self.mags[f] = rmag
        else:
            self.mags = utils.stellarMags(sed_template, rmag=rmag)
        cols = [
            "ra_dcr_amp",
            "dec_dcr_amp",
            seeing_col,
            m5_col,
            filter_col,
            "zenithDistance",
            pa_col,
        ]
        units = "arcseconds"
        self.atm_err = atm_err
        super(DcrPrecisionMetric, self).__init__(cols, metric_name=metric_name, units=units, **kwargs)

    def run(self, data_slice, slice_point=None):
        snr = np.zeros(len(data_slice), dtype="float")
        for filt in self.filters:
            in_filt = np.where(data_slice[self.filter_col] == filt)
            snr[in_filt] = mafUtils.m52snr(self.mags[filt], data_slice[self.m5_col][in_filt])

        position_errors = np.sqrt(
            mafUtils.astrom_precision(data_slice[self.seeing_col], snr) ** 2 + self.atm_err**2
        )

        x_coord = np.tan(np.radians(data_slice["zenithDistance"])) * np.sin(
            np.radians(data_slice[self.pa_col])
        )
        x_coord2 = np.tan(np.radians(data_slice["zenithDistance"])) * np.cos(
            np.radians(data_slice[self.pa_col])
        )
        # Things should be the same for RA and dec.
        # Now I want to compute the error if I interpolate/extrapolate to +/-1.

        # function is of form, y=ax. a=y/x. da = dy/x.
        # Only strictly true if we know the unshifted position.
        # But this should be a reasonable approx.
        slope_uncerts = position_errors / x_coord
        slope_uncerts2 = position_errors / x_coord2

        total_slope_uncert = 1.0 / np.sqrt(np.sum(1.0 / slope_uncerts**2) + np.sum(1.0 / slope_uncerts2**2))

        # So, this will be the uncertainty in the RA or Dec offset at
        # x= +/- 1. A.K.A., the uncertainty in the slope
        # of the line made by tan(zd)*sin(PA) vs RA offset
        # or the line tan(zd)*cos(PA) vs Dec offset
        # Assuming we know the unshfted position of the object
        # (or there's little covariance if we are fitting for both)
        result = total_slope_uncert

        return result
