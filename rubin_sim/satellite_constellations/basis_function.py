__all__ = ("SatelliteAvoidBasisFunction",)

import healpy as hp
import numpy as np
import rubin_scheduler.scheduler.basis_functions as bf


class SatelliteAvoidBasisFunction(bf.BaseBasisFunction):
    """Uses satellite position information from the Conditions object
    and then avoids streaks.

    Parameters
    ----------
    forecast_time : `float`
        The time ahead to forecast satellite streaks (minutes).
    smooth_fwhm : `float`
        The smoothing full width half max to use (degrees)
    """

    def __init__(self, nside=32, forecast_time=90.0, smooth_fwhm=3.5):
        super().__init__(nside=nside)
        self.forecast_time = forecast_time / 60.0 / 24  # To days
        self.smooth_fwhm = np.radians(smooth_fwhm)

    def _calc_value(self, conditions, indx=None):
        result = 0
        # find the indices that are relevant
        indx_min = np.min(np.searchsorted(conditions.satellite_mjds, conditions.mjd))
        indx_max = np.max(np.searchsorted(conditions.satellite_mjds, conditions.mjd + self.forecast_time))

        if indx_max > indx_min:
            result = np.sum(conditions.satellite_maps[indx_min:indx_max], axis=0)
            result = hp.smoothing(result, fwhm=self.smooth_fwhm)
            result = hp.ud_grade(result, self.nside)
            result[np.where(result < 0)] = 0
            # Make it negative, so positive weights will result
            # in avoiding satellites
            result *= -1

        return result
