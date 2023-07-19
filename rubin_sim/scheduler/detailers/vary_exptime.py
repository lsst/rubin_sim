__all__ = ("VaryExptDetailer", "calc_target_m5s")

import healpy as hp
import numpy as np

from rubin_sim.scheduler.detailers import BaseDetailer
from rubin_sim.utils import m5_flat_sed, ra_dec2_hpid


def calc_target_m5s(alt=65.0, fiducial_seeing=0.9, exptime=20.0):
    """Use the skybrightness model to find some good target m5s.

    Parameters
    ----------
    alt : `float`, opt
        Altitude for the target, degrees. Default 65.
    fiducial_seeing : `float`, opt
        Fiducial FWHMeff seeing, arcseconds. Default 0.9.
    exptime : `float`, opt
        Exposure time for the comparison, seconds. Default 20.

    Returns
    -------
    goal_m5 : `dict` of `float`
        dictionary of expected m5 values keyed by filtername
    """
    import rubin_sim.skybrightness as sb

    sm = sb.SkyModel(moon=False, twilight=False, mags=True)
    sm.set_ra_dec_mjd(np.array([0.0]), np.array([alt]), 49353.177645, degrees=True, azAlt=True)
    sky_mags = sm.return_mags()
    airmass = 1.0 / np.cos(np.pi / 2.0 - np.radians(alt))

    goal_m5 = {}
    for filtername in sky_mags:
        goal_m5[filtername] = m5_flat_sed(filtername, sky_mags[filtername], fiducial_seeing, exptime, airmass)

    return goal_m5


class VaryExptDetailer(BaseDetailer):
    """Vary the exposure time on observations to try and keep each observation at uniform depth.

    Parameters
    ----------
    min_expt : `float` (20.)
        The minimum exposure time to use (seconds).
    max_expt : `float` (100.)
        The maximum exposure time to use
    target_m5 : `dict` (None)
        Dictionary with keys of filternames as str and target 5-sigma depth values as floats.
        If none, the target_m5s are set to a min_expt exposure at X=1.1 in dark time.

    """

    def __init__(self, nside=32, min_expt=20.0, max_expt=100.0, target_m5=None):
        """"""
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside
        self.min_exp = min_expt
        self.max_exp = max_expt
        if target_m5 is None:
            self.target_m5 = {
                "g": 24.381615425253738,
                "i": 23.41810142458083,
                "r": 23.964359143049755,
                "u": 22.978794343692783,
                "y": 21.755612950787068,
                "z": 22.80377793629767,
            }
        else:
            self.target_m5 = target_m5

    def __call__(self, observation_list, conditions):
        """
        Parameters
        ----------
        observation_list : `list` of observations
            The observations to detail.
        conditions : `rubin_sim.scheduler.conditions` object

        Returns
        -------
        List of observations.
        """
        obs_array = np.concatenate(observation_list)
        hpids = ra_dec2_hpid(self.nside, obs_array["RA"], obs_array["dec"])
        new_expts = np.zeros(obs_array.size, dtype=float)
        for filtername in np.unique(obs_array["filter"]):
            in_filt = np.where(obs_array["filter"] == filtername)
            delta_m5 = self.target_m5[filtername] - conditions.M5Depth[filtername][hpids[in_filt]]
            # We can get NaNs because dithering pushes the center of the pointing into masked regions.
            nan_indices = np.argwhere(np.isnan(delta_m5)).ravel()
            for indx in nan_indices:
                bad_hp = hpids[in_filt][indx]
                # Note this might fail if we run at higher resolution, then we'd need to look farther for
                # pixels to interpolate.
                near_pix = hp.get_all_neighbours(conditions.nside, bad_hp)
                vals = conditions.M5Depth[filtername][near_pix]
                if True in np.isfinite(vals):
                    estimate_m5 = np.mean(vals[np.isfinite(vals)])
                    delta_m5[indx] = self.target_m5[filtername] - estimate_m5
                else:
                    raise ValueError("Failed to find a nearby unmasked sky value.")

            new_expts[in_filt] = conditions.exptime * 10 ** (delta_m5 / 1.25)
        new_expts = np.clip(new_expts, self.min_exp, self.max_exp)
        # I'm not sure what level of precision we can expect, so let's just limit to seconds
        new_expts = np.round(new_expts)

        for i, observation in enumerate(observation_list):
            observation["exptime"] = new_expts[i]

        return observation_list
