__all__ = ("M5OptimalStacker", "generate_sky_slopes")

import numpy as np
from rubin_scheduler.utils import Site

from .base_stacker import BaseStacker


def generate_sky_slopes():
    """Fit a line to how the sky brightness changes with airmass."""
    import healpy as hp

    import rubin_sim.skybrightness as sb

    sm = sb.SkyModel(mags=True, moon=False, twilight=False, zodiacal=False)
    mjd = 57000
    nside = 32
    airmass_limit = 2.0
    dec, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    dec = np.pi / 2 - dec
    sm.set_ra_dec_mjd(ra, dec, mjd)
    mags = sm.return_mags()
    filters = mags.dtype.names
    filter_slopes = {}
    for filter_name in filters:
        good = np.where((~np.isnan(mags[filter_name])) & (sm.airmass < airmass_limit))
        pf = np.polyfit(sm.airmass[good], mags[filter_name][good], 1)
        filter_slopes[filter_name] = pf[0]
    print(filter_slopes)


class M5OptimalStacker(BaseStacker):
    """Make a new m5 column as if observations were taken on the meridian.
    If the moon is up, assume sky brightness stays the same.

    Assumes seeing scales as airmass^0.6. Uses linear fits for sky
    and airmass relation.

    Parameters
    ----------
    airmass_col : `str`
        Column name for the airmass per pointing.
    dec_col : `str`
        Column name for the pointing declination.
    sky_bright_col : `str`
        Column name for the sky brighntess per pointing.
    filter_col : `str`
        Column name for the filter name.
    m5_col : `str`
        Colum name for the five sigma limiting depth per pointing.
    moon_alt_col : `str`
        Column name for the moon altitude per pointing.
    sun_alt_col : `str`
        Column name for the sun altitude column.
    site : `str`
        Name of the site.
    """

    cols_added = ["m5_optimal"]

    def __init__(
        self,
        airmass_col="airmass",
        dec_col="fieldDec",
        sky_bright_col="skyBrightness",
        seeing_col="seeingFwhmEff",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        moon_alt_col="moonAlt",
        sun_alt_col="sunAlt",
        site="LSST",
    ):
        self.site = Site(site)
        self.units = ["mags"]
        self.airmass_col = airmass_col
        self.dec_col = dec_col
        self.sky_bright_col = sky_bright_col
        self.seeing_col = seeing_col
        self.filter_col = filter_col
        self.moon_alt_col = moon_alt_col
        self.sun_alt_col = sun_alt_col
        self.m5_col = m5_col
        self.cols_req = [
            airmass_col,
            dec_col,
            sky_bright_col,
            seeing_col,
            filter_col,
            moon_alt_col,
            sun_alt_col,
        ]
        self.cols_req = list(set(self.cols_req))

    def _run(self, sim_data, cols_present=False):
        # k_atm values from rubin_sim.operations gen_output.py
        k_atm = {"u": 0.50, "g": 0.21, "r": 0.13, "i": 0.10, "z": 0.07, "y": 0.18}
        # Linear fits to sky brightness change,
        # no moon, twilight, or zodiacal components
        # Use generate_sky_slopes to regenerate if needed.
        sky_slopes = {
            "g": -0.52611780327408397,
            "i": -0.67898669252082422,
            "r": -0.61378749766766827,
            "u": -0.27840980367303503,
            "y": -0.69635091524779691,
            "z": -0.69652846002009128,
        }
        min_z_possible = np.abs(np.radians(sim_data[self.dec_col]) - self.site.latitude_rad)
        min_airmass_possible = 1.0 / np.cos(min_z_possible)
        for filter_name in np.unique(sim_data[self.filter_col]):
            delta_sky = sky_slopes[filter_name] * (sim_data[self.airmass_col] - min_airmass_possible)
            delta_sky[
                np.where((sim_data[self.moon_alt_col] > 0) | (sim_data[self.sun_alt_col] > np.radians(-18.0)))
            ] = 0
            # Using Approximation that FWHM~X^0.6.
            # So seeing term in m5 of: 0.25 * log (7.0/FWHMeff )
            # Goes to 0.15 log(FWHM_min / FWHM_eff) in the difference
            m5_optimal = (
                sim_data[self.m5_col]
                - 0.5 * delta_sky
                - 0.15 * np.log10(min_airmass_possible / sim_data[self.airmass_col])
                - k_atm[filter_name] * (min_airmass_possible - sim_data[self.airmass_col])
            )
            good = np.where(sim_data[self.filter_col] == filter_name)
            sim_data["m5_optimal"][good] = m5_optimal[good]
        return sim_data
