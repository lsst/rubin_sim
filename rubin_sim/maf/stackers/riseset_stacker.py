__all__ = ("compute_gen_oblique_ascension", "RiseSetStacker")

import numpy as np
from rubin_scheduler.utils import Site, calc_lmst

from .base_stacker import BaseStacker

SIDEREAL_DAY = 0.9972696  # solar days per sidereal day


def compute_gen_oblique_ascension(ra, dec, lat, alt):
    """Compute the generalized oblique ascension.

    The generalized oblique ascension is the local mean sidereal time at which
    a point with coordinates (ra, dec) crosses altitude alt at a site with
    geographic latitude lat.  It equals ra minus the ascensional difference at
    that altitude.

    Parameters
    ----------
    ra : `float` or `numpy.ndarray`
        Right ascension, in radians.
    dec : `float` or `numpy.ndarray`
        Declination, in radians.
    lat : `float`
        Geographic latitude of the observer, in radians.
    alt : `float`
        Altitude of the crossing, in radians.

    Returns
    -------
    oblique_ascension : `float` or `numpy.ndarray`
        Generalized oblique ascension in radians.  NaN where the crossing
        altitude is never reached (object is circumpolar with respect to alt,
        or never rises to alt).

    Notes
    -----
    This computes a "generalized" oblique ascension: the traditional oblique
    ascension corresponds to the special case alt = 0 (rising or setting on
    the horizon).  Right ascension is a special case of oblique ascension
    where latitude is zero: at the equator every point rises and sets at
    HA = +/-90 deg, so the LMST at rising is always RA - 90 deg, i.e. the
    zero-latitude oblique ascension is a fixed offset from RA.

    The key intermediate quantity is the ascensional difference, the hour
    angle at which the point crosses altitude alt:

        cos(ascensional_difference) =
            (sin(alt) - sin(dec) * sin(lat)) / (cos(dec) * cos(lat))

    When alt = 0 this reduces to the classical formula
    cos(D) = -tan(dec) * tan(lat).
    """
    cos_asc_diff = (np.sin(alt) - np.sin(dec) * np.sin(lat)) / (np.cos(dec) * np.cos(lat))

    # Values outside [-1, 1] mean the altitude is never reached.
    normal = (cos_asc_diff >= -1.0) & (cos_asc_diff <= 1.0)
    ascensional_difference = np.where(normal, np.arccos(np.clip(cos_asc_diff, -1.0, 1.0)), np.nan)

    return ra - ascensional_difference


class RiseSetStacker(BaseStacker):
    """Add rise_mjd and set_mjd columns for each observation.

    rise_mjd is the most recent MJD before observationStartMJD at which the
    field rose above alt_limit (from below).

    set_mjd is the next MJD after observationStartMJD at which the field will
    fall below alt_limit.

    Both are NaN for fields that are circumpolar relative to alt_limit
    (always above) or that never rise above alt_limit.

    The calculation is purely geometric: it ignores refraction and uses
    sidereal (not solar) time for rise/set crossing times.

    Parameters
    ----------
    mjd_col : `str`, optional
        Column name for observation start MJD. Default 'observationStartMJD'.
    ra_col : `str`, optional
        Column name for RA. Default 'fieldRA'.
    dec_col : `str`, optional
        Column name for Dec. Default 'fieldDec'.
    degrees : `bool`, optional
        If True, ra/dec are in degrees. Default True.
    site : `str` or `rubin_scheduler.utils.Site`, optional
        Observatory name or Site object. Default 'LSST'.
    alt_limit : `float`, optional
        Limiting altitude in degrees. Default 20.0.
    """

    cols_added = ["rise_mjd", "set_mjd"]

    def __init__(
        self,
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        site="LSST",
        alt_limit=20.0,
    ):
        self.mjd_col = mjd_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        self.alt_limit = alt_limit
        self.cols_req = [mjd_col, ra_col, dec_col]
        self.units = ["MJD", "MJD"]
        self.cols_added_dtypes = [float, float]

        if isinstance(site, str):
            self.site = Site(name=site)
        else:
            self.site = site

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data

        mjd = sim_data[self.mjd_col]
        ra = sim_data[self.ra_col]
        dec = sim_data[self.dec_col]

        if self.degrees:
            ra = np.radians(ra)
            dec = np.radians(dec)

        lat = self.site.latitude_rad
        lon = self.site.longitude_rad
        alt = np.radians(self.alt_limit)

        # Compute LMST (returned in hours by calc_lmst) then convert to
        # radians.
        lmst_rad = calc_lmst(mjd, lon) / 12.0 * np.pi

        # Oblique ascension at rise (LMST when the field crosses alt from
        # below) and at set (LMST when it crosses from above).  The
        # ascensional difference embedded in compute_gen_oblique_ascension
        # is the HA offset from the meridian to the crossing; negating it
        # gives the set LMST.
        oa_rise = compute_gen_oblique_ascension(ra, dec, lat, alt)
        oa_set = 2.0 * ra - oa_rise  # ra + ascensional_difference

        normal = np.isfinite(oa_rise)

        # HA elapsed since last rise / remaining until next set,
        # both in [0, 2*pi).
        ha_since_rise = (lmst_rad - oa_rise) % (2.0 * np.pi)
        ha_until_set = (oa_set - lmst_rad) % (2.0 * np.pi)

        rise_mjd = mjd - ha_since_rise / (2.0 * np.pi) * SIDEREAL_DAY
        set_mjd = mjd + ha_until_set / (2.0 * np.pi) * SIDEREAL_DAY

        sim_data["rise_mjd"] = np.where(normal, rise_mjd, np.nan)
        sim_data["set_mjd"] = np.where(normal, set_mjd, np.nan)

        return sim_data
