import numpy as np
from rubin_sim.utils import _galactic_from_equatorial, calc_lmst_last
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from .base_stacker import BaseStacker
from .dither_stackers import wrap_ra
from astropy.time import Time


__all__ = ["ra_dec2_alt_az", "GalacticStacker", "EclipticStacker"]


def ra_dec2_alt_az(ra, dec, lat, lon, mjd, altonly=False):
    """Convert RA/Dec (and telescope site lat/lon) to alt/az.

    This uses simple equations and ignores aberation, precession, nutation, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    altonly : bool, optional
        Calculate altitude only.

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    lmst, last = calc_lmst_last(mjd, lon)
    lmst = lmst / 12.0 * np.pi  # convert to rad
    ha = lmst - ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec * sinlat + np.cos(dec) * coslat * np.cos(ha)
    # make sure sinalt is in the expected range.
    sinalt = np.where(sinalt < -1, -1, sinalt)
    sinalt = np.where(sinalt > 1, 1, sinalt)
    alt = np.arcsin(sinalt)
    if altonly:
        az = None
    else:
        cosaz = (sindec - np.sin(alt) * sinlat) / (np.cos(alt) * coslat)
        cosaz = np.where(cosaz < -1, -1, cosaz)
        cosaz = np.where(cosaz > 1, 1, cosaz)
        az = np.arccos(cosaz)
        signflip = np.where(np.sin(ha) > 0)
        az[signflip] = 2.0 * np.pi - az[signflip]
    return alt, az


class GalacticStacker(BaseStacker):
    """Add the galactic coordinates of each RA/Dec pointing: gall, galb

    Parameters
    ----------
    ra_col : str, optional
        Name of the RA column. Default fieldRA.
    dec_col : str, optional
        Name of the Dec column. Default fieldDec.
    """

    cols_added = ["gall", "galb"]

    def __init__(self, ra_col="fieldRA", dec_col="fieldDec", degrees=True):
        self.cols_req = [ra_col, dec_col]
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        if self.degrees:
            self.units = ["degrees", "degrees"]
        else:
            self.units = ["radians", "radians"]

    def _run(self, sim_data, cols_present=False):
        # raCol and DecCol in radians, gall/b in radians.
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        if self.degrees:
            sim_data["gall"], sim_data["galb"] = _galactic_from_equatorial(
                np.radians(sim_data[self.ra_col]), np.radians(sim_data[self.dec_col])
            )
        else:
            sim_data["gall"], sim_data["galb"] = _galactic_from_equatorial(
                sim_data[self.ra_col], sim_data[self.dec_col]
            )
        return sim_data


class EclipticStacker(BaseStacker):
    """Add the ecliptic coordinates of each RA/Dec pointing: eclipLat, eclipLon
    Optionally subtract off the sun's ecliptic longitude and wrap.

    Parameters
    ----------
    mjd_col : str, optional
        Name of the MJD column. Default expMJD.
    ra_col : str, optional
        Name of the RA column. Default fieldRA.
    dec_col : str, optional
        Name of the Dec column. Default fieldDec.
    subtract_sun_lon : bool, optional
        Flag to subtract the sun's ecliptic longitude. Default False.
    """

    cols_added = ["eclipLat", "eclipLon"]

    def __init__(
        self,
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        subtract_sun_lon=False,
    ):

        self.cols_req = [mjd_col, ra_col, dec_col]
        self.subtract_sun_lon = subtract_sun_lon
        self.degrees = degrees
        if self.degrees:
            self.units = ["degrees", "degrees"]
        else:
            self.units = ["radians", "radians"]
        self.mjd_col = mjd_col
        self.ra_col = ra_col
        self.dec_col = dec_col

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        for i in np.arange(sim_data.size):
            if self.degrees:
                coord = SkyCoord(
                    ra=sim_data[self.ra_col] * u.degree,
                    dec=sim_data[self.dec_col] * u.degree,
                )
            else:
                coord = SkyCoord(
                    ra=sim_data[self.ra_col] * u.rad, dec=sim_data[self.dec_col] * u.rad
                )
            coord_ecl = coord.geocentricmeanecliptic
            sim_data["eclipLat"] = coord_ecl.lat.rad

            if self.subtract_sun_lon:
                times = Time(sim_data[self.mjd_col])
                sun = get_sun(times)
                sun_ecl = sun.geocentricmeanecliptic
                lon = wrap_ra(coord_ecl.lon.rad - sun_ecl.lon.rad)
                sim_data["eclipLon"] = lon
            else:
                sim_data["eclipLon"] = coord_ecl.lon.rad
        if self.degrees:
            sim_data["eclipLon"] = np.degrees(sim_data["eclipLon"])
            sim_data["eclipLat"] = np.degrees(sim_data["eclipLat"])
        return sim_data
