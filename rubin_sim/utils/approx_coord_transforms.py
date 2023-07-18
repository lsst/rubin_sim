__all__ = (
    "_approx_alt_az2_ra_dec",
    "_approx_ra_dec2_alt_az",
    "approx_alt_az2_ra_dec",
    "approx_ra_dec2_alt_az",
    "_approx_altaz2pa",
    "approx_altaz2pa",
)

import numpy as np

from .coordinate_transformations import calc_lmst_last


def _approx_altaz2pa(alt_rad, az_rad, latitude_rad):
    """A fast calculation of parallactic angle

    Parameters
    ----------
    alt_rad : `float`
        Altitude (radians)
    az_rad : `float`
        Azimuth (radians)
    latitude_rad : `float`
        The latitude of the observatory (radians)

    Returns
    -------
    pa : `float`
        Parallactic angle (radians)
    """

    y = np.sin(-az_rad) * np.cos(latitude_rad)
    x = np.cos(alt_rad) * np.sin(latitude_rad) - np.sin(alt_rad) * np.cos(latitude_rad) * np.cos(-az_rad)
    pa = np.arctan2(y, x)
    # Make it run from 0-360 deg instead of of -180 to 180
    pa = pa % (2.0 * np.pi)
    return pa


def approx_altaz2pa(alt_deg, az_deg, latitude_deg):
    """A fast calculation of parallactic angle

    Parameters
    ----------
    alt_rad : `float`
        Altitude (degrees)
    az_rad : `float`
        Azimuth (degrees)
    latitude_rad : `float`
        The latitude of the observatory (degrees)

    Returns
    -------
    pa : `float`
        Parallactic angle (degrees)
    """
    pa = _approx_altaz2pa(np.radians(alt_deg), np.radians(az_deg), np.radians(latitude_deg))
    return np.degrees(pa)


def approx_alt_az2_ra_dec(alt, az, lat, lon, mjd, lmst=None):
    """Convert alt, az to RA, Dec without taking into account aberration, precession, diffraction, etc.

    Parameters
    ----------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Degrees.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Must be same length as `alt`. Degrees.
    lat : float
        Latitude of the observatory in degrees.
    lon : float
        Longitude of the observatory in degrees.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    ra : array_like
        RA, in degrees.
    dec : array_like
        Dec, in degrees.
    """
    ra, dec = _approx_alt_az2_ra_dec(
        np.radians(alt),
        np.radians(az),
        np.radians(lat),
        np.radians(lon),
        mjd,
        lmst=lmst,
    )
    return np.degrees(ra), np.degrees(dec)


def _approx_alt_az2_ra_dec(alt, az, lat, lon, mjd, lmst=None):
    """
    Convert alt, az to RA, Dec without taking into account aberration, precession, diffraction, etc.

    Parameters
    ----------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Must be same length as `alt`. Radians.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians.
    """
    if lmst is None:
        lmst, last = calc_lmst_last(mjd, lon)
    lmst = lmst / 12.0 * np.pi  # convert to rad
    sindec = np.sin(lat) * np.sin(alt) + np.cos(lat) * np.cos(alt) * np.cos(az)
    sindec = np.clip(sindec, -1, 1)
    dec = np.arcsin(sindec)
    ha = np.arctan2(
        -np.sin(az) * np.cos(alt),
        -np.cos(az) * np.sin(lat) * np.cos(alt) + np.sin(alt) * np.cos(lat),
    )
    ra = lmst - ha
    raneg = np.where(ra < 0)
    ra[raneg] = ra[raneg] + 2.0 * np.pi
    raover = np.where(ra > 2.0 * np.pi)
    ra[raover] -= 2.0 * np.pi
    return ra, dec


def approx_ra_dec2_alt_az(ra, dec, lat, lon, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore aberration, precession, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in degrees.
    dec : array_like
        Dec, in degrees. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in degrees.
    lon : float
        Longitude of the observatory in degrees.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. degrees.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. degrees.
    """
    alt, az = _approx_ra_dec2_alt_az(
        np.radians(ra),
        np.radians(dec),
        np.radians(lat),
        np.radians(lon),
        mjd,
        lmst=lmst,
    )
    return np.degrees(alt), np.degrees(az)


def _approx_ra_dec2_alt_az(ra, dec, lat, lon, mjd, lmst=None, return_pa=False):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore aberration, precession, nutation, nutrition, etc.

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
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    if lmst is None:
        lmst, last = calc_lmst_last(mjd, lon)
    lmst = lmst / 12.0 * np.pi  # convert to rad
    ha = lmst - ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec * sinlat + np.cos(dec) * coslat * np.cos(ha)
    sinalt = np.clip(sinalt, -1, 1)
    alt = np.arcsin(sinalt)
    cosaz = (sindec - np.sin(alt) * sinlat) / (np.cos(alt) * coslat)
    cosaz = np.clip(cosaz, -1, 1)
    az = np.arccos(cosaz)
    if np.size(ha) < 2:
        if np.sin(ha) > 0:
            az = 2.0 * np.pi - az
    else:
        signflip = np.where(np.sin(ha) > 0)
        az[signflip] = 2.0 * np.pi - az[signflip]
    if return_pa:
        pa = _approx_altaz2pa(alt, az, lat)
        return alt, az, pa
    return alt, az
