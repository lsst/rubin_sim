"""
This file contains coordinate transformations that rely on both
palpy and the contents of AstrometryUtils.py (basically, coordinate
transformations that need to transform between observed geocentric RA, DEC
and ICRS RA, Dec)
"""
__all__ = (
    "_alt_az_pa_from_ra_dec",
    "alt_az_pa_from_ra_dec",
    "_ra_dec_from_alt_az",
    "ra_dec_from_alt_az",
    "get_rot_tel_pos",
    "_get_rot_tel_pos",
    "get_rot_sky_pos",
    "_get_rot_sky_pos",
)

import numpy as np
import palpy

from .code_utilities import _validate_inputs
from .coordinate_transformations import calc_lmst_last
from .wcs_utils import _icrs_from_observed, _observed_from_icrs


def alt_az_pa_from_ra_dec(ra, dec, obs, include_refraction=True):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    Parameters
    ----------
    ra : `Unknown`
        is RA in degrees.  Can be a numpy array or a single value.
        Assumed to be in the International Celestial Reference System.
    dec : `Unknown`
        is Dec in degrees.  Can be a numpy array or a single value.
        Assumed to be in the International Celestial Reference System.
    obs : `Unknown`
        is an ObservationMetaData characterizing
        the site of the telescope and the MJD of the observation
    include_refraction : `Unknown`
        is a `bool` that turns refraction on and off
        (default True)
    altitude : `Unknown`
        in degrees
    azimuth : `Unknown`
        in degrees
    parallactic : `Unknown`
        angle in degrees

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical celestial
    sphere.
    """

    alt, az, pa = _alt_az_pa_from_ra_dec(
        np.radians(ra), np.radians(dec), obs, include_refraction=include_refraction
    )

    return np.degrees(alt), np.degrees(az), np.degrees(pa)


def _alt_az_pa_from_ra_dec(ra_rad, dec_rad, obs, include_refraction=True):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    Parameters
    ----------
    ra_rad : `Unknown`
        is RA in radians.  Can be a numpy array or a single value.
        Assumed to be in the International Celestial Reference System.
    dec_rad : `Unknown`
        is Dec in radians.  Can be a numpy array or a single value.
        Assumed to be in the International Celestial Reference System.
    obs : `Unknown`
        is an ObservationMetaData characterizing
        the site of the telescope and the MJD of the observation
    include_refraction : `Unknown`
        is a `bool` that turns refraction on and off
        (default True)
    altitude : `Unknown`
        in radians
    azimuth : `Unknown`
        in radians
    parallactic : `Unknown`
        angle in radians

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plan
    to positions on the celestial sphere.
    """

    are_arrays = _validate_inputs([ra_rad, dec_rad], ["ra", "dec"], "alt_az_pa_from_ra_dec")

    ra_obs, dec_obs = _observed_from_icrs(
        ra_rad,
        dec_rad,
        obs_metadata=obs,
        epoch=2000.0,
        include_refraction=include_refraction,
    )

    lst = calc_lmst_last(obs.mjd.ut1, obs.site.longitude_rad)
    last = lst[1]
    ha_rad = np.radians(last * 15.0) - ra_obs

    if are_arrays:
        az, azd, azdd, alt, altd, altdd, pa, pad, padd = palpy.altazVector(
            ha_rad, dec_obs, obs.site.latitude_rad
        )
    else:
        az, azd, azdd, alt, altd, altdd, pa, pad, padd = palpy.altaz(ha_rad, dec_obs, obs.site.latitude_rad)

    return alt, az, pa


def ra_dec_from_alt_az(alt, az, obs, include_refraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    Parameters
    ----------
    alt : `Unknown`
        is the altitude in degrees.  Can be a numpy array or a single value.
    az : `Unknown`
        is the azimuth in degrees.  Cant be a numpy array or a single value.
    obs : `Unknown`
        is an ObservationMetaData characterizing
        the site of the telescope and the MJD of the observation
    include_refraction : `Unknown`
        is a `bool` that turns refraction on and off
        (default True)
    RA : `Unknown`
        in degrees (in the International Celestial Reference System)
    Dec : `Unknown`
        in degrees (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    ra, dec = _ra_dec_from_alt_az(np.radians(alt), np.radians(az), obs, include_refraction=include_refraction)

    return np.degrees(ra), np.degrees(dec)


def _ra_dec_from_alt_az(alt_rad, az_rad, obs, include_refraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    Parameters
    ----------
    alt_rad : `Unknown`
        is the altitude in radians.  Can be a numpy array or a single value.
    az_rad : `Unknown`
        is the azimuth in radians.  Cant be a numpy array or a single value.
    obs : `Unknown`
        is an ObservationMetaData characterizing
        the site of the telescope and the MJD of the observation
    include_refraction : `Unknown`
        is a `bool` that turns refraction on and off
        (default True)
    RA : `Unknown`
        in radians (in the International Celestial Reference System)
    Dec : `Unknown`
        in radians (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    with np.errstate(invalid="ignore", divide="ignore"):
        are_arrays = _validate_inputs([alt_rad, az_rad], ["alt_rad", "az_rad"], "ra_dec_from_alt_az")

        lst = calc_lmst_last(obs.mjd.ut1, obs.site.longitude_rad)
        last = lst[1]
        sin_alt = np.sin(alt_rad)
        cos_lat = np.cos(obs.site.latitude_rad)
        sin_lat = np.sin(obs.site.latitude_rad)
        dec_obs = np.arcsin(sin_lat * sin_alt + cos_lat * np.cos(alt_rad) * np.cos(az_rad))
        costheta = (sin_alt - np.sin(dec_obs) * sin_lat) / (np.cos(dec_obs) * cos_lat)
        if are_arrays:
            ha_rad0 = np.arccos(costheta)
            # Make sure there were no NaNs
            nan_spots = np.where(np.isnan(ha_rad0))[0]
            if np.size(nan_spots) > 0:
                ha_rad0[nan_spots] = 0.5 * np.pi * (1.0 - np.sign(costheta[nan_spots]))
        else:
            ha_rad0 = np.arccos(costheta)
            if np.isnan(ha_rad0):
                if np.sign(costheta) > 0.0:
                    ha_rad0 = 0.0
                else:
                    ha_rad0 = np.pi

        ha_rad = np.where(np.sin(az_rad) >= 0.0, -1.0 * ha_rad0, ha_rad0)
        ra_obs = np.radians(last * 15.0) - ha_rad

        ra_rad, dec_rad = _icrs_from_observed(
            ra_obs,
            dec_obs,
            obs_metadata=obs,
            epoch=2000.0,
            include_refraction=include_refraction,
        )

    return ra_rad, dec_rad


def get_rot_sky_pos(ra, dec, obs, rot_tel):
    """
    Parameters
    ----------
    ra : `Unknown`
        is the RA in degrees.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    dec : `Unknown`
        is Dec in degrees.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    obs : `Unknown`
        is an ObservationMetaData characterizing the telescope pointing
        and site.
    rot_tel : `Unknown`
        is rotTelPos in degrees
        (the angle of the camera rotator).  Can be a numpy array or a single value.
        If a numpy array, should have the same length as ra and dec.  In this case,
        each rot_tel will be associated with the corresponding ra, dec pair.
    rot_sky_pos : `Unknown`
        in degrees

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """

    rot_sky = _get_rot_sky_pos(np.radians(ra), np.radians(dec), obs, np.radians(rot_tel))

    return np.degrees(rot_sky)


def _get_rot_sky_pos(ra_rad, dec_rad, obs, rot_tel_rad):
    """
    Parameters
    ----------
    ra_rad : `Unknown`
        is the RA in radians.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)

    Parameters
    ----------
    dec_rad : `Unknown`
        is Dec in radians.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    obs : `Unknown`
        is an ObservationMetaData characterizing the telescope pointing
        and site.
    rot_tel_rad : `Unknown`
        is rotTelPos in radians
        (the angle of the camera rotator).  Can be a numpy array or a single value.
        If a numpy array, should have the same length as ra_rad and dec_rad.  In this case,
        each rot_tel_rad will be associated with the corresponding ra_rad, dec_rad pair.
    rot_sky_pos : `Unknown`
        in radians

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    alt_rad, az_rad, pa_rad = _alt_az_pa_from_ra_dec(ra_rad, dec_rad, obs)

    return (rot_tel_rad - pa_rad) % (2.0 * np.pi)


def get_rot_tel_pos(ra, dec, obs, rot_sky):
    """
    Parameters
    ----------
    ra : `Unknown`
        is RA in degrees.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    dec : `Unknown`
        is Dec in degrees.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    obs : `Unknown`
        is an ObservationMetaData characterizing the telescope pointing
        and site.
    rot_sky : `Unknown`
        is rot_sky_pos in degrees
        (the angle of the field of view relative to the South pole given a
        rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
        have the same length as ra and dec.  In this case, each rot_sky_pos
        will be associated with the corresponding ra, dec pair.
    rotTelPos : `Unknown`
        in degrees.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    rot_tel = _get_rot_tel_pos(np.radians(ra), np.radians(dec), obs, np.radians(rot_sky))

    return np.degrees(rot_tel)


def _get_rot_tel_pos(ra_rad, dec_rad, obs, rot_sky_rad):
    """
    Parameters
    ----------
    ra_rad : `Unknown`
        is RA in radians.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    dec_rad : `Unknown`
        is Dec in radians.  Can be a numpy array or a single value.
        (In the International Celestial Reference System)
    obs : `Unknown`
        is an ObservationMetaData characterizing the telescope pointing
        and site.
    rot_sky_rad : `Unknown`
        is rot_sky_pos in radians
        (the angle of the field of view relative to the South pole given a
        rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
        have the same length as ra_rad and dec_rad.  In this case, each rot_sky_pos
        will be associated with the corresponding ra_rad, dec_rad pair.
    rotTelPos : `Unknown`
        in radians.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    alt_rad, az_rad, pa_rad = _alt_az_pa_from_ra_dec(ra_rad, dec_rad, obs)

    return (rot_sky_rad + pa_rad) % (2.0 * np.pi)
