"""
This file contains coordinate transformations that rely on both
palpy and the contents of AstrometryUtils.py (basically, coordinate
transformations that need to transform between observed geocentric RA, DEC
and ICRS RA, Dec)
"""
import numpy as np
import palpy
from rubin_sim.utils.code_utilities import _validate_inputs
from rubin_sim.utils import _icrs_from_observed, _observed_from_icrs, calc_lmst_last

__all__ = [
    "_alt_az_pa_from_ra_dec",
    "alt_az_pa_from_ra_dec",
    "_ra_dec_from_alt_az",
    "ra_dec_from_alt_az",
    "get_rot_tel_pos",
    "_get_rot_tel_pos",
    "get_rot_sky_pos",
    "_get_rot_sky_pos",
]


def alt_az_pa_from_ra_dec(ra, dec, obs, include_refraction=True):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] include_refraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] altitude in degrees

    @param [out] azimuth in degrees

    @param [out] parallactic angle in degrees

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

    @param [in] ra_rad is RA in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] dec_rad is Dec in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] include_refraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] altitude in radians

    @param [out] azimuth in radians

    @param [out] parallactic angle in radians

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plan
    to positions on the celestial sphere.
    """

    are_arrays = _validate_inputs(
        [ra_rad, dec_rad], ["ra", "dec"], "alt_az_pa_from_ra_dec"
    )

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
        az, azd, azdd, alt, altd, altdd, pa, pad, padd = palpy.altaz(
            ha_rad, dec_obs, obs.site.latitude_rad
        )

    return alt, az, pa


def ra_dec_from_alt_az(alt, az, obs, include_refraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] alt is the altitude in degrees.  Can be a numpy array or a single value.

    @param [in] az is the azimuth in degrees.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] include_refraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] RA in degrees (in the International Celestial Reference System)

    @param [out] Dec in degrees (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    ra, dec = _ra_dec_from_alt_az(
        np.radians(alt), np.radians(az), obs, include_refraction=include_refraction
    )

    return np.degrees(ra), np.degrees(dec)


def _ra_dec_from_alt_az(alt_rad, az_rad, obs, include_refraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] alt_rad is the altitude in radians.  Can be a numpy array or a single value.

    @param [in] az_rad is the azimuth in radians.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] include_refraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] RA in radians (in the International Celestial Reference System)

    @param [out] Dec in radians (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    with np.errstate(invalid="ignore", divide="ignore"):
        are_arrays = _validate_inputs(
            [alt_rad, az_rad], ["alt_rad", "az_rad"], "ra_dec_from_alt_az"
        )

        lst = calc_lmst_last(obs.mjd.ut1, obs.site.longitude_rad)
        last = lst[1]
        sin_alt = np.sin(alt_rad)
        cos_lat = np.cos(obs.site.latitude_rad)
        sin_lat = np.sin(obs.site.latitude_rad)
        dec_obs = np.arcsin(
            sin_lat * sin_alt + cos_lat * np.cos(alt_rad) * np.cos(az_rad)
        )
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
    @param [in] ra is the RA in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rot_tel is rotTelPos in degrees
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as ra and dec.  In this case,
    each rot_tel will be associated with the corresponding ra, dec pair.

    @param [out] rot_sky_pos in degrees

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """

    rot_sky = _get_rot_sky_pos(
        np.radians(ra), np.radians(dec), obs, np.radians(rot_tel)
    )

    return np.degrees(rot_sky)


def _get_rot_sky_pos(ra_rad, dec_rad, obs, rot_tel_rad):
    """
    @param [in] ra_rad is the RA in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec_rad is Dec in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rot_tel_rad is rotTelPos in radians
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as ra_rad and dec_rad.  In this case,
    each rot_tel_rad will be associated with the corresponding ra_rad, dec_rad pair.

    @param [out] rot_sky_pos in radians

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
    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rot_sky is rot_sky_pos in degrees
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as ra and dec.  In this case, each rot_sky_pos
    will be associated with the corresponding ra, dec pair.

    @param [out] rotTelPos in degrees.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    rot_tel = _get_rot_tel_pos(
        np.radians(ra), np.radians(dec), obs, np.radians(rot_sky)
    )

    return np.degrees(rot_tel)


def _get_rot_tel_pos(ra_rad, dec_rad, obs, rot_sky_rad):
    """
    @param [in] ra_rad is RA in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec_rad is Dec in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rot_sky_rad is rot_sky_pos in radians
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as ra_rad and dec_rad.  In this case, each rot_sky_pos
    will be associated with the corresponding ra_rad, dec_rad pair.

    @param [out] rotTelPos in radians.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rot_sky_pos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    alt_rad, az_rad, pa_rad = _alt_az_pa_from_ra_dec(ra_rad, dec_rad, obs)

    return (rot_sky_rad + pa_rad) % (2.0 * np.pi)
