"""
This file contains coordinate transformations that rely on both
palpy and the contents of AstrometryUtils.py (basically, coordinate
transformations that need to transform between observed geocentric RA, DEC
and ICRS RA, Dec)
"""
import numpy as np
import palpy
from rubin_sim.utils.CodeUtilities import _validate_inputs
from rubin_sim.utils import _icrsFromObserved, _observedFromICRS, calcLmstLast

__all__ = ["_altAzPaFromRaDec", "altAzPaFromRaDec",
           "_raDecFromAltAz", "raDecFromAltAz",
           "getRotTelPos", "_getRotTelPos",
           "getRotSkyPos", "_getRotSkyPos"]


def altAzPaFromRaDec(ra, dec, obs, includeRefraction=True):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] includeRefraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] altitude in degrees

    @param [out] azimuth in degrees

    @param [out] parallactic angle in degrees

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical celestial
    sphere.
    """

    alt, az, pa = _altAzPaFromRaDec(np.radians(ra), np.radians(dec),
                                    obs, includeRefraction=includeRefraction)

    return np.degrees(alt), np.degrees(az), np.degrees(pa)


def _altAzPaFromRaDec(raRad, decRad, obs, includeRefraction=True):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] includeRefraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] altitude in radians

    @param [out] azimuth in radians

    @param [out] parallactic angle in radians

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plan
    to positions on the celestial sphere.
    """

    are_arrays = _validate_inputs([raRad, decRad], ['ra', 'dec'],
                                  "altAzPaFromRaDec")

    raObs, decObs = \
    _observedFromICRS(raRad, decRad, obs_metadata=obs,
                      epoch=2000.0, includeRefraction=includeRefraction)

    lst = calcLmstLast(obs.mjd.UT1, obs.site.longitude_rad)
    last = lst[1]
    haRad = np.radians(last * 15.0) - raObs

    if are_arrays:
        az, azd, azdd, \
            alt, altd, altdd, \
            pa, pad, padd = palpy.altazVector(
                haRad, decObs, obs.site.latitude_rad)
    else:
        az, azd, azdd, \
            alt, altd, altdd, \
            pa, pad, padd = palpy.altaz(haRad, decObs, obs.site.latitude_rad)

    return alt, az, pa


def raDecFromAltAz(alt, az, obs, includeRefraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] alt is the altitude in degrees.  Can be a numpy array or a single value.

    @param [in] az is the azimuth in degrees.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] includeRefraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] RA in degrees (in the International Celestial Reference System)

    @param [out] Dec in degrees (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    ra, dec = _raDecFromAltAz(np.radians(alt), np.radians(az), obs,
                              includeRefraction=includeRefraction)

    return np.degrees(ra), np.degrees(dec)


def _raDecFromAltAz(altRad, azRad, obs, includeRefraction=True):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] altRad is the altitude in radians.  Can be a numpy array or a single value.

    @param [in] azRad is the azimuth in radians.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [in] includeRefraction is a `bool` that turns refraction on and off
    (default True)

    @param [out] RA in radians (in the International Celestial Reference System)

    @param [out] Dec in radians (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    with np.errstate(invalid='ignore', divide='ignore'):
        are_arrays = _validate_inputs(
            [altRad, azRad], ['altRad', 'azRad'], "raDecFromAltAz")

        lst = calcLmstLast(obs.mjd.UT1, obs.site.longitude_rad)
        last = lst[1]
        sinAlt = np.sin(altRad)
        cosLat = np.cos(obs.site.latitude_rad)
        sinLat = np.sin(obs.site.latitude_rad)
        decObs = np.arcsin(sinLat * sinAlt + cosLat *
                           np.cos(altRad) * np.cos(azRad))
        costheta = (sinAlt - np.sin(decObs) * sinLat) / (np.cos(decObs) * cosLat)
        if are_arrays:
            haRad0 = np.arccos(costheta)
            # Make sure there were no NaNs
            nanSpots = np.where(np.isnan(haRad0))[0]
            if np.size(nanSpots) > 0:
                haRad0[nanSpots] = 0.5 * np.pi * \
                    (1.0 - np.sign(costheta[nanSpots]))
        else:
            haRad0 = np.arccos(costheta)
            if np.isnan(haRad0):
                if np.sign(costheta) > 0.0:
                    haRad0 = 0.0
                else:
                    haRad0 = np.pi

        haRad = np.where(np.sin(azRad) >= 0.0, -1.0 * haRad0, haRad0)
        raObs = np.radians(last * 15.) - haRad

        raRad, decRad = _icrsFromObserved(raObs, decObs,
                                          obs_metadata=obs, epoch=2000.0,
                                          includeRefraction=includeRefraction)

    return raRad, decRad


def getRotSkyPos(ra, dec, obs, rotTel):
    """
    @param [in] ra is the RA in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotTel is rotTelPos in degrees
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as ra and dec.  In this case,
    each rotTel will be associated with the corresponding ra, dec pair.

    @param [out] rotSkyPos in degrees

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """

    rotSky = _getRotSkyPos(np.radians(ra), np.radians(dec),
                           obs, np.radians(rotTel))

    return np.degrees(rotSky)


def _getRotSkyPos(raRad, decRad, obs, rotTelRad):
    """
    @param [in] raRad is the RA in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotTelRad is rotTelPos in radians
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as raRad and decRad.  In this case,
    each rotTelRad will be associated with the corresponding raRad, decRad pair.

    @param [out] rotSkyPos in radians

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, obs)

    return (rotTelRad - paRad) % (2. * np.pi)


def getRotTelPos(ra, dec, obs, rotSky):
    """
    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotSky is rotSkyPos in degrees
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as ra and dec.  In this case, each rotSkyPos
    will be associated with the corresponding ra, dec pair.

    @param [out] rotTelPos in degrees.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    rotTel = _getRotTelPos(np.radians(ra), np.radians(dec),
                           obs, np.radians(rotSky))

    return np.degrees(rotTel)


def _getRotTelPos(raRad, decRad, obs, rotSkyRad):
    """
    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.
    (In the International Celestial Reference System)

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotSkyRad is rotSkyPos in radians
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as raRad and decRad.  In this case, each rotSkyPos
    will be associated with the corresponding raRad, decRad pair.

    @param [out] rotTelPos in radians.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, obs)

    return (rotSkyRad + paRad) % (2. * np.pi)
