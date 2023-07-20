__all__ = (
    "_solar_ra_dec",
    "solar_ra_dec",
    "_distance_to_sun",
    "distance_to_sun",
    "apply_refraction",
    "refraction_coefficients",
    "_apply_precession",
    "apply_precession",
    "_apply_proper_motion",
    "apply_proper_motion",
    "_app_geo_from_icrs",
    "app_geo_from_icrs",
    "_icrs_from_app_geo",
    "icrs_from_app_geo",
    "_observed_from_app_geo",
    "observed_from_app_geo",
    "_app_geo_from_observed",
    "app_geo_from_observed",
    "_observed_from_icrs",
    "observed_from_icrs",
    "_icrs_from_observed",
    "icrs_from_observed",
)

import numpy as np
import palpy

from .code_utilities import _validate_inputs
from .coordinate_transformations import (
    arcsec_from_radians,
    cartesian_from_spherical,
    haversine,
    radians_from_arcsec,
    spherical_from_cartesian,
)


def _solar_ra_dec(mjd, epoch=2000.0):
    """
    Return the RA and Dec of the Sun in radians

    Parameters
    ----------
    mjd : `ModifiedJulianDate`
        is the date represented as a
        ModifiedJulianDate object.
    epoch : `float`
        is the mean epoch of the coordinate system
        (default is 2000.0)

    Returns
    -------
    RA : `float`
        RA of Sun in radians
    Dec : `float`
        Declination of Sun in radians
    """

    params = palpy.mappa(epoch, mjd.TDB)
    # params[4:7] is a unit vector pointing from the Sun
    # to the Earth (see the docstring for palpy.mappa)

    return palpy.dcc2s(-1.0 * params[4:7])


def solar_ra_dec(mjd, epoch=2000.0):
    """
    Return the RA and Dec of the Sun in degrees

    Parameters
    ----------
    mjd : `ModifiedJulianDate`
        is the date represented as a
        ModifiedJulianDate object.
    epoch : `float`
        is the mean epoch of the coordinate system
        (default is 2000.0)

    Returns
    ----------
    RA : `float`
        RA of Sun in degrees
    Dec : `float`
        Declination of Sun in degress
    """

    solar_ra, solar_dec = _solar_ra_dec(mjd, epoch=epoch)
    return np.degrees(solar_ra), np.degrees(solar_dec)


def _distance_to_sun(ra, dec, mjd, epoch=2000.0):
    """
    Calculate the distance from an (ra, dec) point to the Sun (in radians).

    Parameters
    ----------
    ra : `float`
        RA in radians
    dec : `float`
        Dec in radians
    mjd : `ModifiedJulianDate`
        is the date represented as a
        ModifiedJulianDate object.
    epoch : `float`
        is the epoch of the coordinate system
        (default is 2000.0)

    Returns
    -------
    distance : `float`
        on the sky to the Sun in radians
    """

    sun_ra, sun_dec = _solar_ra_dec(mjd, epoch=epoch)

    return haversine(ra, dec, sun_ra, sun_dec)


def distance_to_sun(ra, dec, mjd, epoch=2000.0):
    """
    Calculate the distance from an (ra, dec) point to the Sun (in degrees).

    Parameters
    ----------
    ra : `float`
        RA in degrees
    dec : `float`
        Dec in degrees
    mjd : `ModifiedJulianDate`
        is the date represented as a
        ModifiedJulianDate object.
    epoch : `float`
        is the epoch of the coordinate system
        (default is 2000.0)

    Returns
    -------
    distance : `float`
        on the sky to the Sun in degrees
    """

    return np.degrees(_distance_to_sun(np.radians(ra), np.radians(dec), mjd, epoch=epoch))


def refraction_coefficients(wavelength=0.5, site=None):
    """Calculate the refraction using PAL's refco routine

    This calculates the refraction at 2 angles and derives a tanz and tan^3z
    coefficient for subsequent quick calculations. Good for zenith distances < 76 degrees

    Parameters
    ----------
    wavelength : `float`
        is effective wavelength in microns (default 0.5)
    site : `Site`
        is an instantiation of the Site class defined in
        sims_utils/../Site.py

    Returns
    -------
    a, b : `float`
        Coefficients of refractions.

    Notes
    -----
    One should call PAL refz to apply the coefficients calculated here
    """
    precision = 1.0e-10

    if site is None:
        raise RuntimeError("Cannot call refraction_coefficients; no site information")

    # TODO the latitude in refco needs to be astronomical latitude,
    # not geodetic latitude
    _refco_output = palpy.refco(
        site.height,
        site.temperature_kelvin,
        site.pressure,
        site.humidity,
        wavelength,
        site.latitude_rad,
        site.lapse_rate,
        precision,
    )

    return _refco_output[0], _refco_output[1]


def apply_refraction(zenith_distance, tanz_coeff, tan3z_coeff):
    """Calculate refracted Zenith Distance

    uses the quick PAL refco routine which approximates the refractin calculation

    Parameters
    ----------
    zenith_distance : `float`
        is unrefracted zenith distance of the source in radians.
        Can either be a number or a numpy array (not a list).
    tanz_coeff : `float`
        is the first output from refraction_coefficients (above)
    tan3z_coeff : `float`
        is the second output from refraction_coefficients (above)

    Returns
    -------
    refracted_zenith : `float`
        is the refracted zenith distance in radians
    """

    if isinstance(zenith_distance, list):
        raise RuntimeError(
            "You passed a list of zenithDistances to "
            + "apply_refraction.  The method won't know how to "
            + "handle that.  Pass a numpy array."
        )

    if isinstance(zenith_distance, np.ndarray):
        refracted_zenith = palpy.refzVector(zenith_distance, tanz_coeff, tan3z_coeff)
    else:
        refracted_zenith = palpy.refz(zenith_distance, tanz_coeff, tan3z_coeff)

    return refracted_zenith


def apply_precession(ra, dec, epoch=2000.0, mjd=None):
    """
    apply_precession() applies precesion and nutation to coordinates between two epochs.
    Accepts RA and dec as inputs.  Returns corrected RA and dec (in degrees).

    Assumes FK5 as the coordinate system
    units:  ra_in (degrees), dec_in (degrees)

    The precession-nutation matrix is calculated by the palpy.prenut method
    which uses the IAU 2006/2000A model

    Parameters
    ----------
    ra : `float`
        RA in degrees
    dec : `float`
        Dec in degrees
    epoch : `float`
        is the epoch of the mean equinox (in years; default 2000)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `Unknown`
        2-D numpy array in which the first row is the RA
        corrected for precession and nutation and the second row is the
        Dec corrected for precession and nutation (both in degrees)

    """

    output = _apply_precession(np.radians(ra), np.radians(dec), epoch=epoch, mjd=mjd)

    return np.degrees(output)


def _apply_precession(ra, dec, epoch=2000.0, mjd=None):
    """
    _apply_precession() applies precesion and nutation to coordinates between two epochs.
    Accepts RA and dec as inputs.  Returns corrected RA and dec (in radians).

    Assumes FK5 as the coordinate system
    units:  ra_in (radians), dec_in (radians)

    The precession-nutation matrix is calculated by the palpy.prenut method
    which uses the IAU 2006/2000A model

    Parameters
    ----------
    ra : `float`
        RA in radians
    dec : `float`
        Dec in radians
    epoch : `float`
        is the epoch of the mean equinox (in years; default 2000)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the RA
        corrected for precession and nutation and the second row is the
        Dec corrected for precession and nutation (both in radians)
    """

    if hasattr(ra, "__len__"):
        if len(ra) != len(dec):
            raise RuntimeError("You supplied %d RAs but %d Decs to apply_precession" % (len(ra), len(dec)))

    if mjd is None:
        raise RuntimeError("You need to supply apply_precession with an mjd")

    # Determine the precession and nutation
    # palpy.prenut takes the julian epoch for the mean coordinates
    # and the MJD for the the true coordinates
    #
    # TODO it is not specified what this MJD should be (i.e. in which
    # time system it should be reckoned)
    rmat = palpy.prenut(epoch, mjd.TT)

    # Apply rotation matrix
    xyz = cartesian_from_spherical(ra, dec)
    xyz = np.dot(rmat, xyz.transpose()).transpose()

    ra_out, dec_out = spherical_from_cartesian(xyz)
    return np.array([ra_out, dec_out])


def apply_proper_motion(ra, dec, pm_ra, pm_dec, parallax, v_rad, epoch=2000.0, mjd=None):
    """Applies proper motion between two epochs.

    units:  ra (degrees), dec (degrees), pm_ra (arcsec/year), pm_dec
    (arcsec/year), parallax (arcsec), v_rad (km/sec, positive if receding),
    epoch (Julian years)

    Returns corrected ra and dec (in radians)

    The function palpy.pm does not work properly if the parallax is below
    0.00045 arcseconds

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        RA in degrees.
    dec : `float` or `np.ndarray`, (N,)
        Dec in degrees.
    pm_ra : `float` or `np.ndarray`, (N,)
        is ra proper motion multiplied by cos(Dec) in arcsec/year.
    pm_dec : `float` or `np.ndarray`, (N,)
        is dec proper motion in arcsec/year.
    parallax : `float` or `np.ndarray`, (N,)
        in arcsec. Can be a number or a numpy array (not a list).
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity in km/sec (positive if the object is receding).
    epoch : `float`
        is epoch in Julian years (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N,N)
        2-D numpy array in which the first row is the RA corrected
        for proper motion and the second row is the Dec corrected for proper motion
        (both in degrees)
    """

    output = _apply_proper_motion(
        np.radians(ra),
        np.radians(dec),
        radians_from_arcsec(pm_ra),
        radians_from_arcsec(pm_dec),
        radians_from_arcsec(parallax),
        v_rad,
        epoch=epoch,
        mjd=mjd,
    )

    return np.degrees(output)


def _apply_proper_motion(ra, dec, pm_ra, pm_dec, parallax, v_rad, epoch=2000.0, mjd=None):
    """Applies proper motion between two epochs.

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (radians), v_rad (km/sec, positive if receding),
    epoch (Julian years)

    Returns corrected ra and dec (in radians)

    The function palpy.pm does not work properly if the parallax is below
    0.00045 arcseconds

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        in radians.  Can be a number or a numpy array (not a list).
    dec : `float` or `np.ndarray`, (N,)
        in radians.  Can be a number or a numpy array (not a list).
    pm_ra : `float` or `np.ndarray`, (N,)
        is ra proper motion multiplied by cos(Dec) in radians/year.
        Can be a number or a numpy array (not a list).
    pm_dec : `float` or `np.ndarray`, (N,)
        is dec proper motion in radians/year.
        Can be a number or a numpy array (not a list).
    parallax : `float` or `np.ndarray`, (N,)
        in radians. Can be a number or a numpy array (not a list).
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity in km/sec (positive if the object is receding).
        Can be a number or a numpy array (not a list).
    epoch : `float`
        is epoch in Julian years (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `float` or `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the RA corrected
        for proper motion and the second row is the Dec corrected for proper motion
        (both in radians)
    """

    if (
        isinstance(ra, list)
        or isinstance(dec, list)
        or isinstance(pm_ra, list)
        or isinstance(pm_dec, list)
        or isinstance(parallax, list)
        or isinstance(v_rad, list)
    ):
        raise RuntimeError(
            "You tried to pass lists to applyPm. "
            + "The method does not know how to handle lists. "
            + "Use numpy arrays."
        )

    if mjd is None:
        raise RuntimeError("cannot call apply_proper_motion; mjd is None")

    parallax_arcsec = arcsec_from_radians(parallax)
    # convert to Arcsec because that is what PALPY expects

    # Generate Julian epoch from MJD
    #
    # 19 November 2015
    # I am assuming here that the time scale should be
    # Terrestrial Dynamical Time (TT), since that is used
    # as the independent variable for apparent geocentric
    # ephemerides
    julian_epoch = palpy.epj(mjd.TT)

    # because PAL and ERFA expect proper motion in terms of "coordinate
    # angle; not true angle" (as stated in erfa/starpm.c documentation)
    pm_ra_corrected = pm_ra / np.cos(dec)

    if isinstance(ra, np.ndarray):
        if (
            len(ra) != len(dec)
            or len(ra) != len(pm_ra)
            or len(ra) != len(pm_dec)
            or len(ra) != len(parallax_arcsec)
        ) or len(ra) != len(v_rad):
            raise RuntimeError(
                "You passed: "
                + "%d RAs, " % len(ra)
                + "%d Dec, " % len(dec)
                + "%d pm_ras, " % len(pm_ra)
                + "%d pm_decs, " % len(pm_dec)
                + "%d parallaxes, " % len(parallax_arcsec)
                + "%d v_rads " % len(v_rad)
                + "to applyPm; those numbers need to be identical."
            )

        ra_out, dec_out = palpy.pmVector(
            ra,
            dec,
            pm_ra_corrected,
            pm_dec,
            parallax_arcsec,
            v_rad,
            epoch,
            julian_epoch,
        )
    else:
        ra_out, dec_out = palpy.pm(
            ra,
            dec,
            pm_ra_corrected,
            pm_dec,
            parallax_arcsec,
            v_rad,
            epoch,
            julian_epoch,
        )

    return np.array([ra_out, dec_out])


def app_geo_from_icrs(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None, epoch=2000.0, mjd=None):
    """
    Convert the mean position (RA, Dec) in the International Celestial Reference
    System (ICRS) to the mean apparent geocentric position

    units:  ra (degrees), dec (degrees), pm_ra (arcsec/year), pm_dec
    (arcsec/year), parallax (arcsec), v_rad (km/sec; positive if receding),
    epoch (Julian years)

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        in degrees (ICRS).
    dec : `float` or `np.ndarray`, (N,)
        in degrees (ICRS).
    pm_ra : `float` or `np.ndarray`, (N,)
        is ra proper motion multiplied by cos(Dec) in arcsec/year
    pm_dec : `float` or `np.ndarray`, (N,)
        is dec proper motion in arcsec/year
    parallax : `float` or `np.ndarray`, (N,)
        in arcsec
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity in km/sec (positive if the object is receding)
    epoch : `float` or `np.ndarray`, (N,)
        is the julian epoch (in years) of the equinox against which to
        measure RA (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the apparent
        geocentric RA and the second row is the apparent geocentric Dec (both in degrees)
    """

    if pm_ra is not None:
        pm_ra_in = radians_from_arcsec(pm_ra)
    else:
        pm_ra_in = None

    if pm_dec is not None:
        pm_dec_in = radians_from_arcsec(pm_dec)
    else:
        pm_dec_in = None

    if parallax is not None:
        px_in = radians_from_arcsec(parallax)
    else:
        px_in = None

    output = _app_geo_from_icrs(
        np.radians(ra),
        np.radians(dec),
        pm_ra=pm_ra_in,
        pm_dec=pm_dec_in,
        parallax=px_in,
        v_rad=v_rad,
        epoch=epoch,
        mjd=mjd,
    )

    return np.degrees(output)


def _app_geo_from_icrs(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None, epoch=2000.0, mjd=None):
    """
    Convert the mean position (RA, Dec) in the International Celestial Reference
    System (ICRS) to the mean apparent geocentric position

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (radians), v_rad (km/sec; positive if receding),
    epoch (Julian years)

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        in radians (ICRS).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        in radians (ICRS).  Can be a numpy array or a number.
    pm_ra : `float` or `np.ndarray`, (N,)
        is ra proper motion multiplied by cos(Dec) in radians/year.
        Can be a numpy array or a number or None.
    pm_dec : `float` or `np.ndarray`, (N,)
        is dec proper motion in radians/year.
        Can be a numpy array or a number or None.
    parallax : `float` or `np.ndarray`, (N,)
        in radians.  Can be a numpy array or a number or None.
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity in km/sec (positive if the object is receding).
        Can be a numpy array or a number or None.
    epoch : `float`
        is the julian epoch (in years) of the equinox against which to
        measure RA (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the apparent
        geocentric RAand the second row is the apparent geocentric Dec (both in radians)
    """

    if mjd is None:
        raise RuntimeError("cannot call app_geo_from_icrs; mjd is None")

    include_px = False

    if pm_ra is not None or pm_dec is not None or v_rad is not None or parallax is not None:
        include_px = True

        if isinstance(ra, np.ndarray):
            fill_value = np.zeros(len(ra), dtype=float)
        else:
            fill_value = 0.0

        if pm_ra is None:
            pm_ra = fill_value

        if pm_dec is None:
            pm_dec = fill_value

        if v_rad is None:
            v_rad = fill_value

        if parallax is None:
            parallax = fill_value

        are_arrays = _validate_inputs(
            [ra, dec, pm_ra, pm_dec, v_rad, parallax],
            ["ra", "dec", "pm_ra", "pm_dec", "v_rad", "parallax"],
            "app_geo_from_icrs",
        )
    else:
        are_arrays = _validate_inputs([ra, dec], ["ra", "dec"], "app_geo_from_icrs")

    # Define star independent mean to apparent place parameters
    # palpy.mappa calculates the star-independent parameters
    # needed to correct RA and Dec
    # e.g the Earth barycentric and heliocentric position and velocity,
    # the precession-nutation matrix, etc.
    #
    # arguments of palpy.mappa are:
    # epoch of mean equinox to be used (Julian)
    #
    # date (MJD)
    prms = palpy.mappa(epoch, mjd.TDB)

    # palpy.mapqk does a quick mean to apparent place calculation using
    # the output of palpy.mappa
    #
    # Taken from the palpy source code (palMap.c which calls both palMappa and palMapqk):
    # The accuracy is sub-milliarcsecond, limited by the
    # precession-nutation model (see palPrenut for details).

    if include_px:
        # because PAL and ERFA expect proper motion in terms of "coordinate
        # angle; not true angle" (as stated in erfa/starpm.c documentation)
        pm_ra_corrected = pm_ra / np.cos(dec)

    if are_arrays:
        if include_px:
            ra_out, dec_out = palpy.mapqkVector(
                ra,
                dec,
                pm_ra_corrected,
                pm_dec,
                arcsec_from_radians(parallax),
                v_rad,
                prms,
            )
        else:
            ra_out, dec_out = palpy.mapqkzVector(ra, dec, prms)
    else:
        if include_px:
            ra_out, dec_out = palpy.mapqk(
                ra,
                dec,
                pm_ra_corrected,
                pm_dec,
                arcsec_from_radians(parallax),
                v_rad,
                prms,
            )
        else:
            ra_out, dec_out = palpy.mapqkz(ra, dec, prms)

    return np.array([ra_out, dec_out])


def _icrs_from_app_geo(ra, dec, epoch=2000.0, mjd=None):
    """
    Convert the apparent geocentric position in (RA, Dec) to
    the mean position in the International Celestial Reference
    System (ICRS)

    This method undoes the effects of precession, annual aberration,
    and nutation.  It is meant for mapping pointing RA and Dec (which
    presumably include the above effects) back to mean ICRS RA and Dec
    so that the user knows how to query a database of mean RA and Decs
    for objects observed at a given telescope pointing.

    WARNING: This method does not account for apparent motion due to parallax.
    This means it should not be used to invert the ICRS-to-apparent geocentric
    transformation for actual celestial objects.  This method is only useful
    for mapping positions on a theoretical celestial sphere.

    This method works in radians.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        in radians (apparent geocentric).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        in radians (apparent geocentric).  Can be a numpy array or a number.
    epoch : `float`
        is the julian epoch (in years) of the equinox against which to
        measure RA (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the mean ICRS RA and
        the second row is the mean ICRS Dec (both in radians)
    """

    are_arrays = _validate_inputs([ra, dec], ["ra", "dec"], "icrs_from_app_geo")

    # Define star independent mean to apparent place parameters
    # palpy.mappa calculates the star-independent parameters
    # needed to correct RA and Dec
    # e.g the Earth barycentric and heliocentric position and velocity,
    # the precession-nutation matrix, etc.
    #
    # arguments of palpy.mappa are:
    # epoch of mean equinox to be used (Julian)
    #
    # date (MJD)
    params = palpy.mappa(epoch, mjd.TDB)

    if are_arrays:
        ra_out, dec_out = palpy.ampqkVector(ra, dec, params)
    else:
        ra_out, dec_out = palpy.ampqk(ra, dec, params)

    return np.array([ra_out, dec_out])


def icrs_from_app_geo(ra, dec, epoch=2000.0, mjd=None):
    """
    Convert the apparent geocentric position in (RA, Dec) to
    the mean position in the International Celestial Reference
    System (ICRS)

    This method undoes the effects of precession, annual aberration,
    and nutation.  It is meant for mapping pointing RA and Dec (which
    presumably include the above effects) back to mean ICRS RA and Dec
    so that the user knows how to query a database of mean RA and Decs
    for objects observed at a given telescope pointing.

    WARNING: This method does not account for apparent motion due to parallax.
    This means it should not be used to invert the ICRS-to-apparent geocentric
    transformation for actual celestial objects.  This method is only useful
    for mapping positions on a theoretical celestial sphere.

    This method works in degrees.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        in degrees (apparent geocentric).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        in degrees (apparent geocentric).  Can be a numpy array or a number.
    epoch : `float`
        is the julian epoch (in years) of the equinox against which to
        measure RA (default: 2000.0)
    mjd : `ModifiedJulianDate`
        is an instantiation of the ModifiedJulianDate class
        representing the date of the observation

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the mean ICRS RA and
        the second row is the mean ICRS Dec (both in degrees)
    """

    ra_out, dec_out = _icrs_from_app_geo(np.radians(ra), np.radians(dec), epoch=epoch, mjd=mjd)

    return np.array([np.degrees(ra_out), np.degrees(dec_out)])


def observed_from_app_geo(
    ra, dec, include_refraction=True, alt_az_hr=False, wavelength=0.5, obs_metadata=None
):
    """
    Convert apparent geocentric (RA, Dec) to observed (RA, Dec).  More
    specifically: apply refraction and diurnal aberration.

    This method works in degrees.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is geocentric apparent RA (degrees).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is geocentric apparent Dec (degrees).  Can be a numpy array or a number.
    include_refraction : `bool`
        is a `bool` to turn refraction on and off
    alt_az_hr : `bool`
        is a `bool` indicating whether or not to return altitude
        and azimuth
    wavelength : `float`
        is effective wavelength in microns (default: 0.5)
    obs_metadata : `ObservationMetaData`
        is an ObservationMetaData characterizing the
        observation.

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the observed RA
        and the second row is the observed Dec (both in degrees)
        If alt_az_hr is True,
        2-D numpy array in which the first row is the altitude
        and the second row is the azimuth (both in degrees).
    """

    if alt_az_hr:
        ra_dec, alt_az = _observed_from_app_geo(
            np.radians(ra),
            np.radians(dec),
            include_refraction=include_refraction,
            alt_az_hr=alt_az_hr,
            wavelength=wavelength,
            obs_metadata=obs_metadata,
        )

        return np.degrees(ra_dec), np.degrees(alt_az)

    else:
        output = _observed_from_app_geo(
            np.radians(ra),
            np.radians(dec),
            include_refraction=include_refraction,
            alt_az_hr=alt_az_hr,
            wavelength=wavelength,
            obs_metadata=obs_metadata,
        )

        return np.degrees(output)


def _calculate_observatory_parameters(obs_metadata, wavelength, include_refraction):
    """
    Computer observatory-based parameters using palpy.aoppa

    Parameters
    ----------
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData characterizing
        the specific telescope site and pointing
    wavelength : `float`
        is the effective wavelength in microns
    include_refraction : `bool`
        is a `bool` indicating whether or not
        to include the effects of refraction

    Returns
    ----------
    obs_params : `np.ndarray`
        numpy array of observatory Parameters calculated by
        palpy.aoppa
    """

    # Correct site longitude for polar motion slaPolmo
    #
    # 5 January 2016
    #  palAop.c (which calls Aoppa and Aopqk, as we do here) says
    #  *     - The azimuths etc produced by the present routine are with
    #  *       respect to the celestial pole.  Corrections to the terrestrial
    #  *       pole can be computed using palPolmo.
    #
    # As a future issue, we should figure out how to incorporate polar motion
    # into these calculations.  For now, we will set polar motion to zero.
    x_polar = 0.0
    y_polar = 0.0

    #
    # palpy.aoppa computes star-independent parameters necessary for
    # converting apparent place into observed place
    # i.e. it calculates geodetic latitude, magnitude of diurnal aberration,
    # refraction coefficients and the like based on data about the observation site
    if include_refraction:
        obs_prms = palpy.aoppa(
            obs_metadata.mjd.utc,
            obs_metadata.mjd.dut1,
            obs_metadata.site.longitude_rad,
            obs_metadata.site.latitude_rad,
            obs_metadata.site.height,
            x_polar,
            y_polar,
            obs_metadata.site.temperature_kelvin,
            obs_metadata.site.pressure,
            obs_metadata.site.humidity,
            wavelength,
            obs_metadata.site.lapse_rate,
        )
    else:
        # we can discard refraction by setting pressure and humidity to zero
        obs_prms = palpy.aoppa(
            obs_metadata.mjd.utc,
            obs_metadata.mjd.dut1,
            obs_metadata.site.longitude_rad,
            obs_metadata.site.latitude_rad,
            obs_metadata.site.height,
            x_polar,
            y_polar,
            obs_metadata.site.temperature,
            0.0,
            0.0,
            wavelength,
            obs_metadata.site.lapse_rate,
        )

    return obs_prms


def _observed_from_app_geo(
    ra, dec, include_refraction=True, alt_az_hr=False, wavelength=0.5, obs_metadata=None
):
    """
    Convert apparent geocentric (RA, Dec) to observed (RA, Dec).  More specifically:
    apply refraction and diurnal aberration.

    This method works in radians.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is geocentric apparent RA (radians).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is geocentric apparent Dec (radians).  Can be a numpy array or a number.
    include_refraction : `bool`
        is a `bool` to turn refraction on and off
    alt_az_hr : `bool`
        is a `bool` indicating whether or not to return altitude
        and azimuth
    wavelength : `float`
        is effective wavelength in microns (default: 0.5)
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData characterizing the
        observation.

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the observed RA
        and the second row is the observed Dec (both in radians)
        2-D numpy array in which the first row is the altitude
        and the second row is the azimuth (both in radians).  Only returned
        if alt_az_hr == True.

    """

    are_arrays = _validate_inputs([ra, dec], ["ra", "dec"], "observed_from_app_geo")

    if obs_metadata is None:
        raise RuntimeError("Cannot call observed_from_app_geo without an obs_metadata")

    if obs_metadata.site is None:
        raise RuntimeError("Cannot call observed_from_app_geo: obs_metadata has no site info")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call observed_from_app_geo: obs_metadata has no mjd")

    obs_prms = _calculate_observatory_parameters(obs_metadata, wavelength, include_refraction)

    # palpy.aopqk does an apparent to observed place
    # correction
    #
    # it corrects for diurnal aberration and refraction
    # (using a fast algorithm for refraction in the case of
    # a small zenith distance and a more rigorous algorithm
    # for a large zenith distance)
    #

    if are_arrays:
        azimuth, zenith, hour_angle, dec_out, ra_out = palpy.aopqkVector(ra, dec, obs_prms)
    else:
        azimuth, zenith, hour_angle, dec_out, ra_out = palpy.aopqk(ra, dec, obs_prms)

    #
    # Note: this is a choke point.  Even the vectorized version of aopqk
    # is expensive (it takes about 0.006 seconds per call)
    #
    # Actually, this is only a choke point if you are dealing with zenith
    # distances of greater than about 70 degrees

    if alt_az_hr:
        #
        # palpy.de2h converts equatorial to horizon coordinates
        #
        if are_arrays:
            az, alt = palpy.de2hVector(hour_angle, dec_out, obs_metadata.site.latitude_rad)
        else:
            az, alt = palpy.de2h(hour_angle, dec_out, obs_metadata.site.latitude_rad)

        return np.array([ra_out, dec_out]), np.array([alt, az])
    return np.array([ra_out, dec_out])


def app_geo_from_observed(ra, dec, include_refraction=True, wavelength=0.5, obs_metadata=None):
    """
    Convert observed (RA, Dec) to apparent geocentric (RA, Dec).  More
    specifically: undo the effects of refraction and diurnal aberration.

    Note: This method is only accurate at zenith distances less than ~75 degrees.

    This method works in degrees.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is observed RA (degrees).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is observed Dec (degrees).  Can be a numpy array or a number.
    include_refraction : `bool`
        is a `bool` to turn refraction on and off
    wavelength : `float`
        is effective wavelength in microns (default: 0.5)
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData characterizing the
        observation.

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the apparent
        geocentric RA and the second row is the apparentGeocentric Dec (both
        in degrees)
    """

    ra_out, dec_out = _app_geo_from_observed(
        np.radians(ra),
        np.radians(dec),
        include_refraction=include_refraction,
        wavelength=wavelength,
        obs_metadata=obs_metadata,
    )

    return np.array([np.degrees(ra_out), np.degrees(dec_out)])


def _app_geo_from_observed(ra, dec, include_refraction=True, wavelength=0.5, obs_metadata=None):
    """
    Convert observed (RA, Dec) to apparent geocentric (RA, Dec).
    More specifically: undo the effects of refraction and diurnal aberration.

    Note: This method is only accurate at zenith distances less than ~ 75 degrees.

    This method works in radians.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is observed RA (radians).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is observed Dec (radians).  Can be a numpy array or a number.
    include_refraction : `bool`
        is a `bool` to turn refraction on and off
    wavelength : `float`
        is effective wavelength in microns (default: 0.5)
    obs_metadata : `rubin_sim.data.ObservationMetaData`
        is an ObservationMetaData characterizing the
        observation.

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the apparent
        geocentric RA and the second row is the apparentGeocentric Dec (both
        in radians)
    """

    are_arrays = _validate_inputs([ra, dec], ["ra", "dec"], "app_geo_from_observed")

    if obs_metadata is None:
        raise RuntimeError("Cannot call app_geo_from_observed without an obs_metadata")

    if obs_metadata.site is None:
        raise RuntimeError("Cannot call app_geo_from_observed: obs_metadata has no site info")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call app_geo_from_observed: obs_metadata has no mjd")

    obs_prms = _calculate_observatory_parameters(obs_metadata, wavelength, include_refraction)

    if are_arrays:
        ra_out, dec_out = palpy.oapqkVector("r", ra, dec, obs_prms)
    else:
        ra_out, dec_out = palpy.oapqk("r", ra, dec, obs_prms)

    return np.array([ra_out, dec_out])


def observed_from_icrs(
    ra,
    dec,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
    v_rad=None,
    obs_metadata=None,
    epoch=None,
    include_refraction=True,
):
    """
    Convert mean position (RA, Dec) in the International Celestial Reference Frame
    to observed (RA, Dec).

    included are precession-nutation, aberration, proper motion, parallax, refraction,
    radial velocity, diurnal aberration.

    This method works in degrees.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is the unrefracted RA in degrees (ICRS).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is the unrefracted Dec in degrees (ICRS).  Can be a numpy array or a number.
    pm_ra : `float` or `np.ndarray`, (N,)
        is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
        Can be a numpy array or a number or None (default=None).
    pm_dec : `float` or `np.ndarray`, (N,)
        is proper motion in dec (arcsec/yr)
        Can be a numpy array or a number or None (default=None).
    parallax : `float` or `np.ndarray`, (N,)
        is parallax in arcsec
        Can be a numpy array or a number or None (default=None).
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData object describing the
        telescope pointing.
    epoch : `float`
        is the julian epoch (in years) against which the mean
        equinoxes are measured.
    include_refraction : `bool`
        toggles whether or not to correct for refraction

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the observed
        RA and the second row is the observed Dec (both in degrees)
    """

    if pm_ra is not None:
        pm_ra_in = radians_from_arcsec(pm_ra)
    else:
        pm_ra_in = None

    if pm_dec is not None:
        pm_dec_in = radians_from_arcsec(pm_dec)
    else:
        pm_dec_in = None

    if parallax is not None:
        parallax_in = radians_from_arcsec(parallax)
    else:
        parallax_in = None

    output = _observed_from_icrs(
        np.radians(ra),
        np.radians(dec),
        pm_ra=pm_ra_in,
        pm_dec=pm_dec_in,
        parallax=parallax_in,
        v_rad=v_rad,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    return np.degrees(output)


def _observed_from_icrs(
    ra,
    dec,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
    v_rad=None,
    obs_metadata=None,
    epoch=None,
    include_refraction=True,
):
    """
    Convert mean position (RA, Dec) in the International Celestial Reference Frame
    to observed (RA, Dec)-like coordinates.

    included are precession-nutation, aberration, proper motion, parallax, refraction,
    radial velocity, diurnal aberration.

    This method works in radians.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is the unrefracted RA in radians (ICRS).  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is the unrefracted Dec in radians (ICRS).  Can be a numpy array or a number.
    pm_ra : `float` or `np.ndarray`, (N,)
        is proper motion in RA multiplied by cos(Dec) (radians/yr)
        Can be a numpy array or a number or None (default=None).
    pm_dec : `float` or `np.ndarray`, (N,)
        is proper motion in dec (radians/yr)
        Can be a numpy array or a number or None (default=None).
    parallax : `float` or `np.ndarray`, (N,)
        is parallax in radians
        Can be a numpy array or a number or None (default=None).
    v_rad : `float` or `np.ndarray`, (N,)
        is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData object describing the
        telescope pointing.
    epoch : `float`
        is the julian epoch (in years) against which the mean
        equinoxes are measured.
    include_refraction : `bool`
        toggles whether or not to correct for refraction

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the observed
        RA and the second row is the observed Dec (both in radians)
    """

    if obs_metadata is None:
        raise RuntimeError("Cannot call observed_from_icrs; obs_metadata is none")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call observed_from_icrs; obs_metadata.mjd is none")

    if epoch is None:
        raise RuntimeError("Cannot call observed_from_icrs; you have not specified an epoch")

    ra_apparent, dec_apparent = _app_geo_from_icrs(
        ra,
        dec,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        parallax=parallax,
        v_rad=v_rad,
        epoch=epoch,
        mjd=obs_metadata.mjd,
    )

    ra_out, dec_out = _observed_from_app_geo(
        ra_apparent,
        dec_apparent,
        obs_metadata=obs_metadata,
        include_refraction=include_refraction,
    )

    return np.array([ra_out, dec_out])


def icrs_from_observed(ra, dec, obs_metadata=None, epoch=None, include_refraction=True):
    """
    Convert observed RA, Dec into mean International Celestial Reference Frame (ICRS)
    RA, Dec.  This method undoes the effects of precession, nutation, aberration (annual
    and diurnal), and refraction.  It is meant so that users can take pointing RA and Decs,
    which will be in the observed reference system, and transform them into ICRS for
    purposes of querying database tables (likely to contain mean ICRS RA, Dec) for objects
    visible from a given pointing.

    Note: This method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.

    WARNING: This method does not account for apparent motion due to parallax.
    This means it should not be used to invert the ICRS-to-observed coordinates
    transformation for actual celestial objects.  This method is only useful
    for mapping positions on a theoretical celestial sphere.

    This method works in degrees.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is the observed RA in degrees.  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is the observed Dec in degrees.  Can be a numpy array or a number.
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData object describing the
        telescope pointing.
    epoch : `float`
        is the julian epoch (in years) against which the mean
        equinoxes are measured.
    include_refraction : `bool`
        toggles whether or not to correct for refraction

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the mean ICRS
        RA and the second row is the mean ICRS Dec (both in degrees)
    """

    ra_out, dec_out = _icrs_from_observed(
        np.radians(ra),
        np.radians(dec),
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    return np.array([np.degrees(ra_out), np.degrees(dec_out)])


def _icrs_from_observed(ra, dec, obs_metadata=None, epoch=None, include_refraction=True):
    """
    Convert observed RA, Dec into mean International Celestial Reference Frame (ICRS)
    RA, Dec.  This method undoes the effects of precession, nutation, aberration (annual
    and diurnal), and refraction.  It is meant so that users can take pointing RA and Decs,
    which will be in the observed reference system, and transform them into ICRS for
    purposes of querying database tables (likely to contain mean ICRS RA, Dec) for objects
    visible from a given pointing.

    Note: This method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.

    WARNING: This method does not account for apparent motion due to parallax.
    This means it should not be used to invert the ICRS-to-observed coordinates
    transformation for actual celestial objects.  This method is only useful
    for mapping positions on a theoretical celestial sphere.

    This method works in radians.

    Parameters
    ----------
    ra : `float` or `np.ndarray`, (N,)
        is the observed RA in radians.  Can be a numpy array or a number.
    dec : `float` or `np.ndarray`, (N,)
        is the observed Dec in radians.  Can be a numpy array or a number.
    obs_metadata : `rubin_sim.utils.ObservationMetaData`
        is an ObservationMetaData object describing the
        telescope pointing.
    epoch : `float`
        is the julian epoch (in years) against which the mean
        equinoxes are measured.
    include_refraction : `bool`
        toggles whether or not to correct for refraction

    Returns
    -------
    a : `np.ndarray`, (N, N)
        2-D numpy array in which the first row is the mean ICRS
        RA and the second row is the mean ICRS Dec (both in radians)
    """

    _validate_inputs([ra, dec], ["ra", "dec"], "icrs_from_observed")

    if obs_metadata is None:
        raise RuntimeError("Cannot call icrs_from_observed; obs_metadata is None")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call icrs_from_observed; obs_metadata.mjd is None")

    if epoch is None:
        raise RuntimeError("Cannot call icrs_from_observed; you have not specified an epoch")

    ra_app, dec_app = _app_geo_from_observed(
        ra, dec, obs_metadata=obs_metadata, include_refraction=include_refraction
    )

    ra_icrs, dec_icrs = _icrs_from_app_geo(ra_app, dec_app, epoch=epoch, mjd=obs_metadata.mjd)

    return np.array([ra_icrs, dec_icrs])
