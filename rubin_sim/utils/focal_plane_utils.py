__all__ = (
    "_pupil_coords_from_observed",
    "_pupil_coords_from_ra_dec",
    "pupil_coords_from_ra_dec",
    "_ra_dec_from_pupil_coords",
    "ra_dec_from_pupil_coords",
    "_observed_from_pupil_coords",
    "observed_from_pupil_coords",
)

import numpy as np
import palpy

from rubin_sim.utils import _icrs_from_observed, _observed_from_icrs, radians_from_arcsec
from rubin_sim.utils.code_utilities import _validate_inputs


def pupil_coords_from_ra_dec(
    ra_in,
    dec_in,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
    v_rad=None,
    include_refraction=True,
    obs_metadata=None,
    epoch=2000.0,
):
    """
    Take an input RA and dec from the sky and convert it to coordinates
    on the focal plane.

    This uses PAL's gnomonic projection routine which assumes that the focal
    plane is perfectly flat.  The output is in Cartesian coordinates, assuming
    that the Celestial Sphere is a unit sphere.

    The RA, Dec accepted by this method are in the International Celestial
    Reference System.  Before applying the gnomonic projection, this method
    transforms those RA, Dec into observed geocentric coordinates, applying
    the effects of precession, nutation, aberration, parallax and refraction.
    This is done, because the gnomonic projection ought to be applied to what
    observers actually see, rather than the idealized, above-the-atmosphere
    coordinates represented by the ICRS.

    Parameters
    ----------
    ra_in : `Unknown`
        is in degrees (ICRS).  Can be either a numpy array or a number.
    dec_in : `Unknown`
        is in degrees (ICRS).  Can be either a numpy array or a number.
    pm_ra : `Unknown`
        is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
        Can be a numpy array or a number or None (default=None).
    pm_dec : `Unknown`
        is proper motion in dec (arcsec/yr)
        Can be a numpy array or a number or None (default=None).
    parallax : `Unknown`
        is parallax in arcsec
        Can be a numpy array or a number or None (default=None).
    v_rad : `Unknown`
        is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).
    include_refraction : `Unknown`
        is a `bool` controlling the application of refraction.
    obs_metadata : `Unknown`
        is an ObservationMetaData instantiation characterizing the
        telescope location and pointing.
    epoch : `Unknown`
        is the epoch of mean ra and dec in julian years (default=2000.0)
    returns : `Unknown`
        a numpy array whose first row is the x coordinate on the pupil in
        radians and whose second row is the y coordinate in radians
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

    return _pupil_coords_from_ra_dec(
        np.radians(ra_in),
        np.radians(dec_in),
        pm_ra=pm_ra_in,
        pm_dec=pm_dec_in,
        parallax=parallax_in,
        v_rad=v_rad,
        include_refraction=include_refraction,
        obs_metadata=obs_metadata,
        epoch=epoch,
    )


def _pupil_coords_from_ra_dec(
    ra_in,
    dec_in,
    pm_ra=None,
    pm_dec=None,
    parallax=None,
    v_rad=None,
    include_refraction=True,
    obs_metadata=None,
    epoch=2000.0,
):
    """
    Take an input RA and dec from the sky and convert it to coordinates
    on the focal plane.

    This uses PAL's gnomonic projection routine which assumes that the focal
    plane is perfectly flat.  The output is in Cartesian coordinates, assuming
    that the Celestial Sphere is a unit sphere.

    The RA, Dec accepted by this method are in the International Celestial
    Reference System.  Before applying the gnomonic projection, this method
    transforms those RA, Dec into observed geocentric coordinates, applying
    the effects of precession, nutation, aberration, parallax and refraction.
    This is done, because the gnomonic projection ought to be applied to what
    observers actually see, rather than the idealized, above-the-atmosphere
    coordinates represented by the ICRS.

    Parameters
    ----------
    ra_in : `Unknown`
        is in radians (ICRS).  Can be either a numpy array or a number.
    dec_in : `Unknown`
        is in radians (ICRS).  Can be either a numpy array or a number.
    pm_ra : `Unknown`
        is proper motion in RA multiplied by cos(Dec) (radians/yr)
        Can be a numpy array or a number or None (default=None).
    pm_dec : `Unknown`
        is proper motion in dec (radians/yr)
        Can be a numpy array or a number or None (default=None).
    parallax : `Unknown`
        is parallax in radians
        Can be a numpy array or a number or None (default=None).
    v_rad : `Unknown`
        is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).
    include_refraction : `Unknown`
        is a `bool` controlling the application of refraction.
    obs_metadata : `Unknown`
        is an ObservationMetaData instantiation characterizing the
        telescope location and pointing.
    epoch : `Unknown`
        is the epoch of mean ra and dec in julian years (default=2000.0)
    returns : `Unknown`
        a numpy array whose first row is the x coordinate on the pupil in
        radians and whose second row is the y coordinate in radians
    """

    are_arrays = _validate_inputs([ra_in, dec_in], ["ra_in", "dec_in"], "pupil_coords_from_ra_dec")

    if obs_metadata is None:
        raise RuntimeError("Cannot call pupil_coords_from_ra_dec without obs_metadata")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call pupil_coords_from_ra_dec; obs_metadata.mjd is None")

    if epoch is None:
        raise RuntimeError("Cannot call pupil_coords_from_ra_dec; epoch is None")

    if obs_metadata.rot_sky_pos is None:
        # there is no observation meta data on which to base astrometry
        raise RuntimeError("Cannot calculate [x,y]_focal_nominal without obs_metadata.rot_sky_pos")

    if obs_metadata.pointing_ra is None or obs_metadata.pointing_dec is None:
        raise RuntimeError("Cannot calculate [x,y]_focal_nominal without pointing_ra and Dec in obs_metadata")

    ra_obs, dec_obs = _observed_from_icrs(
        ra_in,
        dec_in,
        pm_ra=pm_ra,
        pm_dec=pm_dec,
        parallax=parallax,
        v_rad=v_rad,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    return _pupil_coords_from_observed(
        ra_obs,
        dec_obs,
        obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )


def _pupil_coords_from_observed(ra_obs, dec_obs, obs_metadata, epoch=2000.0, include_refraction=True):
    """
    Convert Observed RA, Dec into pupil coordinates

    Parameters
    ----------
    ra_obs is the observed RA in radians

    dec_obs is the observed Dec in radians

    obs_metadata is an ObservationMetaData characterizing the telescope location and pointing

    epoch is the epoch of the mean RA and Dec in julian years (default=2000.0)

    include_refraction is a `bool` controlling the application of refraction.

    Returns
    --------
    A numpy array whose first row is the x coordinate on the pupil in
    radians and whose second row is the y coordinate in radians
    """

    are_arrays = _validate_inputs([ra_obs, dec_obs], ["ra_obs", "dec_obs"], "pupilCoordsFromObserved")

    if obs_metadata.rot_sky_pos is None:
        raise RuntimeError("Cannot call pupilCoordsFromObserved; " "rot_sky_pos is None")

    theta = -1.0 * obs_metadata._rot_sky_pos

    ra_pointing, dec_pointing = _observed_from_icrs(
        obs_metadata._pointing_ra,
        obs_metadata._pointing_dec,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    # palpy.ds2tp performs the gnomonic projection on ra_in and dec_in
    # with a tangent point at (pointing_ra, pointing_dec)
    #
    if not are_arrays:
        try:
            x, y = palpy.ds2tp(ra_obs, dec_obs, ra_pointing, dec_pointing)
        except:
            x = np.NaN
            y = np.NaN
    else:
        try:
            x, y = palpy.ds2tpVector(ra_obs, dec_obs, ra_pointing, dec_pointing)
        except:
            # apparently, one of your ra/dec values was improper; we will have to do this
            # element-wise, putting NaN in the place of the bad values
            x = []
            y = []
            for rr, dd in zip(ra_obs, dec_obs):
                try:
                    xx, yy = palpy.ds2tp(rr, dd, ra_pointing, dec_pointing)
                except:
                    xx = np.NaN
                    yy = np.NaN
                x.append(xx)
                y.append(yy)
            x = np.array(x)
            y = np.array(y)

    # rotate the result by rotskypos (rotskypos being "the angle of the sky relative to
    # camera coordinates" according to phoSim documentation) to account for
    # the rotation of the focal plane about the telescope pointing

    x_out = x * np.cos(theta) - y * np.sin(theta)
    y_out = x * np.sin(theta) + y * np.cos(theta)

    return np.array([x_out, y_out])


def _observed_from_pupil_coords(x_pupil, y_pupil, obs_metadata=None, include_refraction=True, epoch=2000.0):
    """
    Convert pupil coordinates into observed (RA, Dec)

    Parameters
    ----------
    x_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    y_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    obs_metadata : `Unknown`
        -- an instantiation of ObservationMetaData characterizing
        the state of the telescope
    epoch : `Unknown`
        -- julian epoch of the mean equinox used for the coordinate
        transformations (in years; defaults to 2000)
    include_refraction : `Unknown`
        -- a `bool` which controls the effects of refraction
        (refraction is used when finding the observed coordinates of the boresite specified
        by obs_metadata)
    a : `Unknown`
        2-D numpy array in which the first row is observed RA and the second
        row is observed Dec (both in radians).  Note: these are not ICRS coordinates.
        These are RA and Dec-like coordinates resulting from applying precession, nutation,
        diurnal aberration and annual aberration on top of ICRS coordinates.

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plane
    to positions on the celestial sphere.
    """

    are_arrays = _validate_inputs([x_pupil, y_pupil], ["x_pupil", "y_pupil"], "observed_from_pupil_coords")

    if obs_metadata is None:
        raise RuntimeError("Cannot call observed_from_pupil_coords without obs_metadata")

    if epoch is None:
        raise RuntimeError("Cannot call observed_from_pupil_coords; epoch is None")

    if obs_metadata.rot_sky_pos is None:
        raise RuntimeError("Cannot call observed_from_pupil_coords without rot_sky_pos " + "in obs_metadata")

    if obs_metadata.pointing_ra is None or obs_metadata.pointing_dec is None:
        raise RuntimeError(
            "Cannot call observed_from_pupil_coords " + "without pointing_ra, pointing_dec in obs_metadata"
        )

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot calculate RA, Dec without mjd " + "in obs_metadata")

    ra_pointing, dec_pointing = _observed_from_icrs(
        obs_metadata._pointing_ra,
        obs_metadata._pointing_dec,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    # This is the same as theta in pupil_coords_from_ra_dec, except without the minus sign.
    # This is because we will be reversing the rotation performed in that other method.
    theta = obs_metadata._rot_sky_pos

    x_g = x_pupil * np.cos(theta) - y_pupil * np.sin(theta)
    y_g = x_pupil * np.sin(theta) + y_pupil * np.cos(theta)

    # x_g and y_g are now the x and y coordinates
    # can now use the PALPY method palDtp2s to convert to RA, Dec.

    if are_arrays:
        ra_obs, dec_obs = palpy.dtp2sVector(x_g, y_g, ra_pointing, dec_pointing)
    else:
        ra_obs, dec_obs = palpy.dtp2s(x_g, y_g, ra_pointing, dec_pointing)

    return ra_obs, dec_obs


def observed_from_pupil_coords(x_pupil, y_pupil, obs_metadata=None, include_refraction=True, epoch=2000.0):
    """
    Convert pupil coordinates into observed (RA, Dec)

    Parameters
    ----------
    x_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    y_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    obs_metadata : `Unknown`
        -- an instantiation of ObservationMetaData characterizing
        the state of the telescope
    epoch : `Unknown`
        -- julian epoch of the mean equinox used for the coordinate
        transformations (in years; defaults to 2000)
    include_refraction : `Unknown`
        -- a `bool` which controls the effects of refraction
        (refraction is used when finding the observed coordinates of the boresite specified
        by obs_metadata)
    a : `Unknown`
        2-D numpy array in which the first row is observed RA and the second
        row is observed Dec (both in degrees).  Note: these are not ICRS coordinates.
        These are RA and Dec-like coordinates resulting from applying precession, nutation,
        diurnal aberration and annual aberration on top of ICRS coordinates.

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plane
    to positions on the celestial sphere.
    """
    ra_rad, dec_rad = _observed_from_pupil_coords(
        x_pupil,
        y_pupil,
        obs_metadata=obs_metadata,
        include_refraction=include_refraction,
        epoch=2000.0,
    )

    return np.degrees(ra_rad), np.degrees(dec_rad)


def ra_dec_from_pupil_coords(x_pupil, y_pupil, obs_metadata=None, include_refraction=True, epoch=2000.0):
    """
    Parameters
    ----------
    x_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    y_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    obs_metadata : `Unknown`
        -- an instantiation of ObservationMetaData characterizing
        the state of the telescope
    include_refraction : `Unknown`
        -- a `bool` which controls the effects of refraction
        (refraction is used when finding the observed coordinates of the boresite specified
        by obs_metadata)
    epoch : `Unknown`
        -- julian epoch of the mean equinox used for the coordinate
        transformations (in years; defaults to 2000)
    a : `Unknown`
        2-D numpy array in which the first row is RA and the second
        row is Dec (both in degrees; both in the International Celestial Reference System)

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plane
    to positions on the celestial sphere.
    """

    output = _ra_dec_from_pupil_coords(
        x_pupil,
        y_pupil,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    return np.degrees(output)


def _ra_dec_from_pupil_coords(x_pupil, y_pupil, obs_metadata=None, include_refraction=True, epoch=2000.0):
    """
    Parameters
    ----------
    x_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    y_pupil : `Unknown`
        -- pupil coordinates in radians.
        Can be a numpy array or a number.
    obs_metadata : `Unknown`
        -- an instantiation of ObservationMetaData characterizing
        the state of the telescope
    include_refraction : `Unknown`
        -- a `bool` which controls the effects of refraction
        (refraction is used when finding the observed coordinates of the boresite specified
        by obs_metadata)
    epoch : `Unknown`
        -- julian epoch of the mean equinox used for the coordinate
        transformations (in years; defaults to 2000)
    a : `Unknown`
        2-D numpy array in which the first row is RA and the second
        row is Dec (both in radians; both in the International Celestial Reference System)

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical focal plane
    to positions on the celestial sphere.
    """

    are_arrays = _validate_inputs([x_pupil, y_pupil], ["x_pupil", "y_pupil"], "ra_dec_from_pupil_coords")

    if obs_metadata is None:
        raise RuntimeError("Cannot call ra_dec_from_pupil_coords without obs_metadata")

    if epoch is None:
        raise RuntimeError("Cannot call ra_dec_from_pupil_coords; epoch is None")

    if obs_metadata.rot_sky_pos is None:
        raise RuntimeError("Cannot call ra_dec_from_pupil_coords without rot_sky_pos " + "in obs_metadata")

    if obs_metadata.pointing_ra is None or obs_metadata.pointing_dec is None:
        raise RuntimeError(
            "Cannot call ra_dec_from_pupil_coords " + "without pointing_ra, pointing_dec in obs_metadata"
        )

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot calculate RA, Dec without mjd " + "in obs_metadata")

    ra_obs, dec_obs = _observed_from_pupil_coords(
        x_pupil,
        y_pupil,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    ra_icrs, dec_icrs = _icrs_from_observed(
        ra_obs,
        dec_obs,
        obs_metadata=obs_metadata,
        epoch=epoch,
        include_refraction=include_refraction,
    )

    return np.array([ra_icrs, dec_icrs])
