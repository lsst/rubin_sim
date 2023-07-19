"""Second generation hourglass plotting classes.
"""
__all__ = ("riseset_times",)
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

# imports
import warnings

import astropy.coordinates
import astropy.time
import numpy as np
from astropy import units as u
from astropy.utils import iers

# constants

# exception classes

# interface functions


def riseset_times(  # pylint: disable=too-many-locals
    night_mjds,
    which_direction="down",
    which_night="nearest",
    alt=-14.0,
    location=astropy.coordinates.EarthLocation.of_site("Cerro Pachon"),
    body="sun",
    tolerance=1e-8,
    max_iter=10,
):
    """Find morning or evening twilight using Newton's iterative method

    This fails near the poles!

    Args:
        night_mjds : `numpy.array`
           MJDs of nights (integers)
        which_direction : `str`
           'evening' or 'morning'
        which_night : `str`
           'previous', 'nearest', or 'next'
        alt : `float`
            altitude of twilight, in degrees
        location : `astropy.coordinates.EarthLocation`
            location for site
        tolerance : `float`
            tolerance for twilight altitude, in degrees
        max_iter : `int`
            maximum iterations in Newton's method

    Returns:
        numpy array of mjds

    """

    # the astropy sidereal time routines often fail because they rely on
    # online updates to the IERS table, which is usually pointless
    # for this purpose. Use the formula from Meeus instead.
    def _compute_lst(times):
        try:
            lst = times.sidereal_time("apparent", longitude=location.lon)
        except (iers.IERSRangeError, ValueError):
            warnings.warn("Using (lower precision) Meeus formula for sidereal time.")
            mjd = times.mjd
            century = (mjd - 51544.5) / 36525
            gst = (
                280.46061837
                + 360.98564736629 * (mjd - 51544.5)
                + 0.000387933 * century * century
                - century * century * century / 38710000
            )
            lst = ((gst + location.lon.deg) % 360) * u.deg  # pylint: disable=no-member

        return lst

    event_direction = 1 if which_direction == "down" else -1

    night_wraps = {"previous": 0.0, "nearest": 180.0, "next": 360.0}
    night_wrap = night_wraps[which_night]

    mjds = night_mjds

    # Get close (to of order body motion per day)
    times = astropy.time.Time(mjds, scale="utc", format="mjd", location=location)
    lsts = _compute_lst(times)
    crds = astropy.coordinates.get_body(body, times, location=location)
    hour_angles = lsts - crds.ra
    event_hour_angles = event_direction * np.arccos(
        (np.sin(np.radians(alt)) - np.sin(crds.dec) * np.sin(location.lat))
        / (np.cos(crds.dec) * np.cos(location.lat))
    )
    event_hour_angles = astropy.coordinates.Angle(
        event_hour_angles, unit=u.radian  # pylint: disable=no-member
    )
    ha_diff = (event_hour_angles - hour_angles).wrap_at(night_wrap * u.deg)  # pylint: disable=no-member
    mjds = mjds + ha_diff.radian * (0.9972696 / (2 * np.pi))

    # Refine using Newton's method
    for iter_idx in range(max_iter):  # pylint: disable=unused-variable
        times = astropy.time.Time(mjds, scale="utc", format="mjd", location=location)
        crds = astropy.coordinates.get_body(body, times, location=location)
        current_alt = crds.transform_to(astropy.coordinates.AltAz(obstime=times, location=location)).alt
        finished = np.max(np.abs(current_alt.deg - alt)) < tolerance
        if finished:
            break

        current_sinalt = np.sin(current_alt.rad)
        target_sinalt = np.sin(np.radians(alt))

        ha = _compute_lst(times) - crds.ra  # pylint: disable=invalid-name
        # Derivative of the standard formula for sin(alt) in terms of decl, latitude, and HA
        dsinalt_dlst = (-1 * np.cos(crds.dec) * np.cos(location.lat) * np.sin(ha)).value
        dsinalt_dmjd = dsinalt_dlst * (2 * np.pi / 0.9972696)
        mjds = mjds - (current_sinalt - target_sinalt) / dsinalt_dmjd

    if np.max(np.abs(mjds - night_mjds)) > 1:
        warnings.warn("On some nights, found twilight more than a day away from the night mjd")

    if not finished:
        unfinished = np.abs(current_alt.deg - alt) < tolerance
        if np.any(unfinished):
            try:
                first_unfinished_mjd = night_mjds[unfinished][0]
            except IndexError:
                first_unfinished_mjd = night_mjds[unfinished]
            warnings.warn(f"twilight_times did not converge, starting with {first_unfinished_mjd}")

    return mjds


# classes

# internal functions & classes
