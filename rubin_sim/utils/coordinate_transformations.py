"""
This file contains coordinate transformation methods that are very thin wrappers
of palpy methods, or that have no dependence on palpy at all
"""
__all__ = (
    "_galactic_from_equatorial",
    "galactic_from_equatorial",
    "_equatorial_from_galactic",
    "equatorial_from_galactic",
    "spherical_from_cartesian",
    "cartesian_from_spherical",
    "xyz_from_ra_dec",
    "_xyz_from_ra_dec",
    "_ra_dec_from_xyz",
    "ra_dec_from_xyz",
    "xyz_angular_radius",
    "_xyz_angular_radius",
    "rotation_matrix_from_vectors",
    "rot_about_z",
    "rot_about_y",
    "rot_about_x",
    "equation_of_equinoxes",
    "calc_gmst_gast",
    "calc_lmst_last",
    "angular_separation",
    "_angular_separation",
    "haversine",
    "arcsec_from_radians",
    "radians_from_arcsec",
    "arcsec_from_degrees",
    "degrees_from_arcsec",
)

import numbers

import numpy as np
import palpy

from rubin_sim.utils.code_utilities import _validate_inputs


def calc_lmst_last(mjd, long_rad):
    """
    calculates local mean sidereal time and local apparent sidereal time

    Parameters
    ----------
    mjd : `Unknown`
        is the universal time (ut1) expressed as an MJD.
        This can be a numpy array or a single value.
    long_rad : `Unknown`
        is the longitude in radians (positive east of the prime meridian)
        This can be numpy array or a single value.  If a numpy array, should have the same length as mjd.  In that
        case, each long_rad will be applied only to the corresponding mjd.
    lmst : `Unknown`
        is the local mean sidereal time in hours
    last : `Unknown`
        is the local apparent sideral time in hours
    """
    mjd_is_array = False
    long_rad_is_array = False
    if isinstance(mjd, np.ndarray):
        mjd_is_array = True

    if isinstance(long_rad, np.ndarray):
        long_rad_is_array = True

    if long_rad_is_array and mjd_is_array:
        if len(long_rad) != len(mjd):
            raise RuntimeError("In calc_lmst_last mjd and long_rad have different lengths")

    valid_type = False
    if isinstance(mjd, np.ndarray) and isinstance(long_rad, np.ndarray):
        valid_type = True
    elif isinstance(mjd, np.ndarray) and isinstance(long_rad, numbers.Number):
        valid_type = True
    elif isinstance(mjd, numbers.Number) and isinstance(long_rad, numbers.Number):
        valid_type = True

    if not valid_type:
        msg = (
            "Valid input types for calc_lmst_last are:\n"
            "mjd and long_rad as numpy arrays of the same length\n"
            "mjd as a numpy array and long_rad as a number\n"
            "mjd as a number and long_rad as a number\n"
            "You gave mjd: %s\n" % type(mjd) + "and long_rad: %s\n" % type(long_rad)
        )

        raise RuntimeError(msg)

    long_deg0 = np.degrees(long_rad)
    long_deg0 %= 360.0

    if long_rad_is_array:
        long_deg = np.where(long_deg0 > 180.0, long_deg0 - 360.0, long_deg0)
    else:
        if long_deg0 > 180.0:
            long_deg = long_deg0 - 360.0
        else:
            long_deg = long_deg0

    hrs = long_deg / 15.0
    gmstgast = calc_gmst_gast(mjd)
    lmst = gmstgast[0] + hrs
    last = gmstgast[1] + hrs
    lmst %= 24.0
    last %= 24.0
    return lmst, last


def galactic_from_equatorial(ra, dec):
    """Convert RA,Dec (J2000) to Galactic Coordinates

    Parameters
    ----------
    ra : `Unknown`
        is right ascension in degrees, either a number or a numpy array
    dec : `Unknown`
        is declination in degrees, either a number or a numpy array
    g_long : `Unknown`
        is galactic longitude in degrees
    g_lat : `Unknown`
        is galactic latitude in degrees
    """

    g_long, g_lat = _galactic_from_equatorial(np.radians(ra), np.radians(dec))
    return np.degrees(g_long), np.degrees(g_lat)


def _galactic_from_equatorial(ra, dec):
    """Convert RA,Dec (J2000) to Galactic Coordinates

    All angles are in radians

    Parameters
    ----------
    ra : `Unknown`
        is right ascension in radians, either a number or a numpy array
    dec : `Unknown`
        is declination in radians, either a number or a numpy array
    g_long : `Unknown`
        is galactic longitude in radians
    g_lat : `Unknown`
        is galactic latitude in radians
    """

    if isinstance(ra, np.ndarray):
        g_long, g_lat = palpy.eqgalVector(ra, dec)
    else:
        g_long, g_lat = palpy.eqgal(ra, dec)

    return g_long, g_lat


def equatorial_from_galactic(g_long, g_lat):
    """Convert Galactic Coordinates to RA, dec (J2000)

    Parameters
    ----------
    g_long : `Unknown`
        is galactic longitude in degrees, either a number or a numpy array
        (0 <= g_long <= 360.)
    g_lat : `Unknown`
        is galactic latitude in degrees, either a number or a numpy array
        (-90. <= g_lat <= 90.)
    ra : `Unknown`
        is right ascension in degrees
    dec : `Unknown`
        is declination in degrees
    """

    ra, dec = _equatorial_from_galactic(np.radians(g_long), np.radians(g_lat))
    return np.degrees(ra), np.degrees(dec)


def _equatorial_from_galactic(g_long, g_lat):
    """Convert Galactic Coordinates to RA, dec (J2000)

    Parameters
    ----------
    g_long : `Unknown`
        is galactic longitude in radians, either a number or a numpy array
        (0 <= g_long <= 2*pi)
    g_lat : `Unknown`
        is galactic latitude in radians, either a number or a numpy array
        (-pi/2 <= g_lat <= pi/2)
    ra : `Unknown`
        is right ascension in radians (J2000)
    dec : `Unknown`
        is declination in radians (J2000)
    """

    if isinstance(g_long, np.ndarray):
        ra, dec = palpy.galeqVector(g_long, g_lat)
    else:
        ra, dec = palpy.galeq(g_long, g_lat)

    return ra, dec


def cartesian_from_spherical(longitude, latitude):
    """
    Transforms between spherical and Cartesian coordinates.

    Parameters
    ----------
    longitude : `Unknown`
        is a numpy array or a number in radians
    latitude : `Unknown`
        is a numpy array or number in radians
    a : `Unknown`
        numpy array of the (three-dimensional) cartesian coordinates on a unit sphere.

    if inputs are numpy arrays:
    output[i][0] will be the x-coordinate of the ith point
    output[i][1] will be the y-coordinate of the ith point
    output[i][2] will be the z-coordinate of the ith point

    All angles are in radians

    Also, look at xyz_from_ra_dec().
    """
    return _xyz_from_ra_dec(longitude, latitude).transpose()


def spherical_from_cartesian(xyz):
    """
    Transforms between Cartesian and spherical coordinates

    Parameters
    ----------
    xyz : `Unknown`
        is a numpy array of points in 3-D space.
        Each row is a different point.
    returns : `Unknown`
        longitude and latitude

    All angles are in radians

    Also, look at ra_dec_from_xyz().
    """
    if not isinstance(xyz, np.ndarray):
        raise RuntimeError("You need to pass a numpy array to spherical_from_cartesian")

    if len(xyz.shape) > 1:
        return _ra_dec_from_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    else:
        return _ra_dec_from_xyz(xyz[0], xyz[1], xyz[2])


def xyz_from_ra_dec(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space.

    Parameters
    ----------
    ra : float or array
        RA in degrees
    dec : float or array
        Dec in degrees

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    return _xyz_from_ra_dec(np.radians(ra), np.radians(dec))


def _xyz_from_ra_dec(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # It is ok to mix floats and numpy arrays.

    cos_dec = np.cos(dec)
    return np.array([np.cos(ra) * cos_dec, np.sin(ra) * cos_dec, np.sin(dec)])


def _ra_dec_from_xyz(x, y, z):
    """
    Utility to convert x,y,z Cartesian coordinates to RA, dec positions in space.

    Parameters
    ----------
    x : float or array
        The position on the x-axis of the given points on the unit sphere
    y : float or array
        The position on the y-axis of the given points on the unit sphere
    z : float or array
        The position on the z-axis of the given points on the unit sphere

    Returns
    -------
    ra, dec : floats or arrays
        Ra and dec coordinates in radians.
    """
    rad = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / rad)

    return ra, dec


def ra_dec_from_xyz(x, y, z):
    """
    Utility to convert x,y,z Cartesian coordinates to RA, dec positions in space.

    Parameters
    ----------
    x : float or array
        The position on the x-axis of the given points on the unit sphere
    y : float or array
        The position on the y-axis of the given points on the unit sphere
    z : float or array
        The position on the z-axis of the given points on the unit sphere

    Returns
    -------
    ra, dec : floats or arrays
        Ra and dec coordinates in degrees.
    """

    return np.degrees(_ra_dec_from_xyz(x, y, z))


def xyz_angular_radius(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.

    Returns
    -------
    radius : float
    """
    return _xyz_angular_radius(np.radians(radius))


def _xyz_angular_radius(radius):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in radians.

    Returns
    -------
    radius : float
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = _xyz_from_ra_dec(radius, 0)
    result = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    return result


def z_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)
    return np.array([[cc, -ss, 0.0], [ss, cc, 0.0], [0.0, 0.0, 1.0]])


def rot_about_z(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the z axis.
    Theta is in radians.
    Positive theta rotates +x towards +y.
    """
    return np.dot(z_rotation_matrix(theta), vec.transpose()).transpose()


def y_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)
    return np.array([[cc, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, cc]])


def rot_about_y(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the y axis.
    Theta is in radians.
    Positive theta rotates +x towards -z.
    """
    return np.dot(y_rotation_matrix(theta), vec.transpose()).transpose()


def x_rotation_matrix(theta):
    cc = np.cos(theta)
    ss = np.sin(theta)

    return np.array([[1.0, 0.0, 0.0], [0.0, cc, -ss], [0.0, ss, cc]])


def rot_about_x(vec, theta):
    """
    Rotate a Cartesian vector an angle theta about the x axis.
    Theta is in radians.
    Positive theta rotates +y towards +z.
    """
    return np.dot(x_rotation_matrix(theta), vec.transpose()).transpose()


def rotation_matrix_from_vectors(v1, v2):
    """
    Given two vectors v1,v2 calculate the rotation matrix for v1->v2 using the axis-angle approach

    Parameters
    ----------
    v1,v2 : `Unknown`
        Cartesian unit vectors (in three dimensions).
    rot : `Unknown`
        is the rotation matrix that rotates from one to the other
    """

    if np.abs(np.sqrt(np.dot(v1, v1)) - 1.0) > 0.01:
        raise RuntimeError("v1 in rotation_matrix_from_vectors is not a unit vector")

    if np.abs(np.sqrt(np.dot(v2, v2)) - 1.0) > 0.01:
        raise RuntimeError("v2 in rotation_matrix_from_vectors is not a unit vector")

    # Calculate the axis of rotation by the cross product of v1 and v2
    cross = np.cross(v1, v2)
    cross = cross / np.sqrt(np.dot(cross, cross))

    # calculate the angle of rotation via dot product
    angle = np.arccos(np.dot(v1, v2))
    sin_dot = np.sin(angle)
    cos_dot = np.cos(angle)

    # calculate the corresponding rotation matrix
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot = [
        [
            cos_dot + cross[0] * cross[0] * (1 - cos_dot),
            -cross[2] * sin_dot + (1 - cos_dot) * cross[0] * cross[1],
            cross[1] * sin_dot + (1 - cos_dot) * cross[0] * cross[2],
        ],
        [
            cross[2] * sin_dot + (1 - cos_dot) * cross[0] * cross[1],
            cos_dot + (1 - cos_dot) * cross[1] * cross[1],
            -cross[0] * sin_dot + (1 - cos_dot) * cross[1] * cross[2],
        ],
        [
            -cross[1] * sin_dot + (1 - cos_dot) * cross[0] * cross[2],
            cross[0] * sin_dot + (1 - cos_dot) * cross[1] * cross[2],
            cos_dot + (1 - cos_dot) * (cross[2] * cross[2]),
        ],
    ]

    return rot


def equation_of_equinoxes(d):
    """
    The equation of equinoxes. See http://aa.usno.navy.mil/faq/docs/GAST.php

    Parameters
    ----------
    d : `Unknown`
        is either a numpy array or a number that is Terrestrial Time
        expressed as an MJD
    the : `Unknown`
        equation of equinoxes in radians.
    """

    if isinstance(d, np.ndarray):
        return palpy.eqeqxVector(d)
    else:
        return palpy.eqeqx(d)


def calc_gmst_gast(mjd):
    """
    Compute Greenwich mean sidereal time and Greenwich apparent sidereal time
    see: From http://aa.usno.navy.mil/faq/docs/GAST.php

    Parameters
    ----------
    mjd : `Unknown`
        is the universal time (ut1) expressed as an MJD
    gmst : `Unknown`
        Greenwich mean sidereal time in hours
    gast : `Unknown`
        Greenwich apparent sidereal time in hours
    """

    date = np.floor(mjd)
    ut1 = mjd - date
    if isinstance(mjd, np.ndarray):
        gmst = palpy.gmstaVector(date, ut1)
    else:
        gmst = palpy.gmsta(date, ut1)

    eqeq = equation_of_equinoxes(mjd)
    gast = gmst + eqeq

    gmst = gmst * 24.0 / (2.0 * np.pi)
    gmst %= 24.0

    gast = gast * 24.0 / (2.0 * np.pi)
    gast %= 24.0

    return gmst, gast


def _angular_separation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in radians

    Parameters
    ----------
    long1 is the first longitudinal coordinate in radians

    lat1 is the first latitudinal coordinate in radians

    long2 is the second longitudinal coordinate in radians

    lat2 is the second latitudinal coordinate in radians

    Returns
    -------
    The angular separation between the two points in radians

    Calculated based on the haversine formula
    From http://en.wikipedia.org/wiki/Haversine_formula
    """
    are_arrays_1 = _validate_inputs([long1, lat1], ["long1", "lat1"], "angular_separation")

    are_arrays_2 = _validate_inputs([long2, lat2], ["long2", "lat2"], "angular_separation")

    # The code below is necessary because the call to np.radians() in
    # angular_separation() will automatically convert floats
    # into length 1 numpy arrays, confusing validate_inputs()
    if are_arrays_1 and len(long1) == 1:
        are_arrays_1 = False
        long1 = long1[0]
        lat1 = lat1[0]

    if are_arrays_2 and len(long2) == 1:
        are_arrays_2 = False
        long2 = long2[0]
        lat2 = lat2[0]

    if are_arrays_1 and are_arrays_2:
        if len(long1) != len(long2):
            raise RuntimeError(
                "You need to pass arrays of the same length " "as arguments to angular_separation()"
            )

    t1 = np.sin(lat2 / 2.0 - lat1 / 2.0) ** 2
    t2 = np.cos(lat1) * np.cos(lat2) * np.sin(long2 / 2.0 - long1 / 2.0) ** 2
    _sum = t1 + t2

    if isinstance(_sum, numbers.Number):
        if _sum < 0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum < 0.0, 0.0, _sum)

    return 2.0 * np.arcsin(np.sqrt(_sum))


def angular_separation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in degrees

    Parameters
    ----------
    long1 is the first longitudinal coordinate in degrees

    lat1 is the first latitudinal coordinate in degrees

    long2 is the second longitudinal coordinate in degrees

    lat2 is the second latitudinal coordinate in degrees

    Returns
    -------
    The angular separation between the two points in degrees
    """
    return np.degrees(
        _angular_separation(np.radians(long1), np.radians(lat1), np.radians(long2), np.radians(lat2))
    )


def haversine(long1, lat1, long2, lat2):
    """
    DEPRECATED; use angular_separation() instead

    Return the angular distance between two points in radians

    Parameters
    ----------
    long1 : `Unknown`
        is the longitude of point 1 in radians
    lat1 : `Unknown`
        is the latitude of point 1 in radians
    long2 : `Unknown`
        is the longitude of point 2 in radians
    lat2 : `Unknown`
        is the latitude of point 2 in radians
    the : `Unknown`
        angular separation between points 1 and 2 in radians
    """
    return _angular_separation(long1, lat1, long2, lat2)


def arcsec_from_radians(value):
    """
    Convert an angle in radians to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * np.degrees(value)


def radians_from_arcsec(value):
    """
    Convert an angle in arcseconds to radians

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return np.radians(value / 3600.0)


def arcsec_from_degrees(value):
    """
    Convert an angle in degrees to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * value


def degrees_from_arcsec(value):
    """
    Convert an angle in arcseconds to degrees

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return value / 3600.0
