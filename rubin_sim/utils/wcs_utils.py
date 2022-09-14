import numpy as np
import numbers
from rubin_sim.utils import _observed_from_icrs, _icrs_from_observed

__all__ = [
    "_native_lon_lat_from_pointing",
    "_lon_lat_from_native_lon_lat",
    "_native_lon_lat_from_ra_dec",
    "_ra_dec_from_native_lon_lat",
    "native_lon_lat_from_ra_dec",
    "ra_dec_from_native_lon_lat",
]


def _native_lon_lat_from_pointing(lon, lat, lon_pointing, lat_pointing):
    """
    Convert the longitude and latitude of a point into `native'
    longitude and latitude defined by a telescope pointing.

    Native longitude and latitude are defined as what longitude and latitude would be
    if the pole were at the location where the telescope is pointing.
    The transformation is achieved by rotating the vector pointing to the lon
    and lat being transformed once about the x axis and once about the z axis.
    These are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: This method does not assume anything about the input coordinate
    system.  It merely takes a longitude-like coordinate, a latitude-like coordinate
    and a longitude-like coordinate of the pointing and a latitude-like coordinate
    of the pointing and transforms them using spherical geometry.

    @param [in] a longitude-like coordinate in radians

    @param [in] a latitude-like coordinate in radians

    @param [in] a longitude-like coordinate of the telescope pointing in radians

    @param [in] a latitude-like coordinate of the telescope pointing in radians

    @param [out] the native longitude of the transformed point(s) in radians

    @param [out] the native latitude of the transformed point(s) in radians
    """

    x = -1.0 * np.cos(lat) * np.sin(lon)
    y = np.cos(lat) * np.cos(lon)
    z = np.sin(lat)

    alpha = lat_pointing - 0.5 * np.pi
    beta = lon_pointing

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    v2 = np.dot(
        np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -1.0 * sa, ca]]),
        np.dot(
            np.array([[cb, sb, 0.0], [-sb, cb, 0.0], [0.0, 0.0, 1.0]]),
            np.array([x, y, z]),
        ),
    )

    cc = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    lat_out = np.arctan2(v2[2], cc)

    _y = v2[1] / np.cos(lat_out)
    _ra_raw = np.arccos(_y)

    # control for _y=1.0, -1.0 but actually being stored as just outside
    # the bounds of -1.0<=_y<=1.0 because of floating point error
    if hasattr(_ra_raw, "__len__"):
        _ra = np.array(
            [
                rr if not np.isnan(rr) else 0.5 * np.pi * (1.0 - np.sign(yy))
                for rr, yy in zip(_ra_raw, _y)
            ]
        )
    else:
        if np.isnan(_ra_raw):
            if np.sign(_y) < 0.0:
                _ra = np.pi
            else:
                _ra = 0.0
        else:
            _ra = _ra_raw

    _x = -np.sin(_ra)

    if isinstance(_ra, numbers.Number):
        if np.isnan(_ra):
            lon_out = 0.0
        elif (np.abs(_x) > 1.0e-9 and np.sign(_x) != np.sign(v2[0])) or (
            np.abs(_y) > 1.0e-9 and np.sign(_y) != np.sign(v2[1])
        ):

            lon_out = 2.0 * np.pi - _ra
        else:
            lon_out = _ra
    else:
        _lon_out = [
            2.0 * np.pi - rr
            if (np.abs(xx) > 1.0e-9 and np.sign(xx) != np.sign(v2_0))
            or (np.abs(yy) > 1.0e-9 and np.sign(yy) != np.sign(v2_1))
            else rr
            for rr, xx, yy, v2_0, v2_1 in zip(_ra, _x, _y, v2[0], v2[1])
        ]

        lon_out = np.array([0.0 if np.isnan(ll) else ll for ll in _lon_out])

    return lon_out, lat_out


def _lon_lat_from_native_lon_lat(native_lon, native_lat, lon_pointing, lat_pointing):
    """
    Transform a position in native longitude and latitude into
    longitude and latitude in a coordinate system where the telescope pointing
    is defined.  See the doc string for _native_lon_lat_from_pointing for definitions
    of native longitude and latitude.

    @param [in] native_lon is the native longitude in radians

    @param [in] native_lat is the native latitude in radians

    @param [in] lon_pointing is the longitude-like coordinate of the telescope
    pointing in radians

    @param [in] lat_pointing is the latitude-like coordinate of the telescope
    pointing in radians

    @param [out] lat_out is the latitude of the transformed point(s)
    in the same coordinate system as the telescope pointing in radians

    @param [in] lat_out is the latitude of the transformed point(s)
    in the same coordinate system as the telescope pointing in radians
    """
    x = -1.0 * np.cos(native_lat) * np.sin(native_lon)
    y = np.cos(native_lat) * np.cos(native_lon)
    z = np.sin(native_lat)

    alpha = 0.5 * np.pi - lat_pointing
    beta = lon_pointing

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    v2 = np.dot(
        np.array([[cb, -1.0 * sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]]),
        np.dot(
            np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -1.0 * sa, ca]]),
            np.array([x, y, z]),
        ),
    )

    cc = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    lat_out = np.arctan2(v2[2], cc)

    _y = v2[1] / np.cos(lat_out)
    _lon_out = np.arccos(_y)
    _x = -np.sin(_lon_out)

    if isinstance(_lon_out, numbers.Number):
        if np.isnan(_lon_out):
            lon_out = 0.0
        elif (np.abs(_x) > 1.0e-9 and np.sign(_x) != np.sign(v2[0])) or (
            np.abs(_y) > 1.0e-9 and np.sign(_y) != np.sign(v2[1])
        ):

            lon_out = 2.0 * np.pi - _lon_out
        else:
            lon_out = _lon_out
    else:
        _lon_out = [
            2.0 * np.pi - rr
            if (np.abs(xx) > 1.0e-9 and np.sign(xx) != np.sign(v2_0))
            or (np.abs(yy) > 1.0e-9 and np.sign(yy) != np.sign(v2_1))
            else rr
            for rr, xx, yy, v2_0, v2_1 in zip(_lon_out, _x, _y, v2[0], v2[1])
        ]

        lon_out = np.array([0.0 if np.isnan(rr) else rr for rr in _lon_out])

    return lon_out, lat_out


def _native_lon_lat_from_ra_dec(ra_in, dec_in, obs_metadata):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The transformation is achieved by rotating the vector pointing to the RA
    and Dec being transformed once about the x axis and once about the z axis.
    These are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: RA, and Dec are assumed to be in the International Celestial Reference
    System.  Before calculating native longitude and latitude, this method will
    convert RA and Dec to observed geocentric coordinates.

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical
    celestial sphere.

    @param [in] ra is the RA of the star being transformed in radians
    (in the International Celestial Reference System)

    @param [in] dec is the Dec of the star being transformed in radians
    (in the International Celestial Reference System)

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing of
    the telescope.

    @param [out] lonOut is the native longitude in radians

    @param [out] latOut is the native latitude in radians
    """

    ra, dec = _observed_from_icrs(
        ra_in, dec_in, obs_metadata=obs_metadata, epoch=2000.0, include_refraction=True
    )

    ra_pointing, dec_pointing = _observed_from_icrs(
        obs_metadata._pointing_ra,
        obs_metadata._pointing_dec,
        obs_metadata=obs_metadata,
        epoch=2000.0,
        include_refraction=True,
    )

    return _native_lon_lat_from_pointing(ra, dec, ra_pointing, dec_pointing)


def native_lon_lat_from_ra_dec(ra, dec, obs_metadata):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The coordinate basis axes for this system is achieved by taking the true
    coordinate basis axes and rotating them once about the z axis and once about
    the x axis (or, by rotating the vector pointing to the RA and Dec being
    transformed once about the x axis and once about the z axis).  These
    are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: RA, and Dec are assumed to be in the International Celestial Reference
    System.  Before calculating native longitude and latitude, this method will
    convert RA and Dec to observed geocentric coordinates.

    WARNING: This method does not account for apparent motion due to parallax.
    This method is only useful for mapping positions on a theoretical
    celestial sphere.

    @param [in] ra is the RA of the star being transformed in degrees
    (in the International Celestial Reference System)

    @param [in] dec is the Dec of the star being transformed in degrees
    (in the International Celestial Reference System)

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing of
    the telescope.

    @param [out] lonOut is the native longitude in degrees

    @param [out] latOut is the native latitude in degrees
    """

    lon, lat = _native_lon_lat_from_ra_dec(np.radians(ra), np.radians(dec), obs_metadata)

    return np.degrees(lon), np.degrees(lat)


def _ra_dec_from_native_lon_lat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for _native_lon_lat_from_ra_dec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in radians

    @param [in] lat is the native latitude in radians

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing
    of the telescope

    @param [out] ra_out is the RA of the star in radians
    (in the International Celestial Reference System)

    @param [in] dec_out is the Dec of the star in radians
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrs_from_observed, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    ra_pointing, dec_pointing = _observed_from_icrs(
        obs_metadata._pointing_ra,
        obs_metadata._pointing_dec,
        obs_metadata=obs_metadata,
        epoch=2000.0,
        include_refraction=True,
    )

    ra_obs, dec_obs = _lon_lat_from_native_lon_lat(lon, lat, ra_pointing, dec_pointing)

    # convert from observed geocentric coordinates to International Celestial Reference System
    # coordinates

    ra_out, dec_out = _icrs_from_observed(
        ra_obs, dec_obs, obs_metadata=obs_metadata, epoch=2000.0, include_refraction=True
    )

    return ra_out, dec_out


def ra_dec_from_native_lon_lat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for native_lon_lat_from_ra_dec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in degrees

    @param [in] lat is the native latitude in degrees

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    pointing of the telescope

    @param [out] raOut is the RA of the star in degrees
    (in the International Celestial Reference System)

    @param [in] decOut is the Dec of the star in degrees
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrs_from_observed, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    ra, dec = _ra_dec_from_native_lon_lat(np.radians(lon), np.radians(lat), obs_metadata)

    return np.degrees(ra), np.degrees(dec)
