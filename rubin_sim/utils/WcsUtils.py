import numpy as np
import numbers
from rubin_sim.utils import _observedFromICRS, _icrsFromObserved

__all__ = ["_nativeLonLatFromPointing", "_lonLatFromNativeLonLat",
           "_nativeLonLatFromRaDec", "_raDecFromNativeLonLat",
           "nativeLonLatFromRaDec", "raDecFromNativeLonLat"]


def _nativeLonLatFromPointing(lon, lat, lonPointing, latPointing):
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

    alpha = latPointing - 0.5 * np.pi
    beta = lonPointing

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    v2 = np.dot(np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -1.0 * sa, ca]]),
                np.dot(np.array([[cb, sb, 0.0], [-sb, cb, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, z])))

    cc = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    latOut = np.arctan2(v2[2], cc)

    _y = v2[1] / np.cos(latOut)
    _ra_raw = np.arccos(_y)

    # control for _y=1.0, -1.0 but actually being stored as just outside
    # the bounds of -1.0<=_y<=1.0 because of floating point error
    if hasattr(_ra_raw, '__len__'):
        _ra = np.array([rr
                        if not np.isnan(rr)
                        else 0.5 * np.pi * (1.0 - np.sign(yy))
                        for rr, yy in zip(_ra_raw, _y)])
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
            lonOut = 0.0
        elif (np.abs(_x) > 1.0e-9 and np.sign(_x) != np.sign(v2[0])) or \
             (np.abs(_y) > 1.0e-9 and np.sign(_y) != np.sign(v2[1])):

            lonOut = 2.0 * np.pi - _ra
        else:
            lonOut = _ra
    else:
        _lonOut = [2.0 * np.pi - rr
                   if (np.abs(xx) > 1.0e-9 and np.sign(xx) != np.sign(v2_0)) or
                   (np.abs(yy) > 1.0e-9 and np.sign(yy) != np.sign(v2_1))
                   else rr
                   for rr, xx, yy, v2_0, v2_1 in zip(_ra, _x, _y, v2[0], v2[1])]

        lonOut = np.array([0.0 if np.isnan(ll) else ll for ll in _lonOut])

    return lonOut, latOut


def _lonLatFromNativeLonLat(nativeLon, nativeLat, lonPointing, latPointing):
    """
    Transform a position in native longitude and latitude into
    longitude and latitude in a coordinate system where the telescope pointing
    is defined.  See the doc string for _nativeLonLatFromPointing for definitions
    of native longitude and latitude.

    @param [in] nativeLon is the native longitude in radians

    @param [in] nativeLat is the native latitude in radians

    @param [in] lonPointing is the longitude-like coordinate of the telescope
    pointing in radians

    @param [in] latPointing is the latitude-like coordinate of the telescope
    pointing in radians

    @param [out] latOut is the latitude of the transformed point(s)
    in the same coordinate system as the telescope pointing in radians

    @param [in] latOut is the latitude of the transformed point(s)
    in the same coordinate system as the telescope pointing in radians
    """
    x = -1.0 * np.cos(nativeLat) * np.sin(nativeLon)
    y = np.cos(nativeLat) * np.cos(nativeLon)
    z = np.sin(nativeLat)

    alpha = 0.5 * np.pi - latPointing
    beta = lonPointing

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    v2 = np.dot(np.array([[cb, -1.0 * sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]]),
                np.dot(np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -1.0 * sa, ca]]), np.array([x, y, z])))

    cc = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    latOut = np.arctan2(v2[2], cc)

    _y = v2[1] / np.cos(latOut)
    _lonOut = np.arccos(_y)
    _x = -np.sin(_lonOut)

    if isinstance(_lonOut, numbers.Number):
        if np.isnan(_lonOut):
            lonOut = 0.0
        elif (np.abs(_x) > 1.0e-9 and np.sign(_x) != np.sign(v2[0])) or \
             (np.abs(_y) > 1.0e-9 and np.sign(_y) != np.sign(v2[1])):

            lonOut = 2.0 * np.pi - _lonOut
        else:
            lonOut = _lonOut
    else:
        _lonOut = [2.0 * np.pi - rr
                   if (np.abs(xx) > 1.0e-9 and np.sign(xx) != np.sign(v2_0)) or
                   (np.abs(yy) > 1.0e-9 and np.sign(yy) != np.sign(v2_1))
                   else rr
                   for rr, xx, yy, v2_0, v2_1 in zip(_lonOut, _x, _y, v2[0], v2[1])]

        lonOut = np.array([0.0 if np.isnan(rr) else rr for rr in _lonOut])

    return lonOut, latOut


def _nativeLonLatFromRaDec(ra_in, dec_in, obs_metadata):
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

    ra, dec = _observedFromICRS(ra_in, dec_in,
                                obs_metadata=obs_metadata, epoch=2000.0,
                                includeRefraction=True)

    raPointing, decPointing = _observedFromICRS(obs_metadata._pointingRA,
                                                obs_metadata._pointingDec,
                                                obs_metadata=obs_metadata, epoch=2000.0,
                                                includeRefraction=True)

    return _nativeLonLatFromPointing(ra, dec, raPointing, decPointing)


def nativeLonLatFromRaDec(ra, dec, obs_metadata):
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

    lon, lat = _nativeLonLatFromRaDec(np.radians(ra), np.radians(dec),
                                      obs_metadata)

    return np.degrees(lon), np.degrees(lat)


def _raDecFromNativeLonLat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for _nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in radians

    @param [in] lat is the native latitude in radians

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing
    of the telescope

    @param [out] raOut is the RA of the star in radians
    (in the International Celestial Reference System)

    @param [in] decOut is the Dec of the star in radians
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrsFromObserved, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    raPointing, decPointing = _observedFromICRS(obs_metadata._pointingRA,
                                                obs_metadata._pointingDec,
                                                obs_metadata=obs_metadata, epoch=2000.0,
                                                includeRefraction=True)

    raObs, decObs = _lonLatFromNativeLonLat(lon, lat, raPointing, decPointing)

    # convert from observed geocentric coordinates to International Celestial Reference System
    # coordinates

    raOut, decOut = _icrsFromObserved(raObs, decObs, obs_metadata=obs_metadata,
                                      epoch=2000.0, includeRefraction=True)

    return raOut, decOut


def raDecFromNativeLonLat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in degrees

    @param [in] lat is the native latitude in degrees

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    pointing of the telescope

    @param [out] raOut is the RA of the star in degrees
    (in the International Celestial Reference System)

    @param [in] decOut is the Dec of the star in degrees
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrsFromObserved, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    ra, dec = _raDecFromNativeLonLat(np.radians(lon),
                                     np.radians(lat),
                                     obs_metadata)

    return np.degrees(ra), np.degrees(dec)
