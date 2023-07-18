__all__ = ("bearing", "dest_latlon", "point_to_line_distance")

import numpy as np

from .coordinate_transformations import _angular_separation


def bearing(lon1, lat1, lon2, lat2):
    """Bearing between two points
    all radians
    """

    delta_l = lon2 - lon1
    X = np.cos(lat2) * np.sin(delta_l)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_l)
    theta = np.arctan2(X, Y)

    return theta


def dest_latlon(dist, bearing, lat1, lon1):
    """Destination lat and lon given a distance, bearing and starting position
    all radians
    """

    lat2 = np.arcsin(np.sin(lat1) * np.cos(dist) + np.cos(lat1) * np.sin(dist) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(dist) * np.cos(lat1),
        np.cos(dist) - np.sin(lat1) * np.sin(lat2),
    )
    return lat2, lon2


def point_to_line_distance(lon1, lat1, lon2, lat2, lon3, lat3):
    """All radians
    points 1 and 2 define an arc segment,
    this finds the distance of point 3 to the arc segment.
    """

    result = lon1 * 0
    needed = np.ones(result.size, dtype=bool)

    bear12 = bearing(lon1, lat1, lon2, lat2)
    bear13 = bearing(lon1, lat1, lon3, lat3)
    dis13 = _angular_separation(lon1, lat1, lon3, lat3)

    # Is relative bearing obtuse?
    diff = np.abs(bear13 - bear12)
    if np.size(diff) == 1:
        if diff > np.pi:
            diff = 2 * np.pi - diff
        if diff > (np.pi / 2):
            return dis13
    else:
        over = np.where(diff > np.pi)
        diff[over] = 2 * np.pi - diff[over]
        solved = np.where(diff > (np.pi / 2))[0]
        result[solved] = dis13[solved]
        needed[solved] = False

    # Find the cross-track distance.
    dxt = np.arcsin(np.sin(dis13) * np.sin(bear13 - bear12))

    # Is p4 beyond the arc?
    dis12 = _angular_separation(lon1, lat1, lon2, lat2)
    dis14 = np.arccos(np.cos(dis13) / np.cos(dxt))
    if np.size(dis14) == 1:
        if dis14 > dis12:
            return _angular_separation(lon2, lat2, lon3, lat3)
    else:
        solved = np.where((dis14 > dis12) & needed)[0]
        result[solved] = _angular_separation(lon2[solved], lat2[solved], lon3[solved], lat3[solved])
        needed[solved] = False

    if np.size(lon1) == 1:
        return np.abs(dxt)
    else:
        result[needed] = np.abs(dxt[needed])
        return result
