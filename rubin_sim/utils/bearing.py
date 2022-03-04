import numpy as np

__all__ = ["bearing", "dest_latlon"]


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

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(dist) + np.cos(lat1) * np.sin(dist) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(dist) * np.cos(lat1),
        np.cos(dist) - np.sin(lat1) * np.sin(lat2),
    )
    return lat2, lon2
