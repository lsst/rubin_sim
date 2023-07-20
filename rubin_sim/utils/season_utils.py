__all__ = ("calc_season",)

import numpy as np


def calc_season(ra, time):
    """Calculate the 'season' in the survey for a series of ra/time values of an observation.
    Based only on the RA of the point on the sky, it calculates the 'season' based on when the sun
    passes through this RA (this marks the start of a 'season').

    Note that seasons should be calculated using the RA of a fixed point on the sky, such as
    the slice_point['ra'] if calculating season values for a series of opsim pointings on the sky.
    To convert to integer seasons, use np.floor(seasons)

    Parameters
    ----------
    ra : `float`
        The RA (in degrees) of the point on the sky
    time : `np.ndarray`
        The times of the observations, in MJD days

    Returns
    -------
    seasons : `np.array`
        The season values, as floats.
    """
    # A reference time and sun RA location to anchor the location of the Sun
    # This time was chosen as it is close to the expected start of the survey.
    ref_time = 60575.0
    ref_sun_ra = 179.20796047239727
    # Calculate the fraction of the sphere/"year" for this location
    offset = (ra - ref_sun_ra) / 360 * 365.25
    # Calculate when the seasons should begin
    season_began = ref_time + offset
    # Calculate the season value for each point.
    seasons = (time - season_began) / 365.25
    # (usually) Set first season at this point to 0
    seasons = seasons - np.floor(np.min(seasons))
    return seasons
    # The reference values can be evaluated using:
    # from astropy.time import Time
    # from astropy.coordinates import get_sun
    # from astropy.coordinates import EarthLocation
    # loc = EarthLocation.of_site('Cerro Pachon')
    # t = Time('2024-09-22T00:00:00.00', format='isot', scale='utc', location=loc)
    # print('Ref time', t.utc.mjd)
    # print('Ref sun RA', get_sun(t).ra.deg, t.utc.mjd)
    # print('local sidereal time at season start', t.sidereal_time('apparent').deg)
