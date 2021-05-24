import numpy as np

__all__ = ['calcSeason']


def calcSeason(ra, time):
    """Calculate the 'season' in the survey for a series of ra/dec/time values of an observation.
    Based only on the RA of the point on the sky, it calculates the 'season' based on when this
    point would be overhead .. the season is considered +/- 0.5 years around this time.

    Parameters
    ----------
    ra : float
        The RA (in degrees) of the point on the sky
    time : np.ndarray
        The times of the observations, in MJD

    Returns
    -------
    np.ndarray
        The season values
    """
    # Reference RA and equinox to anchor ra/season reference - RA = 0 is overhead at autumnal equinox
    # autumn equinox 2014 happened on september 23 --> equinox MJD
    Equinox = 2456923.5 - 2400000.5
    # convert ra into 'days'
    dayRA = ra / 360 * 365.25
    firstSeasonBegan = Equinox + dayRA - 0.5 * 365.25
    seasons = (time - firstSeasonBegan) / 365.25
    # Set first season to 0
    seasons = seasons - np.floor(np.min(seasons))
    return seasons
