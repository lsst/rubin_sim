import numpy as np

__all__ = ['calcSeason']


def calcSeason(ra, time):
    """Calculate the 'season' in the survey for a series of ra/dec/time values of an observation.
    Based only on the RA of the point on the sky, it calculates the 'season' based on when this
    point would be overhead. To convert to an integer season label, take np.floor of the returned
    float season values.

    Note that seasons should be calculated for a fixed point on the sky, not for each pointing that
    overlaps a point on the sky.  For example, bad things might happen if you compute the season
    for observations that overlap RA=0, but were centered on RA=359.

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
    # A reference RA and equinox to anchor ra/season calculation - RA = 0 is overhead at this (local) time.
    # This time was chosen as it is close to the expected start of the survey.
    # Generally speaking, this is the equinox (RA=0 is overhead at midnight)
    Equinox = 60208.00106863426
    # convert ra into 'days'
    dayRA = ra / 360 * 365.25
    firstSeasonBegan = Equinox + dayRA - 0.5 * 365.25
    seasons = (time - firstSeasonBegan) / 365.25
    # Set first season to 0
    seasons = seasons - np.floor(np.min(seasons))
    return seasons

    # The value for the equinox above was calculated as follows:
    #from astropy.time import Time
    #from astropy.coordinates import EarthLocation
    #loc = EarthLocation.of_site('Cerro Pachon')
    #t = Time('2023-09-21T00:01:32.33', format='isot', scale='utc', location=loc)
    #print(t.sidereal_time('apparent') - loc.lon, t.utc.mjd)
