import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.utils import Site
from astropy.coordinates import SkyCoord, get_sun, get_moon, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time


__all__ = ['NightPointingMetric']


class NightPointingMetric(BaseMetric):
    """
    Gather relevant information for a night to plot.
    """

    def __init__(self, altCol='altitude', azCol='azimuth', filterCol='filter',
                 mjdCol='observationStartMJD', metricName='NightPointing', telescope='LSST', **kwargs):

        cols = [altCol, azCol, filterCol, mjdCol]
        super(NightPointingMetric, self).__init__(col=cols, metricName=metricName, metricDtype='object', **kwargs)
        self.telescope = Site(name=telescope)
        self.altCol = altCol
        self.azCol = azCol
        self.filterCol = filterCol
        self.mjdCol = mjdCol

        self.location = EarthLocation(lat=self.telescope.latitude_rad*u.rad,
                                      lon=self.telescope.longitude_rad*u.rad,
                                      height=self.telescope.height*u.m)

    def run(self, dataSlice, slicePoint=None):

        pad = 30./60./24.
        mjd_min = dataSlice[self.mjdCol].min() - pad
        mjd_max = dataSlice[self.mjdCol].max() + pad

        # How often to plot the moon and things
        step = 20./60./24.
        mjds = Time(np.arange(mjd_min, mjd_max+step, step), format='mjd')

        aa = AltAz(location=self.location, obstime=mjds)

        moon_coords = get_moon(mjds).transform_to(aa)
        sun_coords = get_sun(mjds).transform_to(aa)

        moon_alts = np.array(moon_coords.alt.rad)
        moon_azs = np.array(moon_coords.az.rad)
        mjds = np.array(mjds)
        sun_alts = np.array(sun_coords.alt.rad)
        sun_azs = np.array(sun_coords.az.rad)

        return {'dataSlice': dataSlice, 'moon_alts': moon_alts, 'moon_azs': moon_azs, 'mjds': mjds,
                'sun_alts': sun_alts, 'sun_azs': sun_azs}
