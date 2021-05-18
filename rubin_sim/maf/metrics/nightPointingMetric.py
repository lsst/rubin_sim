import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.utils import Site
import ephem


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

    def run(self, dataSlice, slicePoint=None):

        lsstObs = ephem.Observer()
        lsstObs.lat = self.telescope.latitude_rad
        lsstObs.lon = self.telescope.longitude_rad
        lsstObs.elevation = self.telescope.height

        pad = 30./60./24.
        mjd_min = dataSlice[self.mjdCol].min() - pad
        mjd_max = dataSlice[self.mjdCol].max() + pad

        # How often to plot the moon and things
        step = 20./60./24.
        mjds = np.arange(mjd_min, mjd_max+step, step)
        sun_alts = []
        moon_alts = []
        moon_azs = []
        sun_azs = []

        doff = ephem.Date(0)-ephem.Date('1858/11/17')
        djds = mjds - doff
        for djd in djds:
            lsstObs.date = djd
            moon = ephem.Moon(lsstObs)
            moon_alts.append(moon.alt + 0)
            moon_azs.append(moon.az + 0)
            sun = ephem.Sun(lsstObs)
            sun_alts.append(sun.alt + 0)
            sun_azs.append(sun.az + 0)
        moon_alts = np.array(moon_alts)
        moon_azs = np.array(moon_azs)
        mjds = np.array(mjds)
        sun_alts = np.array(sun_alts)
        sun_azs = np.array(sun_azs)

        return {'dataSlice': dataSlice, 'moon_alts': moon_alts, 'moon_azs': moon_azs, 'mjds': mjds,
                'sun_alts': sun_alts, 'sun_azs': sun_azs}
