import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.utils import Site
from astropy.coordinates import SkyCoord, get_sun, get_moon, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time


__all__ = ['HourglassMetric']


def nearestVal(A, val):
    return A[np.argmin(np.abs(np.array(A)-val))]


class HourglassMetric(BaseMetric):
    """Plot the filters used as a function of time. Must be used with the Hourglass Slicer.
    Will totally fail in the arctic circle."""

    def __init__(self, telescope='LSST', mjdCol='observationStartMJD', filterCol='filter',
                 nightCol='night', delta_t=20., **kwargs):
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.nightCol = nightCol
        cols = [self.mjdCol, self.filterCol, self.nightCol]
        super(HourglassMetric, self).__init__(col=cols, metricDtype='object', **kwargs)
        self.telescope = Site(name=telescope)
        self.delta_t = delta_t/60./24.
        self.location = EarthLocation(lat=self.telescope.latitude_rad*u.rad,
                                      lon=self.telescope.longitude_rad*u.rad,
                                      height=self.telescope.height*u.m)

    def run(self, dataSlice, slicePoint=None):

        dataSlice.sort(order=self.mjdCol)
        unights, uindx = np.unique(dataSlice[self.nightCol], return_index=True)

        names = ['mjd', 'midnight', 'moonPer', 'twi6_rise', 'twi6_set', 'twi12_rise',
                 'twi12_set', 'twi18_rise', 'twi18_set']
        types = ['float']*len(names)
        pernight = np.zeros(len(unights), dtype=list(zip(names, types)))

        pernight['mjd'] = dataSlice[self.mjdCol][uindx]

        left = np.searchsorted(dataSlice[self.nightCol], unights)
        horizons = ['-6', '-12', '-18']
        key = ['twi6', 'twi12', 'twi18']

        # OK, I could just brute force this and compute a bunch of sun alt,az values and interpolate.
        times = Time(np.arange(dataSlice[self.mjdCol].min()-1, dataSlice[self.mjdCol].max()+1, self.delta_t), format='mjd')
        aa = AltAz(location=self.location, obstime=times)
        sun_coords = get_sun(times).transform_to(aa)

        # now to compute all the midnight, and twilight times
        # midnight is where alt< 0 and distance to meridian flips sign
        alt_slope = sun_coords.alt[1:] - sun_coords.alt[:-1]
        delt = alt_slope[:-1]*alt_slope[1:]
        # These are the indices where the altitude is at a local max or min
        switch = np.where((delt < 0) & (sun_coords.alt[2:] < 0))[0] + 1
        midnights = []
        for indx in switch:
            # Let's just take the weighted mean around the minimum
            midnights.append(np.average(times.mjd[indx-1:indx+2], weights=-sun_coords.alt[indx-1:indx+2]).value)

        for hor in [-6, -12, -18]:
            temp_alt = sun_coords.alt.deg - hor
            # Find where it switches from positive to negative
            ack = temp_alt[1:]*temp_alt[:-1]
            slope = sun_coords.alt.deg[1:] - sun_coords.alt.deg[0:-1]

            crossing = np.where((ack < 0) & (slope > 0))[0]
            cross_times = [np.interp(0, temp_alt[cross-1:cross+3], times.mjd[cross-1:cross+3]) for cross in crossing]
            pernight['twi%i_%s' % (abs(hor), 'rise')] = cross_times[0:len(pernight)]

            crossing = np.where((ack < 0) & (slope < 0))[0]
            cross_times = [np.interp(0, temp_alt[cross-1:cross+3][::-1], times.mjd[cross-1:cross+3][::-1]) for cross in crossing]
            pernight['twi%i_%s' % (abs(hor), 'set')] = cross_times[0:len(pernight)]

        pernight['midnight'] = midnights[0:len(pernight)]
        moon_times = Time(pernight['midnight'], format='mjd')
        aa = AltAz(location=self.location, obstime=moon_times)
        moon_coords = get_moon(moon_times).transform_to(aa)
        sun_coords = get_sun(moon_times).transform_to(aa)
        ang_dist = sun_coords.separation(moon_coords)
        pernight['moonPer'] = ang_dist.deg/180

        # Define the breakpoints as where either the filter changes OR
        # there's more than a 2 minute gap in observing
        good = np.where((dataSlice[self.filterCol] != np.roll(dataSlice[self.filterCol], 1)) |
                        (np.abs(np.roll(dataSlice[self.mjdCol], 1) -
                                dataSlice[self.mjdCol]) > 120./3600./24.))[0]
        good = np.concatenate((good, [0], [len(dataSlice[self.filterCol])]))
        good = np.unique(good)
        left = good[:-1]
        right = good[1:]-1
        good = np.ravel(list(zip(left, right)))

        names = ['mjd', 'midnight', 'filter']
        types = ['float', 'float', (np.str_ ,1)]
        perfilter = np.zeros((good.size), dtype=list(zip(names, types)))
        perfilter['mjd'] = dataSlice[self.mjdCol][good]
        perfilter['filter'] = dataSlice[self.filterCol][good]

        # now for each perfilter, find the closes midnight
        midnights = np.array(midnights)
        indx = np.searchsorted(midnights, perfilter['mjd'])
        d1 = np.abs(perfilter['mjd']-midnights[indx-1])
        d2 = np.abs(perfilter['mjd']-midnights[indx])

        perfilter['midnight'] = midnights[indx]
        temp_indx = np.where(d1 < d2)
        perfilter['midnight'][temp_indx] = midnights[indx-1][temp_indx]

        return {'pernight': pernight, 'perfilter': perfilter}
