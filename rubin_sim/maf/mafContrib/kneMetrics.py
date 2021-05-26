import numpy as np
import rubin_sim.maf.metrics as metrics
import os
from rubin_sim.utils import uniformSphere
import rubin_sim.maf.slicers as slicers
import glob
from rubin_sim.photUtils import Dust_values


__all__ = ['KN_lc', 'KNePopMetric', 'generateKNPopSlicer']


class KN_lc(object):
    """
    Read in some KNe lightcurves

    Parameters
    ----------
    file_list : list of str (None)
        List of file paths to load. If None, loads up all the files from data/tde/
    """

    def __init__(self, file_list=None):

        if file_list is None:
            sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
            file_list = glob.glob(os.path.join(sims_maf_contrib_dir, 'data/bns/*.dat'))

        filts = ["u", "g", "r", "i", "z", "y"]
        magidxs = [1, 2, 3, 4, 5, 6]

        # Let's organize the data in to a list of dicts for easy lookup
        self.data = []
        for filename in file_list:
            mag_ds = np.loadtxt(filename)
            t = mag_ds[:, 0]
            new_dict = {}
            for ii, (filt, magidx) in enumerate(zip(filts, magidxs)):
                new_dict[filt] = {'ph': t, 'mag': mag_ds[:, magidx]}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """
        t : array of floats
            The times to interpolate the light curve to.
        filtername : str
            The filter. one of ugrizy
        lc_index : int (0)
            Which file to use.
        """

        result = np.interp(t, self.data[lc_indx][filtername]['ph'],
                           self.data[lc_indx][filtername]['mag'],
                           left=99, right=99)
        return result


class KNePopMetric(metrics.BaseMetric):
    def __init__(self, metricName='KNePopMetric', mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', ptsNeeded=2, file_list=None, mjd0=59853.5,
                 **kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded

        self.lightcurves = KN_lc(file_list=file_list)
        self.mjd0 = mjd0

        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super(KNePopMetric, self).__init__(col=cols, units='Detected, 0 or 1',
                                           metricName=metricName, maps=maps,
                                           **kwargs)

    def _multi_detect(self, dataSlice, slicePoint, mags, t):
        """
        Simple detection criteria: detect at least twice
        """
        result = 1
        # detected in at least two bands
        around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
        filters = dataSlice[self.filterCol][around_peak]
        if np.size(filters) < 2:
            return 0

        return result

    def _multi_color_detect(self, dataSlice, slicePoint, mags, t):
        """
        Color-based simple detection criteria: detect at least twice, with at least two color
        """
        result = 1
        # detected in at least two bands
        around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]
        filters = np.unique(dataSlice[self.filterCol][around_peak])
        if np.size(filters) < 2:
            return 0

        return result

    def run(self, dataSlice, slicePoint=None):
        result = {}
        t = dataSlice[self.mjdCol] - self.mjd0 - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername, lc_indx=slicePoint['file_indx'])
            # Apply dust extinction on the light curve
            A_x = self.Ax1[filtername] * slicePoint['ebv']
            mags[infilt] -= A_x

            distmod = 5*np.log10(slicePoint['distance']*1e6) - 5.0
            mags[infilt] += distmod

        result['multi_detect'] = self._multi_detect(dataSlice, slicePoint, mags, t)
        result['multi_color_detect'] = self._multi_color_detect(dataSlice, slicePoint, mags, t)

        return result

    def reduce_multi_detect(self, metric):
        return metric['multi_detect']

    def reduce_multi_color_detect(self, metric):
        return metric['multi_color_detect']


def generateKNPopSlicer(t_start=1, t_end=3652, n_events=10000, seed=42, n_files=100):
    """ Generate a population of KNe events, and put the info about them into a UserPointSlicer object

    Parameters
    ----------
    t_start : float (1)
        The night to start tde events on (days)
    t_end : float (3652)
        The final night of TDE events
    n_events : int (10000)
        The number of TDE events to generate
    seed : float
        The seed passed to np.random
    n_files : int (7)
        The number of different TDE lightcurves to use
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    ra, dec = uniformSphere(n_events, seed=seed)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0, high=n_files, size=n_events)).astype(int)
    distance = rndm(10, 300, 4, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['file_indx'] = file_indx
    slicer.slicePoints['distance'] = distance

    return slicer
