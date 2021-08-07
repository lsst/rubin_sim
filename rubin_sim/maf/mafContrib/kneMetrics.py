import glob
import os
import numpy as np
from ..metrics import BaseMetric
from ..slicers import UserPointsSlicer
from rubin_sim.utils import uniformSphere
from rubin_sim.photUtils import Dust_values
from rubin_sim.data import get_data_dir

__all__ = ['KN_lc', 'KNePopMetric', 'generateKNPopSlicer']


class KN_lc(object):
    """
    Read in some KNe lightcurves

    Parameters
    ----------
    file_list : list of str (None)
        List of file paths to load. If None, loads up all the files from data/bns/
    """
    def __init__(self, file_list=None):
        if file_list is None:
            datadir = get_data_dir()
            # Get files, model grid developed by M. Bulla
            file_list = glob.glob(os.path.join(datadir, 'maf', 'bns', '*.dat'))

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


class KNePopMetric(BaseMetric):
    def __init__(self, metricName='KNePopMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, file_list=None, mjd0=59853.5, outputLc=False, badval=-666,
                 **kwargs):
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        # `bool` variable, if True the light curve will be exported
        self.outputLc = outputLc

        self.lightcurves = KN_lc(file_list=file_list)
        self.mjd0 = mjd0

        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super(KNePopMetric, self).__init__(col=cols, units='Detected, 0 or 1',
                                           metricName=metricName, maps=maps, badval=badval,
                                           **kwargs)

    def _multi_detect(self, around_peak):
        """
        Simple detection criteria: detect at least a certain number of times
        """
        result = 1
        # Detected data points
        if np.size(around_peak) < self.ptsNeeded:
            return 0

        return result

    def _ztfrest_simple(self, around_peak, mags, t, filters, min_dt=0.125,
                        min_fade=0.3, max_rise=-1., selectRed=False):
        """
        Selection criteria based on rise or decay rate; simplified version of
        the methods employed by the ZTFReST project
        (Andreoni & Coughlin et al., 2021)

        Parameters
        ----------
        around_peak : array
            indexes corresponding to 5sigma detections
        mags : array
            magnitudes obtained interpolating models on the dataSlice
        t : array
            relative times
        filters : array
            filters in which detections happened
        min_dt : float
            minimum time gap between first and last detection in a given band
        min_fade : float
            fade rate threshold (positive, mag/day)
        max_rise : float
            rise rate threshold (negative, mag/day)
        selectRed : bool
            if True, only red 'izy' filters will be considered

        Examples
        ----------
        A transient:
            rising by 0.74 mag/day will pass a threshold max_rise=-0.5
            rising by 0.74 mag/day will not pass a threshold max_rise=-1.0
            fading by 0.6 mag/day will pass a threshold min_fade=0.3
            fading by 0.2 mag/day will not pass a threshold min_fade=0.3
        """
        result = 1

        # Quick check on the number of detected points
        if np.size(around_peak) < self.ptsNeeded:
            return 0
        # Quick check on the time gap between first and last detection
        elif np.max(t[around_peak]) - np.min(t[around_peak]) < min_dt:
            return 0
        else:
            evol_rate = []
            fil = []
            # Check time gaps and rise or fade rate for each band
            for f in set(filters):
                if selectRed is True and not (f in 'izy'):
                    continue
                times_f = t[around_peak][np.where(filters == f)[0]]
                mags_f = mags[around_peak][np.where(filters == f)[0]]
                dt_f = np.max(times_f) - np.min(times_f)
                # Calculate the evolution rate, if the time gap condition is met
                if dt_f > min_dt:
                    evol_rate_f = ((np.max(mags_f) - np.min(mags_f))
                                   / (times_f[np.where(mags_f == np.max(mags_f))[0]][0]
                                      - times_f[np.where(mags_f == np.min(mags_f))[0]][0]))
                    evol_rate.append(evol_rate_f)
                else:
                    evol_rate.append(0)
                fil.append(f)
            if len(evol_rate) == 0:
                return 0
            # Check if the conditions on the evolution rate are met
            if np.max(evol_rate) < min_fade and np.min(evol_rate) > max_rise:
                return 0

        return result

    def _multi_color_detect(self, filters):
        """
        Color-based simple detection criteria: detect at least twice,
        with at least two filters
        """
        result = 1
        # detected in at least two filters
        if np.size(np.unique(filters)) < 2:
            return 0

        return result

    def _red_color_detect(self, filters, min_det=4):
        """
        Detected at least min_det times in either izy colors

        Parameters
        ----------
        filters : array
            filters in which detections happened
        min_det : float or int
            minimum number of detections required in izy bands
        """
        result = 1
        # Number of detected points in izy bands
        n_red_det = np.size(np.where(filters == 'i')[0]) \
                    + np.size(np.where(filters == 'z')[0]) \
                    + np.size(np.where(filters == 'y')[0])
        # Condition
        if n_red_det < min_det:
            return 0

        return result

    def _blue_color_detect(self, filters, min_det=4):
        """
        Detected at least min_det times in either ugr colors

        Parameters
        ----------
        filters : array
            filters in which detections happened
        min_det : float or int
            minimum number of detections required in ugr bands
        """
        result = 1
        # Number of detected points in ugr bands
        n_blue_det = np.size(np.where(filters == 'u')[0]) \
                     + np.size(np.where(filters == 'g')[0]) \
                     + np.size(np.where(filters == 'r')[0])
        # Condition
        if n_blue_det < min_det:
            return 0

        return result

    def run(self, dataSlice, slicePoint=None):
        result = {}
        t = dataSlice[self.mjdCol] - self.mjd0 - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)

        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername,
                                                   lc_indx=slicePoint['file_indx'])
            # Apply dust extinction on the light curve
            A_x = self.Ax1[filtername] * slicePoint['ebv']
            mags[infilt] += A_x

            distmod = 5*np.log10(slicePoint['distance']*1e6) - 5.0
            mags[infilt] += distmod

        # Find the detected points
        around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]        
        # Filters in which the detections happened
        filters = dataSlice[self.filterCol][around_peak]

        result['multi_detect'] = self._multi_detect(around_peak)
        result['ztfrest_simple'] = self._ztfrest_simple(around_peak, mags, t,
                                                        filters,
                                                        selectRed=False)
        result['ztfrest_simple_red'] = self._ztfrest_simple(around_peak, mags,
                                                            t, filters,
                                                            selectRed=True)
        result['multi_color_detect'] = self._multi_color_detect(filters)
        result['red_color_detect'] = self._red_color_detect(filters)
        result['blue_color_detect'] = self._blue_color_detect(filters)

        # Export the light curve
        if self.outputLc is True:
            mags[np.where(mags > 50)[0]] = 99.
            result['lc'] = [dataSlice[self.mjdCol], mags,
                            dataSlice[self.m5Col], dataSlice[self.filterCol]]
            result['lc_colnames'] = ('t', 'mag', 'maglim', 'filter')

        return result

    def reduce_multi_detect(self, metric):
        return metric['multi_detect']

    def reduce_ztfrest_simple(self, metric):
        return metric['ztfrest_simple']

    def reduce_ztfrest_simple_red(self, metric):
        return metric['ztfrest_simple_red']

    def reduce_multi_color_detect(self, metric):
        return metric['multi_color_detect']

    def reduce_red_color_detect(self, metric):
        return metric['red_color_detect']

    def reduce_blue_color_detect(self, metric):
        return metric['blue_color_detect']


def generateKNPopSlicer(t_start=1, t_end=3652, n_events=10000, seed=42,
                        n_files=100, d_min=10, d_max=300):
    """ Generate a population of KNe events, and put the info about them
    into a UserPointSlicer object

    Parameters
    ----------
    t_start : float (1)
        The night to start kilonova events on (days)
    t_end : float (3652)
        The final night of kilonova events
    n_events : int (10000)
        The number of kilonova events to generate
    seed : float
        The seed passed to np.random
    n_files : int (7)
        The number of different kilonova lightcurves to use
    d_min : float or int (10)
        Minimum luminosity distance (Mpc)
    d_max : float or int (300)
        Maximum luminosity distance (Mpc)
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    ra, dec = uniformSphere(n_events, seed=seed)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0, high=n_files,
                                           size=n_events)).astype(int)

    # Define the distance
    distance = rndm(d_min, d_max, 4, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['file_indx'] = file_indx
    slicer.slicePoints['distance'] = distance

    return slicer
