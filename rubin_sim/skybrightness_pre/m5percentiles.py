from builtins import object
import numpy as np
import healpy as hp
import os
from rubin_sim.data import get_data_dir


def ss_split(arr, side='left'):
    """
    make it possible to run searchsorted easily on 2 dims
    """
    result = np.searchsorted(arr[1:], arr[0], side=side)
    return result


class M5percentiles(object):
    """
    Take a map of five-sigma limiting depths and convert it to a map of percentiles
    """

    def __init__(self):

        # Load up the saved maps
        path = os.path.join(get_data_dir(), 'skybrightness_pre/percentile')
        filename = 'percentile_m5_maps.npz'
        temp = np.load(os.path.join(path, filename))
        self.m5_histograms = temp['histograms'].copy().T
        self.histogram_npts = temp['histogram_npts'].copy()
        temp.close()
        self.npix = self.m5_histograms['u'][:, 0].size
        self.nside = hp.npix2nside(self.npix)
        self.nbins = float(self.m5_histograms['u'][0, :].size)
        # The center of each histogram bin
        self.percentiles = np.arange(self.nbins)/self.nbins +1./2/self.nbins

    def dark_map(self, filtername='r', nside_out=64):
        """Return the darkest every healpixel gets
        """
        result = self.m5_histograms[filtername][:, -1]
        if self.nside != nside_out:
            result = hp.ud_grade(result, nside_out=nside_out)
        return result

    def percentile2m5map(self, percentile, filtername='r', nside=None):
        """
        Given a percentile, return the 5-sigma map for that level

        Parameters
        ----------
        percentile : float
             Value between 0-1.
        """
        if nside is None:
            nside = self.nside

        diff = np.abs(percentile - self.percentiles)
        closest = np.where(diff == diff.min())[0].min()
        result = self.m5_histograms[filtername][:, closest]
        result = hp.ud_grade(result, nside)
        return result

    def m5map2percentile(self, m5map, filtername='r'):
        """
        Convert a healpix map to a percentile map
        """
        orig_mask = np.where(m5map == hp.UNSEEN)

        inNside = hp.npix2nside(m5map.size)
        if inNside != self.nside:
            m5map = hp.ud_grade(m5map, nside_out=self.nside, pess=False)

        goodPix = np.where(m5map != hp.UNSEEN)[0]

        result = np.empty(self.npix, dtype=float)
        result.fill(hp.UNSEEN)
        temp_array = np.column_stack((m5map[goodPix], self.m5_histograms[filtername][goodPix, :]))
        result[goodPix] = np.apply_along_axis(ss_split, 1, temp_array)/self.nbins

        # convert the percentiles back to the right nside if needed
        # XXX--I should make a better linear interpolation upgrade function.
        if inNside != self.nside:
            result = hp.ud_grade(result, nside_out=inNside)
            result[orig_mask] = hp.UNSEEN
        return result

