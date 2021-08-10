import numpy as np
from .baseMetric import BaseMetric
from rubin_sim.photUtils import Dust_values
import healpy as hp
from rubin_sim.maf.maps import TrilegalDensityMap

__all__ = ['NgalScaleMetric', 'NlcPointsMetric']


class NgalScaleMetric(BaseMetric):
    """Approximate number of galaxies, scaled by median seeing.

    Parameters
    ----------
    A_max : float (0.2)
        The maximum dust extinction to allow. Anything with higher dust
        extinction is considered to have zero usable galaxies.
    m5min : float (26)
        The minimum coadded 5-sigma depth to allow. Anything less is
        considered to have zero usable galaxies.
    filter : str ("i")
        The filter to use. Any visits in other filters are ignored.
    """
    def __init__(self, seeingCol='seeingFwhmEff', m5Col='fiveSigmaDepth',
                 metricName='NgalScale', filtername='i', A_max=0.2,
                 m5min=26., filterCol='filter', **kwargs):
        maps = ['DustMap']
        units = 'N gals'
        self.seeingCol = seeingCol
        self.m5Col = m5Col
        self.filtername = filtername
        self.filterCol = filterCol
        self.A_max = A_max
        self.m5min = m5min

        super().__init__(col=[self.m5Col, self.filterCol, self.seeingCol], maps=maps, units=units, 
                         metricName=metricName, **kwargs)
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

    def run(self, dataSlice, slicePoint):

        # I'm a little confused why there's a dust cut and an M5 cut, but whatever
        A_x = self.Ax1[dataSlice[self.filterCol][0]] * slicePoint['ebv']
        if A_x > self.A_max:
            return 0

        in_filt = np.where(dataSlice[self.filterCol] == self.filtername)
        coadd_m5 = 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5Col][in_filt])))
        if coadd_m5 < self.m5min:
            return 0

        theta = np.median(dataSlice[self.seeingCol])
        # N gals per arcmin2
        ngal_per_arcmin2 = 57 * (0.75/theta)**1.5

        area = hp.nside2pixarea(slicePoint['nside'], degrees=True)*3600.

        ngal = ngal_per_arcmin2*area
        return ngal


class NlcPointsMetric(BaseMetric):
    """Number of points in stellar light curves

    Parameters
    ----------
    ndpmin : int (10)
        The number of points to demand on a lightcurve in a single
        filter to have that light curve qualify.
    mags : float (21)
        The magnitude of our fiducial object (maybe make it a dict in the
        future to support arbitrary colors).
    maps : list of map objects (None)
        List of stellar density maps to use. Default of None loads Trilegal maps.
    nside : int (128)
        The nside is needed to make sure the loaded maps match the slicer nside.
    """
    def __init__(self, ndpmin=10, mags=21., m5Col='fiveSigmaDepth', filterCol='filter',
                 metricName='NlcPoints', maps=None, nside=128, **kwargs):
        units = 'N LC points'
        self.ndpmin = ndpmin
        self.mags = mags
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nside = nside
        if maps is None:
            maps = [TrilegalDensityMap(filtername=fn, nside=nside) for fn in 'ugrizy']
        super().__init__(col=[self.m5Col, self.filterCol], maps=maps, units=units, 
                         metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint):
        if self.nside != slicePoint['nside']:
            raise ValueError('nside of metric does not match slicer')
        pix_area = hp.nside2pixarea(slicePoint['nside'], degrees=True)

        nlcpoints = 0
        # Let's do it per filter
        for filtername in np.unique(dataSlice[self.filterCol]):
            in_filt = np.where((dataSlice[self.filterCol] == filtername) &
                               (dataSlice[self.m5Col] > self.mags))[0]
            n_obs = np.size(in_filt)
            if n_obs > self.ndpmin:
                nstars = np.interp(self.mags, slicePoint[f'starMapBins_{filtername}'][1:],
                                   slicePoint[f'starLumFunc_{filtername}']) * pix_area
                nlcpoints += n_obs * nstars

        return nlcpoints
