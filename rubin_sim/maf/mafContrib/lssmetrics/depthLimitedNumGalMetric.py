import numpy as np
import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended \
    as GalaxyCountsMetric

class DepthLimitedNumGalMetric(metrics.BaseMetric):
    """
    This metric calculates the number of galaxies while accounting for the extragalactic footprint.

    Parameters
    ----------
    m5Col: str, optional
        Name of column for depth in the data. Default: 'fiveSigmaDepth'
    filterCol: str, optional
        Name of column for filter in the data. Default: 'filter'
    maps: list, optional
        List of map names. Default: ['DustMap']
    nside: int, optional
        HEALpix resolution parameter. Default: 256. This should match slicer nside.
    filterBand: str, optional
        Filter to use to calculate galaxy counts. Any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    redshiftBin: str, optional
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5', '3.5<z<4.0',
        'all' for no redshift restriction (so consider 0.<z<4.0)
        Default: 'all'
    nfilters_needed: int, optional
        Number of filters in which to require coverage. Default: 6
    lim_mag_i_ptsrc: float, optional
        Point-source limiting mag for the i-band coadded dust-corrected depth. Default: 26.0
    lim_ebv: float, optional
        Limiting EBV value. Default: 0.2

    Returns
    -------
    Number of galaxies in healpix if the slicePoint passes the extragalactic cuts; otherwise self.badval
    """
    def __init__(self, m5Col='fiveSigmaDepth', filterCol='filter',
                 nside=128, filterBand='i', redshiftBin='all',
                 nfilters_needed=6, lim_mag_i_ptsrc=26.0, lim_ebv=0.2,
                 metricName='DepthLimitedNumGalaxiesMetric', units='Galaxy Counts', **kwargs):
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.filterBand = filterBand
        # set up the extended source limiting mag
        # galaxies are x2 as seeing: seeing is generally 0.7arcsec and a typical galaxies is 1arcsec
        # => for extended source limiting mag of x, we'd need x + 0.7 as the point-source limiting mag;
        # 0.7 comes from $\sqrt{1/2}$;
        # basically have x2 difference in magnitudes between point source and extended source.
        lim_mag_i_extsrc = lim_mag_i_ptsrc - 0.7
        # set up the metric for galaxy counts
        self.galmetric = GalaxyCountsMetric(m5Col=self.m5Col, nside=nside,
                                            upperMagLimit=lim_mag_i_extsrc,
                                            includeDustExtinction=True,
                                            filterBand=self.filterBand, redshiftBin=redshiftBin,
                                            CFHTLSCounts=False,
                                            normalizedMockCatalogCounts=True)
        # set up the metric for extragalactic footprint
        self.eg_metric = metrics.ExgalM5_with_cuts(m5Col=self.m5Col, filterCol=self.filterCol,
                                                   lsstFilter=self.filterBand, nFilters=nfilters_needed,
                                                   extinction_cut=lim_ebv, depth_cut=lim_mag_i_ptsrc)
        maps = self.eg_metric.maps + self.galmetric.maps
        maps = set(maps)
        super().__init__(col=[self.m5Col, self.filterCol], maps=maps,
                         metricName=metricName, units=units,
                         **kwargs)

    def run(self, dataslice, slicePoint):
        # see if this slicePoint is in the extragalactic footprint
        pass_egcuts = self.eg_metric.run(dataslice, slicePoint=slicePoint)

        if pass_egcuts == self.badval: # failed dust/depth cuts
            return self.badval

        # Otherwise, find the galaxy counts
        in_filt = np.where(dataslice[self.filterCol] == self.filterBand)[0]
        return self.galmetric.run(dataslice[in_filt], slicePoint=slicePoint)
