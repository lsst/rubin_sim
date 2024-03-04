import numpy as np

import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.maf_contrib.lss_obs_strategy.galaxy_counts_metric_extended import (
    GalaxyCountsMetricExtended as GalaxyCountsMetric,
)


class DepthLimitedNumGalMetric(metrics.BaseMetric):
    """This metric calculates the number of galaxies while accounting for the
    extragalactic footprint.

    Parameters
    ----------
    m5_col : `str`, optional
        Name of column for depth in the data. Default: 'fiveSigmaDepth'
    filter_col : `str`, optional
        Name of column for filter in the data. Default: 'filter'
    maps : `list` [`str`], optional
        List of map names. Default: ['DustMap']
    nside : `int`, optional
        HEALpix resolution parameter. Default: 256.
        This should match slicer nside.
    filter_band : `str`, optional
        Filter to use to calculate galaxy counts.
        Any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    redshiftBin: `str`, optional
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5',
        '3.5<z<4.0', 'all' for no redshift restriction
        (so consider 0.<z<4.0)
        Default: 'all'
    nfilters_needed : `int`, optional
        Number of filters in which to require coverage. Default: 6
    lim_mag_i_ptsrc : `float`, optional
        Point-source limiting mag for the i-band coadded dust-corrected depth.
        Default: 26.0
    lim_ebv : `float`, optional
        Limiting EBV value. Default: 0.2

    Returns
    -------
    ngal : `float`
        Number of galaxies in healpix if the slice_point passes the
        extragalactic cuts; otherwise self.badval
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        nside=128,
        filter_band="i",
        redshift_bin="all",
        nfilters_needed=6,
        lim_mag_i_ptsrc=26.0,
        lim_ebv=0.2,
        metric_name="DepthLimitedNumGalaxiesMetric",
        units="Galaxy Counts",
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.filter_band = filter_band
        # set up the extended source limiting mag
        # galaxies are x2 as seeing: seeing is generally 0.7arcsec
        # and a typical galaxies is 1arcsec
        # => for extended source limiting mag of x, we'd need x + 0.7
        # as the point-source limiting mag;
        # 0.7 comes from $\sqrt{1/2}$;
        # basically have x2 difference in magnitudes between
        # point source and extended source.
        lim_mag_i_extsrc = lim_mag_i_ptsrc - 0.7
        # set up the metric for galaxy counts
        self.galmetric = GalaxyCountsMetric(
            m5_col=self.m5_col,
            nside=nside,
            upper_mag_limit=lim_mag_i_extsrc,
            include_dust_extinction=True,
            filter_band=self.filter_band,
            redshift_bin=redshift_bin,
            cfht_ls_counts=False,
            normalized_mock_catalog_counts=True,
        )
        # set up the metric for extragalactic footprint
        self.eg_metric = metrics.ExgalM5WithCuts(
            m5_col=self.m5_col,
            filter_col=self.filter_col,
            lsst_filter=self.filter_band,
            n_filters=nfilters_needed,
            extinction_cut=lim_ebv,
            depth_cut=lim_mag_i_ptsrc,
        )
        maps = self.eg_metric.maps + self.galmetric.maps
        maps = set(maps)
        super().__init__(
            col=[self.m5_col, self.filter_col], maps=maps, metric_name=metric_name, units=units, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # see if this slice_point is in the extragalactic footprint
        pass_egcuts = self.eg_metric.run(data_slice, slice_point=slice_point)

        if pass_egcuts == self.badval:  # failed dust/depth cuts
            return self.badval

        # Otherwise, find the galaxy counts
        in_filt = np.where(data_slice[self.filter_col] == self.filter_band)[0]
        return self.galmetric.run(data_slice[in_filt], slice_point=slice_point)
