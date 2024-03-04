# An extension to the GalaxyCountsMetric from Lynne Jones:
# maf_contrib/mafContrib/lssMetrics.py
#
# Purpose: Estimate the number of galaxies expected at a particular
# coadded depth, accounting for
# dust extinction, magnitude cuts, as well as redshift-bin-specific
# powerlaws (based on mock catalogs
# from Nelson D. Padilla et al.).
#
# Includes functionality to calculate the galaxy counts from CFHTLS power
# law from LSST Science Book
# as well as to normalize the galaxy counts from mock catalogs to match
# those with CFHTLS power law at i<25.5.
#
# Need constantsForPipeline.py to import the power law constants and
# the normalization factor.
#
# Humna Awan: humna.awan@rutgers.edu

__all__ = ("GalaxyCountsMetricExtended",)

import warnings

import numpy as np
import scipy

from rubin_sim.maf.maf_contrib.lss_obs_strategy.constants_for_pipeline import (
    normalization_constant,
    power_law_const_a,
    power_law_const_b,
)
from rubin_sim.maf.metrics import BaseMetric, Coaddm5Metric, ExgalM5


class GalaxyCountsMetricExtended(BaseMetric):
    """Estimate galaxy counts per HEALpix pixel.

    Accommodates for dust extinction, magnitude cuts,
    and specification of the galaxy LF to specific redshift bin to consider.
    Dependency (aside from MAF): constantsForPipeline.py

    Parameters
    ------------
    m5_col : `str`
        name of column for depth in the data. Default: 'fiveSigmaDepth'
    nside : `int`, opt
        HEALpix resolution parameter. Default: 128
    upper_mag_limit : `float`
        upper limit on magnitude when calculating the galaxy counts.
        Default: 32.0
    include_dust_extinction : `bool`
        set to False if do not want to include dust extinction.
        Default: True
    filter_band : `str`, opt
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    redshift_bin : `str`, opt
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66,
        '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5',
        '3.5<z<4.0',
        'all' for no redshift restriction (i.e. 0.<z<4.0)
        Default: 'all'
    cfht_ls_counts : `bool`, opt
        set to True if want to calculate the total galaxy counts from CFHTLS
        powerlaw from LSST Science Book. Must be run with redshift_bin= 'all'
        Default: False
    normalized_mock_catalog_counts : `bool`, opt
     set to False if  want the raw/un-normalized galaxy counts from
     mock catalogs.
     Default: True
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        nside=128,
        metric_name="GalaxyCountsMetricExtended",
        units="Galaxy Counts",
        upper_mag_limit=32.0,
        include_dust_extinction=True,
        filter_band="i",
        redshift_bin="all",
        cfht_ls_counts=False,
        normalized_mock_catalog_counts=True,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.upper_mag_limit = upper_mag_limit
        self.include_dust_extinction = include_dust_extinction
        self.redshift_bin = redshift_bin
        self.filter_band = filter_band
        self.cfhtls_counts = cfht_ls_counts
        self.normalized_mock_catalog_counts = normalized_mock_catalog_counts
        # Use the coadded depth metric to calculate the coadded depth
        # at each point.
        # Specific band (e.g. r-band) will be provided by the sql constraint.
        if self.include_dust_extinction:
            # include dust extinction when calculating the co-added depth
            self.coaddmetric = ExgalM5(m5_col=self.m5_col)
        else:
            self.coaddmetric = Coaddm5Metric(m5_col=self.m5_col)

        # Need to scale down to indivdual HEALpix pixels.
        # Galaxy count from the coadded depth is per 1 square degree.
        # Number of galaxies ~= 41253 sq. degrees in the full sky divided
        # by number of HEALpix pixels.
        self.scale = 41253.0 / (int(12) * nside**2)
        # Consider power laws from various redshift bins: importing
        # the constant
        # General power law form: 10**(a*m+b).
        self.power_law_const_a = power_law_const_a
        self.power_law_const_b = power_law_const_b

        super().__init__(
            col=[self.m5_col, self.filter_col],
            metric_name=metric_name,
            maps=self.coaddmetric.maps,
            units=units,
            **kwargs,
        )

    # ------------------------------------------------------------------------
    # set up the integrand to calculate galaxy counts
    def _gal_count(self, apparent_mag, coaddm5):
        # calculate the change in the power law constant based on the band
        # colors assumed here: (u-g)=(g-r)=(r-i)=(i-z)= (z-y)=0.4
        factor = 0.4
        band_correction_dict = {
            "u": -3.0 * factor,
            "g": -2.0 * factor,
            "r": -1.0 * factor,
            "i": 0.0,
            "z": factor,
            "y": 2.0 * factor,
        }
        if self.filter_band not in band_correction_dict:
            warnings.warn("Invalid band in GalaxyCountsMetricExtended. " "Assuming i-band instead.")
        band_correction = band_correction_dict.get(self.filter_band, 0.0)

        # check to make sure that the z-bin assigned is valid.
        if (self.redshift_bin != "all") and (self.redshift_bin not in list(self.power_law_const_a.keys())):
            warnings.warn(
                "Invalid redshift bin in GalaxyCountsMetricExtended. " "Defaulting to all redshifts."
            )
            self.redshift_bin = "all"

        # consider the power laws
        if self.redshift_bin == "all":
            if self.cfhtls_counts:
                # LSST power law: eq. 3.7 from LSST Science Book
                # converted to per sq degree:
                # (46*3600)*10^(0.31(i-25))
                dn_gal = 46.0 * 3600.0 * np.power(10.0, 0.31 * (apparent_mag + band_correction - 25.0))
            else:
                # full z-range considered here: 0.<z<4.0
                # sum the galaxy counts from each individual z-bin
                dn_gal = 0.0
                for key in list(self.power_law_const_a.keys()):
                    dn_gal += np.power(
                        10.0,
                        self.power_law_const_a[key] * (apparent_mag + band_correction)
                        + self.power_law_const_b[key],
                    )
        else:
            dn_gal = np.power(
                10.0,
                self.power_law_const_a[self.redshift_bin] * (apparent_mag + band_correction)
                + self.power_law_const_b[self.redshift_bin],
            )

        completeness = 0.5 * scipy.special.erfc(apparent_mag - coaddm5)
        return dn_gal * completeness

    # ------------------------------------------------------------------------
    def run(self, data_slice, slice_point=None):
        # Calculate the coadded depth.
        infilt = np.where(data_slice[self.filter_col] == self.filter_band)[0]
        # If there are no visits in this filter,
        # return immediately with a flagged value
        if len(infilt) == 0:
            return self.badval

        coaddm5 = self.coaddmetric.run(data_slice[infilt], slice_point)

        # some coaddm5 values are really small (i.e. min=10**-314).
        # Zero them out.
        if coaddm5 < 1:
            num_gal = 0

        else:
            num_gal, int_err = scipy.integrate.quad(
                self._gal_count, -np.inf, self.upper_mag_limit, args=coaddm5
            )
            # Normalize the galaxy counts (per sq deg)
            if self.normalized_mock_catalog_counts and not self.cfhtls_counts:
                num_gal = normalization_constant * num_gal
            if num_gal < 1.0:
                num_gal = 0.0
            # scale down to individual HEALpix pixel instead of per sq deg
            num_gal *= self.scale
        return num_gal
