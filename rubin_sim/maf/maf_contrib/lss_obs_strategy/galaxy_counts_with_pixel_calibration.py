# Purpose: Calculate the galaxy counts for each Healpix pixel directly.
# Necessary when accounting for pixel-specific calibration errors
# (as they modify the magnitude limit
# to which incompleteness-corrected galaxy LF is integrated over).
#
# Similar to GalaxyCountsMetric_extended but does the analysis on each
# HEALpix pixel individually,
# without communicating with the slicer. Like a psuedo-metric.
# Accommodates individual redshift
# bins; galaxy LF powerlaws based on mock catalogs from Padilla et al.
#
# Need constantsForPipeline.py to import the power law constants
# and the normalization factor.
#
# Humna Awan: humna.awan@rutgers.edu

__all__ = ("galaxy_counts_with_pixel_calibration",)

import warnings

import numpy as np
import scipy

from rubin_sim.maf.maf_contrib.lss_obs_strategy.constants_for_pipeline import (
    normalization_constant,
    power_law_const_a,
    power_law_const_b,
)


def galaxy_counts_with_pixel_calibration(
    coaddm5,
    upper_mag_limit,
    nside=128,
    filter_band="i",
    redshift_bin="all",
    cfhtls_counts=False,
    normalized_mock_catalog_counts=True,
):
    """Estimate galaxy counts for a given HEALpix pixel directly
    (without a slicer).

    Parameters
    ----------
    coaddm5 : `float`
        coadded 5sigma limiting magnitude for the pixel.
    upper_mag_limit : `float`
        upper limit on the magnitude, used to calculate num_gal.
    nside: `int`, opt
        HEALpix resolution parameter. Default: 128
    filter_band : `str`, opt
        Any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    redshift_bin : `str`, opt
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66,
        '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5',
        '3.5<z<4.0',
        'all' for no redshift restriction (i.e. 0.<z<4.0)
        Default: 'all'
    cfhtls_counts : `bool`, opt
        set to True if want to calculate the total galaxy counts from CFHTLS
        powerlaw from LSST Science Book. Must be run with redshift_bin= 'all'
        Default: False
    normalized_mock_catalog_counts: `bool`, opt
        set to False if  want the raw/un-normalized galaxy counts from
        mock catalogs.
        Default: True

    """
    # Need to scale down to indivdual HEALpix pixels.
    # Galaxy count from the Coadded depth is per 1 square degree.
    # Number of galaxies ~= 41253 sq. degrees in the full sky
    # divided by number of HEALpix pixels.
    scale = 41253.0 / (int(12) * nside**2)

    # ------------------------------------------------------------------------
    # calculate the change in the power law constant based on the band
    # colors assumed here: (u-g)=(g-r)=(r-i)=(i-z)= (z-y)=0.4
    band_correction = -100
    if filter_band == "u":
        # dimmer than i: u-g= 0.4 => g= u-0.4 => i= u-0.4*3
        band_correction = -0.4 * 3.0
    elif filter_band == "g":
        # dimmer than i: g-r= 0.4 => r= g-0.4 => i= g-0.4*2
        band_correction = -0.4 * 2.0
    elif filter_band == "r":
        # dimmer than i: i= r-0.4
        band_correction = -0.4
    elif filter_band == "i":
        # i
        band_correction = 0.0
    elif filter_band == "z":
        # brighter than i: i-z= 0.4 => i= z+0.4
        band_correction = 0.4
    elif filter_band == "y":
        # brighter than i: z-y= 0.4 => z= y+0.4 => i= y+0.4*2
        band_correction = 0.4 * 2.0
    else:
        print("ERROR: Invalid band in GalaxyCountsMetric_withPixelCalibErrors. Assuming i-band.")
        band_correction = 0

    # ------------------------------------------------------------------------
    # check to make sure that the z-bin assigned is valid.
    if (redshift_bin != "all") and (redshift_bin not in list(power_law_const_a.keys())):
        print(
            "ERROR: Invalid redshift bin in GalaxyCountsMetric_withPixelCalibration. "
            "Defaulting to all redshifts."
        )
        redshift_bin = "all"

    # ------------------------------------------------------------------------
    # set up the functions for the integrand
    # when have a redshift slice
    def gal_count_bin(apparent_mag, coaddm5):
        dn_gal = np.power(
            10.0,
            power_law_const_a[redshift_bin] * (apparent_mag + band_correction)
            + power_law_const_b[redshift_bin],
        )
        completeness = 0.5 * scipy.special.erfc(apparent_mag - coaddm5)
        return dn_gal * completeness

    # when have to consider the entire z-range
    def gal_count_all(apparent_mag, coaddm5):
        if cfhtls_counts:
            # LSST power law: eq. 3.7 from LSST Science Book
            # converted to per sq degree:
            # (46*3600)*10^(0.31(i-25))
            dn_gal = 46.0 * 3600.0 * np.power(10.0, 0.31 * (apparent_mag + band_correction - 25.0))
        else:
            # full z-range considered here: 0.<z<4.0
            # sum the galaxy counts from each individual z-bin
            dn_gal = 0.0
            for key in list(power_law_const_a.keys()):
                dn_gal += np.power(
                    10.0,
                    power_law_const_a[key] * (apparent_mag + band_correction) + power_law_const_b[key],
                )
        completeness = 0.5 * scipy.special.erfc(apparent_mag - coaddm5)
        return dn_gal * completeness

    # ------------------------------------------------------------------------
    # some coaddm5 values come out really small (i.e. min= 10**-314).
    # Zero them out.
    if coaddm5 < 1:
        coaddm5 = 0

    # Calculate the number of galaxies.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # set up parameters to consider individual redshift range
        if redshift_bin == "all":
            num_gal, int_err = scipy.integrate.quad(gal_count_all, -np.inf, upper_mag_limit, args=coaddm5)
        else:
            num_gal, int_err = scipy.integrate.quad(gal_count_bin, -np.inf, upper_mag_limit, args=coaddm5)

    if normalized_mock_catalog_counts and not cfhtls_counts:
        # Normalize the counts from mock catalogs to match up to
        # CFHTLS counts fori<25.5 galaxy catalog
        # Found the scaling factor separately.
        num_gal = normalization_constant * num_gal

    # coaddm5=0 implies no observation. Set no observation to zero to num_gal.
    if coaddm5 < 1.0:
        num_gal = 0.0
    if num_gal < 1.0:
        num_gal = 0.0

    # scale down to individual HEALpix pixel
    num_gal *= scale
    return num_gal
