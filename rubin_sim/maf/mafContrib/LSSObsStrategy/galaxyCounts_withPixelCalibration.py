#####################################################################################################
# Purpose: Calculate the galaxy counts for each Healpix pixel directly.
# Necessary when accounting for pixel-specific calibration errors (as they modify the magnitude limit
# to which incompleteness-corrected galaxy LF is integrated over).
#
# Similar to GalaxyCountsMetric_extended but does the analysis on each HEALpix pixel individually,
# without communicating with the slicer. Like a psuedo-metric. Accomodates individual redshift
# bins; galaxy LF powerlaws based on mock catalogs from Padilla et al.
#
# Need constantsForPipeline.py to import the power law constants and the normalization factor.
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
import numpy as np
import scipy
import warnings
from rubin_sim.maf.mafContrib.LSSObsStrategy.constantsForPipeline import powerLawConst_a, powerLawConst_b, normalizationConstant

__all__ = ['GalaxyCounts_withPixelCalibration']

def GalaxyCounts_withPixelCalibration(coaddm5, upperMagLimit, nside=128,
                                      filterBand='i', redshiftBin='all',
                                      CFHTLSCounts=False,
                                      normalizedMockCatalogCounts=True):
    """

    Estimate galaxy counts for a given HEALpix pixel directly (without a slicer).

    Dependency (aside from MAF): constantsForPipeline.py
    ----------

    Parameters
    --------------------
      * coaddm5: float:coadded 5sigma limiting magnitude for the pixel.
      * upperMagLimit: float: upper limit on the magnitude, used to calculate numGal.
    
    Parameters
    --------------------
      * nside: int: HEALpix resolution parameter. Default: 128
      * filterBand: str: any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
      * redshiftBin: str: options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
                          '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5', '3.5<z<4.0',
                          'all' for no redshift restriction (i.e. 0.<z<4.0)
                          Default: 'all'
      * CFHTLSCounts: `bool`: set to True if want to calculate the total galaxy counts from CFHTLS
                               powerlaw from LSST Science Book. Must be run with redshiftBin= 'all'
                               Default: False
      * normalizedMockCatalogCounts: `bool`: set to False if  want the raw/un-normalized galaxy
                                              counts from mock catalogs. Default: True

    """
    # Need to scale down to indivdual HEALpix pixels. Galaxy count from the Coadded depth is per 1 square degree. 
    # Number of galaxies ~= 41253 sq. degrees in the full sky divided by number of HEALpix pixels.
    scale = 41253.0/(int(12)*nside**2)
    # Reset units (otherwise uses magnitudes).
    units = 'Galaxy Counts'

    # ------------------------------------------------------------------------
    # calculate the change in the power law constant based on the band
    # colors assumed here: (u-g)=(g-r)=(r-i)=(i-z)= (z-y)=0.4
    bandCorrection = -100
    if (filterBand=='u'):   # dimmer than i: u-g= 0.4 => g= u-0.4 => i= u-0.4*3
        bandCorrection = -0.4*3.
    elif (filterBand=='g'):   # dimmer than i: g-r= 0.4 => r= g-0.4 => i= g-0.4*2
        bandCorrection= -0.4*2.
    elif (filterBand=='r'):   # dimmer than i: i= r-0.4
        bandCorrection = -0.4
    elif (filterBand=='i'):   # i 
        bandCorrection = 0.
    elif (filterBand=='z'):   # brighter than i: i-z= 0.4 => i= z+0.4
        bandCorrection = 0.4
    elif (filterBand=='y'):   # brighter than i: z-y= 0.4 => z= y+0.4 => i= y+0.4*2
        bandCorrection = 0.4*2.
    else:
        print('ERROR: Invalid band in GalaxyCountsMetric_withPixelCalibErrors. Assuming i-band.')
        bandCorrection = 0

    # ------------------------------------------------------------------------
    # check to make sure that the z-bin assigned is valid.
    if ((redshiftBin != 'all') and (redshiftBin not in list(powerLawConst_a.keys()))):
        print('ERROR: Invalid redshift bin in GalaxyCountsMetric_withPixelCalibration. Defaulting to all redshifts.')
        redshiftBin = 'all'

    # ------------------------------------------------------------------------
    # set up the functions for the integrand
    # when have a redshift slice
    def galCount_bin(apparent_mag, coaddm5):
        dn_gal = np.power(10., powerLawConst_a[redshiftBin]*(apparent_mag+bandCorrection)+powerLawConst_b[redshiftBin])
        completeness = 0.5*scipy.special.erfc(apparent_mag-coaddm5)
        return dn_gal*completeness

    # when have to consider the entire z-range
    def galCount_all(apparent_mag, coaddm5):
        if CFHTLSCounts:
            # LSST power law: eq. 3.7 from LSST Science Book converted to per sq degree:
            # (46*3600)*10^(0.31(i-25))
            dn_gal = 46.*3600.*np.power(10., 0.31*(apparent_mag+bandCorrection-25.))
        else:
            # full z-range considered here: 0.<z<4.0
            # sum the galaxy counts from each individual z-bin
            dn_gal = 0.
            for key in list(powerLawConst_a.keys()):
                dn_gal += np.power(10., powerLawConst_a[key]*(apparent_mag+bandCorrection)+powerLawConst_b[key])
        completeness = 0.5*scipy.special.erfc(apparent_mag-coaddm5)
        return dn_gal*completeness

    # ------------------------------------------------------------------------
    # some coaddm5 values come out really small (i.e. min= 10**-314). Zero them out.
    if (coaddm5 <1): coaddm5 = 0
            
    # Calculate the number of galaxies.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # set up parameters to consider individual redshift range
        if (redshiftBin == 'all'):
            numGal, intErr = scipy.integrate.quad(galCount_all, -np.inf,
                                                  upperMagLimit, args=coaddm5)
        else:
            numGal, intErr = scipy.integrate.quad(galCount_bin, -np.inf,
                                                  upperMagLimit, args=coaddm5)

    if (normalizedMockCatalogCounts and not CFHTLSCounts):
        # Normalize the counts from mock catalogs to match up to CFHTLS counts fori<25.5 galaxy catalog
        # Found the scaling factor separately.
        numGal = normalizationConstant*numGal

    # coaddm5=0 implies no observation. Set no observation to zero to numGal.
    if (coaddm5 < 1.): numGal = 0.
    if (numGal < 1.): numGal = 0.
        
    # scale down to individual HEALpix pixel
    numGal *= scale
    return numGal
