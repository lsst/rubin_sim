#####################################################################################################
# An extension to the GalaxyCountsMetric from Lynne Jones: sims_maf_contrib/mafContrib/lssMetrics.py
#
# Purpose: Estimate the number of galaxies expected at a particular coadded depth, accounting for
# dust extinction, magnitude cuts, as well as redshift-bin-specific powerlaws (based on mock catalogs
# from Nelson D. Padilla et al.).
#
# Includes functionality to calculate the galaxy counts from CFHTLS power law from LSST Science Book
# as well as to normalize the galaxy counts from mock catalogs to match those with CFHTLS power law
# at i<25.5.
#
# Need constantsForPipeline.py to import the power law constants and the normalization factor.
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
import numpy as np
import scipy
from rubin_sim.maf.metrics import BaseMetric, Coaddm5Metric, ExgalM5
from rubin_sim.maf.mafContrib.LSSObsStrategy.constantsForPipeline import powerLawConst_a, powerLawConst_b,\
    normalizationConstant

__all__ = ['GalaxyCountsMetric_extended']

class GalaxyCountsMetric_extended(BaseMetric):
    """

    Estimate galaxy counts per HEALpix pixel. Accomodates for dust extinction, magnitude cuts,
    and specification of the galaxy LF to specific redshift bin to consider.
    
    Dependency (aside from MAF): constantsForPipeline.py

    Parameters
    ------------
      * m5Col: str: name of column for depth in the data. Default: 'fiveSigmaDepth'
      * nside: int: HEALpix resolution parameter. Default: 128
      * upperMagLimit: float: upper limit on magnitude when calculating the galaxy counts. 
                              Default: 32.0
      * includeDustExtinction: `bool`: set to False if do not want to include dust extinction.
                                        Default: True
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
    def __init__(self, m5Col='fiveSigmaDepth', filterCol='filter', nside=128,
                 metricName='GalaxyCountsMetric_extended',
                 units='Galaxy Counts',
                 upperMagLimit=32.0,
                 includeDustExtinction=True,
                 filterBand='i', redshiftBin='all',
                 CFHTLSCounts=False,
                 normalizedMockCatalogCounts=True, **kwargs):
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.upperMagLimit = upperMagLimit
        self.includeDustExtinction = includeDustExtinction
        self.redshiftBin = redshiftBin
        self.filterBand = filterBand
        self.CFHTLSCounts = CFHTLSCounts
        self.normalizedMockCatalogCounts = normalizedMockCatalogCounts
        # Use the coadded depth metric to calculate the coadded depth at each point.
        # Specific band (e.g. r-band) will be provided by the sql constraint.
        if self.includeDustExtinction:
            # include dust extinction when calculating the co-added depth
            self.coaddmetric = ExgalM5(m5Col=self.m5Col)
        else:
            self.coaddmetric = Coaddm5Metric(m5Col=self.m5Col)

        # Need to scale down to indivdual HEALpix pixels.
        # Galaxy count from the coadded depth is per 1 square degree.
        # Number of galaxies ~= 41253 sq. degrees in the full sky divided by number of HEALpix pixels.
        self.scale = 41253.0/(int(12)*nside**2)
        # Consider power laws from various redshift bins: importing the constant
        # General power law form: 10**(a*m+b).  
        self.powerLawConst_a = powerLawConst_a
        self.powerLawConst_b = powerLawConst_b

        super().__init__(col=[self.m5Col, self.filterCol], metricName=metricName, maps=self.coaddmetric.maps,
                         units=units, **kwargs)

    # ------------------------------------------------------------------------
    # set up the integrand to calculate galaxy counts
    def _galCount(self, apparent_mag, coaddm5):
        # calculate the change in the power law constant based on the band
        # colors assumed here: (u-g)=(g-r)=(r-i)=(i-z)= (z-y)=0.4
        if (self.filterBand=='u'):   # dimmer than i: u-g= 0.4 => g= u-0.4 => i= u-0.4*3
            bandCorrection = -0.4*3.
        elif (self.filterBand=='g'):   # dimmer than i: g-r= 0.4 => r= g-0.4 => i= g-0.4*2
            bandCorrection = -0.4*2.
        elif (self.filterBand=='r'):   # dimmer than i: i= r-0.4
            bandCorrection = -0.4
        elif (self.filterBand=='i'):   # i 
            bandCorrection = 0.
        elif (self.filterBand=='z'):   # brighter than i: i-z= 0.4 => i= z+0.4
            bandCorrection = 0.4
        elif (self.filterBand=='y'):   # brighter than i: z-y= 0.4 => z= y+0.4 => i= y+0.4*2
            bandCorrection = 0.4*2.
        else:
            print('ERROR: Invalid band in GalaxyCountsMetric_extended. Assuming i-band.')
            bandCorrection = 0
    
        # check to make sure that the z-bin assigned is valid.
        if ((self.redshiftBin != 'all') and (self.redshiftBin not in list(self.powerLawConst_a.keys()))):
            print('ERROR: Invalid redshift bin in GalaxyCountsMetric_extended. Defaulting to all redshifts.')
            self.redshiftBin = 'all'
        
        # consider the power laws
        if (self.redshiftBin == 'all'):
            if self.CFHTLSCounts: 
                # LSST power law: eq. 3.7 from LSST Science Book converted to per sq degree:
                # (46*3600)*10^(0.31(i-25))
                dn_gal = 46.*3600.*np.power(10., 0.31*(apparent_mag+bandCorrection-25.))
            else:
                # full z-range considered here: 0.<z<4.0
                # sum the galaxy counts from each individual z-bin
                dn_gal = 0.
                for key in list(self.powerLawConst_a.keys()):
                    dn_gal += np.power(10., self.powerLawConst_a[key]*(apparent_mag+bandCorrection) + \
                                       self.powerLawConst_b[key])
        else:
            dn_gal = np.power(10., self.powerLawConst_a[self.redshiftBin]*(apparent_mag+bandCorrection) + \
                              self.powerLawConst_b[self.redshiftBin])
                
        completeness = 0.5*scipy.special.erfc(apparent_mag-coaddm5)
        return dn_gal*completeness

    # ------------------------------------------------------------------------
    def run(self, dataSlice, slicePoint=None):
        # Calculate the coadded depth.
        infilt = np.where(dataSlice[self.filterCol] == self.filterBand)[0]
        coaddm5 = self.coaddmetric.run(dataSlice[infilt], slicePoint)

        # some coaddm5 values are really small (i.e. min=10**-314). Zero them out.
        if (coaddm5 < 1):
            coaddm5 = 0
            numGal = 0

        else:
            numGal, intErr = scipy.integrate.quad(self._galCount, -np.inf,
                                              self.upperMagLimit, args=coaddm5)
            # Normalize the galaxy counts (per sq deg)
            if (self.normalizedMockCatalogCounts and not self.CFHTLSCounts):
                numGal = normalizationConstant*numGal
            if (numGal < 1.):
                numGal = 0.
            # scale down to individual HEALpix pixel instead of per sq deg
            numGal *= self.scale
        return numGal
