################################################################################################
# Metric to evaluate for each HEALpix, the fraction of exposure time spent in each filter as
# a fraction of the total exposure time dedicated to that HEALpix.  The metric sums this over
# all HEALpix in the Galactic Plane/Magellanic Clouds region of interest and all filters and
# presents the result as a fraction of the value expected from the optimal survey strategy.
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord

MAP_FILE_ROOT_NAME = '../../data/galPlane_priority_maps'

class GalPlaneTimePerFilter(BaseMetric):

    def __init__(self, cols=['fieldRA','fieldDec','filter',
                             'observationStartMJD','visitExposureTime'],
                       metricName='GalPlaneTimePerFilter',
                       **kwargs):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.mjdCol = 'observationStartMJD'
        self.exptCol = 'visitExposureTime'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.load_maps()
        self.calc_idealfExpT()

        super(GalPlaneFootprintMetric,self).__init__(col=cols, metricName=metricName)

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(NSIDE)
        for f in self.filters:
            setattr(self, 'map_'+str(f), hp.read_map(MAP_FILE_ROOT_NAME+'_'+str(f)+'.fits')

    def calc_idealfExpT(self):
        """Method to calculate the optimal value of the fExpT metric, in the event
        that a survey strategy perfectly overlaps the desired Galactic Plane footprint
        and the exposure time spent in the different filters at the desired relative
        proportions."""

        # For each HEALpix in the sky, sum the value of the priority weighting
        # per filter.  This gives us the normalization factor to calculate the
        # fractional proportional of exposure time:
        coadded_map = np.zeros(self.map_i.shape)
        for i,f in enumerate(self.filters):
            coadded_map += getattr(self, 'map_'+f)

        fexpt_per_filter_per_pixel = np.zeros([self.map_i.shape,len(self.filters)])
        for i,f in enumerate(self.filters):
            fexpt_per_filter_per_pixel[:,i] = getattr(self, 'map_'+f) / coadded_map

        invalid = np.isnan(fexpt_per_filter_per_pixel)
        fexpt_per_filter_per_pixel[invalid] = 0.0

        # The value of the fExpT metric in ideal circumstance is the sum of all values:
        idealfExpT = fexpt_per_filter_per_pixel.sum()

        self.idealfExpT = idealfExpT

    def run(self, dataSlice, slicePoint=None):

        # Load the list of pointings included in this OpSim dataSlice and
        # construct this as a HEALpix map
        coords = SkyCoord(dataSlice[self.ra_col][0], dataSlice[self.dec_col][0], frame=Galactic())
        ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())
        pixels = ahp.skycoord_to_healpix(coords)

        # Identify those pixels that overlap the desired Galactic Plane survey region:
        # Returns list of pixel indices?
        GP_overlap_pixels = np.where(self.map_i[pixels] > 0, 1, 0)

        # CHECK THIS RETURNS THIS VALUE
        total_expt_per_pixel = dataSlice[self.exptCol]

        # Calculate the exposure time per filter per pixel as the fraction of the total
        # exposure time per pixel
        fexpt_per_filter_per_pixel = np.zeros(len(self.filters))
        for i,f in enumerate(self.filters):
            match = np.where(dataSlice[self.filterCol] == f)[0]
            fexpt = dataSlice[self.exptCol][match].sum()/total_expt_per_pixel
            fexpt_per_filter_per_pixel[i] = fexpt

        # Sum the fraction of exposure time over all pixels within the desired
        # survey region and filters:
        fExpT = fexpt_per_filter_per_pixel.sum()

        # Metric result returns fExpT as a fraction of the value of this metric
        # calculated for the optimal survey footprint overlap and filter time proportion:
        metric_value = fExpT / self.idealfExpT

        return metric_value
