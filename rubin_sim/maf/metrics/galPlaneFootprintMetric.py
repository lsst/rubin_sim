#######################################################################################################################
# Metric to evaluate whether a given pointing falls within the region of highest priority for Galactic Plane
# stellar astrophysics
#
# Rachel Street: rstreet@lco.global
#######################################################################################################################
import numpy as np
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord


class GalPlaneFootprintMetric(maf.BaseMetric):
    """Metric to evaluate the survey overlap with desired regions in the Galactic Plane
    and Magellanic Clouds, by referencing the pre-computed priority maps provided.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    """
    def __init__(self, cols=['fieldRA','fieldDec','filter'],
                       metricName='GalPlaneFootprintMetric',
                       **kwargs):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.MAP_FILE_ROOT_NAME = '../../data/galPlane_priority_maps'
        self.load_maps()

        super(GalPlaneFootprintMetric,self).__init__(col=cols, metricName=metricName)

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        for f in self.filters:
            setattr(self, 'map_'+str(f), hp.read_map(self.MAP_FILE_ROOT_NAME+'_'+str(f)+'.fits'))

    def run(self, dataSlice, slicePoint=None):

        coords_icrs = SkyCoord(dataSlice[self.ra_col][0], dataSlice[self.dec_col][0], frame='icrs', unit=(u.deg, u.deg))
        coords_gal = coords_icrs.transform_to(Galactic())

        ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())

        pixels = ahp.skycoord_to_healpix(coords_gal)

        combined_map = np.zeros(self.NPIX)
        for f in self.filters:
            weighted_map = getattr(self, 'map_'+str(f))
            combined_map += weighted_map[pixels]

        metric_value = combined_map.sum()

        # Normalize by assuming a nominal value of 1 per pixel per filter
        metric_value /= self.NPIX*len(self.filters)

        return metric_value
