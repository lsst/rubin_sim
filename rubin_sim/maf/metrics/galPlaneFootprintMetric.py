#######################################################################################################################
# Metric to evaluate whether a given pointing falls within the region of highest priority for Galactic Plane
# stellar astrophysics
#
# Rachel Street: rstreet@lco.global
#######################################################################################################################
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord


class galPlaneFootprintMetric(maf.BaseMetric):
    """Metric to evaluate the survey overlap with desired regions in the Galactic Plane
    and Magellanic Clouds, by referencing the pre-computed priority maps provided.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    """

    def __init__(self, cols=['fieldRA','fieldDec','filter','fiveSigmaDepth'],
                       metricName='GalPlaneFootprintMetric',
                       **kwargs):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.filterCol = 'filter'
        self.m5Col = 'fiveSigmaDepth'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.magCuts = {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
        cwd = os.getcwd()
        self.MAP_DIR = os.path.join(cwd,'../../data/galPlane_priority_maps')
        self.MAP_FILE_ROOT_NAME = 'GalPlane_priority_map'
        self.load_maps()

        super().__init__(col=cols, metricName=metricName)

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        self.ideal_combined_map = np.zeros(self.NPIX)
        for f in self.filters:
            fmap = hp.read_map(os.path.join(self.MAP_DIR,self.MAP_FILE_ROOT_NAME+'_'+str(f)+'.fits'))
            setattr(self, 'map_'+str(f), fmap)
            self.ideal_combined_map += fmap

    def run(self, dataSlice, slicePoint=None):

        combined_map = np.zeros(self.NPIX)

        for f in self.filters:
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            coords_icrs = SkyCoord(dataSlice[self.ra_col][match], dataSlice[self.dec_col][match], frame='icrs', unit=(u.deg, u.deg))
            coords_gal = coords_icrs.transform_to(Galactic())
            ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())
            pixels = ahp.skycoord_to_healpix(coords_gal)

            weighted_map = getattr(self, 'map_'+str(f))
            combined_map[pixels] += weighted_map[pixels]
            
        metric_value = combined_map.sum()

        # Normalize by full weighted map summed over all filters and pixels:
        metric_value /= self.ideal_combined_map[pixels].sum()

        return metric_value
