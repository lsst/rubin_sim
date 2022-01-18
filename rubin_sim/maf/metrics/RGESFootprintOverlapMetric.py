import os
import numpy as np
import healpy as hp
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
import rubin_sim.maf as maf

class overlapRGESFootprintMetric(maf.BaseMetric):
    """Metric to evaluate the survey footprint overlap survey region for the Roman Galactic Exoplanet Survey

    Derived from the SpacialOverlapMetric by Michael Lund and revised and updated by Rachel Street

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    """

    def __init__(self, cols=['fieldRA','fieldDec','filter','fiveSigmaDepth'],
                       metricName='overlapRGESFootprintMetric',
                       **kwargs):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = 'fieldRA'
        self.dec_col = 'fieldDec'
        self.filterCol = 'filter'
        self.m5Col = 'fiveSigmaDepth'
        self.filters = ['u','g', 'r', 'i', 'z', 'y']
        self.magLimit = 22.0
        cwd = os.getcwd()
        self.create_map()
        self.load_RGES_footprint()

        super().__init__(col=cols, metricName=metricName)

    def create_map(self):
        self.NSIDE = 64
        self.ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())

    def load_RGES_footprint(self):

        # Location of the RGES survey field in Galactic Coordinates.
        # This is defined as a single pointing, since the survey
        # region will be ~2sq deg and fully encompassed within a single Rubin pointing
        l_center = 2.216
        b_center = -3.14
        l_width = 3.5
        b_height = 3.5
        n_points = 50

        # Represent coordinate pointings within this region as a meshgrid
        halfwidth_l = l_width / 2.0
        halfheight_b = b_height / 2.0

        l_min = max( (l_center-halfwidth_l), 0 )
        l_max = min( (l_center+halfwidth_l), 360.0 )
        b_min = max( (b_center-halfheight_b), -90.0 )
        b_max = min( (b_center+halfheight_b), 90.0 )

        l = np.linspace(l_min, l_max, n_points) * u.deg
        b = np.linspace(b_min, b_max, n_points) * u.deg

        LL,BB = np.meshgrid(l, b)

        coords = SkyCoord(LL, BB, frame=Galactic())

        # Calculate the corresponding HEALpixels
        pixels = self.ahp.skycoord_to_healpix(coords)
        self.RGES_pixels = np.unique(pixels.flatten())

    def run(self, dataSlice, slicePoint=None):

        # Only count observations with adequate S/N
        match = np.where(dataSlice[self.m5Col] >= self.magLimit)[0]

        # Extract the RA,Dec coordinates of the fields surveyed by matching observations the dataSlice
        # and calculate which HEALpixels these correspond to
        coords_icrs = SkyCoord(dataSlice[self.ra_col][match], dataSlice[self.dec_col][match], frame='icrs', unit=(u.deg, u.deg))
        coords_gal = coords_icrs.transform_to(Galactic())

        surveyed_pixels = self.ahp.skycoord_to_healpix(coords_gal)
        surveyed_pixels = np.unique(surveyed_pixels.flatten())

        # Calculate the fraction of the RGES survey pixels included in the surveyed pixels
        overlapping_pixels = set(self.RGES_pixels).intersection(set(surveyed_pixels))

        metric_value = float(len(overlapping_pixels)) / float(len(self.RGES_pixels))

        return metric_value
