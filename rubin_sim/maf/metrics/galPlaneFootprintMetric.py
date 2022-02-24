#######################################################################################################################
# Metric to evaluate whether a given pointing falls within the region of highest priority for Galactic Plane
# stellar astrophysics
#
# Rachel Street: rstreet@lco.global
#######################################################################################################################
import os
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
from astropy.io import fits
from rubin_sim.data import get_data_dir


class galPlaneFootprintMetric(maf.BaseMetric):
    """Metric to evaluate the survey overlap with desired regions in the Galactic Plane
    and Magellanic Clouds, by referencing the pre-computed priority maps provided.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    """

    def __init__(
        self,
        cols=["fieldRA", "fieldDec", "filter", "fiveSigmaDepth"],
        metricName="GalPlaneFootprintMetric",
        **kwargs
    ):
        """Kwargs must contain:
        science_map   string  Name of the priority footprint map to use from
                                the column headers contained in the
                                priority_GalPlane_footprint_map_data tables
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.filters = ["u", "g", "r", "i", "z", "y"]
        self.science_map = kwargs["science_map"]
        self.magCuts = {
            "u": 22.7,
            "g": 24.1,
            "r": 23.7,
            "i": 23.1,
            "z": 22.2,
            "y": 21.4,
        }
        cwd = os.getcwd()
        self.MAP_DIR = get_data_dir()
        self.MAP_FILE_NAME = "priority_GalPlane_footprint_map_data_sum.fits"
        self.load_maps()

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        file_path = os.path.join(
            self.MAP_DIR,
            "maf",
            self.MAP_FILE_NAME,
        )
        with fits.open(file_path) as hdul:
            self.map_data_table = hdul[1].data

    def run(self, dataSlice, slicePoint):
        """Metric extracts the scientific priority of the given HEALpixel from
        the prepared maps of the desired survey footprint for galactic science.
        The priority map used has been summed over all filters for the
        science case indicated by the kwargs.  This is normalized using the
        summed priority for the map combining the priorities of all science
        cases."""

        metric_data = {}

        pixPriority = self.map_data_table[self.science_map][slicePoint["sid"]]

        iObservations = []
        for f in self.filters:
            idx = list(np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0])
            if len(idx) > 0:
                iObservations += idx

        metric_data["nObservations"] = len(iObservations)
        metric_data["priority"] = pixPriority

        return metric_data
