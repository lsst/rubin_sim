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
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.io import fits
from rubin_sim.data import get_data_dir
import readGalPlaneMaps


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
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.filters = ["u", "g", "r", "i", "z", "y"]
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
        self.MAP_FILE_ROOT_NAME = "priority_GalPlane_footprint_map_data"
        self.load_maps()

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        self.ideal_combined_map = np.zeros(self.NPIX)
        for f in self.filters:
            file_path = os.path.join(
                self.MAP_DIR, "maf", self.MAP_FILE_ROOT_NAME + "_" + str(f) + ".fits",
            )
            map_data_table = readGalPlaneMaps.load_map_data(file_path)

            setattr(self, "map_" + str(f), map_data_table["combined_map"])
            setattr(self, "map_data_" + str(f), map_data_table)
            self.ideal_combined_map += map_data_table["combined_map"]

    def run(self, dataSlice, slicePoint=None):

        # Initialize holding array for map pixels in the dataSlice that
        # overlap with the desired survey area, summed over all filters
        dataslice_map = np.zeros(self.NPIX)

        for f in self.filters:
            # Select from the dataSlice observations that meet the limiting magnitude
            # requirements for the science concerned for this filter
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            # Calculate the ICRS coordinates of the observed fields and
            # convert these to galactic coordinates
            coords_icrs = SkyCoord(
                dataSlice[self.ra_col][match],
                dataSlice[self.dec_col][match],
                frame="icrs",
                unit=(u.deg, u.deg),
            )
            coords_gal = coords_icrs.transform_to(Galactic())

            # Calculate which HEALpixels in the sky map are covered by these
            # observations
            ahp = HEALPix(nside=self.NSIDE, order="ring", frame=TETE())
            pixels = ahp.skycoord_to_healpix(coords_gal)

            # Add the priority values for these HEALpixels from the map of
            # the desired footprint to the combined_map array
            weighted_map = getattr(self, "map_" + str(f))
            dataslice_map[pixels] += weighted_map[pixels]

        # This loop computes the main metric value over all HEALpixels in the sky map
        # ("combined_map") as well as over the HEALpixels of the specific regions
        # of interest for different science cases
        map_data = getattr(self, "map_data_" + str(f))
        metric_data = {}

        for col in map_data.columns:

            region_pixels = np.where(map_data[col.name] > 0.0)

            # To return a single metric value for the whole map, sum the total
            # priority of all desired pixels included in the survey observations:
            metric = dataslice_map[region_pixels].sum()

            # Normalize by full weighted map summed over all filters and pixels:
            metric /= self.ideal_combined_map[region_pixels].sum()

            metric_data[col.name] = metric

        return metric_data
