################################################################################################
# Metric to evaluate for each HEALpix, the fraction of exposure time spent in each filter as
# a fraction of the total exposure time dedicated to that HEALpix.  The metric sums this over
# all HEALpix in the Galactic Plane/Magellanic Clouds region of interest and all filters and
# presents the result as a fraction of the value expected from the optimal survey strategy.
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import os
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from rubin_sim.data import get_data_dir


class galPlaneTimePerFilter(maf.BaseMetric):
    """Metric to evaluate for each HEALpix, the fraction of exposure time spent in each filter as
     a fraction of the total exposure time dedicated to that HEALpix.  The metric sums this over
     all HEALpix in the Galactic Plane/Magellanic Clouds region of interest and all filters and
     presents the result as a fraction of the value expected from the optimal survey strategy.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    observationStartMJD : float, MJD timestamp of the start of a given observation
    visitExposureTime : float, exposure time in seconds
    """

    def __init__(
        self,
        cols=[
            "fieldRA",
            "fieldDec",
            "filter",
            "observationStartMJD",
            "visitExposureTime",
            "fiveSigmaDepth",
        ],
        metricName="galPlaneTimePerFilter",
        **kwargs
    ):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.mjdCol = "observationStartMJD"
        self.exptCol = "visitExposureTime"
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
        self.MAP_FILE_ROOT_NAME = "GalPlane_priority_map"
        self.load_maps()
        self.calc_coaddmap()

        super().__init__(col=cols, metricName=metricName)

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        for f in self.filters:
            setattr(
                self,
                "map_" + str(f),
                hp.read_map(
                    os.path.join(
                        self.MAP_DIR,
                        "maf",
                        self.MAP_FILE_ROOT_NAME + "_" + str(f) + ".fits",
                    )
                ),
            )

    def calc_coaddmap(self):
        """For each HEALpix in the sky, sum the value of the priority weighting
        per filter.  This gives us the normalization factor to calculate the
        fractional proportional of exposure time"""

        coadded_map = np.zeros(self.map_i.shape)
        for i, f in enumerate(self.filters):
            coadded_map += getattr(self, "map_" + f)

        self.coadded_map = coadded_map

    def calc_idealfExpT(self, filter_name, pixels):
        """Method to calculate the optimal value of the fExpT metric, in the event
        that a survey strategy perfectly overlaps the desired Galactic Plane footprint
        and the exposure time spent in the different filters at the desired relative
        proportions.  Calculation is made over all the pixels in a given dataSlice,
        so that the metric can be accurately normalized later."""

        fexpt_per_pixel = np.zeros(len(pixels))
        fmap = getattr(self, "map_" + filter_name)
        fexpt_per_pixel = fmap[pixels] / self.coadded_map[pixels]

        invalid = np.isnan(fexpt_per_pixel)
        fexpt_per_pixel[invalid] = 0.0

        # The value of the fExpT metric in ideal circumstance is the sum of all values:
        idealfExpT = fexpt_per_pixel.sum()

        return idealfExpT

    def run(self, dataSlice, slicePoint=None):

        # Identify those pixels that overlap the desired Galactic Plane survey region:
        # Returns list of pixel indices?
        # GP_overlap_pixels = np.where(self.map_i[pixels] > 0, 1, 0)

        # DOES A DATASLICE CONTAIN JUST A SUBSET OF PIXELS?
        total_expt_per_pixel = dataSlice[self.exptCol].sum()

        # Calculate the exposure time per filter per pixel as the fraction of the total
        # exposure time per pixel
        fexpt_per_filter = np.zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            coords_icrs = SkyCoord(
                dataSlice[self.ra_col][match],
                dataSlice[self.dec_col][match],
                frame="icrs",
                unit=(u.deg, u.deg),
            )
            coords_gal = coords_icrs.transform_to(Galactic())
            ahp = HEALPix(nside=self.NSIDE, order="ring", frame=TETE())
            pixels = ahp.skycoord_to_healpix(coords_gal)

            idealfExpT = self.calc_idealfExpT(f, pixels)

            fexpt = dataSlice[self.exptCol][match].sum() / total_expt_per_pixel
            fexpt_per_filter[i] = fexpt / idealfExpT

        # Sum the fraction of exposure time over all pixels within the desired
        # survey region and filters:
        metric_value = fexpt_per_filter.sum()

        return metric_value
