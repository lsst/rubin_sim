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
import readGalPlaneMaps

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
        self.MAP_FILE_ROOT_NAME = "priority_GalPlane_footprint_map_data"
        self.load_maps()
        self.calc_coaddmap()

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        for f in self.filters:
            file_path = os.path.join(
                self.MAP_DIR,
                "maf",
                self.MAP_FILE_ROOT_NAME + "_" + str(f) + ".fits",
                )
            map_data_table = readGalPlaneMaps.load_map_data(file_path)

            setattr(self, "map_" + str(f), map_data_table['combined_map'])
            setattr(self, "map_data_" + str(f), map_data_table)

    def calc_coaddmap(self):
        """For each HEALpix in the sky, sum the value of the priority weighting
        per filter.  This gives us the normalization factor to calculate the
        fractional proportional of exposure time"""

        self.coadded_maps = {}

        for i, f in enumerate(self.filters):
            # Fetch the map data for this filter
            map_data_table = getattr(self, 'map_data_'+str(f))

            for col in map_data_table.columns:
                if col.name in self.coadded_maps.keys():
                    coadded_map = self.coadded_maps[col.name]
                else:
                    coadded_map = np.zeros(self.NPIX)

                # Extract the HEALpix priority data for the current science case
                fmap = map_data_table[col.name]

                # Sum the priority values for each HEALpix in this science region
                # over all filters
                coadded_map += fmap

                self.coadded_maps[col.name] = coadded_map

    def calc_idealfExpT(self, filter_name, pixels, col):
        """Method to calculate the optimal value of the fExpT metric, in the event
        that a survey strategy perfectly overlaps the desired Galactic Plane footprint
        and the exposure time spent in the different filters at the desired relative
        proportions.  Calculation is made over all the pixels in a given dataSlice,
        so that the metric can be accurately normalized later."""

        fexpt_per_pixel = np.zeros(len(pixels))

        # Extract the HEALpix priority data for the current science case in
        # the appropriate filter.  This returns a list of priority values for
        # all HEALpixels, with zero values for regions outside the area of interest
        map_data_table = getattr(self, 'map_data_'+str(filter_name))
        fmap = map_data_table[col.name]

        # First select the pixels within the science region of interest for the
        # current filter.  The survey region for a given science case can vary
        # between filters
        idx = np.where(fmap > 0.0)[0]

        # The ideal ratio of observations in this filter, relative to all observations
        # for a given pixel is estimated from the ratio of priority per pixel to
        # the priority of each pixel summed over all filters.
        idealfExpT = fmap[idx].sum() / self.coadded_maps[col.name][idx].sum()

        return idealfExpT

    def calculate_overlap_survey_region(self,region_pixels, dataSlice, match):

        if len(match) > 0:
            # Calculate which HEALpixels observations in the dataSlice correspond to
            coords_icrs = SkyCoord(
                dataSlice[self.ra_col][match],
                dataSlice[self.dec_col][match],
                frame="icrs",
                unit=(u.deg, u.deg),
            )
            coords_gal = coords_icrs.transform_to(Galactic())
            ahp = HEALPix(nside=self.NSIDE, order="ring", frame=TETE())
            pixels = ahp.skycoord_to_healpix(coords_gal)

            # Calculate the overlap between the observed HEALpixels and
            # those from the desired survey region
            overlap_pixels = list(set(pixels.tolist()).intersection(set(region_pixels.tolist())))

            # Identify which observations from the dataSlice correspond to
            # the overlapping survey region.  This may produce multiple
            # indices in the array, referred to different observations
            match = np.array(match)
            match_obs = []
            for p in overlap_pixels:
                ip = np.where(pixels == p)[0]
                match_obs += match[ip].tolist()

        else:
            overlap_pixels = []
            match_obs = []

        return overlap_pixels, match_obs

    def run(self, dataSlice, slicePoint=None, verbose=False):

        # Pre-calculating data that will be used later
        total_expt_per_pixel = dataSlice[self.exptCol].sum()

        # This metric is calculated both for the combined priority map as well
        # as the maps for the component science cases, hence the returned
        # metric_data is a dictionary.  ideal_data contains the corresponding
        # metric values if the survey strategy returned ideal results for
        # each given science case; these data are used for normalization
        metric_data = {}
        for f in self.filters:
            metric_data[f] = {}

        for i, f in enumerate(self.filters):
            # Fetch the map data for this filter
            map_data_table = getattr(self, 'map_data_'+str(f))

            # Select observations within the OpSim for the current filter
            # which match the S/N requirement, and extract the exposure times
            # for those observations
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            for col in map_data_table.columns:

                # Pixels within the current region of scientific interest:
                region_pixels = np.where(map_data_table[col.name] > 0.0)[0]

                # Now find the overlap between the HEALpix covered by the dataSlice
                # and the ideal survey region for this science case, and the
                # dataSlice observations corresponding to this region
                (overlap_pixels, match_obs) = self.calculate_overlap_survey_region(region_pixels, dataSlice, match)

                # Calculate the expected fraction of the total exposure time
                # that would be dedicated to observations in this filter
                # summed over all HEALpixels in the desired region, if the
                # survey strategy exactly matched the needs of the current science case
                idealfExpT = self.calc_idealfExpT(f, overlap_pixels, col)
                if verbose:
                    print('ideal f/ExpT = ',idealfExpT)

                # Now calculate the actual fraction of exposure time spend
                # in this filter summed over all HEALpix from the overlap region.
                # If no exposures are expected in this filter, this returns 1
                # on the principle that 100% of the expected observations are
                # provided, and additional data in other filters is usually welcome
                if verbose:
                    print('Exposure time for matching exposures = ',dataSlice[self.exptCol][match_obs].sum())
                    print('Total exposure time per pixel = ',total_expt_per_pixel)
                fexpt = dataSlice[self.exptCol][match_obs].sum() / total_expt_per_pixel
                if verbose:
                    print('Exp fraction = ',fexpt)
                if idealfExpT > 0:
                    metric = fexpt / idealfExpT
                else:
                    metric = 1.0
                if verbose:
                    print('Exp fraction relative to ideal fraction = ',metric,f,col.name)

                # Accumulate the product of this metric over all filters for each science region
                metric_data_filter = metric_data[f]
                if col.name not in metric_data_filter.keys():
                    metric_data_filter[col.name] = metric
                else:
                    metric_data_filter[col.name] *= metric
                metric_data[f] = metric_data_filter

        return metric_data
