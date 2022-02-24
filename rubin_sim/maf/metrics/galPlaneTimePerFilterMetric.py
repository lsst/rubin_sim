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
from rubin_sim.data import get_data_dir
from astropy.io import fits

class galPlaneTimePerFilter(maf.BaseMetric):
    """Metric to evaluate for each HEALpix, the fraction of exposure time spent in each filter as
     a fraction of the total exposure time dedicated to that HEALpix (fExpT).  The metric sums this over
     all HEALpix in the Galactic Plane/Magellanic Clouds region of interest and all filters and
     presents the result as a fraction of the value expected from the optimal survey strategy.


    WARNING: This metric should only be applied to a HEALpix slicer, otherwise
    the metric value retrieved may refer to the wrong part of the sky

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
        science_map   string  Name of the priority footprint map to use from
                                the column headers contained in the
                                priority_GalPlane_footprint_map_data tables
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.mjdCol = "observationStartMJD"
        self.exptCol = "visitExposureTime"
        self.filters = ["u", "g", "r", "i", "z", "y"]
        self.science_map = kwargs['science_map']
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
        self.calc_idealfExpT()

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def load_maps(self):
        self.NSIDE = 64
        self.NPIX = hp.nside2npix(self.NSIDE)
        self.map_data = {}
        for f in self.filters:
            file_path = os.path.join(
                self.MAP_DIR,
                "maf",
                self.MAP_FILE_ROOT_NAME + "_" + str(f) + ".fits",
                )
            with fits.open(file_path) as hdul:
                self.map_data[f] = hdul[1].data

        file_path = os.path.join(
            self.MAP_DIR,
            "maf",
            self.MAP_FILE_ROOT_NAME + "_sum.fits",
            )
        with fits.open(file_path) as hdul:
            self.map_data['sum'] = hdul[1].data

    def calc_idealfExpT(self):
        """Method to calculate the optimal value of the fExpT metric for each
        HEALpixel in the sky.  This calculation is made for each filter
        and for the selected science map."""

        self.ideal_fExpT = {}

        for f in self.filters:
            self.ideal_fExpT[f] = self.map_data[f][self.science_map]/self.map_data['sum'][self.science_map]

    def run(self, dataSlice, slicePoint):

        # Pre-calculating data that will be used later
        total_expt = dataSlice[self.exptCol].sum()

        metric_data = {}
        for f in self.filters:
            metric_data[f] = {}

        for i, f in enumerate(self.filters):
            # Select observations within the OpSim for the current filter
            # which match the S/N requirement, and extract the exposure times
            # for those observations
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))

            # Now calculate the actual fraction of exposure time spent
            # in this filter for the current slicePoint, relative to the total
            # exposure time spent on this slicePoint.
            # Note that this includes dithered observations.
            # If no exposures are expected in this filter, this returns 1
            # on the principle that 100% of the expected observations are
            # provided, and additional data in other filters is usually welcome
            fexpt = dataSlice[self.exptCol][match].sum() / total_expt

            # This value is normalized against the ideal fExpT predicted for this
            # slicePoint based on the priority map data.
            # If no exposures are expected in this filter for this location,
            # this metric returns zero.
            if self.ideal_fExpT[f][slicePoint['sid']] > 0:
                metric_data[f] = fexpt / self.ideal_fExpT[f][slicePoint['sid']]
            else:
                metric_data[f] = 0.0

        return metric_data
