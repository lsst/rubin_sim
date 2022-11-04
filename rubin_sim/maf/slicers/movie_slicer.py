from builtins import str

# cumulative one dimensional movie slicer
import os
import warnings
import subprocess
from subprocess import CalledProcessError
import numpy as np
from functools import wraps

from rubin_sim.maf.utils import percentileClipping, optimalBins
from rubin_sim.maf.stackers import ColInfo
from .base_slicer import BaseSlicer

__all__ = ["MovieSlicer"]


class MovieSlicer(BaseSlicer):
    def __init__(
        self,
        slice_col_name=None,
        slice_col_units=None,
        bins=None,
        bin_min=None,
        binMax=None,
        binsize=None,
        verbose=True,
        badval=0,
        cumulative=True,
        forceNoFfmpeg=False,
    ):
        """
        The 'movieSlicer' acts similarly to the OneDSlicer (slices on one data column).
        However, the data slices from the movieSlicer are intended to be fed to another slicer, which then
        (together with a set of Metrics) calculates metric values + plots at each slice created by the movieSlicer.
        The job of the movieSlicer is to track those slices and put them together into a movie.

        'slice_col_name' is the name of the data column to use for slicing.
        'slice_col_units' lets the user set the units (for plotting purposes) of the slice column.
        'bins' can be a numpy array with the binpoints for sliceCol or a single integer value
        (if a single value, this will be used as the number of bins, together with data min/max or bin_min/Max),
        as in numpy's histogram function.
        If 'binsize' is used, this will override the bins value and will be used together with the data min/max
        or bin_min/Max to set the binpoint values.

        Bins work like numpy histogram bins: the last 'bin' value is end value of last bin;
          all bins except for last bin are half-open ([a, b>), the last one is ([a, b]).

        The movieSlicer stitches individual frames together into a movie using ffmpeg. Thus, on
        instantiation it checks that ffmpeg is available and will raise and exception if not.
        This behavior can be overriden using forceNoFfmpeg = True (in order to create a movie later perhaps).
        """
        # Check for ffmpeg.
        if not forceNoFfmpeg:
            try:
                fnull = open(os.devnull, "w")
                subprocess.check_call(["which", "ffmpeg"], stdout=fnull)
            except CalledProcessError:
                raise Exception(
                    "Could not find ffmpeg on the system, so will not be able to create movie."
                    " Use forceNoFfmpeg=True to override this error and create individual images."
                )
        super().__init__(verbose=verbose, badval=badval)
        self.slice_col_name = slice_col_name
        self.columns_needed = [slice_col_name]
        self.bins = bins
        self.bin_min = bin_min
        self.binMax = binMax
        self.binsize = binsize
        self.cumulative = cumulative
        if slice_col_units is None:
            co = ColInfo()
            self.slice_col_units = co.getUnits(self.slice_col_name)
        self.slicer_init = {
            "slice_col_name": self.slice_col_name,
            "slice_col_units": slice_col_units,
            "badval": badval,
        }

    def setup_slicer(self, sim_data, maps=None):
        """
        Set up bins in slicer.
        """
        if self.slice_col_name is None:
            raise Exception("slice_col_name was not defined when slicer instantiated.")
        sliceCol = sim_data[self.slice_col_name]
        # Set bin min/max values.
        if self.bin_min is None:
            self.bin_min = sliceCol.min()
        if self.binMax is None:
            self.binMax = sliceCol.max()
        # Give warning if bin_min = binMax, and do something at least slightly reasonable.
        if self.bin_min == self.binMax:
            warnings.warn(
                "bin_min = binMax (maybe your data is single-valued?). "
                "Increasing binMax by 1 (or 2*binsize, if binsize set)."
            )
            if self.binsize is not None:
                self.binMax = self.binMax + 2 * self.binsize
            else:
                self.binMax = self.binMax + 1
        # Set bins.
        # Using binsize.
        if self.binsize is not None:
            if self.bins is not None:
                warnings.warn(
                    "Both binsize and bins have been set; Using binsize %f only."
                    % (self.binsize)
                )
            self.bins = np.arange(
                self.bin_min,
                self.binMax + self.binsize / 2.0,
                float(self.binsize),
                "float",
            )
        # Using bins value.
        else:
            # Bins was a sequence (np array or list)
            if hasattr(self.bins, "__iter__"):
                self.bins = np.sort(self.bins)
            # Or bins was a single value.
            else:
                if self.bins is None:
                    self.bins = optimalBins(sliceCol, self.bin_min, self.binMax)
                nbins = int(self.bins)
                self.binsize = (self.binMax - self.bin_min) / float(nbins)
                self.bins = np.arange(
                    self.bin_min, self.binMax + self.binsize / 2.0, self.binsize, "float"
                )
        # Set nbins to be one less than # of bins because last binvalue is RH edge only
        self.nslice = len(self.bins) - 1
        # Set slicePoint metadata.
        self.slicePoints["sid"] = np.arange(self.nslice)
        self.slicePoints["bins"] = self.bins
        # Add metadata from maps.
        self._run_maps(maps)
        # Set up data slicing.
        self.sim_idxs = np.argsort(sim_data[self.slice_col_name])
        simFieldsSorted = np.sort(sim_data[self.slice_col_name])
        # "left" values are location where simdata == bin value
        self.left = np.searchsorted(simFieldsSorted, self.bins[:-1], "left")
        self.left = np.concatenate(
            (
                self.left,
                np.array(
                    [
                        len(self.sim_idxs),
                    ]
                ),
            )
        )
        # Set up _slice_sim_data method for this class.
        if self.cumulative:

            @wraps(self._slice_sim_data)
            def _slice_sim_data(islice):
                """
                Slice sim_data on oneD sliceCol, to return relevant indexes for slicepoint.
                """
                # this is the important part. The ids here define the pieces of data that get
                # passed on to subsequent slicers
                # cumulative version of 1D slicing
                idxs = self.sim_idxs[0 : self.left[islice + 1]]
                return {
                    "idxs": idxs,
                    "slicePoint": {
                        "sid": islice,
                        "binLeft": self.bins[0],
                        "binRight": self.bins[islice + 1],
                    },
                }

            setattr(self, "_slice_sim_data", _slice_sim_data)
        else:

            @wraps(self._slice_sim_data)
            def _slice_sim_data(islice):
                """
                Slice sim_data on oneD sliceCol, to return relevant indexes for slicepoint.
                """
                idxs = self.sim_idxs[self.left[islice] : self.left[islice + 1]]
                return {
                    "idxs": idxs,
                    "slicePoint": {
                        "sid": islice,
                        "binLeft": self.bins[islice],
                        "binRight": self.bins[islice + 1],
                    },
                }

            setattr(self, "_slice_sim_data", _slice_sim_data)

    def __eq__(self, other_slicer):
        """
        Evaluate if slicers are equivalent.
        """
        if isinstance(other_slicer, MovieSlicer):
            return np.array_equal(
                other_slicer.slicePoints["bins"], self.slicePoints["bins"]
            )
        else:
            return False

    def makeMovie(
        self,
        outfileroot,
        sliceformat,
        plotType,
        figformat,
        outDir="Output",
        ips=10.0,
        fps=10.0,
    ):
        """
        Takes in metric and slicer metadata and calls ffmpeg to stitch together output files.
        """
        if not os.path.isdir(outDir):
            raise Exception(
                "Cannot find output directory %s with movie input files." % (outDir)
            )
        # make video
        # ffmpeg -r 30 -i [image names - FilterColors_%03d_SkyMap.png] -r 30
        #    -pix_fmt yuv420p -crf 18 -preset slower outfile
        callList = [
            "ffmpeg",
            "-r",
            str(ips),
            "-i",
            os.path.join(
                outDir, "%s_%s_%s.%s" % (outfileroot, sliceformat, plotType, figformat)
            ),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "slower",
            os.path.join(
                outDir, "%s_%s_%s_%s.mp4" % (outfileroot, plotType, str(ips), str(fps))
            ),
        ]
        print("Attempting to call ffmpeg with:")
        print(" ".join(callList))
        p = subprocess.check_call(callList)
        # make thumbnail gif
        callList = [
            "ffmpeg",
            "-i",
            os.path.join(
                outDir, "%s_%s_%s_%s.mp4" % (outfileroot, plotType, str(ips), str(fps))
            ),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-t",
            str(10),
            "-r",
            str(10),
            os.path.join(
                outDir, "%s_%s_%s_%s.gif" % (outfileroot, plotType, str(ips), str(fps))
            ),
        ]
        print("converting to animated gif with:")
        print(" ".join(callList))
        p2 = subprocess.check_call(callList)
