import os
import numpy as np
import pandas as pd
import warnings
import datetime

from rubin_sim.phot_utils import Bandpass
from rubin_sim.phot_utils import Sed
from rubin_sim.utils import angular_separation
from rubin_sim.data import get_data_dir

from .ooephemerides import PyOrbEphemerides
from rubin_sim.utils import LsstCameraFootprint

__all__ = ["BaseObs"]


class BaseObs(object):
    """
    Base class to generate observations of a set of moving objects.

    Parameters
    ----------
    footPrint: `str`, optional
        Specify the footprint for the FOV. Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint (following a rough cut with a circular FOV).
        Default is camera FOV.
    r_fov : `float`, optional
        If footprint is "circular", this is the radius of the fov (in degrees).
        Default 1.75 degrees (only used for circular fov).
    x_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of the (on-sky) fov in the RA
        direction (in degrees).
        Default 5 degrees. (so size of footprint in degrees will be 10 degrees in the RA direction).
    y_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of the fov in Declination (in degrees).
        Default is 3 degrees (so size of footprint in degrees will be 6 degrees in the Dec direction).
    eph_mode: `str`, optional
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    eph_type: `str`, optional
        Type of ephemerides to generate - full or basic.
        Full includes all values calculated by openorb; Basic includes a more basic set.
        Default is Basic.  (this includes enough information for most standard MAF metrics).
    eph_file: `str` or None, optional
        The name of the planetary ephemerides file to use for ephemeris generation.
        Default (None) will use the default for PyOrbEphemerides.
    obs_code: `str`, optional
        Observatory code for ephemeris generation. Default is "I11" - Cerro Pachon.
    obs_time_col: `str`, optional
        Name of the time column in the obsData. Default 'observationStartMJD'.
    obs_time_scale: `str`, optional
        Type of timescale for MJD (TAI or UTC currently). Default TAI.
    seeing_col: `str`, optional
        Name of the seeing column in the obsData. Default 'seeingFwhmGeom'.
        This should be the geometric/physical seeing as it is used for the trailing loss calculation.
    visit_exp_time_col: `str`, optional
        Name of the visit exposure time column in the obsData. Default 'visitExposureTime'.
    obs_ra: `str`, optional
        Name of the RA column in the obsData. Default 'fieldRA'.
    obs_dec: `str`, optional
        Name of the Dec column in the obsData. Default 'fieldDec'.
    obs_rot_sky_pos: `str`, optional
        Name of the Rotator column in the obsData. Default 'rotSkyPos'.
    obs_degrees: `bool`, optional
        Whether the observational data is in degrees or radians. Default True (degrees).
    outfile_name : `str`, optional
        The output file name.
        Default is 'lsst_obs.dat'.
    obs_metadata : `str`, optional
        A string that captures provenance information about the observations.
        For example: 'baseline_v2.0_10yrs, years 0-5' or 'baseline2018a minus NES'
        Default ''.
    """

    def __init__(
        self,
        footprint="camera",
        r_fov=1.75,
        x_tol=5,
        y_tol=3,
        eph_mode="nbody",
        eph_type="basic",
        obs_code="I11",
        eph_file=None,
        obs_time_col="observationStartMJD",
        obs_time_scale="TAI",
        seeing_col="seeingFwhmGeom",
        visit_exp_time_col="visitExposureTime",
        obs_ra="fieldRA",
        obs_dec="fieldDec",
        obs_rot_sky_pos="rotSkyPos",
        obs_degrees=True,
        outfile_name="lsst_obs.dat",
        obs_metadata="",
        camera_footprint_file=None,
    ):
        # Strings relating to the names of columns in the observation metadata.
        self.obs_code = obs_code
        self.obs_time_col = obs_time_col
        self.obs_time_scale = obs_time_scale
        self.seeing_col = seeing_col
        self.visit_exp_time_col = visit_exp_time_col
        self.obs_ra = obs_ra
        self.obs_dec = obs_dec
        self.obs_rot_sky_pos = obs_rot_sky_pos
        self.obs_degrees = obs_degrees
        # Save a space for the standard object colors.
        self.colors = {}
        self.outfile_name = outfile_name
        if obs_metadata == "":
            self.obs_metadata = "unknown simdata source"
        else:
            self.obs_metadata = obs_metadata
        # Values for identifying observations.
        self.footprint = footprint.lower()
        if self.footprint == "camera":
            self._setup_camera(camera_footprint_file=camera_footprint_file)
        self.r_fov = r_fov
        self.x_tol = x_tol
        self.y_tol = y_tol
        # Values for ephemeris generation.
        if eph_mode.lower() not in ("2body", "nbody"):
            raise ValueError("Ephemeris generation must be 2body or nbody.")
        self.eph_mode = eph_mode
        self.eph_type = eph_type
        self.eph_file = eph_file

    def _setup_camera(self, camera_footprint_file=None):
        self.camera = LsstCameraFootprint(
            units="degrees", footprint_file=camera_footprint_file
        )

    def setup_ephemerides(self):
        """Initialize the ephemeris generator. Save the setup PyOrbEphemeris class.

        This uses the default engine, pyoorb - however this could be overwritten to use another generator.
        """
        self.ephems = PyOrbEphemerides(ephfile=self.eph_file)

    def generate_ephemerides(self, sso, times, eph_mode=None, eph_type=None):
        """Generate ephemerides for 'sso' at times 'times' (assuming MJDs, with timescale self.obs_time_scale).

        The default engine here is pyoorb, however this method could be overwritten to use another ephemeris
        generator, such as ADAM.

        The initialized pyoorb class (PyOrbEphemerides) is saved, to skip setup on subsequent calls.

        Parameters
        ----------
        sso : `rubin_sim.movingObjects.Orbits`
            Typically this will be a single object.
        times: `np.ndarray`
            The times at which to generate ephemerides. MJD.
        eph_mode: `str` or None, optional
            Potentially override default eph_mode (self.eph_mode). Must be '2body' or 'nbody'.

        Returns
        -------
        ephs : `pd.Dataframe`
            Ephemerides of the sso.
        """
        if not hasattr(self, "ephems"):
            self.setup_ephemerides()
        if eph_mode is None:
            eph_mode = self.eph_mode
        if eph_type is None:
            eph_type = self.eph_type
        self.ephems.set_orbits(sso)
        ephs = self.ephems.generate_ephemerides(
            times,
            time_scale=self.obs_time_scale,
            obscode=self.obs_code,
            eph_mode=eph_mode,
            eph_type=eph_type,
            by_object=True,
        )
        return ephs

    def calc_trailing_losses(self, velocity, seeing, texp=30.0):
        """Calculate the detection and SNR trailing losses.

        'Trailing' losses = loss in sensitivity due to the photons from the source being
        spread over more pixels; thus more sky background is included when calculating the
        flux from the object and thus the SNR is lower than for an equivalent brightness
        stationary/PSF-like source. dmagTrail represents this loss.

        'Detection' trailing losses = loss in sensitivity due to the photons from the source being
        spread over more pixels, in a non-stellar-PSF way, while source detection is (typically) done
        using a stellar PSF filter and 5-sigma cutoff values based on assuming peaks from stellar PSF's
        above the background; thus the SNR is lower than for an equivalent brightness stationary/PSF-like
        source (and by a greater factor than just the simple SNR trailing loss above).
        dmag_detect represents this loss.

        Parameters
        ----------
        velocity : `np.ndarray` or `float`
            The velocity of the moving objects, in deg/day.
        seeing : `np.ndarray` or `float`
            The seeing of the images, in arcseconds.
        texp : `np.ndarray` or `float`, optional
            The exposure time of the images, in seconds. Default 30.

        Returns
        -------
        dmag Trail, dmag_detect : (`np.ndarray`, `np.ndarray`) or (`float`, `float`)
            dmag_trail and dmag_detect for each set of velocity/seeing/texp values.
        """
        a_trail = 0.761
        b_trail = 1.162
        a_det = 0.420
        b_det = 0.003
        x = velocity * texp / seeing / 24.0
        dmag_trail = 1.25 * np.log10(1 + a_trail * x**2 / (1 + b_trail * x))
        dmag_detect = 1.25 * np.log10(1 + a_det * x**2 / (1 + b_det * x))
        return (dmag_trail, dmag_detect)

    def read_filters(
        self,
        filter_dir=None,
        bandpass_root="total_",
        bandpass_suffix=".dat",
        filterlist=("u", "g", "r", "i", "z", "y"),
        v_dir=None,
        v_filter="harris_V.dat",
    ):
        """
        Read (LSST) and Harris (V) filter throughput curves.

        Only the defaults are LSST specific; this can easily be adapted for any survey.

        Parameters
        ----------
        filter_dir : `str`, optional
            Directory containing the filter throughput curves ('total*.dat')
            Default set by 'LSST_THROUGHPUTS_BASELINE' env variable.
        bandpass_root : `str`, optional
            Rootname of the throughput curves in filterlist.
            E.g. throughput curve names are bandpass_root + filterlist[i] + bandpass_suffix
            Default total\_ (appropriate for LSST throughput repo).
        bandpass_suffix : `str`, optional
            Suffix for the throughput curves in filterlist.
            Default '.dat' (appropriate for LSST throughput repo).
        filterlist : `list`, optional
            List containing the filter names to use to calculate colors.
            Default ('u', 'g', 'r', 'i', 'z', 'y')
        v_dir : `str`, optional
            Directory containing the V band throughput curve.
            Default None = $SIMS_MOVINGOBJECTS_DIR/data.
        v_filter : `str`, optional
            Name of the V band filter curve.
            Default harris_V.dat.
        """
        if filter_dir is None:
            filter_dir = os.path.join(get_data_dir(), "throughputs/baseline")
        if v_dir is None:
            v_dir = os.path.join(get_data_dir(), "movingObjects")
        self.filterlist = filterlist
        # Read filter throughput curves from disk.
        self.lsst = {}
        for f in self.filterlist:
            self.lsst[f] = Bandpass()
            self.lsst[f].readThroughput(
                os.path.join(filter_dir, bandpass_root + f + bandpass_suffix)
            )
        self.vband = Bandpass()
        self.vband.readThroughput(os.path.join(v_dir, v_filter))

    def calc_colors(self, sedname="C.dat", sed_dir=None):
        """Calculate the colors for a given SED.

        If the sedname is not already in the dictionary self.colors, this reads the
        SED from disk and calculates all V-[filter] colors for all filters in self.filterlist.
        The result is stored in self.colors[sedname][filter], so will not be recalculated if
        the SED + color is reused for another object.

        Parameters
        ----------
        sedname : `str`, optional
            Name of the SED. Default 'C.dat'.
        sed_dir : `str`, optional
            Directory containing the SEDs of the moving objects.
            Default None = $SIMS_MOVINGOBJECTS_DIR/data.

        Returns
        -------
        colors : `dict`
            Dictionary of the colors in self.filterlist for this particular Sed.
        """
        if sedname not in self.colors:
            if sed_dir is None:
                sed_dir = os.path.join(get_data_dir(), "movingObjects")
            mo_sed = Sed()
            mo_sed.readSED_flambda(os.path.join(sed_dir, sedname))
            vmag = mo_sed.calcMag(self.vband)
            self.colors[sedname] = {}
            for f in self.filterlist:
                self.colors[sedname][f] = mo_sed.calcMag(self.lsst[f]) - vmag
        return self.colors[sedname]

    def sso_in_circle_fov(self, ephems, obs_data):
        """Determine which observations are within a circular fov for a series of observations.
        Note that ephems and obs_data must be the same length.

        Parameters
        ----------
        ephems : `np.recarray`
            Ephemerides for the objects.
        obs_data : `np.recarray`
            The observation pointings.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        return self._sso_in_circle_fov(ephems, obs_data, self.r_fov)

    def _sso_in_circle_fov(self, ephems, obs_data, r_fov):
        if not self.obs_degrees:
            sep = angular_separation(
                ephems["ra"],
                ephems["dec"],
                np.degrees(obs_data[self.obs_ra]),
                np.degrees(obs_data[self.obs_dec]),
            )
        else:
            sep = angular_separation(
                ephems["ra"],
                ephems["dec"],
                obs_data[self.obs_ra],
                obs_data[self.obs_dec],
            )
        idx_obs = np.where(sep <= r_fov)[0]
        return idx_obs

    def sso_in_rectangle_fov(self, ephems, obs_data):
        """Determine which observations are within a rectangular FoV for a series of observations.
        Note that ephems and obs_data must be the same length.

        Parameters
        ----------
        ephems : `np.recarray`
            Ephemerides for the objects.
        obs_data : `np.recarray`
            The observation pointings.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        return self._sso_in_rectangle_fov(ephems, obs_data, self.x_tol, self.y_tol)

    def _sso_in_rectangle_fov(self, ephems, obs_data, x_tol, y_tol):
        delta_dec = np.abs(ephems["dec"] - obs_data[self.obs_dec])
        delta_ra = np.abs(
            (ephems["ra"] - obs_data[self.obs_ra])
            * np.cos(np.radians(obs_data[self.obs_dec]))
        )
        idx_obs = np.where((delta_dec <= y_tol) & (delta_ra <= x_tol))[0]
        return idx_obs

    def sso_in_camera_fov(self, ephems, obs_data):
        """Determine which observations are within the actual camera footprint for a series of observations.
        Note that ephems and obs_data must be the same length.

        Parameters
        ----------
        ephems : `np.ndarray`
            Ephemerides for the objects.
        obs_data : `np.ndarray`
            Observation pointings.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        if not hasattr(self, "camera"):
            self._setup_camera()

        if not self.obs_degrees:
            idx = self.camera(
                ephems["ra"],
                ephems["dec"],
                np.degrees(obs_data[self.obs_ra]),
                np.degrees(obs_data[self.obs_dec]),
                np.degrees(obs_data[self.obs_rot_sky_pos]),
            )
        else:
            idx = self.camera(
                ephems["ra"],
                ephems["dec"],
                obs_data[self.obs_ra],
                obs_data[self.obs_dec],
                obs_data[self.obs_rot_sky_pos],
            )
        return idx

    def sso_in_fov(self, ephems, obs_data):
        """Convenience layer - determine which footprint method to apply (from self.footprint) and use it.

        Parameters
        ----------
        ephems : `np.ndarray`
            Ephemerides for the objects.
        obs_data : `np.ndarray`
            Observation pointings.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object observations which are inside the fov.
        """
        if self.footprint == "camera":
            return self.sso_in_camera_fov(ephems, obs_data)
        elif self.footprint == "rectangle":
            return self.sso_in_rectangle_fov(ephems, obs_data)
        elif self.footprint == "circle":
            return self.sso_in_circle_fov(ephems, obs_data)
        else:
            warnings.warn("Using circular fov; could not match specified footprint.")
            self.footprint = "circle"
            return self.sso_in_circle_fov(ephems, obs_data)

    # Put together the output.
    def _open_output(self):
        # Make sure the directory exists to write the output file into.
        out_dir = os.path.split(self.outfile_name)[0]
        if len(out_dir) > 0:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        # Open the output file for writing.
        self.outfile = open(self.outfile_name, "w")
        self.outfile.write("# Started at %s" % (datetime.datetime.now()))
        # Write metadata into the header, using # to identify as comment lines.
        self.outfile.write("# %s\n" % self.obs_metadata)
        self.outfile.write("# %s\n" % self.outfile_name)
        # Write some generic ephemeris generation information.
        self.outfile.write(
            "# ephemeris generation via %s\n" % self.ephems.__class__.__name__
        )
        self.outfile.write("# planetary ephemeris file %s \n" % self.ephems.ephfile)
        self.outfile.write("# obscode %s\n" % self.obs_code)
        # Write some class-specific metadata about observation generation.
        self._header_meta()
        # Write the footprint information.
        self.outfile.write("# pointing footprint %s\n" % (self.footprint))
        if self.footprint == "circle":
            self.outfile.write("# rfov %f\n" % self.r_fov)
        if self.footprint == "rectangle":
            self.outfile.write("# xTol %f yTol %f\n" % (self.x_tol, self.y_tol))
        # Record columns used from simulation data
        self.outfile.write(
            "# obsRA %s obsDec %s obsRotSkyPos %s obsDeg %s\n"
            % (self.obs_ra, self.obs_dec, self.obs_rot_sky_pos, self.obs_degrees)
        )
        self.outfile.write(
            "# obsMJD %s obsTimeScale %s seeing %s expTime %s\n"
            % (
                self.obs_time_col,
                self.obs_time_scale,
                self.seeing_col,
                self.visit_exp_time_col,
            )
        )

        self.wrote_header = False

    def _header_meta(self):
        # Generic class header metadata, should be overriden with class specific version.
        self.outfile.write("# generic header metadata\n")
        self.outfile.write("# ephMode %s\n" % (self.eph_mode))

    def write_obs(self, obj_id, obj_ephs, obs_data, sedname="C.dat"):
        """
        Call for each object; write out the observations of each object.

        This method is called once all of the ephemeris values for each observation are calculated;
        the calling method should have already done the matching between ephemeris & simulated observations
        to find the observations where the object is within the specified fov.
        Inside this method, the trailing losses and color terms are calculated and added to the output
        observation file.

        The first time this method is called, a header will be added to the output file.

        Parameters
        ----------
        obj_id : `str` or `int` or `float`
            The identifier for the object (from the orbital parameters)
        obj_ephs : `np.ndarray`
            The ephemeris values of the object at each observation.
            Note that the names of the columns are encoded in the numpy structured array,
            and any columns included in the returned ephemeris array will also be propagated to the output.
        obs_data : `np.ndarray`
            The observation details from the simulated pointing history, for all observations of
            the object. All columns automatically propagated to the output file.
        sedname : `str`, out
            The sed_filename for the object (from the orbital parameters).
            Used to calculate the appropriate color terms for the output file.
            Default "C.dat".
        """
        # Return if there's nothing to write out.
        if len(obj_ephs) == 0:
            return
        # Open file if needed.
        if not hasattr(self, "outfile"):
            self._open_output()
        # Calculate the extra columns we want to write out
        # (dmag due to color, trailing loss, and detection loss)
        # First calculate and match the color dmag term.
        dmag_color = np.zeros(len(obs_data), float)
        dmag_color_dict = self.calc_colors(sedname)
        filterlist = np.unique(obs_data["filter"])
        for f in filterlist:
            if f not in dmag_color_dict:
                raise UserWarning(
                    "Could not find filter %s in calculated colors!" % (f)
                )
            match = np.where(obs_data["filter"] == f)[0]
            dmag_color[match] = dmag_color_dict[f]
        # Calculate trailing and detection loses.
        dmag_trail, dmag_detect = self.calc_trailing_losses(
            obj_ephs["velocity"],
            obs_data[self.seeing_col],
            obs_data[self.visit_exp_time_col],
        )
        # Turn into a recarray so it's easier below.
        dmags = np.rec.fromarrays(
            [dmag_color, dmag_trail, dmag_detect],
            names=["dmag_color", "dmag_trail", "dmag_detect"],
        )

        obs_data_names = list(obs_data.dtype.names)
        obs_data_names.sort()

        out_cols = (
            [
                "obj_id",
            ]
            + list(obj_ephs.dtype.names)
            + obs_data_names
            + list(dmags.dtype.names)
        )

        if not self.wrote_header:
            writestring = ""
            for col in out_cols:
                writestring += "%s " % (col)
            self.outfile.write("%s\n" % (writestring))
            self.wrote_header = True

        # Write results.
        # XXX--should remove nested for-loops. Looks like there is a hodgepodge
        # of arrays, structured arrays, and record arrays. Probably a good idea to
        # refactor to eliminate the rec arrays, then it should be easy to stack things
        # and use np.savetxt to eliminate all the loops.
        for eph, simdat, dm in zip(obj_ephs, obs_data, dmags):
            writestring = "%s " % (obj_id)
            for col in eph.dtype.names:
                writestring += "%s " % (eph[col])
            for col in obs_data_names:
                writestring += "%s " % (simdat[col])
            for col in dm.dtype.names:
                writestring += "%s " % (dm[col])
            self.outfile.write("%s\n" % (writestring))

    def _close_output(self):
        self.outfile.write("# Finished at %s" % (datetime.datetime.now()))
