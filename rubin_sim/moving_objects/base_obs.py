__all__ = ("BaseObs",)

import os
import warnings

import numpy as np
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import LsstCameraFootprint, angular_separation

from rubin_sim.phot_utils import Bandpass, Sed

from .ooephemerides import PyOrbEphemerides


class BaseObs:
    """
    Base class to generate observations of a set of moving objects.

    Parameters
    ----------
    footPrint: `str`, optional
        Specify the footprint for the FOV.
        Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint
        (following a rough cut with a circular FOV).
        Default is camera FOV.
    r_fov : `float`, optional
        If footprint is "circular", this is the radius of the fov (in degrees).
        Default 1.75 degrees (only used for circular fov).
    x_tol : `float`, optional
        If footprint is "rectangular", this is half of the width
        of the (on-sky) fov in the RA direction (in degrees).
        Default 5 degrees.
    y_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of
        the fov in Declination (in degrees).
        Default is 3 degrees
    eph_mode: `str`, optional
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    eph_type: `str`, optional
        Type of ephemerides to generate - full or basic.
        Full includes all values calculated by openorb;
        Basic includes a more basic set.
        Default is Basic.
    eph_file: `str` or None, optional
        The name of the planetary ephemerides file to use
        for ephemeris generation.
        Default (None) will use the default for PyOrbEphemerides.
    obs_code: `str`, optional
        Observatory code for ephemeris generation.
        Default is "I11" - Cerro Pachon.
    obs_time_col: `str`, optional
        Name of the time column in the obsData.
        Default 'observationStartMJD'.
    obs_time_scale: `str`, optional
        Type of timescale for MJD (TAI or UTC currently).
        Default TAI.
    seeing_col: `str`, optional
        Name of the seeing column in the obsData.
        Default 'seeingFwhmGeom'.
        This should be the geometric/physical seeing
        as it is used for the trailing loss calculation.
    visit_exp_time_col: `str`, optional
        Name of the visit exposure time column in the obsData.
        Default 'visitExposureTime'.
    obs_ra: `str`, optional
        Name of the RA column in the obsData. Default 'fieldRA'.
    obs_dec: `str`, optional
        Name of the Dec column in the obsData. Default 'fieldDec'.
    obs_rot_sky_pos: `str`, optional
        Name of the Rotator column in the obsData. Default 'rotSkyPos'.
    obs_degrees: `bool`, optional
        Whether the observational data is in degrees or radians.
        Default True (degrees).
    outfile_name : `str`, optional
        The output file name.
        Default is 'lsst_obs.dat'.
    obs_info : `str`, optional
        A string that captures provenance information about the observations.
        For example: 'baseline_v2.0_10yrs, years 0-5'
        or 'baseline2018a minus NES'
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
        obs_info="",
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

        self._setup_info(obs_info=obs_info)

    def _setup_info(self, obs_info=""):
        """Generate a dict to record relevant settings"""
        info = {
            "obs_info": obs_info,
            "footprint": self.footprint,
            "eph_mode": self.eph_mode,
            "eph_type": self.eph_type,
        }
        if self.footprint == "circle":
            info["r_fov"] = self.r_fov
        if self.footprint == "rectangle":
            info["xtol"] = self.x_tol
            info["ytol"] = self.y_tol

        # convert to numpy array in anticipation of saving
        names = list(info.keys())
        types = [np.array(info[key]).dtype for key in names]
        self.info = np.zeros(1, dtype=list(zip(names, types)))
        for key in names:
            self.info[key] = info[key]

    def _setup_camera(self, camera_footprint_file=None):
        self.camera = LsstCameraFootprint(units="degrees", footprint_file=camera_footprint_file)

    def setup_ephemerides(self):
        """Initialize the ephemeris generator.
        Save the setup PyOrbEphemeris class.

        This uses the default engine, pyoorb -
        however this could be overwritten to use another generator.
        """
        self.ephems = PyOrbEphemerides(ephfile=self.eph_file)

    def generate_ephemerides(self, sso, times, eph_mode=None, eph_type=None):
        """Generate ephemerides for 'sso' at times 'times'
        (assuming MJDs, with timescale self.obs_time_scale).

        The default engine here is pyoorb, however other ephemeris generation
        could be used with a matching API to PyOrbEphemerides.

        The initialized pyoorb class (PyOrbEphemerides) is saved,
        to skip setup on subsequent calls.

        Parameters
        ----------
        sso : `rubin_sim.movingObjects.Orbits`
            Typically this will be a single object.
        times: `np.ndarray`
            The times at which to generate ephemerides. MJD.
        eph_mode: `str` or None, optional
            Potentially override default eph_mode (self.eph_mode).
            Must be '2body' or 'nbody'.

        Returns
        -------
        ephs : `pd.Dataframe`
            Results from propigating the orbit(s) to the specified times.
            Columns like:
            obj_id, sedname, time, ra, dec, dradt, ddecdt, phase, solarelon.
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

        'Trailing' losses = loss in sensitivity due to the photons from the
        source being spread over more pixels; thus more sky background is
        included when calculating the flux from the object and thus the SNR
        is lower than for an equivalent brightness stationary/PSF-like source.
        dmagTrail represents this loss.

        'Detection' trailing losses = loss in sensitivity due to the photons
        from the source being spread over more pixels, in a non-stellar-PSF
        way, while source detection is (typically) done using a stellar PSF
        filter and 5-sigma cutoff values based on assuming peaks from
        stellar PSF's above the background; thus the SNR is lower than for an
        equivalent brightness stationary/PSF-like source (and by a greater
        factor than just the simple SNR trailing loss above).
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
        dmag Trail, dmag_detect : (`np.ndarray` `np.ndarray`)
        or (`float`, `float`)
            dmag_trail and dmag_detect for each set of
            velocity/seeing/texp values.
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
        """Read (LSST) and Harris (V) filter throughput curves.

        Only the defaults are LSST specific;
        this can easily be adapted for any survey.

        Parameters
        ----------
        filter_dir : `str`, optional
            Directory containing the filter throughput curves ('total*.dat')
            Default set by 'LSST_THROUGHPUTS_BASELINE' env variable.
        bandpass_root : `str`, optional
            Rootname of the throughput curves in filterlist.
            E.g. throughput curve names are bandpass_root + filterlist[i]
            + bandpass_suffix
            Default `total_` (appropriate for LSST throughput repo).
        bandpass_suffix : `str`, optional
            Suffix for the throughput curves in filterlist.
            Default '.dat' (appropriate for LSST throughput repo).
        filterlist : `list`, optional
            List containing the filter names to use to calculate colors.
            Default ('u', 'g', 'r', 'i', 'z', 'y')
        v_dir : `str`, optional
            Directory containing the V band throughput curve.
            Default None = $RUBIN_SIM_DATA_DIR/movingObjects
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
            self.lsst[f].read_throughput(os.path.join(filter_dir, bandpass_root + f + bandpass_suffix))
        self.vband = Bandpass()
        self.vband.read_throughput(os.path.join(v_dir, v_filter))

    def calc_colors(self, sedname="C.dat", sed_dir=None):
        """Calculate the colors for a given SED.

        If the sedname is not already in the dictionary self.colors,
        this reads the SED from disk and calculates all V-[filter] colors
        for all filters in self.filterlist.
        The result is stored in self.colors[sedname][filter], so will not
        be recalculated if the SED + color is reused for another object.

        Parameters
        ----------
        sedname : `str`, optional
            Name of the SED. Default 'C.dat'.
        sed_dir : `str`, optional
            Directory containing the SEDs of the moving objects.
            Default None = $RUBIN_SIM_DATA_DIR/movingObjects,

        Returns
        -------
        colors : `dict` {'filter': color}}
            Dictionary of the colors in self.filterlist.
        """
        if sedname not in self.colors:
            if sed_dir is None:
                sed_dir = os.path.join(get_data_dir(), "movingObjects")
            mo_sed = Sed()
            mo_sed.read_sed_flambda(os.path.join(sed_dir, sedname))
            vmag = mo_sed.calc_mag(self.vband)
            self.colors[sedname] = {}
            for f in self.filterlist:
                self.colors[sedname][f] = mo_sed.calc_mag(self.lsst[f]) - vmag
        return self.colors[sedname]

    def sso_in_circle_fov(self, ephems, obs_data):
        """Determine which observations are within a circular fov
        for a series of observations.
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
            Returns the indexes of the numpy array of the object
            observations which are inside the fov.
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
        """Determine which observations are within a rectangular FoV
        for a series of observations.
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
            Returns the indexes of the numpy array of the object
            observations which are inside the fov.
        """
        return self._sso_in_rectangle_fov(ephems, obs_data, self.x_tol, self.y_tol)

    def _sso_in_rectangle_fov(self, ephems, obs_data, x_tol, y_tol):
        delta_dec = np.abs(ephems["dec"] - obs_data[self.obs_dec])
        delta_ra = np.abs((ephems["ra"] - obs_data[self.obs_ra]) * np.cos(np.radians(obs_data[self.obs_dec])))
        idx_obs = np.where((delta_dec <= y_tol) & (delta_ra <= x_tol))[0]
        return idx_obs

    def sso_in_camera_fov(self, ephems, obs_data):
        """Determine which observations are within the actual
        camera footprint for a series of observations.
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
            Returns the indexes of the numpy array of the object
            observations which are inside the fov.
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
        """Convenience layer - determine which footprint method to
        apply (from self.footprint) and use it.

        Parameters
        ----------
        ephems : `np.ndarray`
            Ephemerides for the objects.
        obs_data : `np.ndarray`
            Observation pointings.

        Returns
        -------
        indices : `np.ndarray`
            Returns the indexes of the numpy array of the object
            observations which are inside the fov.
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
