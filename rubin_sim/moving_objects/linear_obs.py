import logging
import datetime
import numpy as np
from scipy import interpolate

from .base_obs import BaseObs

__all__ = ["LinearObs"]


class LinearObs(BaseObs):
    """Generate observations for a set of Orbits using linear interpolation.

    Uses linear interpolations between grid of true ephemerides.
    Ephemerides can be generated using 2-body or n-body integration.

    Parameters
    ----------
    footPrint: `str`, optional
        Specify the footprint for the FOV. Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint (following a rough cut with a circular FOV).
        Default is circular FOV.
    r_fov : `float`, optional
        If footprint is "circle", this is the radius of the fov (in degrees).
        Default 1.75 degrees.
    x_tol : `float`, optional
        If footprint is "rectangle", this is half of the width of the (on-sky) fov in the RA
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
    obs_rot_sky_pos: str, optional
        Name of the Rotator column in the obsData. Default 'rotSkyPos'.
    obs_degrees: `bool`, optional
        Whether the observational data is in degrees or radians. Default True (degrees).
    outfile_name : `str`, optional
        The output file name.
        Default is 'lsst_obs.dat'.
    obs_metadata : `str`, optional
        A string that captures provenance information about the observations.
        For example: 'baseline_v2.0_10yrs', MJD 59853-61677' or 'baseline2018a minus NES'
        Default ''.
    tstep : `float`, optional
        The time between points in the ephemeris grid, in days.
        Default 2 hours.
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
        tstep=2.0 / 24.0,
    ):
        super().__init__(
            footprint=footprint,
            r_fov=r_fov,
            x_tol=x_tol,
            y_tol=y_tol,
            eph_mode=eph_mode,
            eph_type=eph_type,
            obs_code=obs_code,
            eph_file=eph_file,
            obs_time_col=obs_time_col,
            obs_time_scale=obs_time_scale,
            seeing_col=seeing_col,
            visit_exp_time_col=visit_exp_time_col,
            obs_ra=obs_ra,
            obs_dec=obs_dec,
            obs_rot_sky_pos=obs_rot_sky_pos,
            obs_degrees=obs_degrees,
            outfile_name=outfile_name,
            obs_metadata=obs_metadata,
        )
        self.tstep = tstep

    def _header_meta(self):
        # Linear obs header metadata.
        self.outfile.write("# linear obs header metadata\n")
        self.outfile.write(
            "# observation generation via %s\n" % self.__class__.__name__
        )
        self.outfile.write("# ephMode %s\n" % (self.eph_mode))
        self.outfile.write("# time step for ephemeris grid %f\n" % self.tstep)

    # Linear interpolation
    def _make_interps(self, ephs):
        """Generate the interpolation functions for the linear interpolation.

        Parameters
        ----------
        ephs : `np.ndarray`
            Grid of actual ephemerides, for a single object.

        Returns
        -------
        interpfuncs : `dict`
            Dictionary of the interpolation functions.
        """
        interpfuncs = {}
        for n in ephs.dtype.names:
            if n == "time":
                continue
            interpfuncs[n] = interpolate.interp1d(
                ephs["time"], ephs[n], kind="linear", assume_sorted=True, copy=False
            )
        return interpfuncs

    def _interp_ephs(self, interpfuncs, times, columns=None):
        """Calculate the linear interpolation approximations of the ephemeride columns.

        Parameters
        ----------
        interpfuncs : dict
            Dictionary of the linear interpolation functions.
        times : np.ndarray
            Times at which to generate ephemerides.
        columns : list of str, optional
            List of the values to generate ephemerides for.
            Default None = generate all values.

        Returns
        -------
        np.recarray
            Array of interpolated ephemerides.
        """
        if columns is None:
            columns = interpfuncs.keys()
        dtype = []
        for col in columns:
            dtype.append((col, "<f8"))
        dtype.append(("time", "<f8"))
        ephs = np.recarray([len(times)], dtype=dtype)
        for col in columns:
            ephs[col] = interpfuncs[col](times)
        ephs["time"] = times
        return ephs

    def run(self, orbits, obs_data):
        """Find and write the observations of each object to disk.

        For each object, identify the observations where the object is
        within rFOV of the pointing boresight (potentially, also in the camera footprint),
        and write the ephemeris values and observation metadata to disk.
        Uses linear interpolation between ephemeris gridpoints.

        Parameters
        ----------
        orbits : `rubin_sim.moving_objects.Orbits`
            The orbits to generate ephemerides for.
        obs_data : `np.ndarray`
            The simulated pointing history data.
        """
        # Set the times for the ephemeris grid.
        time_step = float(self.tstep)
        time_start = obs_data[self.obs_time_col].min() - time_step
        time_end = obs_data[self.obs_time_col].max() + time_step
        times = np.arange(time_start, time_end + time_step / 2.0, time_step)
        logging.info(
            "Generating ephemerides on a grid of %f day timesteps, then will extrapolate "
            "to opsim times." % (time_step)
        )
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for i, sso in enumerate(orbits):
            objid = sso.orbits["obj_id"].iloc[0]
            sedname = sso.orbits["sed_filename"].iloc[0]
            # Generate ephemerides on a grid.
            logging.debug(
                ("%d/%d   id=%s : " % (i, len(orbits), objid))
                + datetime.datetime.now().strftime("Start: %Y-%m-%d %H:%M:%S")
                + " nTimes: %s" % len(times)
            )
            ephs = self.generate_ephemerides(
                sso, times, eph_mode=self.eph_mode, eph_type=self.eph_type
            )[0]
            interpfuncs = self._make_interps(ephs)
            ephs = self._interp_ephs(
                interpfuncs, times=obs_data[self.obs_time_col], columns=["ra", "dec"]
            )
            logging.debug(
                ("%d/%d   id=%s : " % (i, len(orbits), objid))
                + datetime.datetime.now().strftime("Interp end: %Y-%m-%d %H:%M:%S")
            )
            # Find objects in the chosen footprint (circular, rectangular or camera)
            idx_obs = self.sso_in_fov(ephs, obs_data)
            logging.info(
                ("Object %d/%d   id=%s : " % (i, len(orbits), objid))
                + "Object in %d visits" % (len(idx_obs))
            )
            if len(idx_obs) > 0:
                obsdat = obs_data[idx_obs]
                ephs = self._interp_ephs(interpfuncs, times=obsdat[self.obs_time_col])
                # Write these observations to disk.
                self.write_obs(objid, ephs, obsdat, sedname=sedname)
