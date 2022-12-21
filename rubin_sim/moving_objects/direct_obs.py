import logging
import numpy as np
import datetime

from .base_obs import BaseObs

__all__ = ["DirectObs"]


class DirectObs(BaseObs):
    """
    Generate observations of a set of moving objects: exact ephemeris at the times of each observation.

    First generates observations on a rough grid and looks for observations within a specified tolerance
    of the actual observations; for the observations which pass this cut, generates a precise ephemeris
    and checks if the object is within the FOV.

    Parameters
    ----------
    footprint: `str`, optional
        Specify the footprint for the FOV. Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint (following a rough cut with a circular FOV).
        Default is circular FOV.
    r_fov : `float`, optional
        If footprint is "circular", this is the radius of the fov (in degrees).
        Default 1.75 degrees.
    x_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of the (on-sky) fov in the RA
        direction (in degrees).
        Default 5 degrees. (so size of footprint in degrees will be 10 degrees in the RA direction).
    y_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of the fov in Declination (in degrees).
        Default is 3 degrees (so size of footprint in degrees will be 6 degrees in the Dec direction).
    eph_mode: `str`, optional
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    prelim_eph_mode: str, optional
        Mode for preliminary ephemeris generation, if any is done. Default is 2body.
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
        For example: 'baseline_v2.0_10yrs, MJD 59853-61677' or 'baseline2018a minus NES'
        Default ''.
    tstep: `float`, optional
        The time between initial (rough) ephemeris generation points, in days.
        Default 1 day.
    rough_tol: `float`, optional
        The initial rough tolerance value for positions, used as a first cut to identify potential
        observations (in degrees).
        Default 10 degrees.
    """

    def __init__(
        self,
        footprint="camera",
        r_fov=1.75,
        x_tol=5,
        y_tol=3,
        eph_mode="nbody",
        prelim_eph_mode="2body",
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
        tstep=1.0,
        rough_tol=10.0,
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
        self.rough_tol = rough_tol
        if prelim_eph_mode not in ("2body", "nbody"):
            raise ValueError("Ephemeris generation must be 2body or nbody.")
        self.prelim_eph_mode = prelim_eph_mode

    def _header_meta(self):
        # Specific header information for direct obs.
        self.outfile.write("# direct obs header metadata\n")
        self.outfile.write(
            "# observation generation via %s\n" % self.__class__.__name__
        )
        self.outfile.write(
            "# ephMode %s prelimEphMode %s\n" % (self.eph_mode, self.prelim_eph_mode)
        )
        self.outfile.write(
            "# rough tolerance for preliminary match %f\n" % self.rough_tol
        )
        self.outfile.write("# time step for preliminary match %f\n" % self.tstep)

    def run(self, orbits, obs_data):
        """Find and write the observations of each object to disk.

        For each object, generate a very rough grid of ephemeris points (typically using 2body integration).
        Then identify pointings in obs_data which are within

        Parameters
        ----------
        orbits : `rubin_sim.moving_objects.Orbits`
            The orbits to generate ephemerides for.
        obs_data : `np.ndarray`
            The simulated pointing history data.
        """
        # Set the times for the rough ephemeris grid.
        time_step = float(self.tstep)
        time_start = (
            np.floor(obs_data[self.obs_time_col].min() + 0.16 - 0.5) - time_step
        )
        time_end = np.ceil(obs_data[self.obs_time_col].max() + 0.16 + 0.5) + time_step
        rough_times = np.arange(time_start, time_end + time_step / 2.0, time_step)
        logging.info(
            "Generating preliminary ephemerides on a grid of %f day timesteps."
            % (time_step)
        )
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for i, sso in enumerate(orbits):
            objid = sso.orbits["obj_id"].iloc[0]
            sedname = sso.orbits["sed_filename"].iloc[0]
            # Generate ephemerides on the rough grid.
            logging.debug(
                ("%d/%d   id=%s : " % (i, len(orbits), objid))
                + datetime.datetime.now().strftime("Prelim start: %Y-%m-%d %H:%M:%S")
                + " nRoughTimes: %s" % len(rough_times)
            )
            ephs = self.generate_ephemerides(
                sso, rough_times, eph_mode=self.prelim_eph_mode, eph_type=self.eph_type
            )[0]
            mu = ephs["velocity"]
            logging.debug(
                ("%d/%d   id=%s : " % (i, len(orbits), objid))
                + datetime.datetime.now().strftime("Prelim end: %Y-%m-%d %H:%M:%S")
                + " π(median, max), min(geo_dist): %.2f, %.2f deg/day  %.2f AU"
                % (np.median(mu), np.max(mu), np.min(ephs["geo_dist"]))
            )

            # Find observations which come within roughTol of the fov.
            ephs_idxs = np.searchsorted(ephs["time"], obs_data[self.obs_time_col])
            rough_idx_obs = self._sso_in_circle_fov(
                ephs[ephs_idxs], obs_data, self.rough_tol
            )
            if len(rough_idx_obs) > 0:
                # Generate exact ephemerides for these times.
                times = obs_data[self.obs_time_col][rough_idx_obs]
                logging.debug(
                    ("%d/%d   id=%s : " % (i, len(orbits), objid))
                    + datetime.datetime.now().strftime("Exact start: %Y-%m-%d %H:%M:%S")
                    + " nExactTimes: %s" % len(times)
                )
                ephs = self.generate_ephemerides(
                    sso, times, eph_mode=self.eph_mode, eph_type=self.eph_type
                )[0]
                logging.debug(
                    ("%d/%d   id=%s : " % (i, len(orbits), objid))
                    + datetime.datetime.now().strftime("Exact end: %Y-%m-%d %H:%M:%S")
                )
                # Identify the objects which fell within the specific footprint.
                idx_obs = self.sso_in_fov(ephs, obs_data[rough_idx_obs])
                logging.info(
                    ("%d/%d   id=%s : " % (i, len(orbits), objid))
                    + "Object in %d out of %d potential fields (%.2f%% success rate)"
                    % (
                        len(idx_obs),
                        len(times),
                        100.0 * float(len(idx_obs)) / len(times),
                    )
                )
                # Write these observations to disk.
                self.write_obs(
                    objid,
                    ephs[idx_obs],
                    obs_data[rough_idx_obs][idx_obs],
                    sedname=sedname,
                )
