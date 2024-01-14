__all__ = ("DirectObs",)

import datetime
import logging
import warnings

import numpy as np
from rubin_scheduler.utils import angular_separation

from .base_obs import BaseObs


class DirectObs(BaseObs):
    """
    Generate observations of a set of moving objects:
    exact ephemeris at the times of each observation.

    First generates observations on a rough grid and looks for
    observations within a specified tolerance
    of the actual observations; for the observations which pass this cut,
    generates a precise ephemeris and checks if the object is within the FOV.

    Parameters
    ----------
    footprint : `str`, optional
        Specify the footprint for the FOV.
        Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint
        (following a rough cut with a circular FOV).
        Default is circular FOV.
    r_fov : `float`, optional
        If footprint is "circular", this is the radius of the fov (in degrees).
        Default 1.75 degrees.
    x_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of
        the (on-sky) fov in the RA direction (in degrees).
        Default 5 degrees.
    y_tol : `float`, optional
        If footprint is "rectangular", this is half of the width of
        the fov in Declination (in degrees).
        Default is 3 degrees
    eph_mode: `str`, optional
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    prelim_eph_mode: str, optional
        Mode for preliminary ephemeris generation, if any is done.
        Default is 2body.
    eph_type: `str`, optional
        Type of ephemerides to generate - full or basic.
        Full includes all values calculated by openorb;
        Basic includes a more basic set.
        Default is Basic.
        (this includes enough information for most standard MAF metrics).
    eph_file: `str` or None, optional
        The name of the planetary ephemerides file to use in ephemeris
        generation. Default (None) will use the default for PyOrbEphemerides.
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
        This should be the geometric/physical seeing as it is used
        for the trailing loss calculation.
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
        For example: 'baseline_v2.0_10yrs, MJD 59853-61677'
        or 'baseline2018a minus NES'
        Default ''.
    tstep: `float`, optional
        The time between initial (rough) ephemeris generation points, in days.
        Default 1 day.
    rough_tol: `float`, optional
        The initial rough tolerance value for positions, used as a first
        cut to identify potential observations (in degrees).
        Default 10 degrees.
    pre_comp_tol : float (2.08)
        The radial tolerance to add when using pre-computed orbits. Should be
        larger than the full field of view extent.
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
        obs_info="",
        tstep=1.0,
        rough_tol=10.0,
        verbose=False,
        night_col="night",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        obs_id_col="observationId",
        pre_comp_tol=2.08,
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
            obs_info=obs_info,
        )
        self.verbose = verbose
        self.pre_comp_tol = pre_comp_tol
        self.filter_col = filter_col
        self.night_col = night_col
        self.m5_col = m5_col
        self.obs_id_col = obs_id_col
        self.tstep = tstep
        self.rough_tol = rough_tol
        if prelim_eph_mode not in ("2body", "nbody"):
            raise ValueError("Ephemeris generation must be 2body or nbody.")
        self.prelim_eph_mode = prelim_eph_mode

    def run(self, orbits, obs_data, object_positions=None, object_mjds=None):
        """Find and write the observations of each object to disk.

        For each object, a rough grid of ephemeris points are either
        generated on the fly or read from a pre-calculated grid;
        If the rough grids indicate that an object may be present
        in an observation, then a more precise position is generated
        for the time of the observation.


        Parameters
        ----------
        orbits : `rubin_sim.moving_objects.Orbits`
            The orbits for which to generate ephemerides.
        obs_data : `np.ndarray`
            The simulated pointing history data.
        object_positions : `np.ndarray`
            Pre-computed RA,dec positions for each object in orbits (degrees).
        object_mjds : `np.ndarray`
            MJD values for each pre-computed position.
        """

        # If we are trying to use pre-computed positions,
        # check that the MJDs span the necessary time.
        if object_mjds is not None:
            if (obs_data[self.obs_time_col].min() < object_mjds.min()) | (
                obs_data[self.obs_time_col].max() > object_mjds.max()
            ):
                warnings.warn(
                    "Pre-computed position times do not cover MJD range of %i-%i."
                    % (obs_data[self.obs_time_col].min(), obs_data[self.obs_time_col].max())
                )
                object_mjds = None
            # calculate angular motion for each object at each timestep
            # how much does it move going to time step forward
            move1 = angular_separation(
                object_positions["ra"][:, 0:-1],
                object_positions["dec"][:, 0:-1],
                object_positions["ra"][:, 1:],
                object_positions["dec"][:, 1:],
            )
            # Need to take care of ends
            d1 = object_positions["ra"] * 0
            d1[:, 0:-1] = move1
            d2 = object_positions["ra"] * 0
            d2[:, 1:] = move1
            # Now we can use a small tolerance for objects when they are moving
            # slowly across the sky.
            object_tol = np.maximum(d1, d2) + self.pre_comp_tol

        # output dtype
        names = [
            "obj_id",
            self.obs_id_col,
            "sedname",
            "time",
            "ra",
            "dec",
            "dradt",
            "ddecdt",
            "phase",
            "solarelon",
            "helio_dist",
            "geo_dist",
            "magV",
            "trueAnomaly",
            "velocity",
            "dmag_color",
            "dmag_trail",
            "dmag_detect",
        ]
        types = [int, int, "<U40"] + [float] * (len(names) - 3)

        # XXX--for now, copy over some observation info. In the future,
        # just index it to the observations and only transfer observationId
        transfer_cols = [
            self.visit_exp_time_col,
            self.m5_col,
            self.seeing_col,
            self.obs_time_col,
            self.night_col,
            self.filter_col,
        ]
        transfer_types = [float] * (len(transfer_cols) - 2)
        transfer_types += [int, "<U40"]
        names += transfer_cols
        types += transfer_types

        output_dtype = list(zip(names, types))

        # Set the times for the rough ephemeris grid.
        time_step = float(self.tstep)
        time_start = np.floor(obs_data[self.obs_time_col].min() + 0.16 - 0.5) - time_step
        time_end = np.ceil(obs_data[self.obs_time_col].max() + 0.16 + 0.5) + time_step
        rough_times = np.arange(time_start, time_end + time_step / 2.0, time_step)
        if self.verbose:
            logging.info("Generating preliminary ephemerides on a grid of %f day timesteps." % (time_step))
        # list to hold results for each object
        result = []
        # save indx to match observation indx to object indx
        indx_map_visit_to_object = []
        # For each object, identify observations where the object is
        # within the FOV (or camera footprint).
        for i, sso in enumerate(orbits):
            objid = sso.orbits["obj_id"].iloc[0]
            sedname = sso.orbits["sed_filename"].iloc[0]
            # Generate ephemerides on the rough grid.
            if self.verbose:
                logging.debug(
                    ("%d/%d   id=%s : " % (i, len(orbits), objid))
                    + datetime.datetime.now().strftime("Prelim start: %Y-%m-%d %H:%M:%S")
                    + " nRoughTimes: %s" % len(rough_times)
                )
            # Not using pre-computed positions
            if object_mjds is None:
                ephs = self.generate_ephemerides(
                    sso,
                    rough_times,
                    eph_mode=self.prelim_eph_mode,
                    eph_type=self.eph_type,
                )[0]
                mu = ephs["velocity"]
                if self.verbose:
                    logging.debug(
                        ("%d/%d   id=%s : " % (i, len(orbits), objid))
                        + datetime.datetime.now().strftime("Prelim end: %Y-%m-%d %H:%M:%S")
                        + " π(median, max), min(geo_dist): %.2f, %.2f deg/day  %.2f AU"
                        % (np.median(mu), np.max(mu), np.min(ephs["geo_dist"]))
                    )

                # Find observations which come within roughTol of the fov.
                ephs_idxs = np.searchsorted(ephs["time"], obs_data[self.obs_time_col])
                rough_idx_obs = self._sso_in_circle_fov(ephs[ephs_idxs], obs_data, self.rough_tol)
            else:
                # Nearest neighbor search for the object_mjd closest to
                # obs_data mjd
                pos = np.searchsorted(object_mjds, obs_data[self.obs_time_col], side="left")
                pos_right = pos - 1
                object_indx = pos + 0
                d_left = obs_data[self.obs_time_col] - object_mjds[pos]
                d_right = obs_data[self.obs_time_col] - object_mjds[pos_right]
                r_better = np.where(np.abs(d_right) < np.abs(d_left))[0]
                object_indx[r_better] = pos_right[r_better]

                rough_idx_obs = self._sso_in_circle_fov(
                    object_positions[i][object_indx],
                    obs_data,
                    self.r_fov + object_tol[i][object_indx],
                )
            if len(rough_idx_obs) > 0:
                # Generate exact ephemerides for these times.
                times = obs_data[self.obs_time_col][rough_idx_obs]
                if self.verbose:
                    logging.debug(
                        ("%d/%d   id=%s : " % (i, len(orbits), objid))
                        + datetime.datetime.now().strftime("Exact start: %Y-%m-%d %H:%M:%S")
                        + " nExactTimes: %s" % len(times)
                    )
                ephs = self.generate_ephemerides(sso, times, eph_mode=self.eph_mode, eph_type=self.eph_type)[
                    0
                ]
                logging.debug(
                    ("%d/%d   id=%s : " % (i, len(orbits), objid))
                    + datetime.datetime.now().strftime("Exact end: %Y-%m-%d %H:%M:%S")
                )
                # Identify the objects which fell within the footprint.
                idx_obs = self.sso_in_fov(ephs, obs_data[rough_idx_obs])
                if self.verbose:
                    logging.info(
                        ("%d/%d   id=%s : " % (i, len(orbits), objid))
                        + "Object in %d out of %d potential fields (%.2f%% success rate)"
                        % (
                            len(idx_obs),
                            len(times),
                            100.0 * float(len(idx_obs)) / len(times),
                        )
                    )
                object_observations = np.zeros(idx_obs.size, dtype=output_dtype)
                object_observations["obj_id"] = objid
                object_observations[self.obs_id_col] = obs_data[rough_idx_obs][idx_obs][self.obs_id_col]
                object_observations["sedname"] = sedname

                for key in ephs.dtype.names:
                    object_observations[key] = ephs[key][idx_obs].copy()
                result.append(object_observations)
                indx_map_visit_to_object.append(rough_idx_obs[idx_obs])

        if len(result) > 0:
            result = np.concatenate(result)
            indx_map_visit_to_object = np.concatenate(indx_map_visit_to_object)
            # add on any additional info we want here, dmags, etc
            filterlist = np.unique(obs_data[self.filter_col][indx_map_visit_to_object])
            for sname in np.unique(result["sedname"]):
                dmag_color_dict = self.calc_colors(sname)
                for f in filterlist:
                    match = np.where(
                        (obs_data[self.filter_col][indx_map_visit_to_object] == f)
                        & (result["sedname"] == sname)
                    )[0]
                    if np.size(match) > 0:
                        result["dmag_color"][match] = dmag_color_dict[f]
            # Calculate trailing and detection loses.
            result["dmag_trail"], result["dmag_detect"] = self.calc_trailing_losses(
                result["velocity"],
                obs_data[self.seeing_col][indx_map_visit_to_object],
                obs_data[self.visit_exp_time_col][indx_map_visit_to_object],
            )

            # Transfer over info from pointing info array to result array
            for key in transfer_cols:
                result[key] = obs_data[key][indx_map_visit_to_object]

        return result
