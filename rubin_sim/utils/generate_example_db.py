__all__ = ("gen_greedy_surveys", "generate_blobs", "nes_light_footprints", "run_sched")

import argparse
import os
import subprocess
import sys

import healpy as hp
import matplotlib.pylab as plt
import numpy as np

import rubin_sim.scheduler.basis_functions as bf
import rubin_sim.scheduler.detailers as detailers
from rubin_sim.scheduler import sim_runner
from rubin_sim.scheduler.modelObservatory import Model_observatory
from rubin_sim.scheduler.schedulers import Core_scheduler, simple_filter_sched
from rubin_sim.scheduler.surveys import Blob_survey, Greedy_survey, generate_dd_surveys
from rubin_sim.scheduler.utils import Footprint, generate_goal_map, standard_goals


def gen_greedy_surveys(
    nside=32,
    nexp=2,
    exptime=30.0,
    filters=["r", "i", "z", "y"],
    camera_rot_limits=[-80.0, 80.0],
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs="DD",
    m5_weight=3.0,
    footprint_weight=0.3,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    footprints=None,
):
    """
    Make a quick set of greedy surveys

    This is a convienence function to generate a list of survey objects that can be used with
    lsst.sims.featureScheduler.schedulers.Core_scheduler.
    To ensure we are robust against changes in the sims_featureScheduler codebase, all kwargs are
    explicitly set.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filters : list of str (['r', 'i', 'z', 'y'])
        Which filters to generate surveys for.
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    """
    # Define the extra parameters that are used in the greedy survey. I
    # think these are fairly set, so no need to promote to utility func kwargs
    greed_survey_params = {
        "block_size": 1,
        "smoothing_kernel": None,
        "seed": 42,
        "camera": "LSST",
        "dither": True,
        "survey_name": "greedy",
    }

    surveys = []
    detailer = detailers.Camera_rot_detailer(
        min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
    )

    for filtername in filters:
        bfs = []
        bfs.append((bf.M5_diff_basis_function(filtername=filtername, nside=nside), m5_weight))
        bfs.append(
            (
                bf.Footprint_basis_function(
                    filtername=filtername,
                    footprint=footprints,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight,
            )
        )
        bfs.append(
            (
                bf.Slewtime_basis_function(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.Zenith_shadow_mask_basis_function(
                    nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt
                ),
                0,
            )
        )
        bfs.append(
            (
                bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance),
                0,
            )
        )

        bfs.append((bf.Filter_loaded_basis_function(filternames=filtername), 0))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0))

        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        surveys.append(
            Greedy_survey(
                basis_functions,
                weights,
                exptime=exptime,
                filtername=filtername,
                nside=nside,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=[detailer],
                **greed_survey_params,
            )
        )

    return surveys


def generate_blobs(
    nside,
    nexp=2,
    exptime=30.0,
    filter1s=["u", "u", "g", "r", "i", "z", "y"],
    filter2s=["g", "r", "r", "i", "z", "y", "y"],
    pair_time=22.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=3,
    season=300.0,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs="DD",
    m5_weight=6.0,
    footprint_weight=0.6,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    template_weight=12.0,
    footprints=None,
):
    """
    Generate surveys that take observations in blobs.

    Parameters
    ----------
    nside : int (32)
        The HEALpix nside to use
    nexp : int (1)
        The number of exposures to use in a visit.
    exptime : float (30.)
        The exposure time to use per visit (seconds)
    filter1s : list of str
        The filternames for the first set
    filter2s : list of str
        The filter names for the second in the pair (None if unpaired)
    pair_time : float (22)
        The ideal time between pairs (minutes)
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    n_obs_template : int (3)
        The number of observations to take every season in each filter
    season : float (300)
        The length of season (i.e., how long before templates expire) (days)
    season_start_hour : float (-4.)
        For weighting how strongly a template image needs to be observed (hours)
    sesason_end_hour : float (2.)
        For weighting how strongly a template image needs to be observed (hours)
    shadow_minutes : float (60.)
        Used to mask regions around zenith (minutes)
    max_alt : float (76.
        The maximium altitude to use when masking zenith (degrees)
    moon_distance : float (30.)
        The mask radius to apply around the moon (degrees)
    ignore_obs : str or list of str ('DD')
        Ignore observations by surveys that include the given substring(s).
    m5_weight : float (3.)
        The weight for the 5-sigma depth difference basis function
    footprint_weight : float (0.3)
        The weight on the survey footprint basis function.
    slewtime_weight : float (3.)
        The weight on the slewtime basis function
    stayfilter_weight : float (3.)
        The weight on basis function that tries to stay avoid filter changes.
    template_weight : float (12.)
        The weight to place on getting image templates every season
    """

    blob_survey_params = {
        "slew_approx": 7.5,
        "filter_change_approx": 140.0,
        "read_approx": 2.0,
        "min_pair_time": 15.0,
        "search_radius": 30.0,
        "alt_max": 85.0,
        "az_range": 90.0,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": True,
    }

    surveys = []

    times_needed = [pair_time, pair_time * 2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(
            detailers.Camera_rot_detailer(
                min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
            )
        )
        detailer_list.append(detailers.Close_alt_detailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append(
                (
                    bf.M5_diff_basis_function(filtername=filtername, nside=nside),
                    m5_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.M5_diff_basis_function(filtername=filtername2, nside=nside),
                    m5_weight / 2.0,
                )
            )

        else:
            bfs.append(
                (
                    bf.M5_diff_basis_function(filtername=filtername, nside=nside),
                    m5_weight,
                )
            )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.Footprint_basis_function(
                        filtername=filtername,
                        footprint=footprints,
                        out_of_bounds_val=np.nan,
                        nside=nside,
                    ),
                    footprint_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.Footprint_basis_function(
                        filtername=filtername2,
                        footprint=footprints,
                        out_of_bounds_val=np.nan,
                        nside=nside,
                    ),
                    footprint_weight / 2.0,
                )
            )
        else:
            bfs.append(
                (
                    bf.Footprint_basis_function(
                        filtername=filtername,
                        footprint=footprints,
                        out_of_bounds_val=np.nan,
                        nside=nside,
                    ),
                    footprint_weight,
                )
            )

        bfs.append(
            (
                bf.Slewtime_basis_function(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append((bf.Strict_filter_basis_function(filtername=filtername), stayfilter_weight))

        if filtername2 is not None:
            bfs.append(
                (
                    bf.N_obs_per_year_basis_function(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template,
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.N_obs_per_year_basis_function(
                        filtername=filtername2,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername2),
                        n_obs=n_obs_template,
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight / 2.0,
                )
            )
        else:
            bfs.append(
                (
                    bf.N_obs_per_year_basis_function(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template,
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight,
                )
            )
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.Zenith_shadow_mask_basis_function(
                    nside=nside,
                    shadow_minutes=shadow_minutes,
                    max_alt=max_alt,
                    penalty=np.nan,
                    site="LSST",
                ),
                0.0,
            )
        )
        bfs.append(
            (
                bf.Moon_avoidance_basis_function(nside=nside, moon_distance=moon_distance),
                0.0,
            )
        )
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.Filter_loaded_basis_function(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.Time_to_twilight_basis_function(time_needed=time_needed), 0.0))
        bfs.append((bf.Not_twilight_basis_function(), 0.0))
        bfs.append((bf.Planet_mask_basis_function(nside=nside), 0.0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = "blob, %s" % filtername
        else:
            survey_name = "blob, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.Take_as_pairs_detailer(filtername=filtername2))
        surveys.append(
            Blob_survey(
                basis_functions,
                weights,
                filtername1=filtername,
                filtername2=filtername2,
                exptime=exptime,
                ideal_pair_time=pair_time,
                survey_note=survey_name,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **blob_survey_params,
            )
        )

    return surveys


def nes_light_footprints(nside=None):
    """
    A quick function to generate the "standard" goal maps. This is the traditional WFD/mini survey footprint.
    """

    nes_scaledown = 2.0
    scp_scaledown = 1.5

    result = {}
    result["u"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.0 / nes_scaledown,
        wfd_fraction=0.31,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["g"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.2 / nes_scaledown,
        wfd_fraction=0.44,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["r"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.46 / nes_scaledown,
        wfd_fraction=1.0,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["i"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.46 / nes_scaledown,
        wfd_fraction=1.0,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["z"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.4 / nes_scaledown,
        wfd_fraction=0.9,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    result["y"] = generate_goal_map(
        nside=nside,
        nes_fraction=0.0 / nes_scaledown,
        wfd_fraction=0.9,
        scp_fraction=0.15 / scp_scaledown,
        gp_fraction=0.15,
        wfd_dec_min=-62.5,
        wfd_dec_max=3.6,
    )
    return result


def run_sched(
    surveys,
    survey_length=365.25,
    nside=32,
    fileroot="baseline_",
    verbose=False,
    extra_info=None,
    illum_limit=40.0,
):
    years = np.round(survey_length / 365.25)
    scheduler = Core_scheduler(surveys, nside=nside)
    n_visit_limit = None
    filter_sched = simple_filter_sched(illum_limit=illum_limit)
    observatory = Model_observatory(nside=nside)
    observatory, scheduler, observations = sim_runner(
        observatory,
        scheduler,
        survey_length=survey_length,
        filename=fileroot + "%iyrs.db" % years,
        delete_past=True,
        n_visit_limit=n_visit_limit,
        verbose=verbose,
        extra_info=extra_info,
        filter_scheduler=filter_sched,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--survey_length", type=float, default=15)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--maxDither", type=float, default=0.7, help="Dither size for DDFs (deg)")
    parser.add_argument(
        "--moon_illum_limit",
        type=float,
        default=40.0,
        help="illumination limit to remove u-band",
    )
    parser.add_argument("--nexp", type=int, default=2)
    parser.add_argument("--scale_down", dest="scale_down", action="store_true")
    parser.set_defaults(scale_down=False)

    args = parser.parse_args()
    survey_length = args.survey_length  # Days
    out_dir = args.outDir
    verbose = args.verbose
    max_dither = args.maxDither
    illum_limit = args.moon_illum_limit
    nexp = args.nexp
    scale_down = args.scale_down

    nside = 32
    per_night = True  # Dither DDF per night

    camera_ddf_rot_limit = 75.0

    extra_info = {}
    exec_command = ""
    for arg in sys.argv:
        exec_command += " " + arg
    extra_info["exec command"] = exec_command
    try:
        extra_info["git hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        extra_info["git hash"] = "Not in git repo"

    extra_info["file executed"] = os.path.realpath(__file__)

    fileroot = "example_db"
    file_end = "v1.7_"

    if scale_down:
        footprints_hp = nes_light_footprints(nside=nside)
        fileroot = fileroot + "scaleddown_"
    else:
        footprints_hp = standard_goals(nside=nside)

    observatory = Model_observatory(nside=nside)
    conditions = observatory.return_conditions()
    footprints = Footprint(conditions.mjd_start, sun_ra_start=conditions.sun_RA_start, nside=nside)
    for i, key in enumerate(footprints_hp):
        footprints.footprints[i, :] = footprints_hp[key]

    # Set up the DDF surveys to dither
    dither_detailer = detailers.Dither_detailer(per_night=per_night, max_dither=max_dither)
    details = [
        detailers.Camera_rot_detailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
        dither_detailer,
    ]
    euclid_detailers = [
        detailers.Camera_rot_detailer(min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit),
        detailers.Euclid_dither_detailer(),
    ]
    ddfs = generate_dd_surveys(nside=nside, nexp=nexp, detailers=details, euclid_detailers=euclid_detailers)

    greedy = gen_greedy_surveys(nside, nexp=nexp, footprints=footprints)
    blobs = generate_blobs(nside, nexp=nexp, footprints=footprints)
    surveys = [ddfs, blobs, greedy]
    run_sched(
        surveys,
        survey_length=survey_length,
        verbose=verbose,
        fileroot=os.path.join(out_dir, fileroot + file_end),
        extra_info=extra_info,
        nside=nside,
        illum_limit=illum_limit,
    )
