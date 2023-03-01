import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from rubin_sim.scheduler.schedulers import CoreScheduler
from rubin_sim.scheduler.utils import (
    SkyAreaGeneratorGalplane,
    ConstantFootprint,
    make_rolling_footprints,
)
import rubin_sim.scheduler.basis_functions as bf
from rubin_sim.scheduler.surveys import (
    GreedySurvey,
    BlobSurvey,
    ScriptedSurvey,
    LongGapSurvey,
    generate_dd_surveys,
)
import rubin_sim.scheduler.detailers as detailers
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from astropy import units as u
from rubin_sim.utils import _hpid2_ra_dec


__all__ = ["example_scheduler"]


def blob_for_long(
    nside,
    nexp=2,
    exptime=30.0,
    filter1s=["g"],
    filter2s=["i"],
    pair_time=33.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=300.0,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_neo"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    template_weight=12.0,
    u_template_weight=24.0,
    footprints=None,
    u_nexp1=True,
    night_pattern=[True, True],
    time_after_twi=30.0,
    HA_min=12,
    HA_max=24 - 3.5,
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
    pair_time : float (33)
        The ideal time between pairs (minutes)
    camera_rot_limits : list of float ([-80., 80.])
        The limits to impose when rotationally dithering the camera (degrees).
    n_obs_template : int (5)
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
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    u_nexp1 : bool (True)
        Add a detailer to make sure the number of expossures in a visit is always 1 for u observations.
    """

    BlobSurvey_params = {
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
    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 5, "r": 5, "i": 5, "z": 5, "y": 5}

    times_needed = [pair_time, pair_time * 2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(
                min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
            )
        )
        detailer_list.append(detailers.CloseAltDetailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername, nside=nside),
                    m5_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername2, nside=nside),
                    m5_weight / 2.0,
                )
            )

        else:
            bfs.append(
                (bf.M5DiffBasisFunction(filtername=filtername, nside=nside), m5_weight)
            )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                bf.SlewtimeBasisFunction(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append(
            (bf.StrictFilterBasisFunction(filtername=filtername), stayfilter_weight)
        )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername2,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername2),
                        n_obs=n_obs_template[filtername2],
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
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
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
                bf.ZenithShadowMaskBasisFunction(
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
                bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance),
                0.0,
            )
        )
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.FilterLoadedBasisFunction(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0.0))
        bfs.append((bf.AfterEveningTwiBasisFunction(time_after=time_after_twi), 0.0))
        # XXX--move kwargs up
        bfs.append((bf.HaMaskBasisFunction(ha_min=HA_min, ha_max=HA_max), 0.0))
        # don't execute every night
        bfs.append((bf.NightModuloBasisFunction(night_pattern), 0.0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = "blob_long, %s" % filtername
        else:
            survey_name = "blob_long, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))

        if u_nexp1:
            detailer_list.append(detailers.FilterNexp(filtername="u", nexp=1))
        surveys.append(
            BlobSurvey(
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
                **BlobSurvey_params
            )
        )

    return surveys


def gen_long_gaps_survey(
    footprints,
    nside=32,
    night_pattern=[True, True],
    gap_range=[2, 7],
    HA_min=12,
    HA_max=24 - 3.5,
    time_after_twi=120,
):
    """
    Paramterers
    -----------
    HA_min(_max) : float
        The hour angle limits passed to the initial blob scheduler (hours).
    gap_range : list of float ([2, 7])
        The limits of the target time gaps (hours)
    time_after_twi : float (120)
        The amount of time that should have passed after twilight to start
        trying to observe long gaps (minutes).
    """

    surveys = []
    f1 = ["g", "r", "i"]
    f2 = ["r", "i", "z"]
    # Maybe force scripted to not go in twilight?

    for filtername1, filtername2 in zip(f1, f2):
        blob = blob_for_long(
            footprints=footprints,
            nside=nside,
            filter1s=[filtername1],
            filter2s=[filtername2],
            night_pattern=night_pattern,
            time_after_twi=time_after_twi,
            HA_min=HA_min,
            HA_max=HA_max,
        )
        scripted = ScriptedSurvey([], nside=nside, ignore_obs=["blob", "DDF", "twi"])
        surveys.append(
            LongGapSurvey(blob[0], scripted, gap_range=gap_range, avoid_zenith=True)
        )

    return surveys


def gen_greedy_surveys(
    nside=32,
    nexp=2,
    exptime=30.0,
    filters=["r", "i", "z", "y"],
    camera_rot_limits=[-80.0, 80.0],
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_neo"],
    m5_weight=3.0,
    footprint_weight=0.75,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    repeat_weight=-1.0,
    footprints=None,
):
    """
    Make a quick set of greedy surveys

    This is a convienence function to generate a list of survey objects that can be used with
    rubin_sim.scheduler.schedulers.Core_scheduler.
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
    detailer_list = [
        detailers.CameraRotDetailer(
            min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
        )
    ]
    detailer_list.append(detailers.Rottep2RotspDesiredDetailer())

    for filtername in filters:
        bfs = []
        bfs.append(
            (bf.M5DiffBasisFunction(filtername=filtername, nside=nside), m5_weight)
        )
        bfs.append(
            (
                bf.FootprintBasisFunction(
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
                bf.SlewtimeBasisFunction(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append(
            (bf.StrictFilterBasisFunction(filtername=filtername), stayfilter_weight)
        )
        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=2 * 60.0, filtername=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.ZenithShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=shadow_minutes, max_alt=max_alt
                ),
                0,
            )
        )
        bfs.append(
            (bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance), 0)
        )

        bfs.append((bf.FilterLoadedBasisFunction(filternames=filtername), 0))
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0))

        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        surveys.append(
            GreedySurvey(
                basis_functions,
                weights,
                exptime=exptime,
                filtername=filtername,
                nside=nside,
                ignore_obs=ignore_obs,
                nexp=nexp,
                detailers=detailer_list,
                **greed_survey_params
            )
        )

    return surveys


def generate_blobs(
    nside,
    nexp=2,
    exptime=30.0,
    filter1s=["u", "u", "g", "r", "i", "z", "y"],
    filter2s=["g", "r", "r", "i", "z", "y", "y"],
    pair_time=33.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=300.0,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_neo"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    template_weight=12.0,
    u_template_weight=24.0,
    footprints=None,
    u_nexp1=True,
    scheduled_respect=45.0,
    good_seeing={"g": 3, "r": 3, "i": 3},
    good_seeing_weight=3.0,
    mjd_start=1,
    repeat_weight=-20,
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
    pair_time : float (33)
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
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    u_nexp1 : bool (True)
        Add a detailer to make sure the number of expossures in a visit is always 1 for u observations.
    scheduled_respect : float (45)
        How much time to require there be before a pre-scheduled observation (minutes)
    """

    template_weights = {
        "u": u_template_weight,
        "g": template_weight,
        "r": template_weight,
        "i": template_weight,
        "z": template_weight,
        "y": template_weight,
    }

    BlobSurvey_params = {
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
        "twilight_scale": False,
    }

    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 5, "r": 5, "i": 5, "z": 5, "y": 5}

    surveys = []

    times_needed = [pair_time, pair_time * 2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(
                min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
            )
        )
        detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.FlushForSchedDetailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername, nside=nside),
                    m5_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername2, nside=nside),
                    m5_weight / 2.0,
                )
            )

        else:
            bfs.append(
                (bf.M5DiffBasisFunction(filtername=filtername, nside=nside), m5_weight)
            )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                bf.SlewtimeBasisFunction(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append(
            (bf.StrictFilterBasisFunction(filtername=filtername), stayfilter_weight)
        )
        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=3 * 60.0, filtername=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weights[filtername] / 2.0,
                )
            )
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername2,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername2),
                        n_obs=n_obs_template[filtername2],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weights[filtername2] / 2.0,
                )
            )
        else:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight,
                )
            )

        # Insert things for getting good seeing templates
        if filtername2 is not None:
            if filtername in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            filtername=filtername,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(filtername),
                            n_obs_desired=good_seeing[filtername],
                        ),
                        good_seeing_weight,
                    )
                )
            if filtername2 in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            filtername=filtername2,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(filtername2),
                            n_obs_desired=good_seeing[filtername2],
                        ),
                        good_seeing_weight,
                    )
                )
        else:
            if filtername in list(good_seeing.keys()):
                bfs.append(
                    (
                        bf.NGoodSeeingBasisFunction(
                            filtername=filtername,
                            nside=nside,
                            mjd_start=mjd_start,
                            footprint=footprints.get_footprint(filtername),
                            n_obs_desired=good_seeing[filtername],
                        ),
                        good_seeing_weight,
                    )
                )
        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.ZenithShadowMaskBasisFunction(
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
                bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance),
                0.0,
            )
        )
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.FilterLoadedBasisFunction(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append((bf.TimeToTwilightBasisFunction(time_needed=time_needed), 0.0))
        bfs.append((bf.NotTwilightBasisFunction(), 0.0))
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0.0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = "blob, %s" % filtername
        else:
            survey_name = "blob, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))

        if u_nexp1:
            detailer_list.append(detailers.FilterNexp(filtername="u", nexp=1))
        surveys.append(
            BlobSurvey(
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
                **BlobSurvey_params
            )
        )

    return surveys


def generate_twi_blobs(
    nside,
    nexp=2,
    exptime=30.0,
    filter1s=["r", "i", "z", "y"],
    filter2s=["i", "z", "y", "y"],
    pair_time=15.0,
    camera_rot_limits=[-80.0, 80.0],
    n_obs_template=None,
    season=300.0,
    season_start_hour=-4.0,
    season_end_hour=2.0,
    shadow_minutes=60.0,
    max_alt=76.0,
    moon_distance=30.0,
    ignore_obs=["DD", "twilight_neo"],
    m5_weight=6.0,
    footprint_weight=1.5,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    template_weight=12.0,
    footprints=None,
    repeat_night_weight=None,
    wfd_footprint=None,
    scheduled_respect=15.0,
    repeat_weight=-1.0,
    night_pattern=None,
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
    u_template_weight : float (24.)
        The weight to place on getting image templates in u-band. Since there
        are so few u-visits, it can be helpful to turn this up a little higher than
        the standard template_weight kwarg.
    """

    BlobSurvey_params = {
        "slew_approx": 7.5,
        "filter_change_approx": 140.0,
        "read_approx": 2.0,
        "min_pair_time": 10.0,
        "search_radius": 30.0,
        "alt_max": 85.0,
        "az_range": 90.0,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "dither": True,
        "twilight_scale": False,
        "in_twilight": True,
    }

    surveys = []

    if n_obs_template is None:
        n_obs_template = {"u": 3, "g": 5, "r": 5, "i": 5, "z": 5, "y": 5}

    times_needed = [pair_time, pair_time * 2]
    for filtername, filtername2 in zip(filter1s, filter2s):
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(
                min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
            )
        )
        detailer_list.append(detailers.Rottep2RotspDesiredDetailer())
        detailer_list.append(detailers.CloseAltDetailer())
        detailer_list.append(detailers.FlushForSchedDetailer())
        # List to hold tuples of (basis_function_object, weight)
        bfs = []

        if filtername2 is not None:
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername, nside=nside),
                    m5_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.M5DiffBasisFunction(filtername=filtername2, nside=nside),
                    m5_weight / 2.0,
                )
            )

        else:
            bfs.append(
                (bf.M5DiffBasisFunction(filtername=filtername, nside=nside), m5_weight)
            )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                    bf.FootprintBasisFunction(
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
                bf.SlewtimeBasisFunction(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append(
            (bf.StrictFilterBasisFunction(filtername=filtername), stayfilter_weight)
        )
        bfs.append(
            (
                bf.VisitRepeatBasisFunction(
                    gap_min=0, gap_max=2 * 60.0, filtername=None, nside=nside, npairs=20
                ),
                repeat_weight,
            )
        )

        if filtername2 is not None:
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight / 2.0,
                )
            )
            bfs.append(
                (
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername2,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername2),
                        n_obs=n_obs_template[filtername2],
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
                    bf.NObsPerYearBasisFunction(
                        filtername=filtername,
                        nside=nside,
                        footprint=footprints.get_footprint(filtername),
                        n_obs=n_obs_template[filtername],
                        season=season,
                        season_start_hour=season_start_hour,
                        season_end_hour=season_end_hour,
                    ),
                    template_weight,
                )
            )
        if repeat_night_weight is not None:
            bfs.append(
                (
                    bf.AvoidLongGapsBasisFunction(
                        nside=nside,
                        filtername=None,
                        min_gap=0.0,
                        max_gap=10.0 / 24.0,
                        ha_limit=3.5,
                        footprint=wfd_footprint,
                    ),
                    repeat_night_weight,
                )
            )
        # Make sure we respect scheduled observations
        bfs.append((bf.TimeToScheduledBasisFunction(time_needed=scheduled_respect), 0))
        # Masks, give these 0 weight
        bfs.append(
            (
                bf.ZenithShadowMaskBasisFunction(
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
                bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=moon_distance),
                0.0,
            )
        )
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append((bf.FilterLoadedBasisFunction(filternames=filternames), 0))
        if filtername2 is None:
            time_needed = times_needed[0]
        else:
            time_needed = times_needed[1]
        bfs.append(
            (bf.TimeToTwilightBasisFunction(time_needed=time_needed, alt_limit=12), 0.0)
        )
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0.0))

        # Let's turn off twilight blobs on nights where we are
        # doing NEO hunts
        bfs.append((bf.NightModuloBasisFunction(pattern=night_pattern), 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]
        if filtername2 is None:
            survey_name = "blob_twi, %s" % filtername
        else:
            survey_name = "blob_twi, %s%s" % (filtername, filtername2)
        if filtername2 is not None:
            detailer_list.append(detailers.TakeAsPairsDetailer(filtername=filtername2))
        surveys.append(
            BlobSurvey(
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
                **BlobSurvey_params
            )
        )

    return surveys


def ecliptic_target(nside=32, dist_to_eclip=40.0, dec_max=30.0, mask=None):
    """Generate a target map for the area around the ecliptic

    Paramters
    ---------
    dist_to_eclip : float (40)
        How far to extend from the ecliptic (degrees). default 40.
    dec_max : float (30)
        The maximum declination to extend to (degrees). default 30.
    mask : np.array (None)
        A HEALpix map that the result is multiplied by (default None)
    """

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where(
        (np.abs(eclip_lat) < np.radians(dist_to_eclip)) & (dec < np.radians(dec_max))
    )
    result[good] += 1

    if mask is not None:
        result *= mask

    return result


def generate_twilight_neo(
    nside,
    night_pattern=None,
    nexp=1,
    exptime=15,
    ideal_pair_time=5.0,
    max_airmass=2.0,
    camera_rot_limits=[-80.0, 80.0],
    time_needed=10,
    footprint_mask=None,
    footprint_weight=0.1,
    slewtime_weight=3.0,
    stayfilter_weight=3.0,
    area_required=None,
    filters="riz",
    n_repeat=4,
    sun_alt_limit=-14.8,
):
    """Generate a survey for observing NEO objects in twilight

    Parameters
    ----------
    night_pattern : list of bool (None)
        A list of bools that set when the survey will be active. e.g., [True, False]
        for every-other night, [True, False, False] for every third night.
    nexp : int (1)
        Number of snaps in a visit
    exptime : float (15)
        Exposure time of visits
    ideal_pair_time : float (5)
        Ideal time between repeat visits (minutes).
    max_airmass : float (2)
        Maximum airmass to attempt (unitless).
    camera_rot_limits : list of float ([-80, 80])
        The camera rotation limits to use (degrees).
    time_needed : float (10)
        How much time should be available (e.g., before twilight ends) (minutes).
    footprint_mask : np.array (None)
        Mask to apply to the constructed ecliptic target mask (None).
    footprint_weight : float (0.1)
        Weight for footprint basis function
    slewtime_weight : float (3.)
        Weight for slewtime basis function
    stayfilter_weight : float (3.)
        Weight for staying in the same filter basis function
    area_required : float (None)
        The area that needs to be available before the survey will return observations (sq degrees?)
    filters : str ('riz')
        The filters to use, default 'riz'
    n_repeat : int (4)
        The number of times a blob should be repeated, default 4.
    sun_alt_limit : float (-14.8)
        Do not start unless sun is higher than this limit (degrees)
    """
    # XXX finish eliminating magic numbers and document this one
    slew_estimate = 4.5
    survey_name = "twilight_neo"
    footprint = ecliptic_target(nside=nside, mask=footprint_mask)
    constant_fp = ConstantFootprint()
    for filtername in filters:
        constant_fp.set_footprint(filtername, footprint)

    surveys = []
    for filtername in filters:
        detailer_list = []
        detailer_list.append(
            detailers.CameraRotDetailer(
                min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
            )
        )
        detailer_list.append(detailers.CloseAltDetailer())
        # Should put in a detailer so things start at lowest altitude
        detailer_list.append(
            detailers.TwilightTripleDetailer(
                slew_estimate=slew_estimate, n_repeat=n_repeat
            )
        )
        bfs = []

        bfs.append(
            (
                bf.FootprintBasisFunction(
                    filtername=filtername,
                    footprint=constant_fp,
                    out_of_bounds_val=np.nan,
                    nside=nside,
                ),
                footprint_weight,
            )
        )

        bfs.append(
            (
                bf.SlewtimeBasisFunction(filtername=filtername, nside=nside),
                slewtime_weight,
            )
        )
        bfs.append(
            (bf.StrictFilterBasisFunction(filtername=filtername), stayfilter_weight)
        )
        # Need a toward the sun, reward high airmass, with an airmass cutoff basis function.
        bfs.append(
            (bf.NearSunTwilightBasisFunction(nside=nside, max_airmass=max_airmass), 0)
        )
        bfs.append(
            (
                bf.ZenithShadowMaskBasisFunction(
                    nside=nside, shadow_minutes=60.0, max_alt=76.0
                ),
                0,
            )
        )
        bfs.append((bf.MoonAvoidanceBasisFunction(nside=nside, moon_distance=30.0), 0))
        bfs.append((bf.FilterLoadedBasisFunction(filternames=filtername), 0))
        bfs.append((bf.PlanetMaskBasisFunction(nside=nside), 0))
        bfs.append(
            (
                bf.SolarElongationMaskBasisFunction(
                    min_elong=0.0, max_elong=60.0, nside=nside
                ),
                0,
            )
        )

        bfs.append((bf.NightModuloBasisFunction(pattern=night_pattern), 0))
        # Do not attempt unless the sun is getting high
        bfs.append(((bf.SunAltHighLimitBasisFunction(alt_limit=sun_alt_limit)), 0))

        # unpack the basis functions and weights
        weights = [val[1] for val in bfs]
        basis_functions = [val[0] for val in bfs]

        # Set huge ideal pair time and use the detailer to cut down the list of observations to fit twilight?
        surveys.append(
            BlobSurvey(
                basis_functions,
                weights,
                filtername1=filtername,
                filtername2=None,
                ideal_pair_time=ideal_pair_time,
                nside=nside,
                exptime=exptime,
                survey_note=survey_name,
                ignore_obs=["DD", "greedy", "blob"],
                dither=True,
                nexp=nexp,
                detailers=detailer_list,
                az_range=180.0,
                twilight_scale=False,
                area_required=area_required,
            )
        )
    return surveys


def example_scheduler(
    max_dither=0.7,
    nexp=2,
    nslice=2,
    rolling_scale=0.9,
    gsw=3,
    nights_off=6,
    neo_night_pattern=[True, False, False, False],
    neo_filters="riz",
    neo_repeat=4,
    ddf_season_frac=0.2,
    mjd_start=60676.0,
    nside=32,
    per_night=True,
    camera_ddf_rot_limit=75.0,
):
    """Make an example scheduler

    Parameters
    ----------
    max_dither : float (0.7)
        The maximum amount to spacially dither the DDF fields (degrees)
    nexp : int (2)
        Number of snaps to split each visit into (default 2). Does not apply to u-band.
    nslice : int (2)
        Fraction of sky to divide for rolling (default 2).
    rolling_scale : float (0.9)
        The strength of rolling (between 0-1). Default 0.9.
    gsw : float (3)
        The good seeing weight (unitless). Default 3.
    nights_off : int (6)
        How many nights to take off between nights with long-gap observations (days).
    neo_night_pattern : list of bool ([True, False, False, False])
        The pattern of observations to use for twilight NEO observations,
        default [True, False, False, False].
    neo_filters : str ('riz')
        Which filters to use for twilight NEO observations ('riz').
    neo_repeat : int (4)
        How many times a pointing should be repeated when taking NEO observations.
        Default 4.
    ddf_season_frac : (0.2)
        XXX--should be updating to a more intuitive name soon
    mjd_start : float (60676.0)
        The MJD to start the survey on (60676.0)
    nside : int (32)
        The HEALpix nside to use. Default 32.
    per_night : bool (True)
        Dither the DDFs on a per-night basis. Default True.
    camera_ddf_rot_limit : float (75)
        Limit for how far to rotationally dither DDF fields (degrees)

    """

    reverse_neo_night_pattern = [not val for val in neo_night_pattern]

    # Create survey footprint
    sky = SkyAreaGeneratorGalplane(nside=nside, smc_radius=4, lmc_radius=6)
    footprints_hp_array, labels = sky.return_maps()

    wfd_indx = np.where(
        (labels == "lowdust") | (labels == "LMC_SMC") | (labels == "virgo")
    )[0]
    wfd_footprint = footprints_hp_array["r"] * 0
    wfd_footprint[wfd_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    footprint_mask = footprints_hp["r"] * 0
    footprint_mask[np.where(footprints_hp["r"] > 0)] = 1

    repeat_night_weight = None

    # Need to find the sun RA at the start of the survey
    sun = get_sun(Time(mjd_start, format="mjd"))

    footprints = make_rolling_footprints(
        fp_hp=footprints_hp,
        mjd_start=mjd_start,
        sun_ra_start=sun.ra.rad,
        nslice=nslice,
        scale=rolling_scale,
        nside=nside,
        wfd_indx=wfd_indx,
        order_roll=1,
        n_cycles=4,
    )

    gaps_night_pattern = [True] + [False] * nights_off

    long_gaps = gen_long_gaps_survey(
        nside=nside, footprints=footprints, night_pattern=gaps_night_pattern
    )

    # Set up the DDF surveys to dither
    u_detailer = detailers.FilterNexp(filtername="u", nexp=1)
    dither_detailer = detailers.DitherDetailer(
        per_night=per_night, max_dither=max_dither
    )
    details = [
        detailers.CameraRotDetailer(
            min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit
        ),
        dither_detailer,
        u_detailer,
        detailers.Rottep2RotspDesiredDetailer(),
    ]
    euclid_detailers = [
        detailers.CameraRotDetailer(
            min_rot=-camera_ddf_rot_limit, max_rot=camera_ddf_rot_limit
        ),
        detailers.EuclidDitherDetailer(),
        u_detailer,
        detailers.Rottep2RotspDesiredDetailer(),
    ]

    # Note, using old simple DDF rather than the pre-scheduled ones for simplicity.
    # We could put the ddf_grid.npz in rubin_sim_data if we wanted to.
    ddfs = generate_dd_surveys(
        nside=nside, nexp=nexp, detailers=details, euclid_detailers=euclid_detailers
    )

    greedy = gen_greedy_surveys(nside, nexp=nexp, footprints=footprints)
    neo = generate_twilight_neo(
        nside,
        night_pattern=neo_night_pattern,
        filters=neo_filters,
        n_repeat=neo_repeat,
        footprint_mask=footprint_mask,
    )
    blobs = generate_blobs(
        nside,
        nexp=nexp,
        footprints=footprints,
        mjd_start=mjd_start,
        good_seeing_weight=gsw,
    )
    twi_blobs = generate_twi_blobs(
        nside,
        nexp=nexp,
        footprints=footprints,
        wfd_footprint=wfd_footprint,
        repeat_night_weight=repeat_night_weight,
        night_pattern=reverse_neo_night_pattern,
    )
    surveys = [ddfs, long_gaps, blobs, twi_blobs, neo, greedy]

    scheduler = CoreScheduler(surveys, nside=nside)

    return scheduler
