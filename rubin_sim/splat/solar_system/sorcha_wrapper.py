__all__ = ("sorcha_wrapper",)

from sorcha.sorcha import runLSSTSimulation
from sorcha.utilities.sorchaConfigs import (
    inputConfigs,
    simulationConfigs,
    filtersConfigs,
    saturationConfigs,
    phasecurvesConfigs,
    fovConfigs,
    fadingfunctionConfigs,
    linkingfilterConfigs,
    outputConfigs,
    lightcurveConfigs,
    activityConfigs,
    expertConfigs,
    auxiliaryConfigs,
    basesorchaConfigs,
)
from sorcha.utilities.sorchaArguments import sorchaArguments


def sorcha_wrapper(
    pointing_database,
    orbin_file,
    colors_file,
    stats_out_file="",
    ephemerides_type="ar",
    observing_filters="r,g,i,z,u,y",
    query=None,
    seed=42,
):
    """Use sorcha to find overlap between pointing history and
    solar system objects

    Parameters
    ----------
    pointing_database : `str`
        Database to use for selecting visits.
    orbin_file : `str`
        File with orbit info. Expecting csv file with columns:
        ObjID a e inc node argPeri ma epochMJD_TDB FORMAT
    colors_file : `str`
        File with color info for each object in orbin_file.
        expecting columns of:
        ObjID H_r GS u-r g-r i-r z-r y-r
    ephemerides_type : `str`
        ephemerides_type passed to sorcha.utilities.sorchaConfigs.inputConfigs
    observing_filters : `str`
        Passed to various sorcha configs
    query : `str`
        SQL query executed on pointing_database. Default None will
        construct a reasonable selection for everything.
    seed : `float`
        Random number seed passed to sorcha.

    """
    if query is None:
        query = (
            "SELECT observationId, observationStartMJD as observationStartMJD_TAI, "
            "visitTime, visitExposureTime, filter, seeingFwhmGeom as seeingFwhmGeom_arcsec, "
            "seeingFwhmEff as seeingFwhmEff_arcsec, fiveSigmaDepth as fieldFiveSigmaDepth_mag , "
            "fieldRA as fieldRA_deg, fieldDec as fieldDec_deg, rotSkyPos as "
            "fieldRotSkyPos_deg FROM observations order by observationId"
        )

    # I think this just sets the FoV geometry
    survey_name = "rubin_sim"

    input_config = inputConfigs(
        ephemerides_type=ephemerides_type,
        eph_format="csv",
        size_serial_chunk=5000,
        aux_format="whitespace",
        pointing_sql_query=query,
    )

    simulation_config = simulationConfigs(
        ar_ang_fov=2.06,
        ar_fov_buffer=0.2,
        ar_picket=1,
        ar_obs_code="X05",
        ar_healpix_order=6,
        ar_n_sub_intervals=101,
        _ephemerides_type=ephemerides_type,
    )

    filters_config = filtersConfigs(
        observing_filters=observing_filters, survey_name=survey_name, mainfilter=None, othercolours=None
    )

    saturation_config = saturationConfigs(
        bright_limit_on=False, bright_limit=None, _observing_filters=observing_filters
    )
    phasecurves_config = phasecurvesConfigs(phase_function="HG")
    fov_config = fovConfigs(
        camera_model="footprint",
        footprint_path=None,
        fill_factor=None,
        circle_radius=None,
        footprint_edge_threshold=2.0,
        survey_name=survey_name,
    )
    fading_config = fadingfunctionConfigs(
        fading_function_on=True, fading_function_width=0.1, fading_function_peak_efficiency=1.0
    )
    linking_config = linkingfilterConfigs(
        ssp_linking_on=True,
        drop_unlinked=False,
        ssp_detection_efficiency=0.95,
        ssp_number_observations=2,
        ssp_separation_threshold=0.5,
        ssp_maximum_time=0.0625,
        ssp_number_tracklets=3,
        ssp_track_window=15,
        ssp_night_start_utc=16.0,
        survey_name=survey_name,
    )
    output_confg = outputConfigs(
        output_format="csv", output_columns="basic", position_decimals=None, magnitude_decimals=None
    )
    lightcurve_config = lightcurveConfigs(lc_model=None)
    activity_config = activityConfigs(comet_activity=None)
    expert_config = expertConfigs()
    aux_config = auxiliaryConfigs()
    config = basesorchaConfigs(
        survey_name=survey_name,
        input=input_config,
        simulation=simulation_config,
        filters=filters_config,
        saturation=saturation_config,
        phasecurves=phasecurves_config,
        fov=fov_config,
        fadingfunction=fading_config,
        linkingfilter=linking_config,
        output=output_confg,
        lightcurve=lightcurve_config,
        activity=activity_config,
        expert=expert_config,
        auxiliary=aux_config,
    )
    sorcha_args = sorchaArguments(
        cmd_args_dict={
            "paramsinput": colors_file,
            "orbinfile": orbin_file,
            "input_ephemeris_file": None,
            "configfile": colors_file,
            "outpath": "./",
            "outfilestem": "nofile_test",
            # Why is this in here 3x now?
            "visits": pointing_database,
            "pointing_database": pointing_database,
            "visits_database": pointing_database,
            "output_ephemeris_file": None,
            "ar_data_path": None,
            "loglevel": None,
            "stats": stats_out_file,
            "surveyname": survey_name,
            "seed": seed,
        }
    )

    observations, stats = runLSSTSimulation(sorcha_args, config, return_only=True)

    return observations, stats
