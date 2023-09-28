__all__ = ("artificial_structure_calculation",)

#####################################################################################################
# Purpose: calculate artificial structure, i.e. fluctuations in galaxy counts, resulting from
# imperfect observing strategy (OS). Includes the functionality to account for dust extinction,
# photometric calibration errors (simple ansatz implemented here), individual redshift bins (see
# GalaxyCountsMetricExtended for details), as well as poisson noise in the galaxy counts.
#
# Basic workflow, assuming all the functionalities are used:
#       1. HEALpix slicers are set up for survey strategies.
#       2. Using GalaxyCountMetric_extended, which handles dust extinction and calculates galaxy counts
#          based on redshift-bin-specific powerlaws, galaxy counts are found for each HEALpix pixel.
#       3. The shallow borders are masked (based on user-specified 'pixel radius').
#       4. Photometric calibration errors are calculated.
#       5. The galaxy counts in each pixel are recalculated using galaxy_counts_with_pixel_calibration
#          since the calibration errors modify the upper limit on the integral used to calculate
#          galaxy counts. galaxy_counts_with_pixel_calibration takes in each pixel's modified integration
#          limit individually.
#       6. Poisson noise is added to the galaxy counts.
#       7. Fluctuations in the galaxy counts are calculated.
#
# For each pixel i, the photometric calibration errors are modeled as del_i= k*z_i/sqrt(nObs_i),
# where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
# and k is a constant such that var(del_i)= (0.01)^2 -- 0.01 in accordance with LSST goal for relative
# photometric calibration.
#
# Most of the functionalities can be turned on/off, and plots and data can be saved at various points.
# Bordering masking adds significant run time as does the incorporation of photometric calibration
# errors. See the method descrpition for further details.
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
__all__ = ("artificial_structure_calculation",)

import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

try:
    from sympy import Symbol
    from sympy.solvers import solve
except ImportError:
    pass
import copy
import datetime
import time

from matplotlib.ticker import FuncFormatter

import rubin_sim.maf
import rubin_sim.maf.db as db
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as metricBundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as mafStackers  # stackers in sims_maf
from rubin_sim.maf.maf_contrib.lss_obs_strategy.constants_for_pipeline import plot_color, power_law_const_a
from rubin_sim.maf.maf_contrib.lss_obs_strategy.galaxy_counts_metric_extended import (
    GalaxyCountsMetricExtended as GalaxyCountsMetric,
)
from rubin_sim.maf.maf_contrib.lss_obs_strategy.galaxy_counts_with_pixel_calibration import (
    galaxy_counts_with_pixel_calibration as GalaxyCounts_0ptErrors,
)
from rubin_sim.maf.maf_contrib.lss_obs_strategy.masking_algorithm_generalized import (
    masking_algorithm_generalized,
)
from rubin_sim.maf.metrics import CountMetric as NumObsMetric


def artificial_structure_calculation(
    path,
    upper_mag_limit,
    dbfile,
    run_name,
    no_dith_only=False,
    best_dith_only=False,
    specified_dith=None,
    nside=128,
    filter_band="i",
    cut_off_year=None,
    redshift_bin="all",
    cfhtls_counts=False,
    normalized_mock_catalog_counts=True,
    include_dust_extinction=True,
    save_raw_num_gal_data=True,
    pixel_radius_for_masking=5,
    save_num_gal_data_after_masking=False,
    include0pt_errors=True,
    print0pt_information=True,
    plot0pt_plots=True,
    show0pt_plots=False,
    save0pt_plots=True,
    save_num_gal_data_after0pt=False,
    add_poisson_noise=True,
    save_delta_n_by_n_data=True,
    save_cls_for_delta_n_by_n=True,
    show_comp_plots=False,
    return_stuff=False,
):
    """
    Calculate artificial structure, i.e. fluctuations in galaxy counts dN/N, resulting due
    to imperfect observing strategy (OS).
    - Creates an output directory for subdirectories containing the specified things to save.
    - Prints out execution time at key steps (after border-masking, incorporating calibration errors, etc.)
    - Returns the metricBundle object containing the calculated dN/N, the output directory name,
    the results_db object, and (if include0pt_errors=True)  calibration errors for each survey strategy.

    Parameters
    ----------
    path : str
        path to the main directory where output directory is to be saved.
    upper_mag_limit : float
        upper limit on magnitude when calculating the galaxy counts.
    dbfile : str
        path to the OpSim output file, e.g. to a copy of enigma_1189
    run_name : str
        run name tag to identify the output of specified OpSim output.
        Since new OpSim outputs have different columns, the run_name for enigma_1189 **must**
        be 'enigma1189'; can be anything for other outputs, e.g. 'minion1016'
    no_dith_only : `bool`
        set to True if only want to consider the undithered survey. Default: False
    best_dith_only : `bool`
        set to True if only want to consider RandomDitherFieldPerVisit.
        Default: False
    specified_dith : str
        specific dither strategy to run; could be a string or a list of strings.
        Default: None
    nside : int
        HEALpix resolution parameter. Default: 128
    filter_band : str
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    cut_off_year : int
        year cut to restrict analysis to only a subset of the survey.
        Must range from 1 to 9, or None for the full survey analysis (10 yrs).
        Default: None
    redshift_bin : str
        options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
        '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5', '3.5<z<4.0',
        'all' for no redshift restriction (i.e. 0.<z<4.0)
        Default: 'all'
    cfhtls_counts : `bool`
        set to True if want to calculate the total galaxy counts from CFHTLS
        powerlaw from LSST Science Book. Must be run with redshift_bin='all'
        Default: False
    normalized_mock_catalog_counts : `bool`
        set to False if  want the raw/un-normalized galaxy
        counts from mock catalogs. Default: True
    include_dust_extinction : `bool`:
        set to include dust extinction when calculating the coadded
        depth. Default: True
    save_raw_num_gal_data : `bool`
        set to True to save num_gal data right away, i.e. before
        0pt error calibration, bordering masking, or poisson noise.
        Default: True
    pixel_radius_for_masking : int
        number of pixels to mask along the shallow border. Default: 5
    save_num_gal_data_after_masking : `bool`
        set to True to save num_gal data after border masking.
        Default: False
    include0pt_errors : `bool`
        set to True to include photometric calibration errors.
        Default: True
    print0pt_information : `bool`
        set to True to print out some statistics (variance, the k-value, etc.)
        of the calibration errors of every dither strategy.
        Default: True
    plot0pt_plots : `bool`
        generate 0pt plots. Default True.
    save_num_gal_data_after0pt : `bool`
        set to True to save num_gal data after border masking and 0pt calibration. Default: False
    add_poisson_noise : `bool`
        set to True to add poisson noise to the galaxy counts after border masking
        and the incorporation of calibration errors. Default: True
    saveNumGalDataAfterPoisson : `bool`
        set to True to save num_gal data right away, after border masking,
        including the calibration errors, and the  poisson noise.
        Default: True
    showDeltaNByNPlots : `bool`
        set to True to show the plots related to the fluctuations in the galaxy
        counts. Will work only when plotDeltaNByN=True. Default: False
    saveDeltaNByNPlots : `bool`
        set to True to save the plots related to the fluctuations in the galaxy
        counts. Will work only when plotDeltaNByN=True. Default: True
    save_delta_n_by_n_data : `bool`
        set to True to save data for the the fluctuations in the galaxy counts.
        Default: True
    save_cls_for_delta_n_by_n : `bool`
        set to True to save the power spectrum data for the the fluctuations in
        the galaxy counts. Default: True
    show_comp_plots : `bool`
        set to True if want to display the comparison plots (only valid if have more
        han one dither strategy); otherwise, the plots will be saved and not displayed.
        Default: False
    return_stuff : `bool`
        set to True to get the metricBundle object, the out_dir, and results_db object.
        Default: False
    """
    start_time = time.time()
    # ------------------------------------------------------------------------
    # set up the metric
    gal_count_metric = GalaxyCountsMetric(
        upper_mag_limit=upper_mag_limit,
        include_dust_extinction=include_dust_extinction,
        redshift_bin=redshift_bin,
        filter_band=filter_band,
        nside=nside,
        cfhtls_counts=cfhtls_counts,
        normalized_mock_catalog_counts=normalized_mock_catalog_counts,
    )
    # OpSim database
    opsdb = db.OpsimDatabase(dbfile)

    # ------------------------------------------------------------------------
    # set up the out_dir name
    zeropt_tag, dust_tag = "", ""
    if include0pt_errors:
        zeropt_tag = "with0ptErrors"
    else:
        zeropt_tag = "no0ptErrors"

    if include_dust_extinction:
        dust_tag = "withDustExtinction"
    else:
        dust_tag = "noDustExtinction"

    if cut_off_year is not None:
        survey_tag = "%syearCut" % (cut_off_year)
    else:
        survey_tag = "fullSurveyPeriod"

    # check to make sure redshift bin is ok.
    allowed_redshift_bins = list(power_law_const_a.keys()) + ["all"]
    if redshift_bin not in allowed_redshift_bins:
        print("ERROR: Invalid redshift bin. Input bin can only be among %s.\n" % (allowed_redshift_bins))
        return
    zbin_tag = redshift_bin
    if redshift_bin == "all":
        zbin_tag = "allRedshiftData"

    poisson_tag = ""
    if add_poisson_noise:
        poisson_tag = "withPoissonNoise"
    else:
        poisson_tag = "noPoissonNoise"

    counts_tag = ""
    if cfhtls_counts:
        counts_tag = "CFHTLSpowerLaw"
    elif normalized_mock_catalog_counts:
        counts_tag = "normalizedGalaxyCounts"
    else:
        counts_tag = "unnormalizedGalaxyCounts"

    out_dir = (
        f"artificialStructure_{poisson_tag}_nside{nside}"
        f"_pixelRadiusFormasking_{pixel_radius_for_masking}_{zeropt_tag}_{dust_tag}_{filter_band}"
        f"_{upper_mag_limit}_{run_name}_{survey_tag}_{zbin_tag}_{counts_tag}_directory"
    )

    print("# out_dir: %s\n" % out_dir)
    if not os.path.exists("%s%s" % (path, out_dir)):
        os.makedirs("%s%s" % (path, out_dir))

    results_dbname = "resultsDb_%s.db" % np.random.randint(100)
    results_db = db.ResultsDb(database=results_dbname, out_dir="%s%s" % (path, out_dir))

    # ------------------------------------------------------------------------
    # set up the sql constraint
    prop_ids, prop_tags = opsdb.fetchPropInfo()
    wfd_where = opsdb.createSQLWhere("WFD", prop_tags)
    ra_dec_in_deg = opsdb.raDecInDeg
    if cut_off_year is not None:
        night_cut_off = (cut_off_year) * 365.25
        sqlconstraint = '%s and night<=%s and filter=="%s"' % (
            wfd_where,
            night_cut_off,
            filter_band,
        )
    else:
        sqlconstraint = '%s and filter=="%s"' % (wfd_where, filter_band)
    print("# sqlconstraint: %s" % sqlconstraint)

    # ------------------------------------------------------------------------
    # create a ReadMe type file to put info in.
    update = "%s\n" % (datetime.datetime.now())
    update += "\nArtificial structure calculation with %s, %s, and %s " % (
        zeropt_tag,
        dust_tag,
        poisson_tag,
    )
    update += "for %s for %s for %s<%s. " % (
        survey_tag,
        zbin_tag,
        filter_band,
        upper_mag_limit,
    )
    update += "\nWith %s and PixelRadiusForMasking: %s.\n" % (
        counts_tag,
        pixel_radius_for_masking,
    )
    update += "\nsqlconstraint: %s" % sqlconstraint
    update += "\nRunning with %s\n" % run_name
    update += "\nout_dir: %s\n" % out_dir
    update += "\nMAF version: %s\n" % rubin_sim.maf.__version__

    # figure out the readme name
    readme_name = "ReadMe"
    readmes = [f for f in os.listdir("%s%s" % (path, out_dir)) if any([f.endswith(".txt")])]
    num_file = 0
    for f in readmes:
        if f.__contains__("%s_" % readme_name):
            temp = f.split(".txt")[0]
            num_file = max(num_file, int(temp.split("%s_" % readme_name)[1]))
        else:
            num_file = 1
    readme_name = "ReadMe_%s.txt" % (num_file + 1)
    readme = open("%s%s/%s" % (path, out_dir, readme_name), "w")
    readme.write(update)
    readme.close()

    # ------------------------------------------------------------------------
    # setup all the slicers. set up random_seed for random/repRandom strategies through stacker_list.
    slicer = {}
    stacker_list = {}

    if specified_dith is not None:
        # would like to add all the stackers first and then keep only the one that is specified
        best_dith_only, no_dith_only = False, False

    if best_dith_only:
        stacker_list["RandomDitherFieldPerVisit"] = [
            mafStackers.RandomDitherFieldPerVisitStacker(degrees=ra_dec_in_deg, random_seed=1000)
        ]
        slicer["RandomDitherFieldPerVisit"] = slicers.HealpixSlicer(
            lon_col="randomDitherFieldPerVisitRa",
            lat_col="randomDitherFieldPerVisitDec",
            lat_lon_deg=ra_dec_in_deg,
            nside=nside,
            use_cache=False,
        )
    else:
        slicer["NoDither"] = slicers.HealpixSlicer(
            lon_col="fieldRA",
            lat_col="fieldDec",
            lat_lon_deg=ra_dec_in_deg,
            nside=nside,
            use_cache=False,
        )
        if not no_dith_only:
            # random dithers on different timescales
            stacker_list["RandomDitherPerNight"] = [
                mafStackers.RandomDitherPerNightStacker(degrees=ra_dec_in_deg, random_seed=1000)
            ]
            stacker_list["RandomDitherFieldPerNight"] = [
                mafStackers.RandomDitherFieldPerNightStacker(degrees=ra_dec_in_deg, random_seed=1000)
            ]
            stacker_list["RandomDitherFieldPerVisit"] = [
                mafStackers.RandomDitherFieldPerVisitStacker(degrees=ra_dec_in_deg, random_seed=1000)
            ]

            # rep random dithers on different timescales
            # stacker_list['RepulsiveRandomDitherPerNight'] = [myStackers.RepulsiveRandomDitherPerNightStacker(degrees=ra_dec_in_deg,
            #                                                                                                random_seed=1000)]
            # stacker_list['RepulsiveRandomDitherFieldPerNight'] = [myStackers.RepulsiveRandomDitherFieldPerNightStacker(degrees=ra_dec_in_deg,
            #                                                                                                          random_seed=1000)]
            # stacker_list['RepulsiveRandomDitherFieldPerVisit'] = [myStackers.RepulsiveRandomDitherFieldPerVisitStacker(degrees=ra_dec_in_deg,
            #                                                                                                          random_seed=1000)]
            # set up slicers for different dithers
            # random dithers on different timescales
            slicer["RandomDitherPerNight"] = slicers.HealpixSlicer(
                lon_col="randomDitherPerNightRa",
                lat_col="randomDitherPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["RandomDitherFieldPerNight"] = slicers.HealpixSlicer(
                lon_col="randomDitherFieldPerNightRa",
                lat_col="randomDitherFieldPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["RandomDitherFieldPerVisit"] = slicers.HealpixSlicer(
                lon_col="randomDitherFieldPerVisitRa",
                lat_col="randomDitherFieldPerVisitDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            # rep random dithers on different timescales
            # slicer['RepulsiveRandomDitherPerNight'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherPerNightRa',
            #                                                               latCol='repulsiveRandomDitherPerNightDec',
            #                                                               latLonDeg=ra_dec_in_deg, nside=nside, use_cache=False)
            # slicer['RepulsiveRandomDitherFieldPerNight'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerNightRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerNightDec',
            #                                                                    latLonDeg=ra_dec_in_deg, nside=nside,
            #                                                                    use_cache=False)
            # slicer['RepulsiveRandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerVisitRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerVisitDec',
            #                                                                    latLonDeg=ra_dec_in_deg, nside=nside,
            #                                                                    use_cache=False)
            # spiral dithers on different timescales
            slicer["FermatSpiralDitherPerNight"] = slicers.HealpixSlicer(
                lon_col="fermatSpiralDitherPerNightRa",
                lat_col="fermatSpiralDitherPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["FermatSpiralDitherFieldPerNight"] = slicers.HealpixSlicer(
                lon_col="fermatSpiralDitherFieldPerNightRa",
                lat_col="fermatSpiralDitherFieldPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["FermatSpiralDitherFieldPerVisit"] = slicers.HealpixSlicer(
                lon_col="fermatSpiralDitherFieldPerVisitRa",
                lat_col="fermatSpiralDitherFieldPerVisitDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            # hex dithers on different timescales
            slicer["SequentialHexDitherPerNight"] = slicers.HealpixSlicer(
                lon_col="hexDitherPerNightRa",
                lat_col="hexDitherPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["SequentialHexDitherFieldPerNight"] = slicers.HealpixSlicer(
                lon_col="hexDitherFieldPerNightRa",
                lat_col="hexDitherFieldPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["SequentialHexDitherFieldPerVisit"] = slicers.HealpixSlicer(
                lon_col="hexDitherFieldPerVisitRa",
                lat_col="hexDitherFieldPerVisitDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            # per season dithers
            slicer["PentagonDitherPerSeason"] = slicers.HealpixSlicer(
                lon_col="pentagonDitherPerSeasonRa",
                lat_col="pentagonDitherPerSeasonDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["PentagonDiamondDitherPerSeason"] = slicers.HealpixSlicer(
                lon_col="pentagonDiamondDitherPerSeasonRa",
                lat_col="pentagonDiamondDitherPerSeasonDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["SpiralDitherPerSeason"] = slicers.HealpixSlicer(
                lon_col="spiralDitherPerSeasonRa",
                lat_col="spiralDitherPerSeasonDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
    # ------------------------------------------------------------------------
    if specified_dith is not None:
        stacker_list_, slicer_ = {}, {}
        if isinstance(specified_dith, str):
            if specified_dith in slicer.keys():
                if specified_dith.__contains__("Random"):
                    # only Random dithers have a stacker object for rand seed specification
                    stacker_list_[specified_dith] = stacker_list[specified_dith]
                slicer_[specified_dith] = slicer[specified_dith]
        elif isinstance(specified_dith, list):
            for specific in specified_dith:
                if specific in slicer.keys():
                    if specific.__contains__("Random"):
                        # only Random dithers have a stacker object for rand seed specification
                        stacker_list_[specific] = stacker_list[specific]
                    slicer_[specific] = slicer[specific]
        else:
            err = "Invalid value for specified_dith: %s." % specified_dith
            err += "Allowed values include one of the following:\n%s" % (slicer.keys())
            raise ValueError(err)
        stacker_list, slicer = stacker_list_, slicer_

    print("\nRunning the analysis for %s" % slicer.keys())
    # ------------------------------------------------------------------------
    readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
    readme.write("\nObserving strategies considered: %s\n" % (list(slicer.keys())))
    readme.close()
    # ------------------------------------------------------------------------
    # set up bundle for num_gal (and later deltaN/N)
    my_bundles = {}
    dust_map = maps.DustMap(
        interp=False, nside=nside
    )  # include dust_map; actual in/exclusion of dust is handled by the galaxyCountMetric
    for dither in slicer:
        if dither in stacker_list:
            my_bundles[dither] = metricBundles.MetricBundle(
                gal_count_metric,
                slicer[dither],
                sqlconstraint,
                stacker_list=stacker_list[dither],
                run_name=run_name,
                metadata=dither,
                maps_list=[dust_map],
            )
        else:
            my_bundles[dither] = metricBundles.MetricBundle(
                gal_count_metric,
                slicer[dither],
                sqlconstraint,
                run_name=run_name,
                metadata=dither,
                maps_list=[dust_map],
            )
    # ------------------------------------------------------------------------
    # run the metric/slicer combination for galaxy counts (num_gal)
    print("\n# Running my_bundles ...")
    b_group = metricBundles.MetricBundleGroup(
        my_bundles,
        opsdb,
        out_dir="%s%s" % (path, out_dir),
        results_db=results_db,
        save_early=False,
    )
    b_group.run_all()
    b_group.write_all()

    # ------------------------------------------------------------------------
    # print out tot(num_gal) associated with each strategy
    # write to the readme as well
    update = "\n# Before any border masking or photometric error calibration: "
    print(update)
    for dither in my_bundles:
        ind = np.where(my_bundles[dither].metricValues.mask[:] == False)[0]
        print_out = "Total Galaxies for %s: %.9e" % (
            dither,
            sum(my_bundles[dither].metricValues.data[ind]),
        )
        update += "\n %s" % print_out
        print(print_out)
    update += "\n"
    readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
    readme.write(update)
    readme.close()
    print("\n## Time since the start of the calculation: %.2f hrs" % ((time.time() - start_time) / 3600.0))

    # ------------------------------------------------------------------------
    # mask the edges: the data in the masked pixels is not changed
    plot_handler = plots.PlotHandler(
        out_dir="%s%s" % (path, out_dir),
        results_db=results_db,
        thumbnail=False,
        savefig=False,
    )
    print("\n# Masking the edges ...")
    my_bundles, border_pixels_masked = masking_algorithm_generalized(
        my_bundles,
        plot_handler,
        "Number of Galaxies",
        nside=nside,
        pixel_radius=pixel_radius_for_masking,
        plot_intermediate_plots=False,
        plot_final_plots=False,
        print_final_info=True,
        return_border_indices=True,
    )
    # ------------------------------------------------------------------------

    # save the num_gal data.
    if save_num_gal_data_after_masking:
        out_dir_new = "numGalData_afterBorderMasking"
        for b in my_bundles:
            my_bundles[b].write(out_dir=out_dir_new)

    # ------------------------------------------------------------------------
    # print out tot(num_gal) associated with each strategy
    # write to the readme as well
    if pixel_radius_for_masking != 0:
        update = "\n# After border masking: "
        print(update)
        for dither in my_bundles:
            ind = np.where(my_bundles[dither].metricValues.mask[:] == False)[0]
            print_out = "Total Galaxies for %s: %.9e" % (
                dither,
                sum(my_bundles[dither].metricValues.data[ind]),
            )
            print(print_out)
            update += "\n %s" % print_out
        update += "\n"

        readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
        readme.write(update)
        readme.close()
    print("\n## Time since the start of the calculation: %.2f hrs" % ((time.time() - start_time) / 3600.0))

    ################################################################################################################
    # If include 0pt errors
    # Ansatz: for each pixel i, del_i= k*z_i/sqrt(n_obs_i),
    # where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
    # and k is a constant such that var(del_i)= (0.01)^2. 0.01 for the 1% LSST goal.
    # k-constraint equation becomes: k^2*var(z_i/sqrt(n_obs_i))= (0.01)^2    --- equation 1
    if include0pt_errors:
        tablename = "SummaryAllProps"
        if tablename in opsdb.tableNames:
            colname = "seeingFwhmEff"
            if colname not in opsdb.columnNames[tablename]:
                raise ValueError("Unclear which seeing column to use.")
        elif "Summary" in opsdb.tableNames:
            tablename = "Summary"
            colname = "finSeeing"
            if colname not in opsdb.columnNames[tablename]:
                colname = "FWHMeff"
                if colname not in opsdb.columnNames[tablename]:
                    raise ValueError("Unclear which seeing column to use.")

        mean_metric = metrics.MeanMetric(col=colname)  # for avgSeeing per HEALpix pixel

        n_obs_metric = NumObsMetric(nside=nside)  # for numObs per HEALpix pixel
        if include_dust_extinction:
            coadd_metric = metrics.ExgalM5(lsstFilter=filter_band)
        else:
            coadd_metric = metrics.Coaddm5Metric()

        avg_seeing_bundle = {}
        n_obs_bundle = {}
        coadd_bundle = {}

        # can pass dust_map to metricBundle regardless of whether to include dust extinction or not.
        # the metric choice (coadd vs. exGal) takes care of whether to use the dust_map or not.
        dust_map = maps.DustMap(interp=False, nside=nside)
        for dither in slicer:
            if dither in stacker_list:
                avg_seeing_bundle[dither] = metricBundles.MetricBundle(
                    mean_metric,
                    slicer[dither],
                    sqlconstraint,
                    stacker_list=stacker_list[dither],
                    run_name=run_name,
                    metadata=dither,
                )
                n_obs_bundle[dither] = metricBundles.MetricBundle(
                    n_obs_metric,
                    slicer[dither],
                    sqlconstraint,
                    stacker_list=stacker_list[dither],
                    run_name=run_name,
                    metadata=dither,
                )
                coadd_bundle[dither] = metricBundles.MetricBundle(
                    coadd_metric,
                    slicer[dither],
                    sqlconstraint,
                    stacker_list=stacker_list[dither],
                    run_name=run_name,
                    metadata=dither,
                    maps_list=[dust_map],
                )
            else:
                avg_seeing_bundle[dither] = metricBundles.MetricBundle(
                    mean_metric,
                    slicer[dither],
                    sqlconstraint,
                    run_name=run_name,
                    metadata=dither,
                )
                n_obs_bundle[dither] = metricBundles.MetricBundle(
                    n_obs_metric,
                    slicer[dither],
                    sqlconstraint,
                    run_name=run_name,
                    metadata=dither,
                )
                coadd_bundle[dither] = metricBundles.MetricBundle(
                    coadd_metric,
                    slicer[dither],
                    sqlconstraint,
                    run_name=run_name,
                    metadata=dither,
                    maps_list=[dust_map],
                )
        print("\n# Running avg_seeing_bundle ...")
        a_group = metricBundles.MetricBundleGroup(
            avg_seeing_bundle,
            opsdb,
            out_dir="%s%s" % (path, out_dir),
            results_db=results_db,
            save_early=False,
        )
        a_group.run_all()

        print("\n# Running n_obs_bundle ...")
        n_group = metricBundles.MetricBundleGroup(
            n_obs_bundle,
            opsdb,
            out_dir="%s%s" % (path, out_dir),
            results_db=results_db,
            save_early=False,
        )
        n_group.run_all()

        print("\n# Running coadd_bundle ...")
        c_group = metricBundles.MetricBundleGroup(
            coadd_bundle,
            opsdb,
            out_dir="%s%s" % (path, out_dir),
            results_db=results_db,
            save_early=False,
        )
        c_group.run_all()

        # ------------------------------------------------------------------------
        # mask the border pixels
        for dither in slicer:
            avg_seeing_bundle[dither].metricValues.mask[border_pixels_masked[dither]] = True
            n_obs_bundle[dither].metricValues.mask[border_pixels_masked[dither]] = True
            coadd_bundle[dither].metricValues.mask[border_pixels_masked[dither]] = True

        # ------------------------------------------------------------------------
        # calculate averageSeeing over the entrie map
        bundle = {}
        bundle["avg_seeing_across_map"] = metricBundles.MetricBundle(
            mean_metric,
            slicers.UniSlicer(),
            sqlconstraint,
            run_name=run_name,
            metadata="avg_seeing_across_map",
        )
        bundle_group = metricBundles.MetricBundleGroup(
            bundle,
            opsdb,
            out_dir="%s%s" % (path, out_dir),
            results_db=results_db,
            save_early=False,
        )
        bundle_group.run_all()
        avg_seeing_across_map = bundle["avg_seeing_across_map"].metricValues.data[0]
        print_out = "\n# Average seeing across map: %s" % (avg_seeing_across_map)
        print(print_out)

        # add to the readme
        readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
        readme.write(print_out)
        readme.close()

        # find the zero point uncertainties: for each pixel i, del_i=k*z_i/sqrt(n_obs_i),
        # where z_i is the average seeing the pixel minus avgSeeing across map, nObs is the number of observations,
        # and k is a constant such that var(del_i)=(0.01)^2.
        # k-constraint equation becomes: k^2*var(z_i/sqrt(n_obs_i))=(0.01)^2    --- equation 1
        k = Symbol("k")
        zero_pt_error = {}
        k_value = {}

        print(
            "\n# 0pt calculation ansatz: \delta_i=k*z_i/sqrt{n_obs_i}, where k is s.t. var(\delta_i)=(0.01)^$"
        )

        if save0pt_plots:
            out_dir_new = "0pt_plots"
            if not os.path.exists("%s%s/%s" % (path, out_dir, out_dir_new)):
                os.makedirs("%s%s/%s" % (path, out_dir, out_dir_new))

        # ------------------------------------------------------------------------
        # add to the readme
        readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
        readme.write("\n\n0pt Information: ")
        readme.close()

        for dither in avg_seeing_bundle:
            z_i = avg_seeing_bundle[dither].metricValues.data[:] - avg_seeing_across_map
            n_obs_i = n_obs_bundle[dither].metricValues.data[:]
            ind = np.where((n_obs_bundle[dither].metricValues.mask == False) & (n_obs_i != 0.0))[
                0
            ]  # make sure the uncertainty is valid; no division by 0
            temp = np.var(z_i[ind] / np.sqrt(n_obs_i[ind]))  # see equation 1
            k_value[dither] = solve(k**2 * temp - 0.01**2, k)[1]

            err = np.empty(len(z_i))
            err.fill(-500)  # initiate
            err[ind] = (k_value[dither] * z_i[ind]) / np.sqrt(n_obs_i[ind])
            zero_pt_error[dither] = err

            # add to the readme
            readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
            readme.write("\nDith strategy: %s" % dither)
            readme.close()

            # ------------------------------------------------------------------------
            if print0pt_information:
                update = "\n# %s" % dither
                ind = np.where(zero_pt_error[dither] != -500)[0]
                good_error = zero_pt_error[dither][ind]
                update += "var(0pt): %s" % np.var(good_error)
                update += "\n0.01^2 - var(0pt) = %s" % ((0.01) ** 2 - np.var(good_error))
                update += "\nk-value: %s\n" % k_value[dither]
                print(update)
                # add to the readme
                readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
                readme.write(update)
                readme.close()
            # ------------------------------------------------------------------------
            if plot0pt_plots:
                # since not saving the bundle for 0pt errors, must plot out stuff without the plotBundle routine.
                ind = np.where(zero_pt_error[dither] != -500)[0]
                good_error = zero_pt_error[dither][ind]

                for i in range(len(good_error)):
                    good_error[i] = float(good_error[i])

                update = "\n# %s" % dither
                update += "\nMin error: %s" % min(good_error)
                update += "\nMax error: %s" % max(good_error)
                update += "\nMean error: %s" % np.mean(good_error)
                update += "\nStd of error: %s\n" % np.std(good_error)
                print(update)

                # add to the readme
                readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
                readme.write(update)
                readme.close()

                # plot histogram
                bin_size = 0.005
                bins = np.arange(
                    min(good_error) - 5 * bin_size,
                    max(good_error) + 5 * bin_size,
                    bin_size,
                )
                plt.clf()
                plt.hist(good_error, bins=bins)
                plt.xlabel("Zeropoint Uncertainty")
                plt.ylabel("Counts")

                plt.title(
                    "0pt error histogram; bin_size = %s; upper_mag_limit = %s" % (bin_size, upper_mag_limit)
                )
                if save0pt_plots:
                    filename = "0ptHistogram_%s_%s.png" % (filter_band, dither)
                    plt.savefig(
                        "%s%s/%s/%s" % (path, out_dir, out_dir_new, filename),
                        format="png",
                    )
                if show0pt_plots:
                    plt.show()
                else:
                    plt.close()

                # plot skymap
                temp = copy.deepcopy(coadd_bundle[dither])
                temp.metricValues.data[ind] = good_error
                temp.metricValues.mask[:] = True
                temp.metricValues.mask[ind] = False

                in_survey_index = np.where(temp.metricValues.mask == False)[0]
                median = np.median(temp.metricValues.data[in_survey_index])
                stddev = np.std(temp.metricValues.data[in_survey_index])

                color_min = -0.010  # median-1.5*stddev
                color_max = 0.010  # median+1.5*stddev
                n_ticks = 5
                increment = (color_max - color_min) / float(n_ticks)
                ticks = np.arange(color_min + increment, color_max, increment)

                plt.clf()
                hp.mollview(
                    temp.metricValues.filled(temp.slicer.badval),
                    flip="astro",
                    rot=(0, 0, 0),
                    min=color_min,
                    max=color_max,
                    title="",
                    cbar=False,
                )
                hp.graticule(dpar=20, dmer=20, verbose=False)
                plt.title(dither)
                ax = plt.gca()
                im = ax.get_images()[0]
                fig = plt.gcf()
                cbaxes = fig.add_axes([0.1, 0.03, 0.8, 0.04])  # [left, bottom, width, height]
                cb = plt.colorbar(im, orientation="horizontal", ticks=ticks, format="%.3f", cax=cbaxes)
                cb.set_label("Photometric Calibration Error")

                if save0pt_plots:
                    filename = "0ptSkymap_%s.png" % (dither)
                    plt.savefig(
                        "%s%s/%s/%s" % (path, out_dir, out_dir_new, filename),
                        bbox_inches="tight",
                        format="png",
                    )

                if show0pt_plots:
                    plt.show()
                else:
                    plt.close()

                # plot power spectrum
                plt.clf()
                spec = hp.anafast(temp.metricValues.filled(temp.slicer.badval), lmax=500)
                ell = np.arange(np.size(spec))
                condition = ell > 1
                plt.plot(ell, (spec * ell * (ell + 1)) / 2.0 / np.pi)
                plt.title("Photometric Calibration Error: %s" % dither)
                plt.xlabel(r"$\ell$")
                plt.ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
                plt.xlim(0, 500)

                if save0pt_plots:
                    # save power spectrum
                    filename = "0ptPowerSpectrum_%s.png" % (dither)
                    plt.savefig(
                        "%s%s/%s/%s" % (path, out_dir, out_dir_new, filename),
                        bbox_inches="tight",
                        format="png",
                    )

                if show0pt_plots:
                    plt.show()
                else:
                    plt.close()

        print(
            "\n## Time since the start of the calculation: %.2f hrs" % ((time.time() - start_time) / 3600.0)
        )

        # ------------------------------------------------------------------------
        # Now recalculate the num_gal with the fluctuations in depth due to calibation uncertainties.
        print("\n# Recalculating num_gal including 0pt errors on the upper mag limit .. ")
        for dither in my_bundles:
            zero_pt_err = zero_pt_error[dither].copy()
            in_survey = np.where(my_bundles[dither].metricValues.mask == False)[
                0
            ]  # 04/27: only look at in_survey region
            for i in in_survey:  # 4/27
                if zero_pt_err[i] != -500:  # run only when zeroPt was calculated
                    my_bundles[dither].metricValues.data[i] = GalaxyCounts_0ptErrors(
                        coadd_bundle[dither].metricValues.data[i],
                        upper_mag_limit + zero_pt_err[i],
                        redshift_bin=redshift_bin,
                        filter_band=filter_band,
                        nside=nside,
                        cfhtls_counts=cfhtls_counts,
                        normalized_mock_catalog_counts=normalized_mock_catalog_counts,
                    )
        # ------------------------------------------------------------------------

        # save the raw num_gal data.
        if save_num_gal_data_after0pt:
            out_dir_new = "numGalData_afterBorderMasking_after0pt"
            for b in my_bundles:
                my_bundles[b].write(out_dir=out_dir_new)

        # ------------------------------------------------------------------------
        # print out tot(num_gal) associated with each strategy
        # add to the read me as well
        update = "\n# After 0pt error calculation and border masking: "
        print(update)
        for dither in my_bundles:
            ind = np.where(my_bundles[dither].metricValues.mask[:] == False)[0]
            print_out = "Total Galaxies for %s: %.9e" % (
                dither,
                sum(my_bundles[dither].metricValues.data[ind]),
            )
            update += "\n %s" % print_out
            print(print_out)
        update += "\n"
        readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
        readme.write(update)
        readme.close()

    print("\n## Time since the start of the calculation: %.2f hrs" % ((time.time() - start_time) / 3600.0))

    #########################################################################################################
    # add poisson noise?
    if add_poisson_noise:
        print("\n# adding poisson noise to num_gal ... ")
        for dither in my_bundles:
            # make sure the values are valid; sometimes metric leaves negative numbers or nan values.
            out_of_survey = np.where(my_bundles[dither].metricValues.mask == True)[0]
            my_bundles[dither].metricValues.data[out_of_survey] = 0.0

            in_survey = np.where(my_bundles[dither].metricValues.mask == False)[0]
            j = np.where(my_bundles[dither].metricValues.data[in_survey] < 1.0)[0]
            my_bundles[dither].metricValues.data[in_survey][j] = 0.0

            noisy_num_gal = np.random.poisson(lam=my_bundles[dither].metricValues.data, size=None)
            my_bundles[dither].metricValues.data[:] = noisy_num_gal
        # ------------------------------------------------------------------------

        # save the num_gal data.
        if saveNumGalDataAfterPoisson:
            out_dir_new = "numGalData_afterBorderMasking_after0pt_afterPoisson"
            for b in my_bundles:
                my_bundles[b].write(out_dir=out_dir_new)

        # ------------------------------------------------------------------------
        # print out tot(num_gal) associated with each strategy
        # add to the read me as well
        update = "\n# After adding poisson noise: "
        print(update)
        for dither in my_bundles:
            ind = np.where(my_bundles[dither].metricValues.mask[:] == False)[0]
            print_out = "Total Galaxies for %s: %.9e" % (
                dither,
                sum(my_bundles[dither].metricValues.data[ind]),
            )
            update += "\n %s" % print_out
            print(print_out)
        update += "\n"
        readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
        readme.write(update)
        readme.close()

    print("\n## Time since the start of the calculation: %.2f hrs" % ((time.time() - start_time) / 3600.0))
    #########################################################################################################
    plot_handler = plots.PlotHandler(
        out_dir="%s%s" % (path, out_dir),
        results_db=results_db,
        thumbnail=False,
        savefig=False,
    )
    print("\n# Calculating fluctuations in the galaxy counts ...")
    # Change num_gal metric data to deltaN/N
    num_gal = {}
    # add to readme too
    update = "\n"
    for dither in my_bundles:
        # zero out small/nan entries --- problem: should really be zeroed out by the metric ***
        j = np.where(np.isnan(my_bundles[dither].metricValues.data) == True)[0]
        my_bundles[dither].metricValues.data[j] = 0.0
        j = np.where(my_bundles[dither].metricValues.data < 1.0)[0]
        my_bundles[dither].metricValues.data[j] = 0.0
        # calculate the fluctuations
        num_gal[dither] = my_bundles[
            dither
        ].metricValues.data.copy()  # keep track of num_gal for plotting purposes
        valid_pixel = np.where(my_bundles[dither].metricValues.mask == False)[0]
        galaxy_average = sum(num_gal[dither][valid_pixel]) / len(valid_pixel)

        # in place calculation of the fluctuations
        my_bundles[dither].metricValues.data[:] = 0.0
        my_bundles[dither].metricValues.data[valid_pixel] = (
            num_gal[dither][valid_pixel] - galaxy_average
        ) / galaxy_average
        print_out = "# Galaxy Average for %s: %s" % (dither, galaxy_average)
        print(print_out)
        update += "%s\n" % print_out

    readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
    readme.write(update)
    readme.close()

    # ------------------------------------------------------------------------
    # save the deltaN/N data
    if save_delta_n_by_n_data:
        out_dir_new = "deltaNByNData"
        for b in my_bundles:
            my_bundles[b].write(out_dir=out_dir_new)

    # ------------------------------------------------------------------------
    # Calculate total power
    # add to the read me as well
    summarymetric = metrics.TotalPowerMetric()
    update = ""
    for dither in my_bundles:
        my_bundles[dither].set_summary_metrics(summarymetric)
        my_bundles[dither].compute_summary_stats()
        print_out = "# Total power for %s case is %f." % (
            dither,
            my_bundles[dither].summary_values["TotalPower"],
        )
        print(print_out)
        update += "\n%s" % (print_out)
    update += "\n"

    readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
    readme.write(update)
    readme.close()
    # ------------------------------------------------------------------------
    # calculate the power spectra
    cl = {}
    for dither in my_bundles:
        cl[dither] = hp.anafast(
            my_bundles[dither].metricValues.filled(my_bundles[dither].slicer.badval),
            lmax=500,
        )
    # save deltaN/N spectra?
    if save_cls_for_delta_n_by_n:
        out_dir_new = "cls_DeltaByN"
        if not os.path.exists("%s%s/%s" % (path, out_dir, out_dir_new)):
            os.makedirs("%s%s/%s" % (path, out_dir, out_dir_new))

        for dither in my_bundles:
            filename = "cls_deltaNByN_%s_%s" % (filter_band, dither)
            np.save("%s%s/%s/%s" % (path, out_dir, out_dir_new, filename), cl[dither])

    ##########################################################################################################
    # Plots for the fluctuations: power spectra, histogram
    if len(list(my_bundles.keys())) > 1:
        out_dir_new = "artificialFluctuationsComparisonPlots"
        if not os.path.exists("%s%s/%s" % (path, out_dir, out_dir_new)):
            os.makedirs("%s%s/%s" % (path, out_dir, out_dir_new))
        # ------------------------------------------------------------------------
        # power spectra
        for dither in my_bundles:
            ell = np.arange(np.size(cl[dither]))
            condition = ell > 1
            plt.plot(
                ell,
                (cl[dither] * ell * (ell + 1)) / 2.0 / np.pi,
                color=plot_color[dither],
                linestyle="-",
                label=dither,
            )
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
        plt.xlim(0, 500)
        leg = plt.legend(labelspacing=0.001)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.savefig(
            "%s%s/%s/powerspectrum_comparison.png" % (path, out_dir, out_dir_new),
            format="png",
        )
        if show_comp_plots:
            plt.show()
        else:
            plt.close("all")
        # ------------------------------------------------------------------------
        # create the histogram
        scale = hp.nside2pixarea(nside, degrees=True)

        def tick_formatter(y, pos):
            return "%d" % (y * scale)  # convert pixel count to area

        for dither in my_bundles:
            ind = np.where(my_bundles[dither].metricValues.mask == False)[0]
            bin_size = 0.01
            bin_all = int(
                (
                    max(my_bundles[dither].metricValues.data[ind])
                    - min(my_bundles[dither].metricValues.data[ind])
                )
                / bin_size
            )
            plt.hist(
                my_bundles[dither].metricValues.data[ind],
                bins=bin_all,
                label=dither,
                histtype="step",
                color=plot_color[dither],
            )
        # plt.xlim(-0.6,1.2)
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        n_yticks = 10.0
        wanted_y_max = ymax * scale
        wanted_y_max = 10.0 * np.ceil(float(wanted_y_max) / 10.0)
        increment = 5.0 * np.ceil(float(wanted_y_max / n_yticks) / 5.0)
        wanted_array = np.arange(0, wanted_y_max, increment)
        ax.yaxis.set_ticks(wanted_array / scale)
        ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))
        plt.xlabel(r"$\mathrm{\Delta N/\overline{N}}$")
        plt.ylabel("Area (deg$^2$)")
        leg = plt.legend(labelspacing=0.001, bbox_to_anchor=(1, 1))
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        plt.savefig(
            "%s%s/%s/histogram_comparison.png" % (path, out_dir, out_dir_new),
            bbox_inches="tight",
            format="png",
        )
        if show_comp_plots:
            plt.show()
        else:
            plt.close("all")

    # now remove the results db object -- useless
    os.remove("%s%s/%s" % (path, out_dir, results_dbname))
    print("Removed %s from out_dir" % (results_dbname))

    # all done. final update.
    update = "\n## All done. Time since the start of the calculation: %.2f hrs" % (
        (time.time() - start_time) / 3600.0
    )
    print(update)
    readme = open("%s%s/%s" % (path, out_dir, readme_name), "a")
    readme.write(update)
    readme.close()

    if return_stuff:
        if include0pt_errors:
            return my_bundles, out_dir, results_db, zero_pt_error
        else:
            return my_bundles, out_dir, results_db
