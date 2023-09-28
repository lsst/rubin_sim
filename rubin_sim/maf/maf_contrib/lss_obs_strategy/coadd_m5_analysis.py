#####################################################################################################
# Purpose: calculate the coadded 5-sigma depth from various survey strategies. Incudes functionality
# to consider various survey strategies, mask shallow borders, create/save/show relevant plots, do
# an alm analysis, and save data.

__all__ = ("coadd_m5_analysis",)

import copy
import os

import healpy as hp
import matplotlib.pyplot as plt

# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

import rubin_sim.maf.db as db
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as metricBundles
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.stackers as mafStackers  # stackers in sims_maf
from rubin_sim.maf.maf_contrib.lss_obs_strategy.alm_plots import alm_plots
from rubin_sim.maf.maf_contrib.lss_obs_strategy.constants_for_pipeline import plot_color
from rubin_sim.maf.maf_contrib.lss_obs_strategy.masking_algorithm_generalized import (
    masking_algorithm_generalized,
)


def coadd_m5_analysis(
    path,
    dbfile,
    run_name,
    slair=False,
    wf_dand_dd_fs=False,
    no_dith_only=False,
    best_dith_only=False,
    some_dith_only=False,
    specified_dith=None,
    nside=128,
    filter_band="r",
    include_dust_extinction=False,
    saveun_masked_coadd_data=False,
    pixel_radius_for_masking=5,
    cut_off_year=None,
    plot_skymap=True,
    plot_cartview=True,
    unmasked_color_min=None,
    unmasked_color_max=None,
    masked_color_min=None,
    masked_color_max=None,
    n_ticks=5,
    plot_power_spectrum=True,
    show_plots=True,
    save_figs=True,
    alm_analysis=True,
    ra_range=[-50, 50],
    dec_range=[-65, 5],
    save_masked_coadd_data=True,
):
    """

    Analyze the artifacts induced in the coadded 5sigma depth due to imperfect observing strategy.
      - Creates an output directory for subdirectories containing the specified things to save.
      - Creates, shows, and saves comparison plots.
      - Returns the metricBundle object containing the calculated coadded depth, and the output directory name.

    Parameters
    ----------
    path : str
        path to the main directory where output directory is to be saved.
    dbfile : str
        path to the OpSim output file, e.g. to a copy of enigma_1189
    run_name : str
        run name tag to identify the output of specified OpSim output, e.g. 'enigma1189'
    slair : `bool`
        set to True if analysis on a SLAIR output.
        Default: False
    wf_dand_dd_fs : `bool`
        set to True if want to consider both WFD survet and DDFs. Otherwise will only work
        with WFD. Default: False
    no_dith_only : `bool`
        set to True if only want to consider the undithered survey. Default: False
    best_dith_only : `bool`
        set to True if only want to consider RandomDitherFieldPerVisit.
        Default: False
    some_dith_only : `bool`
        set to True if only want to consider undithered and a few dithered surveys.
        Default: False
    specified_dith : str
        specific dither strategy to run.
        Default: None
    nside : int
        HEALpix resolution parameter. Default: 128
    filter_band : str
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'r'
    include_dust_extinction : `bool`
        set to include dust extinction. Default: False
    saveun_masked_coadd_data : `bool`
        set to True to save data before border masking. Default: False
    pixel_radius_for_masking : int
        number of pixels to mask along the shallow border. Default: 5
    cut_off_year : int
        year cut to restrict analysis to only a subset of the survey.
        Must range from 1 to 9, or None for the full survey analysis (10 yrs).
        Default: None
    plot_skymap : `bool`
        set to True if want to plot skymaps. Default: True
    plot_cartview : `bool`
        set to True if want to plot cartview plots. Default: False
    unmasked_color_min : float
        lower limit on the colorscale for unmasked skymaps. Default: None
    unmasked_color_max : float
        upper limit on the colorscale for unmasked skymaps. Default: None
    masked_color_min : float
        lower limit on the colorscale for border-masked skymaps. Default: None
    masked_color_max : float
        upper limit on the colorscale for border-masked skymaps. Default: None
    n_ticks : int
        (number of ticks - 1) on the skymap colorbar. Default: 5
    plot_power_spectrum : `bool`
        set to True if want to plot powerspectra. Default: True
    show_plots : `bool`
        set to True if want to show figures. Default: True
    save_figs : `bool`
        set to True if want to save figures. Default: True
    alm_analysis : `bool`
        set to True to perform the alm analysis. Default: True
    ra_range : float array
        range of right ascention (in degrees) to consider in alm  cartview plot;
        applicable when alm_analysis=True. Default: [-50,50]
    dec_range : float array
        range of declination (in degrees) to consider in alm cartview plot;
        applicable when alm_analysis=True. Default: [-65,5]
    save_masked_coadd_data : `bool`
        set to True to save the coadded depth data after the border
        masking. Default: True
    """
    # ------------------------------------------------------------------------
    # read in the database
    if slair:
        # slair database
        opsdb = db.Database(dbfile, defaultTable="observations")
    else:
        # OpSim database
        opsdb = db.OpsimDatabase(dbfile)

    # ------------------------------------------------------------------------
    # set up the out_dir
    zeropt_tag = ""
    if cut_off_year is not None:
        zeropt_tag = "%syearCut" % cut_off_year
    else:
        zeropt_tag = "fullSurveyPeriod"

    if include_dust_extinction:
        dust_tag = "withDustExtinction"
    else:
        dust_tag = "noDustExtinction"

    region_type = ""
    if wf_dand_dd_fs:
        region_type = "WFDandDDFs_"

    out_dir = "coaddM5Analysis_%snside%s_%s_%spixelRadiusForMasking_%sBand_%s_%s_directory" % (
        region_type,
        nside,
        dust_tag,
        pixel_radius_for_masking,
        filter_band,
        run_name,
        zeropt_tag,
    )
    print("# out_dir: %s" % out_dir)
    results_db = db.ResultsDb(out_dir=out_dir)

    # ------------------------------------------------------------------------
    # set up the sql constraint
    if wf_dand_dd_fs:
        if cut_off_year is not None:
            night_cut_off = (cut_off_year) * 365.25
            sqlconstraint = 'night<=%s and filter=="%s"' % (night_cut_off, filter_band)
        else:
            sqlconstraint = 'filter=="%s"' % filter_band
    else:
        # set up the propID and units on the ra, dec
        if slair:  # no prop ID; only WFD is simulated.
            wfd_where = ""
            ra_dec_in_deg = True
        else:
            prop_ids, prop_tags = opsdb.fetchPropInfo()
            wfd_where = "%s and " % opsdb.createSQLWhere("WFD", prop_tags)
            ra_dec_in_deg = opsdb.raDecInDeg
        # set up the year cutoff
        if cut_off_year is not None:
            night_cut_off = (cut_off_year) * 365.25
            sqlconstraint = '%snight<=%s and filter=="%s"' % (
                wfd_where,
                night_cut_off,
                filter_band,
            )
        else:
            sqlconstraint = '%sfilter=="%s"' % (wfd_where, filter_band)
    print("# sqlconstraint: %s" % sqlconstraint)

    # ------------------------------------------------------------------------
    # setup all the slicers
    slicer = {}
    stacker_list = {}

    if (
        specified_dith is not None
    ):  # would like to add all the stackers first and then keep only the one that is specified
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
        if slair:
            slicer["NoDither"] = slicers.HealpixSlicer(
                lon_col="RA",
                lat_col="dec",
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
        if some_dith_only and not no_dith_only:
            # stacker_list['RepulsiveRandomDitherFieldPerVisit'] = [myStackers.RepulsiveRandomDitherFieldPerVisitStacker(degrees=ra_dec_in_deg,
            #                                                                                                          random_seed=1000)]
            # slicer['RepulsiveRandomDitherFieldPerVisit'] = slicers.HealpixSlicer(lonCol='repulsiveRandomDitherFieldPerVisitRa',
            #                                                                    latCol='repulsiveRandomDitherFieldPerVisitDec',
            #                                                                    latLonDeg=ra_dec_in_deg, nside=nside,
            #                                                                    use_cache=False)
            slicer["SequentialHexDitherFieldPerNight"] = slicers.HealpixSlicer(
                lon_col="hexDitherFieldPerNightRa",
                lat_col="hexDitherFieldPerNightDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
            slicer["PentagonDitherPerSeason"] = slicers.HealpixSlicer(
                lon_col="pentagonDitherPerSeasonRa",
                lat_col="pentagonDitherPerSeasonDec",
                lat_lon_deg=ra_dec_in_deg,
                nside=nside,
                use_cache=False,
            )
        elif not no_dith_only:
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
    if specified_dith is not None:
        stacker_list_, slicer_ = {}, {}
        if specified_dith in slicer.keys():
            if specified_dith.__contains__(
                "Random"
            ):  # only Random dithers have a stacker object for rand seed specification
                stacker_list_[specified_dith] = stacker_list[specified_dith]
            slicer_[specified_dith] = slicer[specified_dith]
        else:
            raise ValueError(
                "Invalid value for specified_dith: %s. Allowed values include one of the following:\n%s"
                % (specified_dith, slicer.keys())
            )
        stacker_list, slicer = stacker_list_, slicer_

    # ------------------------------------------------------------------------
    if slair:
        m5_col = "fivesigmadepth"
    else:
        m5_col = "fiveSigmaDepth"
    # set up the metric
    if include_dust_extinction:
        # include dust extinction when calculating the co-added depth
        coadd_metric = metrics.ExgalM5(m5_col=m5_col, lsstFilter=filter_band)
    else:
        coadd_metric = metrics.Coaddm5Metric(m5col=m5col)
    dust_map = maps.DustMap(
        interp=False, nside=nside
    )  # include dust_map; actual in/exclusion of dust is handled by the galaxyCountMetric

    # ------------------------------------------------------------------------
    # set up the bundle
    coadd_bundle = {}
    for dither in slicer:
        if dither in stacker_list:
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
            coadd_bundle[dither] = metricBundles.MetricBundle(
                coadd_metric,
                slicer[dither],
                sqlconstraint,
                run_name=run_name,
                metadata=dither,
                maps_list=[dust_map],
            )

    # ------------------------------------------------------------------------
    # run the analysis
    if include_dust_extinction:
        print("\n# Running coadd_bundle with dust extinction ...")
    else:
        print("\n# Running coadd_bundle without dust extinction ...")
    c_group = metricBundles.MetricBundleGroup(
        coadd_bundle, opsdb, out_dir=out_dir, results_db=results_db, save_early=False
    )
    c_group.run_all()

    # ------------------------------------------------------------------------
    plot_handler = plots.PlotHandler(out_dir=out_dir, results_db=results_db, thumbnail=False, savefig=False)

    print("# Number of pixels in the survey region (before masking the border):")
    for dither in coadd_bundle:
        print(
            "  %s: %s"
            % (
                dither,
                len(np.where(coadd_bundle[dither].metricValues.mask == False)[0]),
            )
        )

    # ------------------------------------------------------------------------
    # save the unmasked data?
    if saveun_masked_coadd_data:
        out_dir_new = "unmaskedCoaddData"
        for b in coadd_bundle:
            coadd_bundle[b].write(out_dir=out_dir_new)

    # ------------------------------------------------------------------------
    # mask the edges
    print("\n# Masking the edges for coadd ...")
    coadd_bundle = masking_algorithm_generalized(
        coadd_bundle,
        plot_handler,
        data_label="$%s$-band Coadded Depth" % filter_band,
        nside=nside,
        pixel_radius=pixel_radius_for_masking,
        plot_intermediate_plots=False,
        plot_final_plots=False,
        print_final_info=True,
    )
    # ------------------------------------------------------------------------
    # Calculate total power
    summarymetric = metrics.TotalPowerMetric()
    for dither in coadd_bundle:
        coadd_bundle[dither].set_summary_metrics(summarymetric)
        coadd_bundle[dither].compute_summary_stats()
        print(
            "# Total power for %s case is %f." % (dither, coadd_bundle[dither].summary_values["TotalPower"])
        )
    print("")

    # ------------------------------------------------------------------------
    # run the alm analysis
    if alm_analysis:
        alm_plots(
            path,
            out_dir,
            copy.deepcopy(coadd_bundle),
            nside=nside,
            filterband=filter_band,
            ra_range=ra_range,
            dec_range=dec_range,
            show_plots=show_plots,
        )
    # ------------------------------------------------------------------------
    # save the masked data?
    if save_masked_coadd_data and (pixel_radius_for_masking > 0):
        out_dir_new = "maskedCoaddData"
        for b in coadd_bundle:
            coadd_bundle[b].write(out_dir=out_dir_new)

    # ------------------------------------------------------------------------
    # plot comparison plots
    if len(coadd_bundle.keys()) > 1:  # more than one key
        # set up the directory
        out_dir_comp = "coaddM5ComparisonPlots"
        if not os.path.exists("%s%s/%s" % (path, out_dir, out_dir_comp)):
            os.makedirs("%s%s/%s" % (path, out_dir, out_dir_comp))
        # ------------------------------------------------------------------------
        # plot for the power spectra
        cl = {}
        for dither in plot_color:
            if dither in coadd_bundle:
                cl[dither] = hp.anafast(
                    hp.remove_dipole(
                        coadd_bundle[dither].metricValues.filled(coadd_bundle[dither].slicer.badval)
                    ),
                    lmax=500,
                )
                ell = np.arange(np.size(cl[dither]))
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
        fig = plt.gcf()
        fig.set_size_inches(12.5, 10.5)
        leg = plt.legend(labelspacing=0.001)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        filename = "powerspectrum_comparison_all.png"
        plt.savefig(
            "%s%s/%s/%s" % (path, out_dir, out_dir_comp, filename),
            bbox_inches="tight",
            format="png",
        )
        plt.show()

        # create the histogram
        scale = hp.nside2pixarea(nside, degrees=True)

        def tick_formatter(y, pos):
            return "%d" % (y * scale)  # convert pixel count to area

        bin_size = 0.01
        for dither in plot_color:
            if dither in coadd_bundle:
                ind = np.where(coadd_bundle[dither].metricValues.mask == False)[0]
                bin_all = int(
                    (
                        max(coadd_bundle[dither].metricValues.data[ind])
                        - min(coadd_bundle[dither].metricValues.data[ind])
                    )
                    / bin_size
                )
                plt.hist(
                    coadd_bundle[dither].metricValues.data[ind],
                    bins=bin_all,
                    label=dither,
                    histtype="step",
                    color=plot_color[dither],
                )
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        n_yticks = 10.0
        wanted_y_max = ymax * scale
        wanted_y_max = 10.0 * np.ceil(float(wanted_y_max) / 10.0)
        increment = 5.0 * np.ceil(float(wanted_y_max / n_yticks) / 5.0)
        wanted_array = np.arange(0, wanted_y_max, increment)
        ax.yaxis.set_ticks(wanted_array / scale)
        ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter))
        plt.xlabel("$%s$-band Coadded Depth" % filter_band)
        plt.ylabel("Area (deg$^2$)")
        fig = plt.gcf()
        fig.set_size_inches(12.5, 10.5)
        leg = plt.legend(labelspacing=0.001, loc=2)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(2.0)
        filename = "histogram_comparison.png"
        plt.savefig(
            "%s%s/%s/%s" % (path, out_dir, out_dir_comp, filename),
            bbox_inches="tight",
            format="png",
        )
        plt.show()
        # ------------------------------------------------------------------------
        # plot power spectra for the separte panel
        tot_keys = len(list(coadd_bundle.keys()))
        if tot_keys > 1:
            plt.clf()
            n_cols = 2
            n_rows = int(np.ceil(float(tot_keys) / n_cols))
            fig, ax = plt.subplots(n_rows, n_cols)
            plot_row = 0
            plot_col = 0
            for dither in list(plot_color.keys()):
                if dither in list(coadd_bundle.keys()):
                    ell = np.arange(np.size(cl[dither]))
                    ax[plot_row, plot_col].plot(
                        ell,
                        (cl[dither] * ell * (ell + 1)) / 2.0 / np.pi,
                        color=plot_color[dither],
                        label=dither,
                    )
                    if plot_row == n_rows - 1:
                        ax[plot_row, plot_col].set_xlabel(r"$\ell$")
                    ax[plot_row, plot_col].set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
                    ax[plot_row, plot_col].yaxis.set_major_locator(MaxNLocator(3))
                    if dither != "NoDither":
                        ax[plot_row, plot_col].set_ylim(0, 0.0035)
                    ax[plot_row, plot_col].set_xlim(0, 500)
                    plot_row += 1
                    if plot_row > n_rows - 1:
                        plot_row = 0
                        plot_col += 1
            fig.set_size_inches(20, int(n_rows * 30 / 7.0))
            filename = "powerspectrum_sepPanels.png"
            plt.savefig(
                "%s%s/%s/%s" % (path, out_dir, out_dir_comp, filename),
                bbox_inches="tight",
                format="png",
            )
            plt.show()
    return coadd_bundle, out_dir
