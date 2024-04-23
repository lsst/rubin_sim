__all__ = ("radar_limited",)

import healpy as hp
import numpy as np

import rubin_sim.maf as maf
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers

from .common import filter_list, lightcurve_summary, standard_summary


def radar_limited(
    runName="run name",
    nside=64,
    benchmarkArea=18000,
    benchmarkNvisits=825,
    minNvisits=750,
    long_microlensing=True,
    srd_only=False,
    mjd0=None,
):
    """A batch of metrics for looking at survey performance
    relative to the SRD and the main science drivers of LSST.

    Parameters
    ----------
    runName : `str`, optional
        The simulation run name that should appear as plot titles.
    benchmarkArea : `float`, optional
        The area to use for SRD metrics (sq degrees)
    benchmarkNvisits : `int`, optional
        The number of visits to use for SRD metrics.
    minNvisits : `int`, optional
        The minimum number of visits to use for SRD metrics.
    long_microlensing : `bool`, optional
        Add the longer running microlensing metrics to the batch
        (a subset of crossing times only)
    srd_only : `bool`, optional
        Only return the SRD metrics
    mjd0 : float, optional
        The modified Julian date start date of the survey.

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    bundleList = []
    # Get some standard per-filter coloring and sql constraints
    filterlist, colors, filterorders, filtersqls, filterinfo_label = filter_list(all=False)
    (
        allfilterlist,
        allcolors,
        allfilterorders,
        allfiltersqls,
        allfilterinfo_label,
    ) = filter_list(all=True)

    standardStats = standard_summary(with_count=False)

    # This is the default slicer for most purposes in this batch.
    # Note that the cache is on -
    # if the metric requires a dust map, this is not the right slicer to use.
    healpixslicer = slicers.HealpixSlicer(nside=nside, use_cache=True)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    #########################
    #########################
    # SRD, DM, etc
    #########################
    #########################
    # fO metric
    displayDict = {"group": "SRD", "subgroup": "FO metrics", "order": 0}

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountExplimMetric(col="observationStartMJD", metric_name="fO")
    plotDict = {
        "xlabel": "Number of Visits",
        "asky": benchmarkArea,
        "n_visits": benchmarkNvisits,
        "x_min": 0,
        "x_max": 1500,
    }
    summaryMetrics = [
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name="fOArea",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FOArea(
            nside=nside,
            norm=True,
            metric_name="fOArea/benchmark",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FONv(
            nside=nside,
            norm=False,
            metric_name="fONv",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FONv(
            nside=nside,
            norm=True,
            metric_name="fONv/benchmark",
            asky=benchmarkArea,
            n_visit=benchmarkNvisits,
        ),
        metrics.FOArea(
            nside=nside,
            norm=False,
            metric_name=f"fOArea_{minNvisits}",
            asky=benchmarkArea,
            n_visit=minNvisits,
        ),
    ]
    caption = "The FO metric evaluates the overall efficiency of observing. "
    caption += (
        "foNv: out of %.2f sq degrees, the area receives at least X and a median of Y visits "
        "(out of %d, if compared to benchmark). " % (benchmarkArea, benchmarkNvisits)
    )
    caption += (
        "fOArea: this many sq deg (out of %.2f sq deg if compared "
        "to benchmark) receives at least %d visits. " % (benchmarkArea, benchmarkNvisits)
    )
    displayDict["caption"] = caption
    bundle = mb.MetricBundle(
        metric,
        healpixslicer,
        "",
        plot_dict=plotDict,
        display_dict=displayDict,
        summary_metrics=summaryMetrics,
        plot_funcs=[plots.FOPlot()],
    )
    bundleList.append(bundle)

    # Single visit depth distribution
    displayDict["subgroup"] = "Visit Depths"
    # Histogram values over all and per filter.

    value = "fiveSigmaDepth"
    for f in filterlist:
        displayDict["caption"] = "Histogram of %s" % (value)
        displayDict["caption"] += " for %s." % (filterinfo_label[f])
        displayDict["order"] = filterorders[f]
        m = metrics.CountMetric(value, metric_name="%s Histogram" % (value))
        slicer = slicers.OneDSlicer(slice_col_name=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            filtersqls[f],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    displayDict["caption"] = ""
    for f in filterlist:
        slicer = maf.UniSlicer()
        metric = maf.MedianMetric(col="fiveSigmaDepth")
        bundle = mb.MetricBundle(
            metric,
            slicer,
            filtersqls[f],
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    ##############
    # Astrometry
    ###############

    rmags_para = [22.4, 24.0]
    rmags_pm = [20.5, 24.0]

    # Set up parallax/dcr stackers.
    parallaxStacker = maf.ParallaxFactorStacker()
    dcrStacker = maf.DcrStacker()

    # Set up parallax metrics.
    displayDict["subgroup"] = "Parallax"
    displayDict["order"] += 1
    # Expected error on parallax at 10 AU.
    plotmaxVals = (2.0, 15.0)
    good_parallax_limit = 11.5
    summary = [
        metrics.AreaSummaryMetric(
            area=18000,
            reduce_func=np.median,
            decreasing=False,
            metric_name="Median Parallax Uncert (18k)",
        ),
        metrics.AreaThresholdMetric(
            upper_threshold=good_parallax_limit,
            metric_name="Area better than %.1f mas uncertainty" % good_parallax_limit,
        ),
    ]
    summary.append(metrics.PercentileMetric(percentile=95, metric_name="95th Percentile Parallax Uncert"))
    summary.extend(standard_summary())
    for rmag, plotmax in zip(rmags_para, plotmaxVals):
        plotDict = {"x_min": 0, "x_max": plotmax, "color_min": 0, "color_max": plotmax}
        metric = metrics.ParallaxMetric(
            metric_name="Parallax Uncert @ %.1f" % (rmag),
            rmag=rmag,
            normalize=False,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stacker_list=[parallaxStacker],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=summary,
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1

    # Parallax normalized to 'best possible'.
    # This separates the effect of cadence from depth.
    for rmag in rmags_para:
        metric = metrics.ParallaxMetric(
            metric_name="Normalized Parallax Uncert @ %.1f" % (rmag),
            rmag=rmag,
            normalize=True,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stacker_list=[parallaxStacker],
            display_dict=displayDict,
            summary_metrics=standard_summary(),
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Parallax factor coverage.
    for rmag in rmags_para:
        metric = metrics.ParallaxCoverageMetric(metric_name="Parallax Coverage @ %.1f" % (rmag), rmag=rmag)
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stacker_list=[parallaxStacker],
            display_dict=displayDict,
            summary_metrics=standard_summary(),
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Parallax problems can be caused by HA and DCR degeneracies.
    # Check their correlation.
    for rmag in rmags_para:
        metric = metrics.ParallaxDcrDegenMetric(
            metric_name="Parallax-DCR degeneracy @ %.1f" % (rmag), rmag=rmag
        )
        caption = "Correlation between parallax offset magnitude and hour angle for a r=%.1f star." % (rmag)
        caption += " (0 is good, near -1 or 1 is bad)."
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stacker_list=[dcrStacker, parallaxStacker],
            display_dict=displayDict,
            summary_metrics=standard_summary(),
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1

    # Proper Motion metrics.
    displayDict["subgroup"] = "Proper Motion"
    displayDict["order"] = 0
    # Proper motion errors.
    plotmaxVals = (1.0, 5.0)
    summary = [
        metrics.AreaSummaryMetric(
            area=18000,
            reduce_func=np.median,
            decreasing=False,
            metric_name="Median Proper Motion Uncert (18k)",
        )
    ]
    summary.append(metrics.PercentileMetric(metric_name="95th Percentile Proper Motion Uncert"))
    summary.extend(standard_summary())
    for rmag, plotmax in zip(rmags_pm, plotmaxVals):
        plotDict = {"x_min": 0, "x_max": plotmax, "color_min": 0, "color_max": plotmax}
        metric = metrics.ProperMotionMetric(
            metric_name="Proper Motion Uncert @ %.1f" % rmag,
            rmag=rmag,
            normalize=False,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=summary,
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Normalized proper motion.
    for rmag in rmags_pm:
        metric = metrics.ProperMotionMetric(
            metric_name="Normalized Proper Motion Uncert @ %.1f" % rmag,
            rmag=rmag,
            normalize=True,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            display_dict=displayDict,
            summary_metrics=standard_summary(),
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1

    # DCR precision metric
    displayDict["subgroup"] = "DCR"
    displayDict["order"] = 0
    plotDict = {"caption": "Precision of DCR slope.", "percentile_clip": 95}
    metric = metrics.DcrPrecisionMetric()
    bundle = mb.MetricBundle(
        metric,
        healpixslicer,
        "",
        plot_dict=plotDict,
        plot_funcs=subsetPlots,
        display_dict=displayDict,
        summary_metrics=standard_summary(with_count=False),
    )
    bundleList.append(bundle)

    # Rapid Revisit
    displayDict["subgroup"] = "Rapid Revisits"
    displayDict["order"] = 0
    # Calculate the actual number of revisits within 30 minutes.
    dTmax = 30  # time in minutes
    m2 = metrics.NRevisitsMetric(d_t=dTmax, normed=False, metric_name="NumberOfQuickRevisits")
    plotDict = {"color_min": 400, "color_max": 2000, "x_min": 400, "x_max": 2000}
    caption = "Number of consecutive visits with return times faster than %.1f minutes, " % (dTmax)
    caption += "in any filter. "
    displayDict["caption"] = caption
    bundle = mb.MetricBundle(
        m2,
        healpixslicer,
        "",
        plot_dict=plotDict,
        plot_funcs=subsetPlots,
        display_dict=displayDict,
        summary_metrics=standard_summary(with_count=False),
    )
    bundleList.append(bundle)
    displayDict["order"] += 1

    # Better version of the rapid revisit requirements:
    # require a minimum number of visits between
    # dtMin and dtMax, but also a minimum number of visits
    # between dtMin and dtPair (the typical pair time).
    # 1 means the healpix met the requirements (0 means did not).
    dTmin = 40.0 / 60.0  # (minutes) 40s minimum for rapid revisit range
    dTpairs = 20.0  # minutes (time when pairs should start kicking in)
    dTmax = 30.0  # 30 minute maximum for rapid revisit range
    nOne = 82  # Number of revisits between 40s-30m required
    nTwo = 28  # Number of revisits between 40s - tPairs required.
    pix_area = float(hp.nside2pixarea(nside, degrees=True))
    scale = pix_area * hp.nside2npix(nside)
    m1 = metrics.RapidRevisitMetric(
        metric_name="RapidRevisits",
        d_tmin=dTmin / 60.0 / 60.0 / 24.0,
        d_tpairs=dTpairs / 60.0 / 24.0,
        d_tmax=dTmax / 60.0 / 24.0,
        min_n1=nOne,
        min_n2=nTwo,
    )
    plotDict = {
        "x_min": 0,
        "x_max": 1,
        "color_min": 0,
        "color_max": 1,
        "log_scale": False,
    }
    cutoff1 = 0.9
    summaryStats = [metrics.FracAboveMetric(cutoff=cutoff1, scale=scale, metric_name="Area (sq deg)")]
    caption = "Rapid Revisit: area that receives at least %d visits between %.3f and %.1f minutes, " % (
        nOne,
        dTmin,
        dTmax,
    )
    caption += "with at least %d of those visits falling between %.3f and %.1f minutes. " % (
        nTwo,
        dTmin,
        dTpairs,
    )
    caption += (
        'Summary statistic "Area" indicates the area on the sky which meets this requirement.'
        " (SRD design specification is 2000 sq deg)."
    )
    displayDict["caption"] = caption
    displayDict["order"] = 0
    bundle = mb.MetricBundle(
        m1,
        healpixslicer,
        "",
        plot_dict=plotDict,
        plot_funcs=subsetPlots,
        display_dict=displayDict,
        summary_metrics=summaryStats,
    )
    bundleList.append(bundle)

    # For SRD batch only, return here.
    if srd_only:
        for b in bundleList:
            b.set_run_name(runName)
        bundleDict = mb.make_bundles_dict_from_list(bundleList)

        return bundleDict

    # Year Coverage
    displayDict["subgroup"] = "Year Coverage"
    metric = metrics.YearCoverageMetric()
    for f in filterlist:
        plotDict = {"color_min": 7, "color_max": 10, "color": colors[f]}
        displayDict["caption"] = f"Number of years of coverage in {f} band."
        displayDict["order"] = filterorders[f]
        summary = [
            metrics.AreaSummaryMetric(
                area=18000,
                reduce_func=np.mean,
                decreasing=True,
                metric_name="N Years (18k) %s" % f,
            )
        ]
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            filtersqls[f],
            plot_dict=plotDict,
            info_label=filterinfo_label[f],
            display_dict=displayDict,
            summary_metrics=summary,
        )
        bundleList.append(bundle)

    #########################
    #########################
    # Galaxies
    #########################
    #########################

    # Run this per filter, to look at variations in
    # counts of galaxies in blue bands?
    displayDict = {
        "group": "Galaxies",
        "subgroup": "Galaxy Counts",
        "order": 0,
        "caption": None,
    }
    # make sure slicer has cache off
    slicer = slicers.HealpixSlicer(nside=nside, use_cache=False)

    displayDict["subgroup"] = "Surface Brightness"
    summary = [metrics.MedianMetric()]
    for filtername in "ugrizy":
        displayDict["caption"] = "Surface brightness limit in %s, no extinction applied." % filtername
        displayDict["order"] = filterorders[f]
        sql = 'filter="%s"' % filtername
        metric = metrics.SurfaceBrightLimitMetric()
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            display_dict=displayDict,
            summary_metrics=summary,
            plot_funcs=subsetPlots,
        )
        bundleList.append(bundle)

    #########################
    #########################
    # Cosmology
    #########################
    #########################

    bandpass = "i"
    nfilters_needed = 6
    lim_ebv = 0.2
    offset = 0.1
    mag_cuts = {
        10: 26.0 - offset,
    }
    yrs = list(mag_cuts.keys())
    maxYr = max(yrs)

    displayDict = {"group": "Cosmology"}
    subgroupCount = 1

    displayDict["subgroup"] = f"{subgroupCount}: Static Science"
    ## Static Science
    # Calculate the static science metrics - effective survey area,
    # mean/median coadded depth, stdev of
    # coadded depth and the 3x2ptFoM emulator.

    dustmap = maps.DustMap(nside=nside, interp=False)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    summaryMetrics = [
        metrics.MeanMetric(),
        metrics.MedianMetric(),
        metrics.RmsMetric(),
        metrics.CountRatioMetric(norm_val=1 / pix_area, metric_name="Effective Area (deg)"),
    ]
    displayDict["order"] = 0
    slicer = slicers.HealpixSlicer(nside=nside, use_cache=False)
    for yr_cut in yrs:
        ptsrc_lim_mag_i_band = mag_cuts[yr_cut]
        sqlconstraint = "night <= %s" % (yr_cut * 365.25 + 0.5)
        sqlconstraint += ' and scheduler_note not like "DD%"'
        info_label = f"{bandpass} band non-DD year {yr_cut}"
        ThreebyTwoSummary_simple = metrics.StaticProbesFoMEmulatorMetricSimple(
            nside=nside, year=yr_cut, metric_name="3x2ptFoM_simple"
        )
        ThreebyTwoSummary = maf.StaticProbesFoMEmulatorMetric(nside=nside, metric_name="3x2ptFoM")

        m = metrics.ExgalM5WithCuts(
            lsst_filter=bandpass,
            n_filters=nfilters_needed,
            extinction_cut=lim_ebv,
            depth_cut=ptsrc_lim_mag_i_band,
        )
        caption = (
            f"Cosmology/Static science metrics are based on evaluating the region "
            f"of the sky that meets the requirements (in year {yr_cut} of coverage"
            f"in all {nfilters_needed} bands, a lower E(B-V) value than {lim_ebv} "
            f"and at least a coadded depth of {ptsrc_lim_mag_i_band} in {bandpass}. "
            f"From there the effective survey area, coadded depth, standard deviation of "
            f"the depth, and a 3x2pt static science figure of merit emulator are "
            f"calculated using the dust-extinction coadded depth map (over that reduced "
            f"footprint)."
        )
        displayDict["caption"] = caption
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqlconstraint,
            maps_list=[dustmap],
            info_label=info_label,
            summary_metrics=summaryMetrics + [ThreebyTwoSummary, ThreebyTwoSummary_simple],
            display_dict=displayDict,
        )
        displayDict["order"] += 1
        bundleList.append(bundle)

    ## LSS Science
    # The only metric we have from LSS is the NGals metric -
    # which is similar to the GalaxyCountsExtended
    # metric, but evaluated only on the depth/dust cuts footprint.
    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: LSS"
    displayDict["order"] = 0
    plotDict = {"n_ticks": 5}

    ## WL metrics
    # Calculates the number of visits per pointing, after removing
    # parts of the footprint due to dust/depth
    # Count visits in gri bands.
    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: WL"
    displayDict["order"] = 0
    sqlconstraint = 'scheduler_note not like "DD%" and (filter="g" or filter="r" or filter="i")'
    info_label = "gri band non-DD"
    minExpTime = 15
    m = metrics.WeakLensingNvisits(
        lsst_filter=bandpass,
        depth_cut=mag_cuts[maxYr],
        ebvlim=lim_ebv,
        min_exp_time=minExpTime,
        metric_name="WeakLensingNvisits",
    )
    slicer = slicers.HealpixSlicer(nside=nside, use_cache=False)
    displayDict["caption"] = (
        f"The number of visits per pointing, over a similarly reduced footprint as "
        f"described above for the 3x2pt FOM, but allowing areas of sky with "
        f"fewer than {nfilters_needed} filters. "
        f"A cutoff of {minExpTime} removes very short visits."
    )
    bundle = mb.MetricBundle(
        m,
        slicer,
        sqlconstraint,
        maps_list=[dustmap],
        info_label=info_label,
        summary_metrics=standardStats,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Do the weak lensing per year
    for year in [10]:
        sqlconstraint = (
            'scheduler_note not like "DD%"'
            + ' and (filter="g" or filter="r" or filter="i") and night < %i' % (year * 365.25)
        )
        m = metrics.WeakLensingNvisits(
            lsst_filter=bandpass,
            depth_cut=mag_cuts[year],
            ebvlim=lim_ebv,
            min_exp_time=minExpTime,
            metric_name="WeakLensingNvisits_gri_year%i" % year,
        )
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqlconstraint,
            maps_list=[dustmap],
            info_label=info_label,
            summary_metrics=standardStats,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

        m = metrics.RIZDetectionCoaddExposureTime(
            det_bands=["g", "r", "i"], metric_name="gri_exposure_time_year%i" % year
        )
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqlconstraint,
            maps_list=[dustmap],
            info_label=info_label,
            summary_metrics=standardStats,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

        sqlconstraint = (
            'scheduler_note not like "DD%"'
            + ' and (filter="r" or filter="i" or filter="z") and night < %i' % (year * 365.25)
        )
        m = metrics.WeakLensingNvisits(
            lsst_filter=bandpass,
            depth_cut=mag_cuts[year],
            ebvlim=lim_ebv,
            min_exp_time=minExpTime,
            metric_name="WeakLensingNvisits_riz_year%i" % year,
        )
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqlconstraint,
            maps_list=[dustmap],
            info_label=info_label,
            summary_metrics=standardStats,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

        m = metrics.RIZDetectionCoaddExposureTime(
            det_bands=["g", "r", "i"], metric_name="riz_exposure_time_year%i" % year
        )
        bundle = mb.MetricBundle(
            m,
            slicer,
            sqlconstraint,
            maps_list=[dustmap],
            info_label=info_label,
            summary_metrics=standardStats,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: Camera Rotator"
    metric1 = metrics.KuiperMetric("rotSkyPos")
    metric2 = metrics.KuiperMetric("rotTelPos")
    caption_root = "Kuiper statistic (0 is uniform, 1 is delta function) of the "
    for f in filterlist:
        for m in [metric1, metric2]:
            plotDict = {"color": colors[f]}
            displayDict["order"] = filterorders[f]
            displayDict["caption"] = caption_root + f"{m.colname} for visits in {f} band."
            bundleList.append(
                mb.MetricBundle(
                    m,
                    healpixslicer,
                    filtersqls[f],
                    plot_dict=plotDict,
                    display_dict=displayDict,
                    summary_metrics=standardStats,
                    plot_funcs=subsetPlots,
                )
            )

    ##############
    # SNe Ia
    ##############
    displayDict = {
        "group": "Cosmology",
        "subgroup": "5: SNe Ia",
        "order": 0,
        "caption": "Expected discoveries of SNeIa, using the SNNSNMetric.",
    }
    sne_nside = 16
    sn_summary = [
        metrics.MedianMetric(),
        metrics.MeanMetric(),
        metrics.SumMetric(metric_name="Total detected"),
        metrics.CountMetric(metric_name="Total on sky", mask_val=0),
    ]
    snslicer = slicers.HealpixSlicer(nside=sne_nside, use_cache=False)
    metric = metrics.SNNSNMetric(
        n_bef=3,
        n_aft=8,
        coadd_night=True,
        add_dust=False,
        hard_dust_cut=0.25,
        zmin=0.2,
        zmax=0.5,
        z_step=0.03,
        daymax_step=3.0,
        zlim_coeff=0.95,
        gamma_name="gamma_WFD.hdf5",
        verbose=False,
    )
    plotDict = {"percentile_clip": 95, "n_ticks": 5}
    # Run without DDF observations
    bundle = mb.MetricBundle(
        metric,
        snslicer,
        "scheduler_note not like '%DD%'",
        plot_dict=plotDict,
        display_dict=displayDict,
        info_label="DDF excluded",
        summary_metrics=sn_summary,
        plot_funcs=subsetPlots,
    )

    bundleList.append(bundle)

    #########################
    #########################
    # Variables and Transients
    #########################
    #########################

    # Periodic Stars
    displayDict = {"group": "Variables/Transients", "order": 0}

    # Tidal Disruption Events
    displayDict["subgroup"] = "TDE"
    displayDict["caption"] = "TDE lightcurves that could be identified"
    displayDict["order"] = 0
    metric = maf.TdePopMetric(mjd0=mjd0)
    tdeslicer = maf.generate_tde_pop_slicer()
    bundle = mb.MetricBundle(
        metric,
        tdeslicer,
        "",
        run_name=runName,
        summary_metrics=lightcurve_summary(),
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    displayDict["caption"] = "TDE lightcurves quality"
    metric = maf.TdePopMetricQuality(metric_name="TDE_Quality")
    bundle = mb.MetricBundle(
        metric,
        tdeslicer,
        "",
        run_name=runName,
        summary_metrics=lightcurve_summary(),
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Microlensing events

    displayDict["subgroup"] = "Microlensing"
    plotDict = {"nside": 128}

    n_events = 10000
    # Let's evaluate a variety of crossing times
    crossing_times = [
        [1, 5],
        [5, 10],
        [10, 20],
        [20, 30],
        [30, 60],
        [60, 90],
        [100, 200],
        [200, 500],
        [500, 1000],
    ]
    metric = maf.MicrolensingMetric()
    summaryMetrics = maf.batches.lightcurve_summary()
    order = 0
    for crossing in crossing_times:
        displayDict["caption"] = "Microlensing events with crossing times between %i to %i days." % (
            crossing[0],
            crossing[1],
        )
        displayDict["order"] = order
        order += 1
        slicer = maf.generate_microlensing_slicer(
            min_crossing_time=crossing[0],
            max_crossing_time=crossing[1],
            n_events=n_events,
        )
        bundleList.append(
            maf.MetricBundle(
                metric,
                slicer,
                None,
                run_name=runName,
                summary_metrics=summaryMetrics,
                info_label=f"tE {crossing[0]}_{crossing[1]} days",
                display_dict=displayDict,
                plot_dict=plotDict,
                plot_funcs=[plots.HealpixSkyMap()],
            )
        )

    if long_microlensing:
        n_events = 10000
        # Let's evaluate a subset of the crossing times for these
        crossing_times = [
            [10, 20],
            [20, 30],
            [30, 60],
            [200, 500],
        ]
        metric_Npts = maf.MicrolensingMetric(metric_calc="Npts")
        summaryMetrics = maf.batches.microlensing_summary(metric_type="Npts")
        order = 0
        for crossing in crossing_times:
            slicer = maf.generate_microlensing_slicer(
                min_crossing_time=crossing[0],
                max_crossing_time=crossing[1],
                n_events=n_events,
            )
            displayDict["caption"] = "Microlensing events with crossing times between %i to %i days." % (
                crossing[0],
                crossing[1],
            )
            displayDict["order"] = order
            order += 1
            bundleList.append(
                maf.MetricBundle(
                    metric_Npts,
                    slicer,
                    None,
                    run_name=runName,
                    summary_metrics=summaryMetrics,
                    info_label=f"tE {crossing[0]}_{crossing[1]} days",
                    display_dict=displayDict,
                    plot_dict=plotDict,
                    plot_funcs=[],
                )
            )

        metric_Fisher = maf.MicrolensingMetric(metric_calc="Fisher")
        summaryMetrics = maf.batches.microlensing_summary(metric_type="Fisher")
        order = 0
        for crossing in crossing_times:
            displayDict["caption"] = "Microlensing events with crossing times between %i to %i days." % (
                crossing[0],
                crossing[1],
            )
            displayDict["order"] = order
            order += 1
            slicer = maf.generate_microlensing_slicer(
                min_crossing_time=crossing[0],
                max_crossing_time=crossing[1],
                n_events=n_events,
            )
            bundleList.append(
                maf.MetricBundle(
                    metric_Fisher,
                    slicer,
                    None,
                    run_name=runName,
                    summary_metrics=summaryMetrics,
                    info_label=f"tE {crossing[0]}_{crossing[1]} days",
                    display_dict=displayDict,
                    plot_dict=plotDict,
                    plot_funcs=[],
                )
            )

    # Kilonovae metric
    displayDict["group"] = "Variables/Transients"
    displayDict["subgroup"] = "KNe"
    n_events = 10000
    caption = f"KNe metric, injecting {n_events} lightcurves over the entire sky, GW170817-like only."
    caption += " Ignoring DDF observations."
    displayDict["caption"] = caption
    displayDict["order"] = 0
    # Kilonova parameters
    inj_params_list = [
        {"mej_dyn": 0.005, "mej_wind": 0.050, "phi": 30, "theta": 25.8},
    ]
    filename = maf.get_kne_filename(inj_params_list)
    kneslicer = maf.generate_kn_pop_slicer(n_events=n_events, n_files=len(filename), d_min=10, d_max=600)
    metric = maf.KNePopMetric(
        output_lc=False,
        file_list=filename,
        metric_name="KNePopMetric_single",
        mjd0=mjd0,
    )
    bundle = mb.MetricBundle(
        metric,
        kneslicer,
        "scheduler_note not like 'DD%'",
        run_name=runName,
        info_label="single model",
        summary_metrics=lightcurve_summary(),
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    n_events = 10000
    caption = f"KNe metric, injecting {n_events} lightcurves over the entire sky, entire model population."
    caption += " Ignoring DDF observations."
    displayDict["caption"] = caption
    displayDict["order"] = 1
    # Kilonova parameters
    filename = maf.get_kne_filename(None)
    kneslicer_allkne = maf.generate_kn_pop_slicer(
        n_events=n_events, n_files=len(filename), d_min=10, d_max=600
    )
    metric_allkne = maf.KNePopMetric(output_lc=False, file_list=filename, metric_name="KNePopMetric_all")
    bundle = mb.MetricBundle(
        metric_allkne,
        kneslicer_allkne,
        "scheduler_note not like 'DD%'",
        run_name=runName,
        info_label="all models",
        summary_metrics=lightcurve_summary(),
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # General time intervals
    displayDict = {
        "group": "TimeGaps",
        "subgroup": "Time",
        "caption": None,
        "order": 0,
    }

    for f in filterlist:
        gaps = [3.0, 7.0, 24.0]
        for gap in gaps:
            summary_stats = []
            summary_stats.append(
                metrics.AreaSummaryMetric(
                    area=18000,
                    reduce_func=np.median,
                    decreasing=True,
                    metric_name="Median N gaps in %s at %ihr in top 18k" % (f, gap),
                )
            )

            summary_stats.append(
                metrics.AreaSummaryMetric(
                    area=18000,
                    reduce_func=np.mean,
                    decreasing=True,
                    metric_name="Mean N gaps in %s at %ihr in top 18k" % (f, gap),
                )
            )

            summary_stats.append(metrics.MeanMetric())
            summary_stats.append(metrics.MedianMetric())

            m2 = metrics.GapsMetric(
                time_scale=gap,
                metric_name="Gaps_%ihr" % gap,
            )
            plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
            plotDict = {"color_min": 0, "color": colors[f], "percentile_clip": 95}
            displayDict["caption"] = (
                "Number of times the timescale of ~%i hours is sampled in %s band(s)." % (gap, f)
            )
            displayDict["order"] = filterorders[f]
            bundleList.append(
                mb.MetricBundle(
                    m2,
                    healpixslicer,
                    constraint=filtersqls[f],
                    info_label=filterinfo_label[f],
                    run_name=runName,
                    summary_metrics=summary_stats,
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    display_dict=displayDict,
                )
            )

    # XRB metric
    displayDict["subgroup"] = "XRB"
    displayDict["order"] = 0
    displayDict["caption"] = "Number or characterization of XRBs."
    n_events = 10000
    xrbslicer = maf.generate_xrb_pop_slicer(n_events=n_events)
    metric = maf.XRBPopMetric(output_lc=False, mjd0=mjd0)
    xrb_summaryMetrics = [
        maf.SumMetric(metric_name="Total detected"),
        maf.CountMetric(metric_name="Total lightcurves in footprint"),
        maf.CountMetric(metric_name="Total lightcurves on sky", mask_val=0),
        maf.MeanMetric(metric_name="Fraction detected in footprint"),
        maf.MeanMetric(mask_val=0, metric_name="Fraction detected of total"),
        maf.MedianMetric(metric_name="Median"),
        maf.MeanMetric(metric_name="Mean"),
    ]

    bundleList.append(
        maf.MetricBundle(
            metric,
            xrbslicer,
            "",
            run_name=runName,
            summary_metrics=xrb_summaryMetrics,
            display_dict=displayDict,
        )
    )

    #########################
    #########################
    # Milky Way
    #########################
    #########################

    displayDict = {"group": "Milky Way", "subgroup": ""}

    # Brown Dwarf Volume
    displayDict["subgroup"] = "Brown Dwarf"
    displayDict["order"] = 0
    l7_bd_mags = {"i": 20.09, "z": 18.18, "y": 17.13}
    displayDict["caption"] = (
        f"The expected maximum distance at which an L7 brown dwarf with magnitude {l7_bd_mags} "
        f"would have a parallax SNR of 10.0. The summary statistic represents the volume enclosed by "
        f"the result of this metric (BDParallaxMetric)."
    )
    sum_stats = [metrics.VolumeSumMetric(nside=nside)]
    metric = metrics.BDParallaxMetric(mags=l7_bd_mags, metric_name="Brown Dwarf, L7")
    sql = ""
    plotDict = {}
    bundleList.append(
        mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            plot_dict=plotDict,
            summary_metrics=sum_stats,
            display_dict=displayDict,
            run_name=runName,
        )
    )

    l4_bd_mags = {"i": 18.35, "z": 16.68, "y": 15.66}
    displayDict["caption"] = (
        f"The expected maximum distance at which an L4 brown dwarf with magnitude {l4_bd_mags} "
        f"would have a parallax SNR of 10.0. The summary statistic represents the total volume enclosed "
        f"by the result of this metric (BDParallaxMetric)."
    )
    metric = metrics.BDParallaxMetric(mags=l4_bd_mags, metric_name="Brown Dwarf, L4")
    bundleList.append(
        mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            plot_dict=plotDict,
            summary_metrics=sum_stats,
            display_dict=displayDict,
            run_name=runName,
        )
    )

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    bundleDict = mb.make_bundles_dict_from_list(bundleList)

    return bundleDict
