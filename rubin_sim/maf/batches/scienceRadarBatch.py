import numpy as np
import healpy as hp
import astropy.units as u
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.plots as plots
import rubin_sim.maf.maps as maps
import rubin_sim.maf.metricBundles as mb
from .common import standardSummary, extendedSummary, lightcurveSummary, filterList
import rubin_sim.maf as maf


__all__ = ["scienceRadarBatch"]


def scienceRadarBatch(
    runName="opsim",
    nside=64,
    benchmarkArea=18000,
    benchmarkNvisits=825,
    minNvisits=750,
    long_microlensing=False,
    srd_only=False,
):
    """A batch of metrics for looking at survey performance relative to the SRD and the main
    science drivers of LSST.

    Parameters
    ----------
    long_microlensing : `bool` (False)
        Add the longer running microlensing metrics to the batch
    srd_only : `bool` (False)
        Only return the SRD metrics
    """

    bundleList = []
    # Get some standard per-filter coloring and sql constraints
    filterlist, colors, filterorders, filtersqls, filterinfo_label = filterList(
        all=False
    )
    (
        allfilterlist,
        allcolors,
        allfilterorders,
        allfiltersqls,
        allfilterinfo_label,
    ) = filterList(all=True)

    standardStats = standardSummary(withCount=False)

    # This is the default slicer for most purposes in this batch. Note that the cache is off -
    # if the metric requires a dust map, this is not the right slicer to use.
    healpixslicer = slicers.HealpixSlicer(nside=nside, useCache=True)
    subsetPlots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]

    #########################
    #########################
    # SRD, DM, etc
    #########################
    #########################
    # fO metric
    displayDict = {"group": "SRD", "subgroup": "FO metrics", "order": 0}

    # Configure the count metric which is what is used for f0 slicer.
    metric = metrics.CountExplimMetric(col="observationStartMJD", metricName="fO")
    plotDict = {
        "xlabel": "Number of Visits",
        "Asky": benchmarkArea,
        "Nvisit": benchmarkNvisits,
        "xMin": 0,
        "xMax": 1500,
    }
    summaryMetrics = [
        metrics.fOArea(
            nside=nside,
            norm=False,
            metricName="fOArea",
            Asky=benchmarkArea,
            Nvisit=benchmarkNvisits,
        ),
        metrics.fOArea(
            nside=nside,
            norm=True,
            metricName="fOArea/benchmark",
            Asky=benchmarkArea,
            Nvisit=benchmarkNvisits,
        ),
        metrics.fONv(
            nside=nside,
            norm=False,
            metricName="fONv",
            Asky=benchmarkArea,
            Nvisit=benchmarkNvisits,
        ),
        metrics.fONv(
            nside=nside,
            norm=True,
            metricName="fONv/benchmark",
            Asky=benchmarkArea,
            Nvisit=benchmarkNvisits,
        ),
        metrics.fOArea(
            nside=nside,
            norm=False,
            metricName=f"fOArea_{minNvisits}",
            Asky=benchmarkArea,
            Nvisit=minNvisits,
        ),
    ]
    caption = "The FO metric evaluates the overall efficiency of observing. "
    caption += (
        "foNv: out of %.2f sq degrees, the area receives at least X and a median of Y visits "
        "(out of %d, if compared to benchmark). " % (benchmarkArea, benchmarkNvisits)
    )
    caption += (
        "fOArea: this many sq deg (out of %.2f sq deg if compared "
        "to benchmark) receives at least %d visits. "
        % (benchmarkArea, benchmarkNvisits)
    )
    displayDict["caption"] = caption
    bundle = mb.MetricBundle(
        metric,
        healpixslicer,
        "",
        plotDict=plotDict,
        displayDict=displayDict,
        summaryMetrics=summaryMetrics,
        plotFuncs=[plots.FOPlot()],
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
        m = metrics.CountMetric(value, metricName="%s Histogram" % (value))
        slicer = slicers.OneDSlicer(sliceColName=value)
        bundle = mb.MetricBundle(
            m,
            slicer,
            filtersqls[f],
            displayDict=displayDict,
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
            displayDict=displayDict,
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
            metricName="Median Parallax Uncert (18k)",
        ),
        metrics.AreaThresholdMetric(
            upper_threshold=good_parallax_limit,
            metricName="Area better than %.1f mas uncertainty" % good_parallax_limit,
        ),
    ]
    summary.append(
        metrics.PercentileMetric(
            percentile=95, metricName="95th Percentile Parallax Uncert"
        )
    )
    summary.extend(standardSummary())
    for rmag, plotmax in zip(rmags_para, plotmaxVals):
        plotDict = {"xMin": 0, "xMax": plotmax, "colorMin": 0, "colorMax": plotmax}
        metric = metrics.ParallaxMetric(
            metricName="Parallax Uncert @ %.1f" % (rmag),
            rmag=rmag,
            normalize=False,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stackerList=[parallaxStacker],
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=summary,
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1

    # Parallax normalized to 'best possible' if all visits separated by 6 months.
    # This separates the effect of cadence from depth.
    for rmag in rmags_para:
        metric = metrics.ParallaxMetric(
            metricName="Normalized Parallax Uncert @ %.1f" % (rmag),
            rmag=rmag,
            normalize=True,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stackerList=[parallaxStacker],
            displayDict=displayDict,
            summaryMetrics=standardSummary(),
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Parallax factor coverage.
    for rmag in rmags_para:
        metric = metrics.ParallaxCoverageMetric(
            metricName="Parallax Coverage @ %.1f" % (rmag), rmag=rmag
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stackerList=[parallaxStacker],
            displayDict=displayDict,
            summaryMetrics=standardSummary(),
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Parallax problems can be caused by HA and DCR degeneracies. Check their correlation.
    for rmag in rmags_para:
        metric = metrics.ParallaxDcrDegenMetric(
            metricName="Parallax-DCR degeneracy @ %.1f" % (rmag), rmag=rmag
        )
        caption = (
            "Correlation between parallax offset magnitude and hour angle for a r=%.1f star."
            % (rmag)
        )
        caption += " (0 is good, near -1 or 1 is bad)."
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            stackerList=[dcrStacker, parallaxStacker],
            displayDict=displayDict,
            summaryMetrics=standardSummary(),
            plotFuncs=subsetPlots,
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
            metricName="Median Proper Motion Uncert (18k)",
        )
    ]
    summary.append(
        metrics.PercentileMetric(metricName="95th Percentile Proper Motion Uncert")
    )
    summary.extend(standardSummary())
    for rmag, plotmax in zip(rmags_pm, plotmaxVals):
        plotDict = {"xMin": 0, "xMax": plotmax, "colorMin": 0, "colorMax": plotmax}
        metric = metrics.ProperMotionMetric(
            metricName="Proper Motion Uncert @ %.1f" % rmag,
            rmag=rmag,
            normalize=False,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            displayDict=displayDict,
            plotDict=plotDict,
            summaryMetrics=summary,
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1
    # Normalized proper motion.
    for rmag in rmags_pm:
        metric = metrics.ProperMotionMetric(
            metricName="Normalized Proper Motion Uncert @ %.1f" % rmag,
            rmag=rmag,
            normalize=True,
        )
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            "",
            displayDict=displayDict,
            summaryMetrics=standardSummary(),
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)
        displayDict["order"] += 1

    # DCR precision metric
    displayDict["subgroup"] = "DCR"
    displayDict["order"] = 0
    plotDict = {"caption": "Precision of DCR slope."}
    metric = metrics.DcrPrecisionMetric()
    bundle = mb.MetricBundle(
        metric,
        healpixslicer,
        "",
        plotDict=plotDict,
        plotFuncs=subsetPlots,
        displayDict=displayDict,
        summaryMetrics=standardSummary(withCount=False),
    )
    bundleList.append(bundle)

    # Rapid Revisit
    displayDict["subgroup"] = "Rapid Revisits"
    displayDict["order"] = 0
    # Calculate the actual number of revisits within 30 minutes.
    dTmax = 30  # time in minutes
    m2 = metrics.NRevisitsMetric(
        dT=dTmax, normed=False, metricName="NumberOfQuickRevisits"
    )
    plotDict = {"colorMin": 400, "colorMax": 2000, "xMin": 400, "xMax": 2000}
    caption = (
        "Number of consecutive visits with return times faster than %.1f minutes, "
        % (dTmax)
    )
    caption += "in any filter. "
    displayDict["caption"] = caption
    bundle = mb.MetricBundle(
        m2,
        healpixslicer,
        "",
        plotDict=plotDict,
        plotFuncs=subsetPlots,
        displayDict=displayDict,
        summaryMetrics=standardSummary(withCount=False),
    )
    bundleList.append(bundle)
    displayDict["order"] += 1

    # Better version of the rapid revisit requirements: require a minimum number of visits between
    # dtMin and dtMax, but also a minimum number of visits between dtMin and dtPair (the typical pair time).
    # 1 means the healpix met the requirements (0 means did not).
    dTmin = 40.0 / 60.0  # (minutes) 40s minumum for rapid revisit range
    dTpairs = 20.0  # minutes (time when pairs should start kicking in)
    dTmax = 30.0  # 30 minute maximum for rapid revisit range
    nOne = 82  # Number of revisits between 40s-30m required
    nTwo = 28  # Number of revisits between 40s - tPairs required.
    pixArea = float(hp.nside2pixarea(nside, degrees=True))
    scale = pixArea * hp.nside2npix(nside)
    m1 = metrics.RapidRevisitMetric(
        metricName="RapidRevisits",
        dTmin=dTmin / 60.0 / 60.0 / 24.0,
        dTpairs=dTpairs / 60.0 / 24.0,
        dTmax=dTmax / 60.0 / 24.0,
        minN1=nOne,
        minN2=nTwo,
    )
    plotDict = {"xMin": 0, "xMax": 1, "colorMin": 0, "colorMax": 1, "logScale": False}
    cutoff1 = 0.9
    summaryStats = [
        metrics.FracAboveMetric(cutoff=cutoff1, scale=scale, metricName="Area (sq deg)")
    ]
    caption = (
        "Rapid Revisit: area that receives at least %d visits between %.3f and %.1f minutes, "
        % (nOne, dTmin, dTmax)
    )
    caption += (
        "with at least %d of those visits falling between %.3f and %.1f minutes. "
        % (nTwo, dTmin, dTpairs)
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
        plotDict=plotDict,
        plotFuncs=subsetPlots,
        displayDict=displayDict,
        summaryMetrics=summaryStats,
    )
    bundleList.append(bundle)

    # For SRD batch only, return here.
    if srd_only:
        for b in bundleList:
            b.setRunName(runName)
        bundleDict = mb.makeBundlesDictFromList(bundleList)

        return bundleDict

    # Year Coverage
    displayDict["subgroup"] = "Year Coverage"
    metric = metrics.YearCoverageMetric()
    for f in filterlist:
        plotDict = {"colorMin": 7, "colorMax": 10, "color": colors[f]}
        displayDict["caption"] = f"Number of years of coverage in {f} band."
        displayDict["order"] = filterorders[f]
        summary = [
            metrics.AreaSummaryMetric(
                area=18000,
                reduce_func=np.mean,
                decreasing=True,
                metricName="N Years (18k) %s" % f,
            )
        ]
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            filtersqls[f],
            plotDict=plotDict,
            info_label=filterinfo_label[f],
            displayDict=displayDict,
            summaryMetrics=summary,
        )
        bundleList.append(bundle)

    #########################
    #########################
    # Galaxies
    #########################
    #########################

    # Run this per filter, to look at variations in counts of galaxies in blue bands?
    displayDict = {
        "group": "Galaxies",
        "subgroup": "Galaxy Counts",
        "order": 0,
        "caption": None,
    }
    summary = [
        metrics.AreaSummaryMetric(
            area=18000,
            reduce_func=np.sum,
            decreasing=True,
            metricName="N Galaxies (18k)",
        )
    ]
    summary.append(metrics.SumMetric(metricName="N Galaxies (all)"))
    # make sure slicer has cache off
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    for f in filterlist:
        plotDict = {"percentileClip": 95.0, "nTicks": 5, "color": colors[f]}
        metric = maf.GalaxyCountsMetric_extended(
            filterBand=f, redshiftBin="all", nside=nside
        )
        displayDict[
            "caption"
        ] = f"Number of galaxies across the sky, in {f} band. Generally, full survey footprint."
        displayDict["order"] = filterorders[f]
        bundle = mb.MetricBundle(
            metric,
            slicer,
            filtersqls[f],
            plotDict=plotDict,
            displayDict=displayDict,
            summaryMetrics=summary,
            plotFuncs=subsetPlots,
        )
        bundleList.append(bundle)

    displayDict["subgroup"] = "Surface Brightness"
    summary = [metrics.MedianMetric()]
    for filtername in "ugrizy":
        displayDict["caption"] = (
            "Surface brightness limit in %s, no extinction applied." % filtername
        )
        displayDict["order"] = filterorders[f]
        sql = 'filter="%s"' % filtername
        metric = metrics.SurfaceBrightLimitMetric()
        bundle = mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            displayDict=displayDict,
            summaryMetrics=summary,
            plotFuncs=subsetPlots,
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
    mag_cuts = {1: 24.75 - 0.1, 3: 25.35 - 0.1, 6: 25.72 - 0.1, 10: 26.0 - 0.1}
    yrs = list(mag_cuts.keys())
    maxYr = max(yrs)

    displayDict = {"group": "Cosmology"}
    subgroupCount = 1

    displayDict["subgroup"] = f"{subgroupCount}: Static Science"
    ## Static Science
    # Calculate the static science metrics - effective survey area, mean/median coadded depth, stdev of
    # coadded depth and the 3x2ptFoM emulator.

    dustmap = maps.DustMap(nside=nside, interp=False)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    summaryMetrics = [
        metrics.MeanMetric(),
        metrics.MedianMetric(),
        metrics.RmsMetric(),
        metrics.CountRatioMetric(
            normVal=1 / pix_area, metricName="Effective Area (deg)"
        ),
    ]
    displayDict["order"] = 0
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    for yr_cut in yrs:
        ptsrc_lim_mag_i_band = mag_cuts[yr_cut]
        sqlconstraint = "night <= %s" % (yr_cut * 365.25)
        sqlconstraint += ' and note not like "DD%"'
        info_label = f"{bandpass} band non-DD year {yr_cut}"
        ThreebyTwoSummary_simple = metrics.StaticProbesFoMEmulatorMetricSimple(
            nside=nside, year=yr_cut, metricName="3x2ptFoM_simple"
        )
        ThreebyTwoSummary = maf.StaticProbesFoMEmulatorMetric(
            nside=nside, metricName="3x2ptFoM"
        )

        m = metrics.ExgalM5_with_cuts(
            lsstFilter=bandpass,
            nFilters=nfilters_needed,
            extinction_cut=lim_ebv,
            depth_cut=ptsrc_lim_mag_i_band,
        )
        caption = (
            "fCosmology/Static science metrics are based on evaluating the region "
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
            mapsList=[dustmap],
            info_label=info_label,
            summaryMetrics=summaryMetrics
            + [ThreebyTwoSummary, ThreebyTwoSummary_simple],
            displayDict=displayDict,
        )
        displayDict["order"] += 1
        bundleList.append(bundle)

    ## LSS Science
    # The only metric we have from LSS is the NGals metric - which is similar to the GalaxyCountsExtended
    # metric, but evaluated only on the depth/dust cuts footprint.
    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: LSS"
    displayDict["order"] = 0
    plotDict = {"nTicks": 5}
    # Have to include all filters in query, so that we check for all-band coverage.
    # Galaxy numbers calculated using 'bandpass' images only though.
    sqlconstraint = f'note not like "DD%"'
    info_label = f"{bandpass} band galaxies non-DD"
    metric = maf.DepthLimitedNumGalMetric(
        nside=nside,
        filterBand=bandpass,
        redshiftBin="all",
        nfilters_needed=nfilters_needed,
        lim_mag_i_ptsrc=mag_cuts[maxYr],
        lim_ebv=lim_ebv,
    )
    summary = [
        metrics.AreaSummaryMetric(
            area=18000,
            reduce_func=np.sum,
            decreasing=True,
            metricName="N Galaxies (18k)",
        )
    ]
    summary.append(metrics.SumMetric(metricName="N Galaxies (all)"))
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sqlconstraint,
        plotDict=plotDict,
        info_label=info_label,
        mapsList=[dustmap],
        displayDict=displayDict,
        summaryMetrics=summary,
        plotFuncs=subsetPlots,
    )
    bundleList.append(bundle)

    ## WL metrics
    # Calculates the number of visits per pointing, after removing parts of the footprint due to dust/depth
    # Count visits in gri bands.
    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: WL"
    displayDict["order"] = 0
    sqlconstraint = f'note not like "DD%" and (filter="g" or filter="r" or filter="i")'
    info_label = f"gri band non-DD"
    minExpTime = 15
    m = metrics.WeakLensingNvisits(
        lsstFilter=bandpass,
        depthlim=mag_cuts[maxYr],
        ebvlim=lim_ebv,
        min_expTime=minExpTime,
        metricName="WeakLensingNvisits",
    )
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    displayDict["caption"] = (
        f"The number of visits per pointing, over a similarly  reduced footprint as "
        f"described above for the 3x2pt FOM, but allowing areas of sky with "
        f"fewer than {nfilters_needed} filters. "
        f"A cutoff of {minExpTime} removes very short visits."
    )
    bundle = mb.MetricBundle(
        m,
        slicer,
        sqlconstraint,
        mapsList=[dustmap],
        info_label=info_label,
        summaryMetrics=standardStats,
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    subgroupCount += 1
    displayDict["subgroup"] = f"{subgroupCount}: Camera Rotator"
    displayDict[
        "caption"
    ] = "Kuiper statistic (0 is uniform, 1 is delta function) of the "
    metric1 = metrics.KuiperMetric("rotSkyPos")
    metric2 = metrics.KuiperMetric("rotTelPos")
    for f in filterlist:
        for m in [metric1, metric2]:
            plotDict = {"color": colors[f]}
            displayDict["order"] = filterorders[f]
            displayDict["caption"] += f"{m.colname} for visits in {f} band."
            bundleList.append(
                mb.MetricBundle(
                    m,
                    healpixslicer,
                    filtersqls[f],
                    plotDict=plotDict,
                    displayDict=displayDict,
                    summaryMetrics=standardStats,
                    plotFuncs=subsetPlots,
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
        metrics.SumMetric(metricName="Total detected"),
        metrics.CountMetric(metricName="Total on sky", maskVal=0),
    ]
    snslicer = slicers.HealpixSlicer(nside=sne_nside, useCache=False)
    metric = metrics.SNNSNMetric(verbose=False)  # zlim_coeff=0.98)
    bundle = mb.MetricBundle(
        metric,
        snslicer,
        "",
        plotDict=plotDict,
        displayDict=displayDict,
        summaryMetrics=sn_summary,
        plotFuncs=subsetPlots,
    )

    bundleList.append(bundle)

    #########################
    #########################
    # AGN
    #########################
    #########################

    # AGN structure function error
    agnslicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    dustmap = maps.DustMap(nside=nside)
    displayDict = {"group": "AGN", "order": 0}

    # Calculate the number of expected QSOs, in each band
    for f in filterlist:
        sql = filtersqls[f] + ' and note not like "%DD%"'
        md = filterinfo_label[f] + " and non-DD"
        summaryMetrics = [metrics.SumMetric(metricName="Total QSO")]
        zmin = 0.3
        m = metrics.QSONumberCountsMetric(
            f,
            units="mag",
            extinction_cut=1.0,
            qlf_module="Shen20",
            qlf_model="A",
            SED_model="Richards06",
            zmin=zmin,
            zmax=None,
        )
        displayDict["subgroup"] = "nQSO"
        displayDict["caption"] = (
            "The expected number of QSOs in regions of low dust extinction,"
            f"based on detection in {f} bandpass."
        )
        displayDict["order"] = filterorders[f]
        bundleList.append(
            mb.MetricBundle(
                m,
                agnslicer,
                constraint=sql,
                info_label=md,
                runName=runName,
                mapsList=[dustmap],
                summaryMetrics=summaryMetrics,
                displayDict=displayDict,
            )
        )

    # Calculate the expected AGN structure function error
    # These agn test magnitude values are determined by looking at the baseline median m5 depths
    # For v1.7.1 these values are:
    agn_m5 = {"u": 22.89, "g": 23.94, "r": 23.5, "i": 22.93, "z": 22.28, "y": 21.5}
    # And the expected medians SF error at those values is about 0.04 - set the threshold slightly below
    threshold = 0.025
    summaryMetrics = extendedSummary()
    summaryMetrics += [metrics.AreaThresholdMetric(upper_threshold=threshold)]
    for f in filterlist:
        m = metrics.SFUncertMetric(
            mag=agn_m5[f],
            metricName="AGN SF_uncert",
        )
        plotDict = {"color": colors[f], "colorMin": 0, "colorMax": 0.2}
        displayDict["order"] = filterorders[f]
        displayDict["subgroup"] = "SFUncert"
        displayDict["caption"] = (
            "Expected AGN structure function uncertainties, based on observations in "
            f"{f} band, for an AGN of magnitude {agn_m5[f]:.2f}"
        )
        bundleList.append(
            mb.MetricBundle(
                m,
                agnslicer,
                constraint=filtersqls[f],
                info_label=filterinfo_label[f],
                runName=runName,
                mapsList=[dustmap],
                plotDict=plotDict,
                summaryMetrics=summaryMetrics,
                displayDict=displayDict,
            )
        )

    # Run the TimeLag for each filter *and* all filters
    nquist_threshold = 2.2
    lag = 100
    summaryMetrics = extendedSummary()
    summaryMetrics += [metrics.AreaThresholdMetric(lower_threshold=nquist_threshold)]
    m = metrics.AGN_TimeLagMetric(threshold=nquist_threshold, lag=lag)
    for f in allfilterlist:
        plotDict = {
            "color": allcolors[f],
            "colorMin": 0,
            "colorMax": 5,
            "percentileClip": 95,
        }
        displayDict["order"] = allfilterorders[f]
        displayDict["subgroup"] = "Time Lags"
        displayDict["caption"] = (
            f"Comparion of the time between visits compared to a defined sampling gap ({lag} days) in "
            f"{f} band."
        )
        bundleList.append(
            mb.MetricBundle(
                m,
                agnslicer,
                constraint=allfiltersqls[f],
                info_label=allfilterinfo_label[f],
                runName=runName,
                mapsList=[dustmap],
                plotDict=plotDict,
                summaryMetrics=summaryMetrics,
                displayDict=displayDict,
            )
        )

    # Run the TimeLag for each filter *and* all filters but for 5 days
    nquist_threshold = 2.2
    lag = 5
    summaryMetrics = extendedSummary()
    summaryMetrics += [metrics.AreaThresholdMetric(lower_threshold=nquist_threshold)]
    m = metrics.AGN_TimeLagMetric(threshold=nquist_threshold, lag=lag)
    for f in allfilterlist:
        plotDict = {
            "color": allcolors[f],
            "colorMin": 0,
            "colorMax": 5,
            "percentileClip": 95,
        }
        displayDict["order"] = allfilterorders[f]
        displayDict["subgroup"] = "Time Lags"
        displayDict["caption"] = (
            f"Comparion of the time between visits compared to a defined sampling gap ({lag} days) in "
            f"{f} band."
        )
        bundleList.append(
            mb.MetricBundle(
                m,
                agnslicer,
                constraint=allfiltersqls[f],
                info_label=allfilterinfo_label[f],
                runName=runName,
                mapsList=[dustmap],
                plotDict=plotDict,
                summaryMetrics=summaryMetrics,
                displayDict=displayDict,
            )
        )

    #########################
    #########################
    # Strong Lensing
    #########################
    #########################

    # TDC metric
    # Calculate a subset of DESC WFD-related metrics.
    if nside > 64:
        nside_tdc = 64
    else:
        nside_tdc = nside
    displayDict = {"group": "Strong Lensing"}
    displayDict["subgroup"] = "Lens Time Delay"

    tdc_plots = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
    plotDict = {"xMin": 0.01, "colorMin": 0.01, "percentileClip": 70, "nTicks": 5}
    tdc_summary = [metrics.MeanMetric(), metrics.MedianMetric(), metrics.RmsMetric()]
    # Ideally need a way to do better on calculating the summary metrics for the high accuracy area.
    slicer = slicers.HealpixSlicer(nside=nside_tdc, useCache=False)
    tdcMetric = metrics.TdcMetric(metricName="TDC")
    dustmap = maps.DustMap(nside=nside_tdc, interp=False)
    bundle = mb.MetricBundle(
        tdcMetric,
        slicer,
        constraint="",
        displayDict=displayDict,
        plotDict=plotDict,
        plotFuncs=tdc_plots,
        mapsList=[dustmap],
        summaryMetrics=tdc_summary,
    )
    bundleList.append(bundle)

    # Strongly lensed SNe
    displayDict["group"] = "Strong Lensing"
    displayDict["subgroup"] = "SLSN"
    displayDict[
        "caption"
    ] = "Strongly Lensed SNe, evaluated with the addition of galactic dust extinction."
    metric = metrics.SNSLMetric()
    slicer = slicers.HealpixSlicer(nside=nside_tdc, useCache=False)
    plotDict = {}
    bundle = mb.MetricBundle(
        metric,
        slicer,
        "",
        runName=runName,
        plotDict=plotDict,
        summaryMetrics=metrics.SumMetric(metricName="Total detected"),
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    #########################
    #########################
    # Variables and Transients
    #########################
    #########################

    # Periodic Stars
    displayDict = {"group": "Variables/Transients", "order": 0}

    # PeriodicStarModulation metric
    # colors for c type RRLyrae
    displayDict["subgroup"] = "Periodic Star Modulation"
    I_rrc_lmc = 18.9
    V_rrc_lmc = 19.2
    Vi = V_rrc_lmc - (2.742 * 0.08) - 18.5
    Ii = I_rrc_lmc - (1.505 * 0.08) - 18.5
    ii_rrc = Ii + 0.386 * 0.013 + 0.397  # 0.013 = (i-z)_0
    gi_rrc = ii_rrc + 1.481 * (Vi - Ii) - 0.536
    ri_rrc = (1 / 0.565) * (Vi - 0.435 * gi_rrc + 0.016)
    ui_rrc = gi_rrc + 0.575
    zi_rrc = ii_rrc - 0.013
    yi_rrc = zi_rrc
    rrc = np.array([ui_rrc, gi_rrc, ri_rrc, ii_rrc, zi_rrc, yi_rrc])
    time_intervals = (15, 30)
    distMod = (18, 19, 20, 21)
    summaryStats = [metrics.MeanMetric(), metrics.MedianMetric(), metrics.MaxMetric()]
    slicer = slicers.HealpixSlicer(nside=8)
    for time_interval in time_intervals:
        for dM in distMod:
            displayDict["caption"] = (
                "Periodic star modulation metric, evaluates the likelihood of "
                "measuring variation in an RRLyrae periodic variable. "
                "Evaluated based on the full LSST survey data. "
                f"Searching time interval of {time_interval} and distance modulus {dM}."
            )
            displayDict["order"] += 1
            m = maf.PeriodicStarModulationMetric(
                period=0.3,
                amplitude=0.3,
                random_phase=True,
                time_interval=time_interval,
                nMonte=100,
                periodTol=0.002,
                ampTol=0.01,
                means=rrc + dM,
                magTol=0.01,
                nBands=3,
            )
            # Run this on year 1 and on year 3.5-4.5
            bundle = mb.MetricBundle(
                m,
                slicer,
                "night < 365.25",
                displayDict=displayDict,
                runName=runName,
                summaryMetrics=summaryStats,
                info_label=f"dm {dM} interval {time_interval} RRc Year 1",
            )
            bundleList.append(bundle)
            # and on year 3.5-4.5
            bundle = mb.MetricBundle(
                m,
                slicer,
                "night > 365.25*3.5 and night < 365.25*4.5",
                displayDict=displayDict,
                runName=runName,
                summaryMetrics=summaryStats,
                info_label=f"dm {dM} interval {time_interval} RRc Year 3.5",
            )
            bundleList.append(bundle)

    # PulsatingStarRecovery metric (to be added; Marcella)

    # our periodic star metrics
    displayDict["subgroup"] = "Periodic Stars"
    displayDict["order"] = 0
    for period in [0.5, 1, 2]:
        for magnitude in [21.0, 24.0]:
            amplitudes = [0.05, 0.1, 1.0]
            periods = [period] * len(amplitudes)
            starMags = [magnitude] * len(amplitudes)

            plotDict = {"nTicks": 3, "colorMin": 0, "colorMax": 3, "xMin": 0, "xMax": 3}
            info_label = "P_%.1f_Mag_%.0f_Amp_0.05-0.1-1" % (period, magnitude)
            sql = ""
            displayDict["caption"] = (
                "Metric evaluates if a periodic signal of period %.1f days could "
                "be detected for an r=%i star. A variety of amplitudes of periodicity "
                "are tested: [1, 0.1, and 0.05] mag amplitudes, which correspond to "
                "metric values of [1, 2, or 3]. " % (period, magnitude)
            )
            metric = metrics.PeriodicDetectMetric(
                periods=periods,
                starMags=starMags,
                amplitudes=amplitudes,
                metricName="PeriodicDetect",
            )
            bundle = mb.MetricBundle(
                metric,
                healpixslicer,
                sql,
                info_label=info_label,
                displayDict=displayDict,
                plotDict=plotDict,
                plotFuncs=subsetPlots,
                summaryMetrics=standardStats,
            )
            bundleList.append(bundle)
            displayDict["order"] += 1

    # Tidal Disruption Events
    displayDict["subgroup"] = "TDE"
    displayDict["caption"] = "TDE lightcurves that could be identified"
    displayDict["order"] = 0
    metric = maf.TdePopMetric()
    tdeslicer = maf.generateTdePopSlicer()
    bundle = mb.MetricBundle(
        metric,
        tdeslicer,
        "",
        runName=runName,
        summaryMetrics=lightcurveSummary(),
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    displayDict["caption"] = "TDE lightcurves quality"
    metric = maf.TdePopMetricQuality(metricName="TDE_Quality")
    bundle = mb.MetricBundle(
        metric,
        tdeslicer,
        "",
        runName=runName,
        summaryMetrics=lightcurveSummary(),
        displayDict=displayDict,
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
    summaryMetrics = maf.batches.lightcurveSummary()
    order = 0
    for crossing in crossing_times:
        displayDict[
            "caption"
        ] = "Microlensing events with crossing times between %i to %i days." % (
            crossing[0],
            crossing[1],
        )
        displayDict["order"] = order
        order += 1
        slicer = maf.generateMicrolensingSlicer(
            min_crossing_time=crossing[0],
            max_crossing_time=crossing[1],
            n_events=n_events,
        )
        bundleList.append(
            maf.MetricBundle(
                metric,
                slicer,
                None,
                runName=runName,
                summaryMetrics=summaryMetrics,
                info_label=f"tE {crossing[0]}_{crossing[1]} days",
                displayDict=displayDict,
                plotDict=plotDict,
                plotFuncs=[plots.HealpixSkyMap()],
            )
        )

    if long_microlensing:
        metric_Npts = maf.MicrolensingMetric(metricCalc="Npts")
        summaryMetrics = maf.batches.microlensingSummary(metricType="Npts")
        order = 0
        for crossing in crossing_times:
            slicer = maf.generateMicrolensingSlicer(
                min_crossing_time=crossing[0],
                max_crossing_time=crossing[1],
                n_events=n_events,
            )
            displayDict[
                "caption"
            ] = "Microlensing events with crossing times between %i to %i days." % (
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
                    runName=runName,
                    summaryMetrics=summaryMetrics,
                    info_label=f"tE {crossing[0]}_{crossing[1]} days",
                    displayDict=displayDict,
                    plotDict=plotDict,
                    plotFuncs=[],
                )
            )

        metric_Fisher = maf.MicrolensingMetric(metricCalc="Fisher")
        summaryMetrics = maf.batches.microlensingSummary(metricType="Fisher")
        order = 0
        for crossing in crossing_times:
            displayDict[
                "caption"
            ] = "Microlensing events with crossing times between %i to %i days." % (
                crossing[0],
                crossing[1],
            )
            displayDict["order"] = order
            order += 1
            slicer = maf.generateMicrolensingSlicer(
                min_crossing_time=crossing[0],
                max_crossing_time=crossing[1],
                n_events=n_events,
            )
            bundleList.append(
                maf.MetricBundle(
                    metric_Fisher,
                    slicer,
                    None,
                    runName=runName,
                    summaryMetrics=summaryMetrics,
                    info_label=f"tE {crossing[0]}_{crossing[1]} days",
                    displayDict=displayDict,
                    plotDict=plotDict,
                    plotFuncs=[],
                )
            )

    # Kilonovae metric
    displayDict["group"] = "Variables/Transients"
    displayDict["subgroup"] = "KNe"
    n_events = 50000
    displayDict[
        "caption"
    ] = f"KNe metric, injecting {n_events} lightcurves over the entire sky, GW170817-like only."
    displayDict["order"] = 0
    # Kilonova parameters
    inj_params_list = [
        {"mej_dyn": 0.005, "mej_wind": 0.050, "phi": 30, "theta": 25.8},
        # {"mej_dyn": 0.005, "mej_wind": 0.050, "phi": 30, "theta": 0.0}, # no longer preferred
    ]
    filename = maf.get_KNe_filename(inj_params_list)
    kneslicer = maf.generateKNPopSlicer(
        n_events=n_events, n_files=len(filename), d_min=10, d_max=600
    )
    metric = maf.KNePopMetric(outputLc=False, file_list=filename)
    bundle = mb.MetricBundle(
        metric,
        kneslicer,
        "",
        runName=runName,
        info_label="single model",
        summaryMetrics=lightcurveSummary(),
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    n_events = 50000
    displayDict[
        "caption"
    ] = f"KNe metric, injecting {n_events} lightcurves over the entire sky, entire model population."
    displayDict["order"] = 1
    # Kilonova parameters
    filename = maf.get_KNe_filename(None)
    kneslicer_allkne = maf.generateKNPopSlicer(
        n_events=n_events, n_files=len(filename), d_min=10, d_max=600
    )
    metric_allkne = maf.KNePopMetric(outputLc=False, file_list=filename)
    bundle = mb.MetricBundle(
        metric_allkne,
        kneslicer_allkne,
        "",
        runName=runName,
        info_label="all models",
        summaryMetrics=lightcurveSummary(),
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    # General time intervals
    displayDict = {
        "group": "TimeGaps",
        "subgroup": "Time",
        "caption": None,
        "order": 0,
    }

    # Logarithmically spaced gaps from 30s to 5 years
    tMin = 30 / 60 / 60 / 24.0  # 30s
    tMax = 5 * 365.25  # 5 years
    tgaps = np.logspace(np.log10(tMin), np.log10(tMax), 100)

    for f in filterlist:
        m1 = metrics.TgapsMetric(bins=tgaps, allGaps=False)
        plotDict = {
            "bins": tgaps,
            "xscale": "log",
            "yMin": 0,
            "figsize": (8, 6),
            "ylabel": "Number of observation pairs",
            "xlabel": "Time gap between pairs of visits (days)",
            "color": colors[f],
        }
        plotFuncs = [plots.SummaryHistogram()]
        displayDict["caption"] = (
            f"Summed Histogram of time between visits at each point in the sky, "
            f"in {f} band(s)."
        )
        displayDict["order"] = filterorders[f]
        bundleList.append(
            mb.MetricBundle(
                m1,
                healpixslicer,
                constraint=filtersqls[f],
                info_label=filterinfo_label[f],
                runName=runName,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                displayDict=displayDict,
            )
        )

        m2 = metrics.TgapsPercentMetric(
            minTime=2 / 24.0,
            maxTime=14 / 24.0,
            allGaps=False,
            metricName="TgapsPercent_2-14hrs",
        )
        plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
        plotDict = {"colorMin": 0, "color": colors[f], "percentileClip": 95}
        summaryMetrics = extendedSummary()
        displayDict["caption"] = (
            f"Percent of the total time gaps which fall into the interval"
            f" between 2-14 hours, in {f} band(s)."
        )
        displayDict["order"] = filterorders[f]
        bundleList.append(
            mb.MetricBundle(
                m2,
                healpixslicer,
                constraint=filtersqls[f],
                info_label=filterinfo_label[f],
                runName=runName,
                summaryMetrics=summaryMetrics,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                displayDict=displayDict,
            )
        )

        m3 = metrics.TgapsPercentMetric(
            minTime=14.0 / 24.0,
            maxTime=(14.0 / 24 + 1.0),
            allGaps=False,
            metricName="TgapsPercent_1day",
        )
        displayDict["caption"] = (
            f"Percent of the total time gaps which fall into the interval around 1 day,"
            f" in {f} band(s)."
        )
        displayDict["order"] = filterorders[f]
        bundleList.append(
            mb.MetricBundle(
                m3,
                healpixslicer,
                constraint=filtersqls[f],
                info_label=filterinfo_label[f],
                runName=runName,
                summaryMetrics=summaryMetrics,
                plotDict=plotDict,
                plotFuncs=plotFuncs,
                displayDict=displayDict,
            )
        )

    # Presto KNe metric
    displayDict["group"] = "Variables/Transients"
    displayDict["subgroup"] = "Presto KNe"
    displayDict["caption"] = "Probability of detecting and classifying a KNe"
    prestoslicer = maf.generatePrestoPopSlicer(skyregion="extragalactic")
    metric = maf.PrestoColorKNePopMetric(
        skyregion="extragalactic", metricName="PrestoKNe"
    )
    summaryMetrics_kne = [maf.MedianMetric(), maf.SumMetric()]
    bundleList.append(
        maf.MetricBundle(
            metric,
            prestoslicer,
            None,
            runName=runName,
            displayDict=displayDict,
            summaryMetrics=summaryMetrics_kne,
        )
    )

    # XRB metric
    displayDict["subgroup"] = "XRB"
    displayDict["order"] = 0
    displayDict["caption"] = "Number or characterization of XRBs."
    n_events = 10000
    xrbslicer = maf.generateXRBPopSlicer(n_events=n_events)
    metric = maf.XRBPopMetric(outputLc=False)
    xrb_summaryMetrics = [
        maf.SumMetric(metricName="Total detected"),
        maf.CountMetric(metricName="Total lightcurves in footprint"),
        maf.CountMetric(metricName="Total lightcurves on sky", maskVal=0),
        maf.MeanMetric(metricName="Fraction detected in footprint"),
        maf.MeanMetric(maskVal=0, metricName="Fraction detected of total"),
        maf.MedianMetric(metricName="Median"),
    ]

    bundleList.append(
        maf.MetricBundle(
            metric,
            xrbslicer,
            "",
            runName=runName,
            summaryMetrics=xrb_summaryMetrics,
            displayDict=displayDict,
        )
    )

    #########################
    #########################
    # Galactic Plane - TVS/MW
    #########################
    #########################

    displayDict = {"group": "Galactic Plane", "subgroup": ""}

    footprint_summaries = [metrics.SumMetric()]
    footprint_plotDicts = {"percentileClip": 95}
    filter_summaries = [
        metrics.MeanMetric(),
        metrics.MedianMetric(),
        metrics.RmsMetric(),
        metrics.AreaThresholdMetric(lower_threshold=0.8),
    ]
    filter_plotdicts = {"colorMin": 0, "colorMax": 2, "xMin": 0, "xMax": 5}
    timescale_summaries = [
        metrics.SumMetric(),
        metrics.MedianMetric(),
        metrics.AreaThresholdMetric(lower_threshold=0.5),
    ]
    timescale_plotdicts = {"colorMin": 0, "colorMax": 1, "xMin": 0, "xMax": 1}

    galactic_plane_map_keys = maps.galplane_priority_map(nside=64, get_keys=True)
    science_maps = [
        s.replace("galplane_priority_", "").split(":")[0]
        for s in galactic_plane_map_keys
        if "sum" in s
    ]

    slicer = slicers.HealpixSlicer(nside=64, useCache=False)
    sql = None
    bundles = {}
    for m in science_maps:
        displayDict["subgroup"] = m
        displayDict[
            "caption"
        ] = f"Footprint relevant for {m} galactic plane science case."
        footprintmetric = metrics.GalPlaneFootprintMetric(science_map=m)
        bundles[f"{m} footprint"] = mb.MetricBundle(
            footprintmetric,
            slicer,
            sql,
            plotDict=footprint_plotDicts,
            runName=runName,
            summaryMetrics=footprint_summaries,
            displayDict=displayDict,
        )
        displayDict[
            "caption"
        ] = f"Filter balance in region of {m}  galactic plane science case."
        filtermetric = metrics.GalPlaneTimePerFilterMetric(science_map=m)
        bundles[f"{m} filter"] = mb.MetricBundle(
            filtermetric,
            slicer,
            sql,
            plotDict=filter_plotdicts,
            runName=runName,
            summaryMetrics=filter_summaries,
            displayDict=displayDict,
        )
        displayDict[
            "caption"
        ] = f"Timescale intervals in region of {m} galactic plane science case."
        visit_timescalesmetric = metrics.GalPlaneVisitIntervalsTimescaleMetric(
            science_map=m
        )
        bundles[f"{m} visit intervals"] = mb.MetricBundle(
            visit_timescalesmetric,
            slicer,
            sql,
            plotDict=timescale_plotdicts,
            runName=runName,
            summaryMetrics=timescale_summaries,
            displayDict=displayDict,
        )
        displayDict[
            "caption"
        ] = f"Seasonal coverage in region of {m} galactic plane science case."
        season_timescalemetric = metrics.GalPlaneSeasonGapsTimescaleMetric(
            science_map=m
        )
        bundles[f"{m} season gaps"] = mb.MetricBundle(
            season_timescalemetric,
            slicer,
            sql,
            plotDict=timescale_plotdicts,
            runName=runName,
            summaryMetrics=timescale_summaries,
            displayDict=displayDict,
        )

    bundleList += list(bundles.values())

    #########################
    #########################
    # Milky Way
    #########################
    #########################

    displayDict = {"group": "Milky Way", "subgroup": ""}

    displayDict["subgroup"] = "N stars"
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    sum_stats = [metrics.SumMetric(metricName="Total N Stars, crowding")]
    for f in filterlist:
        stellar_map = maps.StellarDensityMap(filtername=f)
        displayDict["order"] = filterorders[f]
        displayDict["caption"] = (
            "Number of stars in %s band with an measurement uncertainty due to crowding "
            "of less than 0.2 mag" % f
        )
        # Configure the NstarsMetric - note 'filtername' refers to the filter in which to evaluate crowding
        metric = metrics.NstarsMetric(
            crowding_error=0.2,
            filtername=f,
            ignore_crowding=False,
            maps=[],
        )
        plotDict = {"nTicks": 5, "logScale": True, "colorMin": 100}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            filtersqls[f],
            runName=runName,
            summaryMetrics=sum_stats,
            plotFuncs=subsetPlots,
            plotDict=plotDict,
            displayDict=displayDict,
            mapsList=[stellar_map],
        )
        bundleList.append(bundle)

    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    sum_stats = [metrics.SumMetric(metricName="Total N Stars, no crowding")]
    for f in filterlist:
        stellar_map = maps.StellarDensityMap(filtername=f)
        displayDict["order"] = filterorders[f]
        displayDict["caption"] = (
            "Number of stars in %s band with an measurement uncertainty "
            "of less than 0.2 mag, not considering crowding" % f
        )
        # Configure the NstarsMetric - note 'filtername' refers to the filter in which to evaluate crowding
        metric = metrics.NstarsMetric(
            crowding_error=0.2,
            filtername=f,
            ignore_crowding=True,
            metricName="Nstars_no_crowding",
            maps=[],
        )
        plotDict = {"nTicks": 5, "logScale": True, "colorMin": 100}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            filtersqls[f],
            runName=runName,
            summaryMetrics=sum_stats,
            plotFuncs=subsetPlots,
            plotDict=plotDict,
            displayDict=displayDict,
            mapsList=[stellar_map],
        )
        bundleList.append(bundle)

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
    metric = metrics.BDParallaxMetric(mags=l7_bd_mags, metricName="Brown Dwarf, L7")
    sql = ""
    plotDict = {}
    bundleList.append(
        mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            plotDict=plotDict,
            summaryMetrics=sum_stats,
            displayDict=displayDict,
            runName=runName,
        )
    )

    l4_bd_mags = {"i": 18.35, "z": 16.68, "y": 15.66}
    displayDict["caption"] = (
        f"The expected maximum distance at which an L4 brown dwarf with magnitude {l4_bd_mags} "
        f"would have a parallax SNR of 10.0. The summary statistic represents the total volume enclosed "
        f"by the result of this metric (BDParallaxMetric)."
    )
    metric = metrics.BDParallaxMetric(mags=l4_bd_mags, metricName="Brown Dwarf, L4")
    bundleList.append(
        mb.MetricBundle(
            metric,
            healpixslicer,
            sql,
            plotDict=plotDict,
            summaryMetrics=sum_stats,
            displayDict=displayDict,
            runName=runName,
        )
    )

    displayDict["subgroup"] = "Young Stellar Objects"
    displayDict["caption"] = (
        "The number of expected Young Stellar Objects with age t<10 Myr and "
        "mass >0.3 solar masses, using coadded depths in g, r and i bands."
    )
    # The underlying dustmap is nside=64, but can be resampled
    nside_yso = 64
    dustmap3d = maps.DustMap3D(nside=nside_yso)
    sql = ""
    # Let's plug in the magnitudes for one type
    metric = maf.mafContrib.NYoungStarsMetric(nside=nside_yso)
    slicer = maf.slicers.HealpixSlicer(nside=nside_yso, useCache=False)
    summaryStats = [maf.metrics.SumMetric()]
    plotDict = {"logScale": True, "colorMin": 1}
    bundleList.append(
        maf.metricBundles.MetricBundle(
            metric,
            slicer,
            sql,
            mapsList=[dustmap3d],
            plotDict=plotDict,
            summaryMetrics=summaryStats,
            runName=runName,
            displayDict=displayDict,
        )
    )

    ## Local Volume Dwarf Satellites
    displayDict["group"] = "Local Volume"
    displayDict["subgroup"] = "LV dwarf satellites"
    displayDict["order"] = 0
    # First the known dwarf satellite galaxies
    displayDict["caption"] = (
        "Predicted magnitude detection limit (including stellar density and "
        "star-galaxy separation) at the location of known LV dwarf galaxies. "
    )
    i_starMap = maf.maps.StellarDensityMap(filtername="i")
    lv_slicer = maf.mafContrib.generateKnownLVDwarfSlicer()
    lv_metric = maf.mafContrib.LVDwarfsMetric()
    sqlconstraint = '(filter = "i" OR filter = "g")'
    info_label = "gi"
    cutoff = -6.4
    summary_metrics = [
        maf.metrics.CountBeyondThreshold(
            lower_threshold=cutoff, metricName=f"Total detected {cutoff}"
        ),
        maf.metrics.CountBeyondThreshold(
            lower_threshold=-7.0, metricName="Total detected -7.0"
        ),
    ]
    plotDict = {"nTicks": 7}
    bundle = maf.MetricBundle(
        lv_metric,
        lv_slicer,
        sqlconstraint,
        mapsList=[i_starMap],
        summaryMetrics=summary_metrics,
        info_label=info_label,
        displayDict=displayDict,
        plotDict=plotDict,
    )
    bundleList.append(bundle)

    displayDict["order"] += 1
    displayDict["caption"] = (
        "Predicted magnitude detection limit (including stellar density and "
        "star-galaxy separation), over the whole sky to a distance of 4 Mpc."
    )
    lv_metric2 = maf.mafContrib.LVDwarfsMetric(
        distlim=4.0 * u.Mpc
    )  # for a distance limit, healpix map
    dustmap = maf.maps.DustMap(nside=32)
    lv_healpix_slicer = maf.slicers.HealpixSlicer(nside=32, useCache=False)
    summary_area = maf.AreaThresholdMetric(
        lower_threshold=cutoff, metricName=f"Area M_v>{cutoff}"
    )
    bundle = maf.MetricBundle(
        lv_metric2,
        lv_healpix_slicer,
        sqlconstraint,
        mapsList=[i_starMap, dustmap],
        info_label=info_label,
        displayDict=displayDict,
        summaryMetrics=summary_area,
        plotDict=plotDict,
    )
    bundleList.append(bundle)

    displayDict["order"] += 1
    displayDict["caption"] = (
        "Predicted magnitude detection limit (including stellar density and "
        "star-galaxy separation), over the southern celestial pole,"
        "to a distance of 0.1 Mpc."
    )
    sqlconstraint = '(filter = "i" OR filter = "g") and fieldDec < -60'
    info_label = "gi SCP"
    lv_metric3 = maf.mafContrib.LVDwarfsMetric(
        distlim=0.1 * u.Mpc
    )  # for a distance limit, healpix map
    summary_area = maf.metrics.AreaThresholdMetric(
        lower_threshold=0.0, metricName="Area M_v>0.0"
    )
    summary_median = maf.metrics.MedianMetric()
    bundle = maf.MetricBundle(
        lv_metric3,
        lv_healpix_slicer,
        sqlconstraint,
        mapsList=[i_starMap, dustmap],
        info_label=info_label,
        displayDict=displayDict,
        plotDict=plotDict,
        summaryMetrics=[summary_area, summary_median],
    )
    bundleList.append(bundle)

    #########################
    #########################
    # Scaling numbers
    #########################
    #########################

    displayDict = {"group": "Scaling Numbers", "subgroup": ""}
    displayDict["subgroup"] = "N gals"
    sql = 'filter="i"'
    metric = metrics.NgalScaleMetric()
    # galaxy counting uses dustmap
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    displayDict["caption"] = (
        f"Approximate number of resolvable galaxies in i band, scaled by the "
        f"coadded depth and median seeing. A dust and magnitude cut has been applied."
    )
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        runName=runName,
        summaryMetrics=[metrics.SumMetric()],
        plotFuncs=subsetPlots,
        plotDict=plotDict,
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    displayDict["subgroup"] = "Lightcurve Pts"
    metric = metrics.NlcPointsMetric(nside=nside)
    # NlcPoints metric uses star density maps
    slicer = slicers.HealpixSlicer(nside=nside, useCache=False)
    displayDict["caption"] = (
        f"Approximate number of expected stellar measurements (nstars * nobs) "
        f"in all filters, where the limiting magnitude for at least 10 visits "
        f"is fainter than 21st magnitude, using a TRILEGAL stellar density map."
    )
    bundle = mb.MetricBundle(
        metric,
        slicer,
        None,
        runName=runName,
        summaryMetrics=[metrics.SumMetric()],
        plotFuncs=subsetPlots,
        plotDict=plotDict,
        displayDict=displayDict,
    )
    bundleList.append(bundle)

    # Set the runName for all bundles and return the bundleDict.
    for b in bundleList:
        b.setRunName(runName)
    bundleDict = mb.makeBundlesDictFromList(bundleList)

    return bundleDict
