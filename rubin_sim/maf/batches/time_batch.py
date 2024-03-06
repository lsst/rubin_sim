"""Sets of metrics to look at time between visits/pairs, etc.
"""

__all__ = ("intraNight", "interNight", "timeGaps", "seasons")

import numpy as np

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.plots as plots
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import filter_list, standard_summary


def intraNight(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
    display_group="IntraNight",
    subgroup="Pairs",
):
    """Generate a set of statistics about the pair/triplet/etc. rate
    within a night.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        Nside for the healpix slicer.
    extraSql : `str` or None, optional
        Additional sql constraint to apply to all metrics.
    extraInfoLabel : `str` or None, optional
        Additional info_label to apply to all results.
    slicer : `maf.BaseSlicer` or None
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    info_label = extraInfoLabel
    if extraSql is not None and len(extraSql) > 0:
        if info_label is None:
            info_label = extraSql

    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]

    bundleList = []
    standardStats = standard_summary()

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, lat_col=decCol, lon_col=raCol, lat_lon_deg=degrees)

    # Look for the fraction of visits in gri where there are pairs
    # within dtMin/dtMax.
    displayDict = {
        "group": display_group,
        "subgroup": subgroup,
        "caption": None,
        "order": 0,
    }
    if extraSql is not None and len(extraSql) > 0:
        sql = '(%s) and (filter="g" or filter="r" or filter="i")' % extraSql
    else:
        sql = 'filter="g" or filter="r" or filter="i"'
    md = "gri"
    if info_label is not None:
        md += " " + info_label
    dtMin = 10.0
    dtMax = 60.0
    metric = metrics.PairFractionMetric(
        mjd_col=colmap["mjd"],
        min_gap=dtMin,
        max_gap=dtMax,
        metric_name="Fraction of visits in pairs (%.0f-%.0f min)" % (dtMin, dtMax),
    )
    displayDict["caption"] = (
        "Fraction of %s visits that have a paired visit"
        "between %.1f and %.1f minutes away. "
        % (
            md,
            dtMin,
            dtMax,
        )
    )
    displayDict["caption"] += "If all visits were in pairs, this fraction would be 1."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        info_label=md,
        summary_metrics=standardStats,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    dtMin = 20.0
    dtMax = 90.0
    metric = metrics.PairFractionMetric(
        mjd_col=colmap["mjd"],
        min_gap=dtMin,
        max_gap=dtMax,
        metric_name="Fraction of visits in pairs (%.0f-%.0f min)" % (dtMin, dtMax),
    )
    displayDict["caption"] = (
        "Fraction of %s visits that have a paired visit"
        "between %.1f and %.1f minutes away. "
        % (
            md,
            dtMin,
            dtMax,
        )
    )
    displayDict["caption"] += "If all visits were in pairs, this fraction would be 1."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        info_label=md,
        summary_metrics=standardStats,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Look at the fraction of visits which have another visit within dtMax gri.
    dtMax = 60.0
    metric = metrics.NRevisitsMetric(
        mjd_col=colmap["mjd"],
        d_t=dtMax,
        normed=True,
        metric_name="Fraction of visits with a revisit < %.0f min" % dtMax,
    )
    displayDict["caption"] = "Fraction of %s visits that have another visit " "within %.1f min. " % (
        md,
        dtMax,
    )
    displayDict["caption"] += "If all visits were in pairs (only), this fraction would be 0.5."
    displayDict["order"] += 1
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        info_label=md,
        summary_metrics=standardStats,
        display_dict=displayDict,
    )
    bundleList.append(bundle)

    # Intranight gap map, all filters. Returns value in hours.
    metric = metrics.IntraNightGapsMetric(
        metric_name="Median Intra-Night Gap",
        mjd_col=colmap["mjd"],
        reduce_func=np.median,
    )
    displayDict["caption"] = "Median gap between consecutive visits within a night, all bands"
    if info_label is None or len(info_label) == 0:
        displayDict["caption"] += ", all visits."
    else:
        displayDict["caption"] += ", %s." % info_label
    displayDict["order"] += 1
    plotDict = {"percentile_clip": 95}
    bundle = mb.MetricBundle(
        metric,
        slicer,
        extraSql,
        info_label=info_label,
        display_dict=displayDict,
        plot_dict=plotDict,
        summary_metrics=standardStats,
    )
    bundleList.append(bundle)

    # Max Timespans (in each night)
    # Run in all filters, u+g, g+r, r+i, i+z and z+y filters,
    # and individual filters

    metric = metrics.NightTimespanMetric(percentile=75, night_col=colmap["night"], mjd_col=colmap["mjd"])
    displayDict["caption"] = "75th percentile value of the maximum intra-night timespan, on each night"
    # individual and all filters
    filterlist, colors, orders, sqls, info_labels = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=info_label
    )
    for f in filterlist:
        if info_labels[f] is None or len(info_labels[f]) == 0:
            displayDict["caption"] += ", all visits."
        else:
            displayDict["caption"] += ", %s." % info_labels[f]
        displayDict["order"] = orders[f]
        plotDict = {"percentile_clip": 98}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_labels[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)
    # subsets of adjacent filters
    filtersubsets = {
        "ug": '(filter = "u" or filter = "g")',
        "gr": '(filter = "g" or filter = "r")',
        "ri": '(filter = "r" or filter = "i")',
        "iz": '(filter = "i" or filter = "z")',
        "zy": '(filter = "z" or filter = "y")',
    }
    sqls = [extraSql]
    if extraSql is not None and len(extraSql) > 0:
        for fi in filtersubsets:
            sqls.append(f"{extraSql} and {filtersubsets[fi]}")
    else:
        for fi in filtersubsets:
            sqls.append(f"{filtersubsets[fi]}")
    md = [info_label]
    if info_label is not None:
        for fi in filtersubsets:
            md.append(f"{info_label} {fi} bands")
    else:
        for fi in filtersubsets:
            md.append(f"{fi} bands")

    for sql, info in zip(sqls, md):
        if info_label is None or len(info_label) == 0:
            displayDict["caption"] += ", all visits."
        else:
            displayDict["caption"] += ", %s." % info_label
        displayDict["order"] += 1
        plotDict = {"percentile_clip": 98}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sql,
            info_label=info,
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # Histogram the number of visits per night.
    countbins = np.arange(0, 10, 1)
    metric = metrics.NVisitsPerNightMetric(
        night_col=colmap["night"], bins=countbins, metric_name="NVisitsPerNight"
    )
    plotDict = {"bins": countbins, "xlabel": "Number of visits each night"}
    displayDict["caption"] = "Histogram of the number of visits in each night, per point on the sky"
    if info_label is None or len(info_label) == 0:
        displayDict["caption"] += ", all proposals."
    else:
        displayDict["caption"] += ", %s." % info_label
    displayDict["order"] = 0
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(
        metric,
        slicer,
        extraSql,
        plot_dict=plotDict,
        display_dict=displayDict,
        info_label=info_label,
        plot_funcs=[plotFunc],
    )
    bundleList.append(bundle)

    # Histogram of the time between revisits (all filters) within two hours.
    binMin = 0
    binMax = 120.0
    bin_size = 5.0
    bins_metric = np.arange(binMin / 60.0 / 24.0, (binMax + bin_size) / 60.0 / 24.0, bin_size / 60.0 / 24.0)
    bins_plot = bins_metric * 24.0 * 60.0
    metric = metrics.TgapsMetric(bins=bins_metric, times_col=colmap["mjd"], metric_name="DeltaT Histogram")
    plotDict = {"bins": bins_plot, "xlabel": "dT (minutes)"}
    displayDict["caption"] = (
        "Histogram of the time between consecutive visits to a given point "
        "on the sky, considering visits between %.1f and %.1f minutes" % (binMin, binMax)
    )
    if info_label is None or len(info_label) == 0:
        displayDict["caption"] += ", all proposals."
    else:
        displayDict["caption"] += ", %s." % info_label
    displayDict["order"] += 1
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(
        metric,
        slicer,
        extraSql,
        plot_dict=plotDict,
        display_dict=displayDict,
        info_label=info_label,
        plot_funcs=[plotFunc],
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def interNight(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
    display_group="InterNight",
    subgroup="Night gaps",
):
    """Generate a set of statistics about the spacing between nights
    with observations.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        Nside for the healpix slicer.
    extraSql : `str` or None, optional
        Additional sql constraint to apply to all metrics.
    extraInfoLabel : `str` or None, optional
        Additional info_label to use for all outputs.
    slicer : `maf.BaseSlicer` or None
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    bundleList = []

    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, lat_col=decCol, lon_col=raCol, lat_lon_deg=degrees)

    displayDict = {
        "group": display_group,
        "subgroup": subgroup,
        "caption": None,
        "order": 0,
    }

    # Histogram of the number of nights between visits.
    bins = np.arange(1, 20.5, 1)
    metric = metrics.NightgapsMetric(bins=bins, night_col=colmap["night"], metric_name="DeltaNight Histogram")
    plotDict = {"bins": bins, "xlabel": "dT (nights)"}
    displayDict["caption"] = (
        "Histogram of the number of nights between consecutive visits to a "
        "given point on the sky, considering separations between %d and %d" % (bins.min(), bins.max())
    )
    if info_label["all"] is None or len(info_label["all"]) == 0:
        displayDict["caption"] += ", all proposals."
    else:
        displayDict["caption"] += ", %s." % info_label["all"]
    plotFunc = plots.SummaryHistogram()
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sqls["all"],
        plot_dict=plotDict,
        display_dict=displayDict,
        info_label=info_label["all"],
        plot_funcs=[plotFunc],
    )
    bundleList.append(bundle)

    standardStats = standard_summary()

    # Look at the total number of unique nights with visits
    metric = metrics.CountUniqueMetric(col=colmap["night"], metric_name="N Unique Nights")
    displayDict["caption"] = "Number of unique nights with visits"
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sqls["all"],
        info_label=info_label["all"],
        display_dict=displayDict,
        plot_dict={"color_min": 0, "color_max": 500},
        summary_metrics=standardStats,
    )
    bundleList.append(bundle)

    # Median inter-night gap (each and all filters)
    metric = metrics.InterNightGapsMetric(
        metric_name="Median Inter-Night Gap",
        mjd_col=colmap["mjd"],
        reduce_func=np.median,
    )
    for f in filterlist:
        displayDict["caption"] = "Median gap between nights with observations, %s." % info_label[f]
        displayDict["order"] = orders[f]
        plotDict = {"color": colors[f], "percentile_clip": 95.0}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # 20th percentile inter-night gap (each and all filters) -
    # aimed at active rolling years
    def rfunc(simdata):
        return np.percentile(simdata, 20)

    metric = metrics.InterNightGapsMetric(
        metric_name="20thPercentile Inter-Night Gap",
        mjd_col=colmap["mjd"],
        reduce_func=rfunc,
    )
    for f in filterlist:
        displayDict["caption"] = "20th percentile gap between nights with observations, %s." % info_label[f]
        displayDict["order"] = orders[f]
        plotDict = {"color": colors[f], "percentile_clip": 95.0}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # Maximum inter-night gap (in each and all filters).
    metric = metrics.InterNightGapsMetric(
        metric_name="Max Inter-Night Gap", mjd_col=colmap["mjd"], reduce_func=np.max
    )
    for f in filterlist:
        displayDict["caption"] = "Maximum gap between nights with observations, %s." % info_label[f]
        displayDict["order"] = orders[f]
        plotDict = {"color": colors[f], "percentile_clip": 95.0, "bin_size": 5}
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def timeGaps(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
    display_group="Time Coverage",
    subgroup=None,
):
    """Generate a set of statistics about the spacing
    between nights with observations.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        Nside for the healpix slicer.
    extraSql : `str`, optional
        Additional sql constraint to apply to all metrics.
    extraInfoLabel : `str`, optional
        Additional info_label to use for all outputs.
    slicer : `rubin_sim.maf.slicer.BaseSlicer`, optional
        Optionally use something other than a HealpixSlicer

    Returns
    -------
    bundle_dict : `dict` of `rubin_sim.maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    bundleList = []

    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, lat_col=decCol, lon_col=raCol, lat_lon_deg=degrees)

    displayDict = {
        "group": display_group,
        "subgroup": subgroup,
        "caption": None,
        "order": 0,
    }

    # Logarithmically spaced gaps from 30s to 5 years
    tMin = 30 / 60 / 60 / 24.0  # 30s
    tMax = 5 * 365.25  # 5 years
    tgaps = np.logspace(np.log10(tMin), np.log10(tMax), 100)

    for f in filterlist:
        m1 = metrics.TgapsMetric(bins=tgaps, all_gaps=False)
        plotDict = {
            "bins": tgaps,
            "xscale": "log",
            "y_min": 0,
            "figsize": (8, 6),
            "ylabel": "Number of observation pairs",
            "xlabel": "Time gap between pairs of visits (days)",
            "color": colors[f],
        }
        plotFuncs = [plots.SummaryHistogram()]
        displayDict["caption"] = (
            f"Summed Histogram of time between visits at each point in the sky, " f"in {f} band(s)."
        )
        displayDict["order"] = orders[f]
        bundleList.append(
            mb.MetricBundle(
                m1,
                slicer,
                constraint=sqls[f],
                info_label=info_label[f],
                run_name=runName,
                plot_dict=plotDict,
                plot_funcs=plotFuncs,
                display_dict=displayDict,
            )
        )

        gaps = [3.0, 7.0, 24.0]
        for gap in gaps:
            summary_stats = []
            summary_stats.append(
                metrics.AreaSummaryMetric(
                    area=18000, reduce_func=np.median, decreasing=True, metric_name="Median (top 18k)"
                )
            )

            summary_stats.append(
                metrics.AreaSummaryMetric(
                    area=18000, reduce_func=np.mean, decreasing=True, metric_name="Mean (top 18k)"
                )
            )

            summary_stats.append(metrics.MeanMetric())
            summary_stats.append(metrics.MedianMetric())

            m2 = metrics.GapsMetric(time_scale=gap, metric_name=f"Gaps at {gap: .1f} hr")
            plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
            plotDict = {"color_min": 0, "color": colors[f], "percentile_clip": 95}
            displayDict["subgroup"] = "Gaps Sampling"
            displayDict["caption"] = (
                f"Number of times the timescale of ~{gap: .1f} hours is sampled in {f} band(s)."
            )
            displayDict["order"] = orders[f]
            bundleList.append(
                mb.MetricBundle(
                    m2,
                    slicer,
                    constraint=sqls[f],
                    info_label=info_label[f],
                    run_name=runName,
                    summary_metrics=summary_stats,
                    plot_dict=plotDict,
                    plot_funcs=plotFuncs,
                    display_dict=displayDict,
                )
            )

    return mb.make_bundles_dict_from_list(bundleList)


def seasons(
    colmap=None,
    runName="opsim",
    nside=64,
    extraSql=None,
    extraInfoLabel=None,
    slicer=None,
):
    """Generate a set of statistics about the length and number of seasons.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    nside : `int`, optional
        Nside for the healpix slicer.
    extraSql : `str` or None, optional
        Additional sql constraint to apply to all metrics.
    extraInfoLabel : `str` or None, optional
        Additional info_label to use for all outputs.
    slicer : `maf.BaseSlicer` or None`
         Optionally use something other than a HealpixSlicer

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    bundleList = []

    # Set up basic all and per filter sql constraints.
    raCol = colmap["ra"]
    decCol = colmap["dec"]
    degrees = colmap["raDecDeg"]
    filterlist, colors, orders, sqls, info_label = filter_list(
        all=True, extra_sql=extraSql, extra_info_label=extraInfoLabel
    )

    if slicer is None:
        slicer = slicers.HealpixSlicer(nside=nside, lat_col=decCol, lon_col=raCol, lat_lon_deg=degrees)

    displayDict = {
        "group": "IntraSeason",
        "subgroup": "Season length",
        "caption": None,
        "order": 0,
    }

    standardStats = standard_summary()

    metric = metrics.SeasonLengthMetric(
        metric_name="Median Season Length", mjd_col=colmap["mjd"], reduce_func=np.median
    )
    for f in filterlist:
        displayDict["caption"] = "Median season length, %s." % info_label[f]
        displayDict["order"] = orders[f]
        maxS = 250
        if f == "all":
            minS = 90
        else:
            minS = 30
        plotDict = {
            "color": colors[f],
            "color_min": minS,
            "color_max": maxS,
            "x_min": minS,
            "x_max": maxS,
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # 80th percentile season length -
    # aimed at finding season length during rolling or long years
    def rfunc(simdata):
        return np.percentile(simdata, 80)

    metric = metrics.SeasonLengthMetric(
        metric_name="80thPercentile Season Length",
        mjd_col=colmap["mjd"],
        reduce_func=rfunc,
    )
    for f in filterlist:
        displayDict["caption"] = "80th percentile season length, %s." % info_label[f]
        displayDict["order"] = orders[f]
        maxS = 350
        if f == "all":
            minS = 90
        else:
            minS = 30
        plotDict = {
            "color": colors[f],
            "color_min": minS,
            "color_max": maxS,
            "x_min": minS,
            "x_max": maxS,
        }
        bundle = mb.MetricBundle(
            metric,
            slicer,
            sqls[f],
            info_label=info_label[f],
            display_dict=displayDict,
            plot_dict=plotDict,
            summary_metrics=standardStats,
        )
        bundleList.append(bundle)

    # Number of seasons
    metric = metrics.CampaignLengthMetric(
        metric_name="NSeasons",
        mjd_col=colmap["mjd"],
        exp_time_col=colmap["exptime"],
        min_exp_time=15,
    )
    displayDict["caption"] = "Number of seasons, any filter."
    displayDict["order"] = 0
    plotDict = {"color": "k", "color_min": 0, "color_max": 11, "x_min": 0, "x_max": 11}
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sqls["all"],
        info_label=info_label["all"],
        display_dict=displayDict,
        plot_dict=plotDict,
        summary_metrics=standardStats,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
