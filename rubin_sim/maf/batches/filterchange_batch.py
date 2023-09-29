__all__ = ("filtersPerNight", "filtersWholeSurvey")

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict
from .common import standard_summary


def setupMetrics(colmap, wholesurvey=False):
    metric_list = []
    caption_list = []
    # Number of filter changes (per slice - either whole survey or X nights)
    if wholesurvey:
        metric_list.append(
            metrics.NChangesMetric(
                col=colmap["filter"],
                order_by=colmap["mjd"],
                metric_name="Total Filter Changes",
            )
        )
    else:
        metric_list.append(
            metrics.NChangesMetric(
                col=colmap["filter"],
                order_by=colmap["mjd"],
                metric_name="Filter Changes",
            )
        )
    caption_list.append("Total filter changes ")
    # Minimum time between filter changes
    metric_list.append(
        metrics.MinTimeBetweenStatesMetric(change_col=colmap["filter"], time_col=colmap["mjd"])
    )
    caption_list.append("Minimum time between filter changes ")
    # Number of filter changes faster than 10 minutes
    metric_list.append(
        metrics.NStateChangesFasterThanMetric(change_col=colmap["filter"], time_col=colmap["mjd"], cutoff=10)
    )
    caption_list.append("Number of filter changes faster than 10 minutes ")
    # Number of filter changes faster than 20 minutes
    metric_list.append(
        metrics.NStateChangesFasterThanMetric(change_col=colmap["filter"], time_col=colmap["mjd"], cutoff=20)
    )
    caption_list.append("Number of filter changes faster than 20 minutes ")
    # Maximum number of filter changes faster than 10 minutes within slice
    metric_list.append(
        metrics.MaxStateChangesWithinMetric(change_col=colmap["filter"], time_col=colmap["mjd"], timespan=10)
    )
    caption_list.append("Max number of filter  changes within a window of 10 minutes ")
    # Maximum number of filter changes faster than 20 minutes within slice
    metric_list.append(
        metrics.MaxStateChangesWithinMetric(change_col=colmap["filter"], time_col=colmap["mjd"], timespan=20)
    )
    caption_list.append("Max number of filter changes within a window of 20 minutes ")
    return metric_list, caption_list


def filtersPerNight(colmap=None, runName="opsim", nights=1, extraSql=None, extraInfoLabel=None):
    """Generate a set of metrics measuring the number and rate of filter
    changes over a given span of nights.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    nights : `int`, optional
        Size of night bin to use when calculating metrics.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    # Set up sql and info_label, if passed any additional information.
    sql = ""
    info_label = "Per"
    if nights == 1:
        info_label += " Night"
    else:
        info_label += " %s Nights" % nights
    metacaption = info_label.lower()
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraInfoLabel is None:
            info_label += " %s" % extraSql
            metacaption += ", with %s selection" % extraSql
    if extraInfoLabel is not None:
        info_label += " %s" % extraInfoLabel
        metacaption += ", %s only" % extraInfoLabel
    metacaption += "."

    displayDict = {"group": "Filter Changes", "subgroup": info_label}
    summaryStats = standard_summary()

    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=nights)
    metricList, captionList = setupMetrics(colmap)
    for m, caption in zip(metricList, captionList):
        displayDict["caption"] = caption + metacaption
        bundle = mb.MetricBundle(
            m,
            slicer,
            sql,
            run_name=runName,
            info_label=info_label,
            display_dict=displayDict,
            summary_metrics=summaryStats,
        )
        bundleList.append(bundle)

    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def filtersWholeSurvey(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None):
    """Generate a set of metrics measuring the number and rate of filter
    changes over the entire survey.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints.
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    # Set up sql and info_label, if passed any additional information.
    sql = ""
    info_label = "Whole Survey"
    metacaption = "over the whole survey"
    if (extraSql is not None) and (len(extraSql) > 0):
        sql = extraSql
        if extraInfoLabel is None:
            info_label += " %s" % extraSql
            metacaption += ", with %s selction" % extraSql
    if extraInfoLabel is not None:
        info_label += " %s" % extraInfoLabel
        metacaption += ", %s only" % (extraInfoLabel)
    metacaption += "."

    displayDict = {"group": "Filter Changes", "subgroup": info_label}

    slicer = slicers.UniSlicer()
    metricList, captionList = setupMetrics(colmap)
    for m, caption in zip(metricList, captionList):
        displayDict["caption"] = caption + metacaption
        bundle = mb.MetricBundle(
            m,
            slicer,
            sql,
            run_name=runName,
            info_label=info_label,
            display_dict=displayDict,
        )
        bundleList.append(bundle)

    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
