"""Evaluate some bulk properties of the sky coverage
"""

__all__ = ("meanRADec", "eastWestBias")

import numpy as np

import rubin_sim.maf.metric_bundles as mb
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers

from .col_map_dict import col_map_dict


def meanRADec(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None):
    """Plot the range of RA/Dec as a function of night.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'night<365')
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    group = "RA Dec coverage"

    subgroup = "All visits"
    if extraInfoLabel is not None:
        subgroup = extraInfoLabel

    displayDict = {"group": group, "subgroup": subgroup, "order": 0}

    ra_metrics = [
        metrics.MeanAngleMetric(colmap["ra"]),
        metrics.FullRangeAngleMetric(colmap["ra"]),
    ]
    dec_metrics = [
        metrics.MeanMetric(colmap["dec"]),
        metrics.MinMetric(colmap["dec"]),
        metrics.MaxMetric(colmap["dec"]),
    ]
    for m in ra_metrics:
        slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1)
        if not colmap["raDecDeg"]:
            plotDict = {"y_min": np.radians(-5), "y_max": np.radians(365)}
        else:
            plotDict = {"y_min": -5, "y_max": 365}
        bundle = mb.MetricBundle(
            m,
            slicer,
            extraSql,
            info_label=extraInfoLabel,
            display_dict=displayDict,
            plot_dict=plotDict,
        )
        bundleList.append(bundle)

    for m in dec_metrics:
        slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1)
        bundle = mb.MetricBundle(m, slicer, extraSql, info_label=extraInfoLabel, display_dict=displayDict)
        bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)


def eastWestBias(colmap=None, runName="opsim", extraSql=None, extraInfoLabel=None):
    """Plot the number of observations to the east vs to the west, per night.

    Parameters
    ----------
    colmap : `dict`, optional
        A dictionary with a mapping of column names.
    runName : `str`, optional
        The name of the simulated survey.
    extraSql : `str`, optional
        Additional constraint to add to any sql constraints (e.g. 'night<365')
    extraInfoLabel : `str`, optional
        Additional info_label to add before any below (i.e. "WFD").

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """
    if colmap is None:
        colmap = col_map_dict()
    bundleList = []

    group = "East vs West"

    subgroup = "All visits"
    if extraInfoLabel is not None:
        subgroup = extraInfoLabel

    displayDict = {"group": group, "subgroup": subgroup, "order": 0}

    eastvswest = 180
    if not colmap["raDecDeg"]:
        eastvswest = np.radians(eastvswest)

    displayDict["caption"] = "Number of visits per night that occur with azimuth <= 180."
    if extraSql is not None:
        displayDict["caption"] += " With additional sql constraint %s." % extraSql
    metric = metrics.CountMetric(colmap["night"], metric_name="Nvisits East")
    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1)
    sql = "%s <= %f" % (colmap["az"], eastvswest)
    if extraSql is not None:
        sql = "(%s) and (%s)" % (sql, extraSql)
    plotDict = {"color": "orange", "label": "East"}
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        info_label=extraInfoLabel,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundleList.append(bundle)

    displayDict["caption"] = "Number of visits per night that occur with azimuth > 180."
    if extraSql is not None:
        displayDict["caption"] += " With additional sql constraint %s." % extraSql
    metric = metrics.CountMetric(colmap["night"], metric_name="Nvisits West")
    slicer = slicers.OneDSlicer(slice_col_name=colmap["night"], bin_size=1)
    sql = "%s > %f" % (colmap["az"], eastvswest)
    if extraSql is not None:
        sql = "(%s) and (%s)" % (sql, extraSql)
    plotDict = {"color": "blue", "label": "West"}
    bundle = mb.MetricBundle(
        metric,
        slicer,
        sql,
        info_label=extraInfoLabel,
        display_dict=displayDict,
        plot_dict=plotDict,
    )
    bundleList.append(bundle)

    # Set the run_name for all bundles and return the bundleDict.
    for b in bundleList:
        b.set_run_name(runName)
    return mb.make_bundles_dict_from_list(bundleList)
